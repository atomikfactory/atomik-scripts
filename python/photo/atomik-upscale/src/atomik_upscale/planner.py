"""Target resolution parsing and multi-pass scale planning.

A 4x network asked for x6 has two bad options: one pass plus a Lanczos
*up*-scale (soft, wastes the network) or two passes to x16 (four times the
pixels anyone needed). The planner takes a third route: chain a 4x pass with a
cheap 2x pass to reach x8, then resample *down* to x6. Downscaling after
super-resolution is close to free and keeps every bit of detail the network
produced.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Long-edge targets. "minimum 4K" means the long edge reaches 3840.
PRESETS: dict[str, int] = {
    "1080p": 1920,
    "2k": 2560,
    "1440p": 2560,
    "4k": 3840,
    "5k": 5120,
    "6k": 6144,
    "8k": 7680,
    "12k": 11520,
    "16k": 15360,
}

_WXH = re.compile(r"^(\d{2,6})\s*[x×]\s*(\d{2,6})$", re.I)
_XN = re.compile(r"^x?(\d+(?:\.\d+)?)x?$", re.I)

MAX_OUTPUT_PIXELS = 800_000_000  # ~28000x28000; refuse beyond this


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Target:
    width: int
    height: int
    multiplier: float
    label: str


@dataclass(frozen=True)
class Step:
    model_id: str
    scale: int


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    target: Target
    net_scale: int          # product of the model scales
    final_resize: bool      # whether a Lanczos pass follows the chain

    @property
    def passes(self) -> int:
        return len(self.steps)

    def describe(self) -> str:
        if not self.steps:
            return f"resample only -> {self.target.width}x{self.target.height}"
        chain = " -> ".join(f"{s.model_id} x{s.scale}" for s in self.steps)
        tail = f" -> Lanczos to {self.target.width}x{self.target.height}" \
            if self.final_resize else ""
        return f"{chain} (x{self.net_scale}){tail}"


def parse_target(spec: str, src_w: int, src_h: int) -> Target:
    """Turn "x4" / "4k" / "3840x2160" / "5000" into concrete output pixels."""
    spec = str(spec).strip().lower()
    if not spec:
        raise PlanError("empty target")

    long_edge = max(src_w, src_h)

    m = _WXH.match(spec)
    if m:
        tw, th = int(m.group(1)), int(m.group(2))
        # Preserve aspect ratio and cover the requested box on both axes.
        mult = max(tw / src_w, th / src_h)
        return _from_multiplier(src_w, src_h, mult, f"{tw}x{th}")

    if spec in PRESETS:
        edge = PRESETS[spec]
        return _from_multiplier(src_w, src_h, edge / long_edge, spec.upper())

    m = _XN.match(spec)
    if m:
        val = float(m.group(1))
        if val <= 0:
            raise PlanError("scale must be positive")
        explicit_x = spec.startswith("x") or spec.endswith("x")
        # "x6" and "6" are multipliers; a bare "3840" is a target width.
        if explicit_x or val < 100:
            if val > 32:
                raise PlanError(f"scale x{val:g} is beyond the x32 ceiling")
            return _from_multiplier(src_w, src_h, val, f"x{val:g}")
        return _from_multiplier(src_w, src_h, val / src_w, f"{int(val)}px wide")

    raise PlanError(
        f"cannot read target {spec!r}. Use x2 / x4 / x6, a preset "
        f"({', '.join(sorted(PRESETS))}), 3840x2160, or a pixel width."
    )


def _from_multiplier(src_w: int, src_h: int, mult: float, label: str) -> Target:
    w = max(1, round(src_w * mult))
    h = max(1, round(src_h * mult))
    if w * h > MAX_OUTPUT_PIXELS:
        raise PlanError(
            f"{label} would produce {w}x{h} ({w * h / 1e6:.0f} MPix), over the "
            f"{MAX_OUTPUT_PIXELS / 1e6:.0f} MPix ceiling."
        )
    return Target(width=w, height=h, multiplier=mult, label=label)


def plan(target: Target, main_model: str, main_scale: int,
         chain_model: str, chain_scale: int = 2,
         *, max_passes: int = 3, allow_downscale_only: bool = True,
         cost_of=None) -> Plan:
    """Choose the cheapest chain of model passes that reaches the target.

    A pass is priced as its output pixel count times that model's measured
    relative cost, so a second heavy pass over an already-quadrupled image is
    correctly recognised as far dearer than two passes of the small 2x net.
    """
    m = target.multiplier

    if m <= 1.0:
        if not allow_downscale_only:
            raise PlanError("target is not larger than the source")
        return Plan(steps=(), target=target, net_scale=1, final_resize=True)

    # The chosen model always takes the first pass. It is the one that decides
    # the quality of the result, so it must see the original pixels rather than
    # another network's output -- and running it while the image is still small
    # is also the cheapest place to spend it.
    first = Step(main_model, main_scale)

    if main_scale >= m:
        return _finish((first,), target)

    options = sorted({main_scale, chain_scale}, reverse=True)
    best: tuple[float, int, float] | None = None
    best_steps: tuple[Step, ...] = (first,)

    def model_for(scale: int) -> str:
        return main_model if scale == main_scale else chain_model

    def weight_of(model_id: str) -> float:
        if cost_of is not None:
            return max(cost_of(model_id), 1e-6)
        from . import models as _models  # local import keeps this module standalone
        spec = _models.REGISTRY.get(model_id)
        return max(spec.cost_weight, 1e-6) if spec else 1.0

    def walk(seq: list[int], product: int, cost: float) -> None:
        nonlocal best, best_steps
        if product >= m:
            key = (cost, len(seq), product / m)
            if best is None or key < best:
                best = key
                best_steps = (first,) + tuple(Step(model_for(s), s) for s in seq)
            return
        if len(seq) + 1 >= max_passes:
            return
        for s in options:
            # A pass costs its output pixel count (in source-pixel units)
            # scaled by how expensive that particular network is. Without the
            # weight, a second pass of a heavy model looks as cheap as two
            # passes of the tiny one -- and it is ~30x dearer.
            out_px = float(product * s) ** 2
            walk(seq + [s], product * s, cost + weight_of(model_for(s)) * out_px)

    walk([], main_scale, weight_of(main_model) * float(main_scale) ** 2)

    if best is None:  # beyond max_passes: take the largest chain allowed
        extra = [main_scale] * (max_passes - 1)
        best_steps = (first,) + tuple(Step(model_for(s), s) for s in extra)

    return _finish(best_steps, target)


def _finish(steps: tuple[Step, ...], target: Target) -> Plan:

    net = 1
    for st in steps:
        net *= st.scale
    # Only resample if the chain does not land exactly on the target.
    exact = abs(net - target.multiplier) < 1e-6
    return Plan(steps=steps, target=target, net_scale=net,
                final_resize=not exact)


def intermediate_sizes(src_w: int, src_h: int, p: Plan) -> list[tuple[int, int]]:
    """Sizes after each pass — used for the memory warning and progress weights."""
    out, w, h = [], src_w, src_h
    for st in p.steps:
        w, h = w * st.scale, h * st.scale
        out.append((w, h))
    return out


def peak_megapixels(src_w: int, src_h: int, p: Plan) -> float:
    sizes = intermediate_sizes(src_w, src_h, p)
    if not sizes:
        return src_w * src_h / 1e6
    return max(w * h for w, h in sizes) / 1e6


def ceil_multiplier_for_4k(src_w: int, src_h: int) -> float:
    return max(1.0, PRESETS["4k"] / max(src_w, src_h))
