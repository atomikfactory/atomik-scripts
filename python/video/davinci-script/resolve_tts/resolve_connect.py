"""Locating, importing and validating the DaVinci Resolve scripting API.

Everything platform-specific about reaching Resolve lives here.  The lookup
order mirrors Blackmagic's own ``DaVinciResolveScript.py`` shim:

1. ``import DaVinciResolveScript`` straight off ``PYTHONPATH``.
2. The same import after adding ``RESOLVE_SCRIPT_API/Modules`` (or the
   platform default Modules directory) to ``sys.path``.
3. Loading ``fusionscript`` directly from ``RESOLVE_SCRIPT_LIB`` or the
   platform default library path.

Nothing here generates speech or touches a timeline; it only hands back a
validated :class:`ResolveContext`.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Optional

from . import ResolveError

__all__ = [
    "MIN_MAJOR_VERSION",
    "ResolveContext",
    "connect",
    "default_library_paths",
    "default_module_paths",
    "import_resolve_module",
    "probe_report",
]

LOGGER = logging.getLogger("resolve_tts.connect")

#: Oldest Resolve major version that has ``Project.GenerateSpeech``.
MIN_MAJOR_VERSION = 20

#: Message printed whenever ``scriptapp("Resolve")`` hands back ``None``.
#: Verified on 21.0.1: scriptapp returned None until "External scripting using"
#: was set to Local *and* Resolve was restarted.  The preference is stored as
#: ``System.Scripting.Mode`` (0 = none, 1 = local) in Resolve's ``config.dat``.
SCRIPTING_DISABLED_HINT = (
    "  Is DaVinci Resolve Studio running?\n"
    "  In Resolve: DaVinci Resolve -> Preferences -> System -> General ->\n"
    "  'External scripting using' must be set to 'Local', and Resolve must then\n"
    "  be RESTARTED (Resolve 21 does not pick the change up until it is)."
)


def _windows_program_data() -> str:
    return os.environ.get("PROGRAMDATA") or "C:\\ProgramData"


def default_module_paths() -> List[str]:
    """Directories that may contain ``DaVinciResolveScript.py``, best first."""
    paths: List[str] = []
    api = os.environ.get("RESOLVE_SCRIPT_API")
    if api:
        paths.append(os.path.join(api, "Modules"))

    if sys.platform.startswith("darwin"):
        paths.append(
            "/Library/Application Support/Blackmagic Design/DaVinci Resolve"
            "/Developer/Scripting/Modules"
        )
    elif sys.platform.startswith("win") or sys.platform.startswith("cygwin"):
        paths.append(
            os.path.join(
                _windows_program_data(),
                "Blackmagic Design",
                "DaVinci Resolve",
                "Support",
                "Developer",
                "Scripting",
                "Modules",
            )
        )
    elif sys.platform.startswith("linux"):
        paths.append("/opt/resolve/Developer/Scripting/Modules")
        paths.append("/home/resolve/Developer/Scripting/Modules")
    return paths


def default_library_paths() -> List[str]:
    """Full paths to ``fusionscript`` (``.dll`` / ``.so``), best first."""
    paths: List[str] = []
    library = os.environ.get("RESOLVE_SCRIPT_LIB")
    if library:
        paths.append(library)

    if sys.platform.startswith("darwin"):
        paths.append(
            "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents"
            "/Libraries/Fusion/fusionscript.so"
        )
    elif sys.platform.startswith("win") or sys.platform.startswith("cygwin"):
        paths.append(
            "C:\\Program Files\\Blackmagic Design\\DaVinci Resolve\\fusionscript.dll"
        )
    elif sys.platform.startswith("linux"):
        paths.append("/opt/resolve/libs/Fusion/fusionscript.so")
        paths.append("/home/resolve/libs/Fusion/fusionscript.so")
    return paths


def _load_extension(module_name: str, file_path: str) -> ModuleType:
    """Import a native extension module from an explicit file path."""
    loader = importlib.machinery.ExtensionFileLoader(module_name, file_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError("could not build an import spec for %s" % file_path)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def import_resolve_module() -> ModuleType:
    """Import Blackmagic's scripting module, trying every documented location.

    Raises:
        ResolveError: The module could not be found or imported anywhere.
    """
    attempts: List[str] = []

    try:
        module = importlib.import_module("DaVinciResolveScript")
        LOGGER.debug("Imported DaVinciResolveScript from PYTHONPATH: %s", module)
        return module
    except ImportError as exc:
        attempts.append("import DaVinciResolveScript from PYTHONPATH -> %s" % exc)

    for directory in default_module_paths():
        candidate = os.path.join(directory, "DaVinciResolveScript.py")
        if not os.path.isfile(candidate):
            attempts.append("%s -> not found" % candidate)
            continue
        if directory not in sys.path:
            sys.path.insert(0, directory)
        try:
            module = importlib.import_module("DaVinciResolveScript")
            LOGGER.debug("Imported DaVinciResolveScript from %s", candidate)
            return module
        except Exception as exc:  # noqa: BLE001 - the shim raises bare ImportError
            attempts.append("%s -> %s: %s" % (candidate, type(exc).__name__, exc))

    for library in default_library_paths():
        if not os.path.isfile(library):
            attempts.append("%s -> not found" % library)
            continue
        try:
            module = _load_extension("fusionscript", library)
            LOGGER.debug("Loaded fusionscript directly from %s", library)
            return module
        except Exception as exc:  # noqa: BLE001 - native loaders raise anything
            attempts.append("%s -> %s: %s" % (library, type(exc).__name__, exc))

    raise ResolveError(
        "Could not import the DaVinci Resolve scripting module.",
        "  Locations tried:\n"
        + "".join("    %s\n" % line for line in attempts)
        + "  Set these environment variables to point at your installation:\n"
        "    RESOLVE_SCRIPT_API  (the ...\\Developer\\Scripting folder)\n"
        "    RESOLVE_SCRIPT_LIB  (the full path to fusionscript.dll/.so)\n"
        "    PYTHONPATH          (append %RESOLVE_SCRIPT_API%\\Modules)\n"
        "  Resolve's scripting README documents the defaults for each platform.",
    )


@dataclass
class ResolveContext:
    """A connected, validated handle on Resolve's object graph."""

    resolve: Any
    project_manager: Any
    project: Optional[Any] = None
    media_pool: Optional[Any] = None
    timeline: Optional[Any] = None
    product_name: str = ""
    version_string: str = ""
    version_fields: List[Any] = field(default_factory=list)

    @property
    def is_studio(self) -> bool:
        """True when the connected application is Resolve *Studio*."""
        return "studio" in (self.product_name or "").lower()

    @property
    def major_version(self) -> Optional[int]:
        """The major version number, when Resolve reported one."""
        if self.version_fields:
            try:
                return int(self.version_fields[0])
            except (TypeError, ValueError):
                return None
        return None

    def has_generate_speech(self) -> bool:
        """True when the project object exposes a callable ``GenerateSpeech``.

        On Resolve Studio 21.0.1 this reports a "Remote Function" object, which
        is callable, so the check is a genuine presence test -- but the bridge
        will happily hand back a proxy for a method that does not exist on older
        builds, which is why :func:`connect` also checks the major version.
        """
        if self.project is None:
            return False
        attribute = getattr(self.project, "GenerateSpeech", None)
        return attribute is not None and callable(attribute)

    def require_project(self) -> Any:
        """Return the current project.

        Raises:
            ResolveError: No project is open.
        """
        if self.project is None:
            raise ResolveError(
                "No project is open in DaVinci Resolve.",
                "  Open (or create) a project in Resolve, then run this tool again.",
            )
        return self.project

    def require_timeline(self) -> Any:
        """Return the current timeline.

        Raises:
            ResolveError: No timeline is open.
        """
        if self.timeline is None:
            raise ResolveError(
                "No timeline is open in the current project.",
                "  Open a timeline on the Edit or Cut page, then run this tool again.",
            )
        return self.timeline

    def require_media_pool(self) -> Any:
        """Return the media pool.

        Raises:
            ResolveError: The media pool could not be obtained.
        """
        if self.media_pool is None:
            raise ResolveError(
                "Could not obtain the Media Pool from the current project.",
                "  This usually means the project failed to load fully; reopen it in Resolve.",
            )
        return self.media_pool


def _call(obj: Any, name: str, *args: Any) -> Any:
    """Call ``obj.name(*args)`` defensively, logging and swallowing failures."""
    method = getattr(obj, name, None)
    if method is None:
        LOGGER.debug("%r has no attribute %s", obj, name)
        return None
    try:
        return method(*args)
    except Exception as exc:  # noqa: BLE001 - the bridge raises opaque errors
        LOGGER.debug("%s(%s) raised %s: %s", name, args, type(exc).__name__, exc)
        return None


def connect(
    require_studio: bool = True,
    require_project: bool = True,
    require_timeline: bool = True,
) -> ResolveContext:
    """Connect to a running Resolve and validate the environment.

    Args:
        require_studio: Fail when the free (non-Studio) edition is running.
        require_project: Fail when no project is open.
        require_timeline: Fail when no timeline is open.

    Raises:
        ResolveError: Resolve is unreachable or fails one of the checks.
    """
    LOGGER.info(
        "Python %s (%s), %s",
        platform.python_version(),
        platform.architecture()[0],
        sys.executable,
    )
    module = import_resolve_module()

    scriptapp = getattr(module, "scriptapp", None)
    if scriptapp is None:
        raise ResolveError(
            "The imported scripting module has no scriptapp() entry point.",
            "  The module that was imported is: %s\n"
            "  Check that RESOLVE_SCRIPT_LIB points at Resolve's fusionscript library."
            % getattr(module, "__file__", module),
        )

    try:
        resolve = scriptapp("Resolve")
    except Exception as exc:  # noqa: BLE001 - the native bridge raises anything
        raise ResolveError(
            "scriptapp('Resolve') failed: %s: %s" % (type(exc).__name__, exc),
            SCRIPTING_DISABLED_HINT,
        )

    if not resolve:
        raise ResolveError(
            "Could not connect to DaVinci Resolve (scriptapp returned None).",
            SCRIPTING_DISABLED_HINT,
        )

    product_name = _call(resolve, "GetProductName") or ""
    version_string = _call(resolve, "GetVersionString") or ""
    version_fields = _call(resolve, "GetVersion") or []
    LOGGER.info("Connected to %s %s", product_name or "DaVinci Resolve", version_string)

    project_manager = _call(resolve, "GetProjectManager")
    if not project_manager:
        raise ResolveError(
            "Resolve did not return a Project Manager.",
            "  Restart Resolve and try again.",
        )

    project = _call(project_manager, "GetCurrentProject") or None
    media_pool = _call(project, "GetMediaPool") if project else None
    timeline = _call(project, "GetCurrentTimeline") if project else None

    context = ResolveContext(
        resolve=resolve,
        project_manager=project_manager,
        project=project or None,
        media_pool=media_pool or None,
        timeline=timeline or None,
        product_name=product_name,
        version_string=version_string,
        version_fields=list(version_fields) if isinstance(version_fields, list) else [],
    )

    if require_studio and product_name and not context.is_studio:
        raise ResolveError(
            "GenerateSpeech requires DaVinci Resolve Studio; this is %s."
            % product_name,
            "  The AI Speech Generator is a Studio-only feature and is not available\n"
            "  in the free edition of DaVinci Resolve.",
        )

    if require_project:
        context.require_project()
    if require_timeline:
        context.require_timeline()

    if require_project:
        major = context.major_version
        if not context.has_generate_speech():
            raise ResolveError(
                "This Resolve build does not expose Project.GenerateSpeech().",
                "  GenerateSpeech was added in DaVinci Resolve Studio 20.x and is present\n"
                "  in 21.x (verified on 21.0.1.11). Detected version: %s%s\n"
                "  Update Resolve Studio, then install the 'AI Speech Generator' package\n"
                "  from the Extras Download Manager."
                % (
                    version_string or "unknown",
                    "" if major is None else " (major %d)" % major,
                ),
            )
        if major is not None and major < MIN_MAJOR_VERSION:
            raise ResolveError(
                "DaVinci Resolve Studio %d or newer is required; this is %s."
                % (MIN_MAJOR_VERSION, version_string or "version unknown"),
                "  The AI Speech Generator (Project.GenerateSpeech) arrived in Resolve\n"
                "  Studio 20. This tool was verified against 21.0.1.11; on an older\n"
                "  build the scripting bridge can still hand back a callable proxy for\n"
                "  GenerateSpeech that does nothing.\n"
                "  Update Resolve Studio and try again.",
            )

    return context


def _track_summary(timeline: Any) -> List[Dict[str, Any]]:
    """Describe every audio track on ``timeline`` for the probe report."""
    tracks: List[Dict[str, Any]] = []
    count = _call(timeline, "GetTrackCount", "audio") or 0
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 0
    for index in range(1, count + 1):
        items = _call(timeline, "GetItemListInTrack", "audio", index) or []
        tracks.append(
            {
                "index": index,
                "name": _call(timeline, "GetTrackName", "audio", index) or "",
                "sub_type": _call(timeline, "GetTrackSubType", "audio", index) or "",
                "locked": bool(_call(timeline, "GetIsTrackLocked", "audio", index)),
                "enabled": bool(_call(timeline, "GetIsTrackEnabled", "audio", index)),
                "item_count": len(items) if isinstance(items, list) else 0,
            }
        )
    return tracks


def probe_report(context: ResolveContext) -> str:
    """Build the human-readable diagnostic report printed by ``--probe``."""
    lines: List[str] = []
    add = lines.append

    add("DaVinci Resolve scripting probe")
    add("=" * 34)
    add("Interpreter        : %s" % sys.executable)
    add(
        "Python             : %s (%s)"
        % (platform.python_version(), platform.architecture()[0])
    )
    add("Platform           : %s" % platform.platform())
    add("Product            : %s" % (context.product_name or "unknown"))
    add("Version            : %s" % (context.version_string or "unknown"))
    add("Version fields     : %s" % (context.version_fields or "unknown"))
    add("Studio edition     : %s" % ("yes" if context.is_studio else "NO"))
    add("Current page       : %s" % (_call(context.resolve, "GetCurrentPage") or "unknown"))

    project = context.project
    if project is None:
        add("Project            : none open")
    else:
        add("Project            : %s" % (_call(project, "GetName") or "unnamed"))
        add(
            "GenerateSpeech     : %s"
            % ("present" if context.has_generate_speech() else "MISSING")
        )
        add(
            "Project frame rate : %s"
            % (_call(project, "GetSetting", "timelineFrameRate") or "unknown")
        )

    timeline = context.timeline
    if timeline is None:
        add("Timeline           : none open")
    else:
        add("Timeline           : %s" % (_call(timeline, "GetName") or "unnamed"))
        add(
            "  frame rate       : %s"
            % (_call(timeline, "GetSetting", "timelineFrameRate") or "unknown")
        )
        add("  start timecode   : %s" % (_call(timeline, "GetStartTimecode") or "unknown"))
        add("  start frame      : %s" % (_call(timeline, "GetStartFrame")))
        add("  end frame        : %s" % (_call(timeline, "GetEndFrame")))
        add("  playhead         : %s" % (_call(timeline, "GetCurrentTimecode") or "unknown"))
        add("  audio tracks     : %d" % (_call(timeline, "GetTrackCount", "audio") or 0))
        for track in _track_summary(timeline):
            add(
                "    A%-2d %-24s %-10s %s%s%d item(s)"
                % (
                    track["index"],
                    track["name"] or "(unnamed)",
                    track["sub_type"] or "?",
                    "locked " if track["locked"] else "",
                    "" if track["enabled"] else "disabled ",
                    track["item_count"],
                )
            )

    media_pool = context.media_pool
    if media_pool is None:
        add("Media pool         : unavailable")
    else:
        root = _call(media_pool, "GetRootFolder")
        folders = _call(root, "GetSubFolderList") if root else None
        names = []
        if isinstance(folders, list):
            for folder in folders:
                names.append(_call(folder, "GetName") or "(unnamed)")
        add("Media pool folders : %s" % (", ".join(names) if names else "(none)"))
        current = _call(media_pool, "GetCurrentFolder")
        add("  current folder   : %s" % (_call(current, "GetName") if current else "unknown"))

    add("")
    add("Environment variables")
    for name in ("RESOLVE_SCRIPT_API", "RESOLVE_SCRIPT_LIB", "PYTHONPATH"):
        add("  %-18s : %s" % (name, os.environ.get(name, "(not set)")))
    return "\n".join(lines)
