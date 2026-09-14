# local-tts

Turn a plain-text script into a natural-sounding voice-over on your own
machine, optionally in **your own cloned voice**. No cloud APIs, no accounts,
no uploads.

```
input/script.txt                voices/my_voice/reference.wav
        │                                  │
        ▼                                  ▼
  sentence-aware chunking  ─────►  local TTS / voice-cloning model (GPU or CPU)
                                           │
                                           ▼
                              per-chunk audio (cached, resumable)
                                           │
                                           ▼
                           stitched, normalised  output/script.wav
```

```bash
python tts.py --input input/script.txt                              # default voice
python tts.py --input input/script.txt --voice-sample me.wav        # clone a voice once
python tts.py --input input/script.txt --voice-profile my_voice     # reuse a saved voice
```

Built and tested on an **NVIDIA RTX 3060 Ti (8 GB)** on Windows 11, Python 3.11.
Everything runs equally on Linux and macOS (CPU or CUDA).

---

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Recommended Model](#recommended-model)
- [Alternative Models](#alternative-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Voice Cloning](#voice-cloning)
- [Voice Profiles](#voice-profiles)
- [Model Selection](#model-selection)
- [Voice Selection](#voice-selection)
- [Long Scripts](#long-scripts)
- [Output](#output)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Privacy](#privacy)
- [Model Licenses](#model-licenses)
- [Third-Party Attributions](#third-party-attributions)
- [Responsible Voice Cloning](#responsible-voice-cloning)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Local TTS** - the whole pipeline runs on your machine; the only network access is the one-time model download from Hugging Face.
- **Voice cloning** - zero-shot cloning from 10-20 seconds of reference audio (Chatterbox, Qwen3-TTS).
- **Reusable voice profiles** - `voices/<name>/reference.wav` once, `--voice-profile <name>` forever. Speaker embeddings are cached per model.
- **Model switching** - `--model chatterbox-turbo | chatterbox | chatterbox-multilingual | chatterbox-nano | kokoro | qwen3-tts | ...`; each model is one isolated module, adding another is a small job.
- **Voice switching** - preset voices where a model has them (`--voice af_heart` for Kokoro's ~50 voices, `--voice Ryan` for Qwen3-TTS CustomVoice).
- **Long-form narration** - paragraphs and sentences are detected, chunks are balanced and never cut mid-sentence, chunks are generated in order, sanity-checked, retried when the model hallucinates, cached so an interrupted run resumes, then stitched with natural sentence/paragraph pauses.
- **GPU acceleration with CPU fallback** - CUDA is detected automatically; `--device cpu` always works (slowly for the large models, comfortably for Kokoro).
- **RTX 3060 Ti friendly** - the default model needs about 3.5 GB of VRAM; nothing here requires more than 8 GB.
- **Sensible audio** - mono 24 kHz WAV (16/24/32-bit) by default, optional FLAC/OGG/MP3, optional resampling to 48 kHz for video editors, peak normalisation, click-free joins.
- **Clean failure modes** - readable errors with suggestions for missing CUDA, out-of-memory, bad reference audio, unknown voices, missing packages. Tracebacks only with `--debug`.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 (3.13 is not yet supported by all model packages) |
| GPU | none (CPU mode) | NVIDIA GPU with 6 GB+ VRAM and a current driver (RTX 3060 Ti 8 GB is the reference card) |
| CUDA | - | any driver that supports CUDA 12.x (driver 528+). You do **not** install the CUDA toolkit; PyTorch wheels bundle it. |
| Disk | 6 GB | 10-15 GB for several models (weights are downloaded to `models/`) |
| RAM | 8 GB | 16 GB |
| ffmpeg | optional | needed only for MP3 export fallback, M4A/AAC input and `--speed` on models without native speed control |

AMD GPUs (ROCm on Linux) and Apple Silicon (MPS) are not tested; use `--device cpu` there.

## Recommended Model

**Default: `chatterbox-turbo`** - Resemble AI's Chatterbox Turbo (350M parameters, MIT license).

The models were compared in September 2026 on the criteria that matter for a
voice-over tool on an 8 GB card: cloning quality, prosody on long passages,
VRAM, speed, licence, install friction and maintenance. Summary:

| Model | Licence (weights) | Cloning | VRAM (inference) | 8 GB card | Notes |
|---|---|---|---|---|---|
| **Chatterbox Turbo** (Dec 2025) | MIT | zero-shot, 5-20 s | ~3.5 GB | yes, comfortable | Fastest of the family, most natural English, `[laugh]`-style tags, pip install, actively maintained (Nano added Apr 2026, Multilingual v3 Jun 2026). Watermarks its output. |
| Chatterbox / Multilingual v2-v3 | MIT | zero-shot | ~4.5 GB | yes | Same family; 23 languages; emotion (`--exaggeration`) and pacing (`--cfg-weight`) knobs. Slower (10-step decoder). |
| Kokoro-82M | Apache-2.0 | **no** (fixed voices) | <1 GB, runs on CPU | yes | Excellent, tiny, very fast; the fallback when you need a clean stock narrator. |
| Qwen3-TTS 0.6B / 1.7B (Jan 2026) | Apache-2.0 | 3 s zero-shot, better with a transcript | ~4.5 / ~7 GB | 0.6B yes, 1.7B tight | Very expressive, 10 languages, voice design. Pins an older `transformers`, so it needs its own virtual environment. |
| VoxCPM2 (Apr 2026) | Apache-2.0 | zero-shot | ~8 GB | no headroom | 48 kHz output, 30 languages. VoxCPM 1.5 (0.6B, ~6 GB) is zh/en only. Heavy dependency set. Not included yet. |
| F5-TTS | CC-BY-NC-4.0 | zero-shot | 6-12 GB | yes | Great prosody, but weights are **non-commercial**. |
| XTTS-v2 (Coqui) | CPML (non-commercial) | zero-shot | ~4-6 GB | yes | Unmaintained upstream, non-commercial licence. |
| Fish Speech / OpenAudio S1, Fish Audio S2 | CC-BY-NC-SA / paid | zero-shot | 6 GB+ | yes | Non-commercial without a paid licence. |
| IndexTTS-2 / 2.5 | Bilibili model licence | zero-shot | ~6-8 GB | tight | Strong for zh/ja; custom licence with usage restrictions, `uv`-only install. |
| Breeze TTS 2 (Aug 2026) | research / non-commercial | zero-shot | 12 GB+ | no | Top of the open-weights arena, but too big and non-commercial. |
| VibeVoice TTS 1.5B | MIT | speaker prompts | ~7 GB+ | tight | Built for multi-speaker podcasts (90 min); Microsoft pulled and re-released the TTS code. Overkill here. |
| CosyVoice 3 | Apache-2.0 | zero-shot | ~6 GB | yes | Good, but git-clone + conda install only; Chinese-centric. |

Why Turbo wins for *this* use case:

1. **Quality per VRAM.** It produces the most natural English narration of the permissively-licensed models that fit an 8 GB card with room to spare.
2. **Cloning.** Zero-shot cloning from a 10-20 s clip, no fine-tuning; the reference is only ever read from disk.
3. **Speed.** The 1-step decoder makes it several times faster than the original Chatterbox, so a 10-minute script is a coffee break, not a lunch break.
4. **Licence.** MIT for code *and* weights, so the repository and your generated voice-overs are safe for commercial use.
5. **Maintenance and install.** One `pip install`, weights auto-downloaded, active upstream.

Trade-offs to know: Turbo is English-only (use `chatterbox-multilingual` for 23 languages), has no emotion knob (use `chatterbox` for `--exaggeration`), and every file it produces carries Resemble's inaudible Perth watermark (a licence-compatible, responsible-AI feature that cannot be turned off through this tool).

## Alternative Models

All of these are selectable with `--model`:

| id | When to use it |
|---|---|
| `chatterbox-turbo` | Default. English voice-over with or without cloning. |
| `chatterbox` | You want `--exaggeration` / `--cfg-weight` control over emotion and pacing (slower). |
| `chatterbox-multilingual` | Non-English scripts (23 languages, `--language de`). PyPI release ships v2 weights; the GitHub version ships v3 (better speaker similarity, fewer hallucinations) - see [Model Selection](#model-selection). |
| `chatterbox-nano` | CPU-only machines. Needs the GitHub version of `chatterbox-tts`. |
| `kokoro` | No cloning needed; you want a fast, clean stock voice, native `--speed`, or CPU-only generation. |
| `qwen3-tts` | Second opinion on cloning quality (Apache-2.0). Best when you also provide the transcript of the reference. Separate virtual environment. |
| `qwen3-tts-1.7b` | Same, larger model. Fits 8 GB only when nothing else uses the GPU. |
| `qwen3-tts-voices` | Nine preset speakers (two English) with `--instruct "warm, slow"` style control. |

`python tts.py --list-models` shows this list with licence, VRAM, download size and whether each one is installed.

## Installation

### 1. Python and Git

Install Python 3.11 (or 3.10/3.12) from https://www.python.org/downloads/ and make sure `python --version` prints it.

### 2. Clone the repository

```bash
git clone https://github.com/<your-account>/local-tts.git
cd local-tts
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

| Platform | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

On Windows, `py -3.11 -m venv .venv` picks a specific interpreter if several are installed.

### 4. Install PyTorch (CUDA build for NVIDIA GPUs)

The default engine pins `torch==2.6.0`. Install it first so pip fetches the GPU-enabled wheel:

```bash
# NVIDIA GPU (RTX 3060 Ti and any other CUDA 12 capable card):
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# No NVIDIA GPU:
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"` should print `True` on a GPU machine.

### 5. Install the default engine (Chatterbox) and core dependencies

```bash
pip install -r requirements-chatterbox.txt
```

Optional extra engines in the **same** environment:

```bash
pip install -r requirements-kokoro.txt          # Kokoro fixed voices (Apache-2.0, CPU friendly)
```

Qwen3-TTS must live in a **separate** environment because `qwen-tts` pins `transformers==4.57.x` while `chatterbox-tts` pins `transformers==5.x`:

```bash
python -m venv .venv-qwen
# activate .venv-qwen, then:
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-qwen3.txt
```

### 6. Optional: ffmpeg

Only needed for MP3 export fallback, reading M4A/AAC voice samples, and `--speed` on models without native speed control (everything except Kokoro). Get it from https://ffmpeg.org/download.html and make sure `ffmpeg -version` works in your terminal.

### 7. Models download themselves

The first run of each model downloads its weights from Hugging Face into `models/` (Chatterbox Turbo ~4 GB, original/multilingual ~3.3 GB each, Kokoro ~330 MB, Qwen3-TTS 0.6B ~2.5 GB). No account or token is required. After that you can work fully offline (`HF_HUB_OFFLINE=1` in `.env` guarantees it).

To reuse an existing Hugging Face cache instead, set `HF_HOME` (see `.env.example`).

### 8. Check the installation

```bash
python tts.py --list-models
python tts.py --input input/example_script.txt --dry-run
python tts.py --text "The installation works." --output output/test.wav
```

## Quick Start

1. Put your script in `input/script.txt` (plain UTF-8 text; blank lines separate paragraphs).
2. Run:

   ```bash
   python tts.py --input input/script.txt
   ```

3. Listen to `output/script.wav`.

Progress looks like this:

```
Processing script...
Chunks: 7 (~1.4 min of speech, rough estimate)
Loading model Chatterbox Turbo on NVIDIA GeForce RTX 3060 Ti...
Model loaded (15.0s).
Voice: built-in default voice
[1/7] Generating (23 chars)... done (2.4s -> 1.6s audio)
[2/7] Generating (268 chars)... done (7.2s -> 14.8s audio)
[3/7] Generating (158 chars)... done (5.0s -> 10.2s audio)
...
[7/7] Generating (73 chars)... done (2.5s -> 4.5s audio)
Combining audio...
Done!
Output: output\example_script.wav
Length: 1m 14.6s @ 24000 Hz | chunks: 7 (0 from cache) | took 55s (1.4x real time) on NVIDIA GeForce RTX 3060 Ti
```

Quick variations:

```bash
python tts.py --input input/script.txt --output renders/episode-01.wav
python tts.py --input input/script.txt --voice-sample recordings/me.wav
python tts.py --input input/script.txt --model kokoro --voice bm_george --speed 1.05
python tts.py --input input/script.txt --format mp3 --sample-rate 48000
```

## Voice Cloning

Provide a recording of the voice you want (your own, or one you have explicit permission to use):

```bash
python tts.py --input input/script.txt --voice-sample recordings/me.wav
```

For repeated use, save it as a [voice profile](#voice-profiles) instead.

### What makes a good reference recording

| Aspect | Recommendation |
|---|---|
| **Duration** | 10-20 seconds of continuous speech. Minimum is 3 s (Chatterbox Turbo/Nano require more than 5 s). Longer than ~30 s does not help: Turbo uses the first 15 s for the speaker embedding and the first 10 s for the acoustic prompt; the tool caps the reference at 30 s (`--max-reference-seconds`). |
| **Content** | Natural, complete sentences spoken the way you want the narration to sound (same pace, energy, mic distance). Avoid lists of words, singing, shouting. |
| **Format** | WAV or FLAC preferred (lossless). MP3/OGG work. M4A/AAC need ffmpeg installed. |
| **Sample rate / bit depth** | Anything from 16 kHz up; 44.1 or 48 kHz, 16- or 24-bit is ideal. The models resample internally; the tool does not resample your file. |
| **Channels** | Mono preferred. Stereo is averaged to mono automatically. |
| **Cleanliness** | Only the target speaker. No music, no other voices, no reverb-heavy rooms, no background noise. If you have a noisy take, clean it in your editor first (e.g. a spectral de-noiser); the tool deliberately does not denoise, because aggressive processing degrades the clone. |
| **Level** | Not clipping, reasonably loud. Turbo normalises reference loudness itself. |
| **Multiple samples** | Chatterbox and Qwen3-TTS read one file. Concatenate your best 2-3 takes into one 15-20 s file if a single take is too short. Keep a couple of alternative profiles (`voices/me_calm/`, `voices/me_energetic/`) - the delivery style of the reference strongly shapes the output. |
| **Transcript** | Optional; only Qwen3-TTS uses it (`--voice-text "..."` or `reference.txt` in the profile) and it improves that model's clone noticeably. |

### What the tool does to the reference

```
reference file  ->  decode  ->  mono  ->  trim leading/trailing silence  ->  cap at 30 s  ->  cached WAV  ->  model
```

Sample rate and loudness are left untouched. The original file is never modified. Use `--no-preprocess-reference` to hand the file to the model completely as-is (WAV only).

### Cloning tips per model

- **chatterbox-turbo**: just the audio. Speaks with the pacing of the reference; if the clone speaks too slowly, use a livelier reference clip.
- **chatterbox / chatterbox-multilingual**: `--exaggeration 0.5` (default) is neutral; `0.7` plus `--cfg-weight 0.3` is more expressive. Higher `--cfg-weight` slows speech. For non-English, the reference should be in the target language or the accent transfers.
- **qwen3-tts**: give the exact transcript of the reference for in-context cloning; without it only the speaker embedding is used.

## Voice Profiles

A profile is a folder in `voices/`:

```
voices/
├── my_voice/
│   ├── reference.wav       # required
│   ├── reference.txt       # optional exact transcript
│   └── profile.json        # optional: {"language": "en", "description": "...", "options": {"exaggeration": 0.6}}
├── narrator/
│   └── reference.flac
└── character_voice/
    └── reference.mp3
```

Create one from the command line (the recording is validated and copied unchanged):

```bash
python tts.py --save-voice-profile my_voice --voice-sample recordings/me.wav --voice-text "what I say in the recording"
```

Use it:

```bash
python tts.py --input input/script.txt --voice-profile my_voice
python tts.py --input input/script.txt --voice my_voice          # shorthand: --voice also accepts profile names
python tts.py --list-voices                                       # profiles + preset voices of the current model
```

Speaker embeddings (Chatterbox "conditionals", Qwen3-TTS clone prompts) are cached in `voices/<name>/.cache/` per model, so later runs skip the reference processing. Delete `.cache/` to force recomputation.

No microphone at hand? You can try the cloning pipeline with a synthetic voice: generate 15-20 s of speech with a Kokoro preset and save that as a profile (Kokoro's voices are Apache-2.0 licensed synthetic voices, not recordings of a person):

```bash
python tts.py --text "Twenty seconds of natural, complete sentences spoken at a steady narration pace..." --model kokoro --voice bm_george --output output/kokoro_george.wav
python tts.py --save-voice-profile demo_george --voice-sample output/kokoro_george.wav --voice-text "Twenty seconds of natural, complete sentences spoken at a steady narration pace..."
python tts.py --input input/example_script.txt --voice-profile demo_george
```

`voices/` (except its README) is in `.gitignore`: recordings are personal data and must never be committed.

## Model Selection

```bash
python tts.py --list-models
python tts.py --input input/script.txt --model kokoro
python tts.py --input input/script.txt --model chatterbox --exaggeration 0.7 --cfg-weight 0.3
python tts.py --input input/script.txt --model chatterbox-multilingual --language fr --voice-profile narrateur
```

Set a permanent default with `LOCAL_TTS_MODEL=...` in `.env`.

### Newest Chatterbox weights (Nano, Multilingual v3)

The pinned PyPI release `chatterbox-tts==0.1.7` (March 2026) includes Turbo, the original model and Multilingual **v2**. Nano and the Multilingual **v3** weights were added to the GitHub repository later. If you want them:

```bash
pip install --upgrade "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git"
```

The tool detects which version is installed and uses v3/Nano automatically when available.

### Adding a model

Engines live in `src/local_tts/engines/`. Copy `kokoro.py` or `chatterbox.py`, implement `load()`, `list_voices()`, `set_voice()` and `synthesize()`, fill in the `ModelInfo` block, and add one line to `_REGISTRY` in `engines/__init__.py`. Nothing else changes. Import heavy packages inside `load()` so `--list-models` and the tests keep working without them.

## Voice Selection

- Cloning models (`chatterbox*`, `qwen3-tts`, `qwen3-tts-1.7b`) have **one built-in default voice** (Chatterbox) or **none** (Qwen3-TTS Base) plus whatever you clone. `--voice default` selects the built-in voice explicitly.
- `kokoro` ships ~50 named voices in 9 language/accent packs; `qwen3-tts-voices` ships 9 speakers.

```bash
python tts.py --list-voices --model kokoro
python tts.py --input input/script.txt --model kokoro --voice af_heart
python tts.py --input input/script.txt --model kokoro --voice "af_heart,af_bella"   # Kokoro voice blend
python tts.py --input input/script.txt --model qwen3-tts-voices --voice Ryan --instruct "calm documentary narrator"
```

Kokoro's language pack is taken from the voice id (`a`=American, `b`=British, `e`=Spanish, `f`=French, `h`=Hindi, `i`=Italian, `j`=Japanese, `p`=Portuguese, `z`=Mandarin). Japanese and Mandarin need `pip install "misaki[ja]"` / `"misaki[zh]"`.

## Long Scripts

There is no size limit on the script. The processing is:

1. **Load** - UTF-8 (with or without BOM) or Windows-1252 text.
2. **Clean** - unify newlines/quotes, strip light Markdown (`#` headings, `-`/`1.` list markers, `**bold**`).
3. **Paragraphs** - blank lines separate paragraphs; single line breaks are soft wraps (`--newline-is-break` treats each line as a paragraph instead).
4. **Sentences** - rule-based splitting that understands abbreviations (`Dr.`, `e.g.`, `p.m.`), initials, decimals, ellipses and URLs.
5. **Chunks** - sentences are packed into balanced chunks below the model's limit (about 300 characters, `--max-chunk-chars`). A chunk never spans paragraphs, a sentence is only split when it alone exceeds the limit (then at `;` `:` `,` `-` and finally spaces).
6. **Generate** - chunks run sequentially in script order. Each result is sanity-checked against the expected duration; a runaway or truncated generation is retried with a new seed (`--retries`, default 2) and the best attempt is kept. Every finished chunk is saved to `.cache/chunks/`.
7. **Resume** - re-running the same command after a crash or Ctrl+C reuses cached chunks (same model, voice, text and seed). `--no-cache` disables this, `--clear-cache` starts fresh.
8. **Stitch** - leading/trailing silence of each chunk is trimmed (`--no-trim` keeps it), a 5 ms fade prevents clicks, then chunks are joined with `--sentence-gap` (0.25 s) inside a paragraph and `--paragraph-gap` (0.6 s) between paragraphs.
9. **Finish** - optional speed change, optional resampling, peak normalisation to -1 dBFS (`--no-normalize`), export.

`--dry-run` prints the chunk plan without loading a model. `--manifest` writes `<output>.json` with the start/end time and text of every chunk - handy for placing markers in a video editor.

## Output

| Option | Default | Notes |
|---|---|---|
| `--output` | `output/<script name>.wav` | A directory path puts the default filename inside it. |
| `--format` | from extension, else `wav` | `wav`, `flac`, `ogg`, `mp3` (MP3 via libsndfile if built with LAME support, otherwise ffmpeg). |
| `--bit-depth` | 16 | 16 / 24 / 32 (float) for WAV and FLAC. |
| `--sample-rate` | model native (24 000 Hz) | e.g. `48000` for video timelines; high-quality soxr resampling. |
| channels | mono | Voice-overs are mono; pan in your editor. |
| loudness | peak -1 dBFS | No compression or LUFS normalisation is applied - do that in your mix. |

## CLI Reference

```
python tts.py --help
```

| Option | Description |
|---|---|
| `-i, --input FILE` | Script text file. |
| `--text TEXT` | Synthesize this string instead of a file. |
| `-o, --output PATH` | Output file or directory. |
| `--format {wav,flac,ogg,mp3}` | Output format. |
| `--bit-depth {16,24,32}` | WAV/FLAC bit depth. |
| `--sample-rate HZ` | Resample the final file. |
| `--manifest` | Write per-chunk timestamps to `<output>.json`. |
| `-m, --model ID` | Model id (`--list-models`). |
| `--device auto\|cuda\|cuda:N\|cpu` | Compute device. |
| `-l, --language CODE` | Language for multilingual models (`en`, `de`, `fr`, ...). |
| `--models-dir DIR` | Where weights are cached. |
| `--voice NAME` | Preset voice or profile name. |
| `--voice-sample FILE` | Reference recording to clone. |
| `--voice-text TEXT` | Transcript of the reference (Qwen3-TTS). |
| `--voice-profile NAME` | Use `voices/NAME/`. |
| `--save-voice-profile NAME` | Create `voices/NAME/` from `--voice-sample`. |
| `--overwrite` | Replace an existing profile. |
| `--no-preprocess-reference` | Pass the reference to the model untouched. |
| `--max-reference-seconds SEC` | Cap the reference length (default 30). |
| `--speed X` | 0.5-2.0. Native for Kokoro; ffmpeg time-stretch for the others. |
| `--exaggeration X` | Emotion intensity, `chatterbox` / `chatterbox-multilingual` (default 0.5). |
| `--cfg-weight X` | Pacing / adherence, same models (default 0.5). |
| `--temperature X` | Sampling temperature for Chatterbox models (default 0.8). |
| `--instruct TEXT` | Style instruction for `qwen3-tts-voices`. |
| `--seed N` | Reproducible output. |
| `--max-chunk-chars N` | Characters per generation (default: model specific). |
| `--newline-is-break` | Every line break is a paragraph break. |
| `--sentence-gap SEC` / `--paragraph-gap SEC` | Pauses inserted between chunks (0.25 / 0.6). |
| `--no-trim` / `--no-normalize` | Keep model silence / skip peak normalisation. |
| `--retries N` | Regeneration attempts per chunk (default 2). |
| `--no-cache` / `--clear-cache` | Disable or reset the resumable chunk cache. |
| `--dry-run` | Show the chunk plan only. |
| `--list-models` / `--list-voices` | Information. |
| `-v, --verbose` / `--debug` / `-q, --quiet` | Output verbosity. `--debug` shows tracebacks and library logs. |

Exit codes: `0` success, `1` user/input error, `2` model/device/dependency error, `3` GPU out of memory, `4` generation failed, `130` interrupted.

## Configuration

Copy `.env.example` to `.env` (ignored by Git) to change defaults without typing options every time:

```
LOCAL_TTS_MODEL=chatterbox-turbo
LOCAL_TTS_DEVICE=auto
LOCAL_TTS_LANGUAGE=en
LOCAL_TTS_MODELS_DIR=models
LOCAL_TTS_VOICES_DIR=voices
LOCAL_TTS_OUTPUT_DIR=output
LOCAL_TTS_CACHE_DIR=.cache
```

Command-line options override `.env`, which overrides the built-in defaults. No secrets are needed anywhere: the models are public downloads.

## Performance

Measured on the development machine: RTX 3060 Ti 8 GB, Windows 11, Python 3.11, `torch 2.6.0+cu124`, `chatterbox-tts 0.1.7`. Your numbers will vary with driver, text and reference voice.

| Model | Device | Model load (weights already downloaded) | Generation speed (after load) | Peak GPU memory (reserved) |
|---|---|---|---|---|
| chatterbox-turbo | RTX 3060 Ti | ~15 s | **~2x real time** (a 250-character chunk -> ~15 s of audio in ~7-8 s; ~54 speech tokens/s) | 3.5 GB |
| kokoro | RTX 3060 Ti | ~8 s | **~10x real time** (20 s of audio in 2 s) | 1.4 GB |
| kokoro | CPU | ~8 s | roughly real time | - |

Example: the 75-second `input/example_script.txt` took 55 s end-to-end with Turbo (15 s load + 40 s generation), and 15 s with Kokoro.

Practical expectations:

- The **first run** of a model is dominated by the download (4 GB for Turbo, 330 MB for Kokoro).
- With Turbo, a **10-minute narration takes about 5-6 minutes** on the 3060 Ti. Cloning a voice adds one or two seconds per run (the speaker embedding is cached afterwards).
- The original and multilingual Chatterbox models use a 10-step decoder and are noticeably slower than Turbo (expect around real time or slower).
- CPU generation with Chatterbox works but is **several times slower than real time**; use `kokoro` (or `chatterbox-nano` from GitHub) for CPU workflows.
- VRAM headroom on 8 GB is comfortable for every model shipped here except `qwen3-tts-1.7b`, which needs the card to itself.

Numbers above were read from the tool's own `--verbose` output on the development machine (`Peak GPU memory used` and per-chunk timings). Run with `--verbose` to see them on your hardware.

## Troubleshooting

**"CUDA was requested but PyTorch cannot see a CUDA GPU" / generation runs on CPU**
`nvidia-smi` must list your GPU. Then check the PyTorch build: `python -c "import torch; print(torch.__version__, torch.version.cuda)"` must show `+cu124` (or another `cu` tag), not `+cpu`. Reinstall with the `--index-url https://download.pytorch.org/whl/cu124` command from the installation steps. Old drivers (< 528) need updating.

**"The GPU ran out of memory"**
Close other GPU programs (browsers with hardware acceleration, games, other AI tools). Use a smaller model (`--model kokoro`, `--model chatterbox-nano`), shorter chunks (`--max-chunk-chars 150`), or `--device cpu`. For Qwen3-TTS pick the 0.6B model.

**"Failed to load ..." / download errors**
The first run downloads several GB; interrupted downloads resume when you run again. Check the connection, proxy settings, or disk space in `models/`. If Hugging Face is unreachable but the weights are already present, set `HF_HUB_OFFLINE=1`.

**"is not installed: Python package 'chatterbox' not found"**
You installed the core requirements but not the engine: `pip install -r requirements-chatterbox.txt` (after PyTorch). Make sure the virtual environment is activated - `python -c "import sys; print(sys.prefix)"` should point into `.venv`.

**Dependency conflicts during pip install**
`chatterbox-tts` pins exact versions (`torch==2.6.0`, `transformers==5.2.0`, `numpy<2`). Install it into a fresh virtual environment. Qwen3-TTS (`transformers==4.57.x`) must go into a separate environment. Python 3.13 is not supported by `kokoro`/`chatterbox` yet - use 3.10-3.12.

**"Invalid reference audio" / "Reference audio is too short"**
Use at least 5-6 s (10-20 s recommended) of speech in WAV/FLAC/MP3/OGG. M4A/AAC need ffmpeg on PATH. Silent or extremely quiet files are rejected.

**"Voice 'x' is not available for model ..." / "Voice profile 'x' not found"**
`python tts.py --list-voices --model <id>` shows what exists. Kokoro voice ids look like `af_heart`; profile folders must contain a `reference.*` audio file.

**Output has odd pauses, repeated words or trailing gibberish in one chunk**
Look at the chunk warnings printed at the end. Regenerate with another seed (`--seed 42`, or `--clear-cache`), shorten chunks (`--max-chunk-chars 200`), or try `--model chatterbox` with `--cfg-weight 0.5 --exaggeration 0.4`. Very short chunks (one or two words) are the most fragile: give headings a full sentence.

**Clone sounds "off" (accent, pace, energy)**
The reference drives everything. Use a cleaner, livelier or longer (15-20 s) sample in the target language; try 2-3 different takes as separate profiles. For Qwen3-TTS add the transcript.

**`EspeakFallback not Enabled: OOD words will be skipped` (Kokoro)**
Rare English words fall back to espeak-ng. `espeakng-loader` ships the binary for most platforms; if the warning persists, install espeak-ng from https://github.com/espeak-ng/espeak-ng/releases.

**Windows: `UnicodeEncodeError` or garbled characters in the console**
Run `chcp 65001` or set `PYTHONIOENCODING=utf-8`. Output files are unaffected.

**Everything else**
Run again with `--debug` for the full traceback and library logs, and include that output when opening an issue.

## Limitations

- **Watermark**: every Chatterbox output contains Resemble's Perth watermark (inaudible). It cannot be disabled through this tool.
- **English focus**: the default model is English-only. Non-English requires `chatterbox-multilingual` (23 languages) or `qwen3-tts` (10 languages); quality varies by language.
- **Chunk-level consistency**: chunks are generated independently, so prosody can shift slightly between chunks even with the same voice; a fixed `--seed` and a consistent reference minimise it.
- **Occasional hallucinations**: autoregressive TTS models sometimes add, drop or repeat words. The sanity check and retries catch the obvious cases (length), not subtle ones. Proof-listen anything you publish.
- **No SSML / phonetic control**: pronunciation of names and acronyms is up to the model; rewrite the text ("N A S A", "Ay-shuh") when needed.
- **Speed for most models is a post-process** (ffmpeg `atempo`), which is transparent for 0.9-1.1x but audible beyond that. Kokoro changes speed natively.
- **Single speaker per run**: dialogue with several voices means several runs (one profile each) and assembly in your editor.
- **Hardware**: tested on an 8 GB NVIDIA card and CPU. ROCm and Apple MPS are untested.

## Privacy

```
your text + your recordings  ->  this Python program  ->  local model  ->  local audio files
```

- No API keys, accounts or telemetry. Nothing in this repository sends data anywhere.
- The only network traffic is `huggingface_hub` downloading public model weights on first use. Set `HF_HUB_OFFLINE=1` afterwards to forbid even that.
- Voice samples, profiles, caches and outputs stay in the project folder and are excluded from Git.
- If an optional cloud backend is ever added, it will be a separate engine module, disabled by default, and clearly labelled.

## Model Licenses

The application code is MIT. The models are downloaded separately by the user and remain under their own licences:

| Model | Weights & code licence | Source | Notes |
|---|---|---|---|
| Chatterbox, Turbo, Nano, Multilingual (Resemble AI) | [MIT](https://github.com/resemble-ai/chatterbox/blob/master/LICENSE) | https://huggingface.co/ResembleAI | Outputs carry the Perth watermark. Resemble asks users not to use the model for harmful or deceptive purposes. |
| Perth watermarker (resemble-perth) | MIT | https://github.com/resemble-ai/Perth | Dependency of chatterbox-tts. |
| Kokoro-82M (hexgrad) | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M) | https://huggingface.co/hexgrad/Kokoro-82M | Voices trained on permissive/synthetic data per the model card. |
| misaki / espeak-ng | MIT / GPL-3.0 | https://github.com/hexgrad/misaki, https://github.com/espeak-ng/espeak-ng | espeak-ng is used as an external phonemiser fallback by Kokoro only; it is not bundled in this repository. |
| Qwen3-TTS (Alibaba) | [Apache-2.0](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE) | https://huggingface.co/Qwen | |
| PyTorch, transformers, soundfile/libsndfile, soxr | BSD-3 / Apache-2.0 / BSD-3+LGPL / LGPL-2.1 | | Standard Python dependencies, not redistributed here. |

No model weights are included in this repository. Models rejected for licence reasons (F5-TTS, XTTS-v2, Fish/OpenAudio, IndexTTS, Breeze) are not referenced by the code.

## Third-Party Attributions

- **Chatterbox** - Resemble AI, https://github.com/resemble-ai/chatterbox (MIT). "Chatterbox" is a trademark of Resemble AI; this project is not affiliated with Resemble AI.
- **Kokoro-82M** - hexgrad, https://github.com/hexgrad/kokoro (Apache-2.0), built on StyleTTS 2 (Yinghao Aaron Li et al.) and ISTFTNet.
- **Qwen3-TTS** - Qwen team, Alibaba Cloud, https://github.com/QwenLM/Qwen3-TTS (Apache-2.0).
- **python-soundfile / libsndfile**, **python-soxr / libsoxr**, **NumPy**, **PyTorch**, **Hugging Face Hub** - see their respective licences.
- The long-form chunk-and-stitch workflow is an independent implementation inspired by the segmenting approach used in DaVinci Resolve narration scripts; no code is shared.

## Responsible Voice Cloning

This tool clones voices. You are responsible for:

- **Consent** - only clone your own voice or a voice whose owner has given you explicit permission. Do not impersonate real people.
- **Law** - comply with the laws that apply to you (personality/publicity rights, fraud, deepfake and AI-disclosure legislation, data protection for recordings of others).
- **Model licences** - respect the terms above; Resemble AI's usage guidelines for Chatterbox prohibit deceptive or harmful use.
- **Platform policies** - YouTube, podcast hosts, app stores and clients may require disclosure of synthetic voices.
- **Disclosure** - label AI-generated narration where your audience would reasonably expect to know.

The repository ships no voice samples and hard-codes no person's identity or voice. Everything in `voices/` is user-provided local data.

## Contributing

Bug reports and pull requests are welcome.

```bash
pip install -r requirements-dev.txt      # pytest + ruff (no model needed)
python -m pytest                          # 80+ unit tests, run in seconds
ruff check src tests
```

- Unit tests must not download models or need a GPU; use the `FakeEngine` in `tests/conftest.py`.
- Keep model-specific code inside `src/local_tts/engines/<model>.py`; the CLI and pipeline stay model-agnostic.
- Do not add dependencies to the core (`requirements.txt`) that only one engine needs.
- Never commit audio, weights, `.env` or anything from `voices/`, `output/`, `models/`, `.cache/`.
- New engines: add a `ModelInfo` with an honest licence/VRAM entry and update the tables in this README.

## License

The code in this repository is released under the **MIT License** (see `LICENSE`). MIT was chosen because it matches the default model's licence, is the most widely understood permissive licence, and puts no obligations on people who build products on top of this tool. If you prefer a copyleft licence for your own fork, that is your call; the model licences remain what they are regardless.
