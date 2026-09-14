# Local Video Transcriber (`transcribe_folder.py`)

Recursively find and transcribe all video files in any directory using OpenAI's open-source **Whisper** model. The script runs entirely on your local machine—no internet connection, no external API calls, and **zero cloud fees**.

---

## Features

- **Recursive Scanning:** Automatically scans subfolders looking for popular video formats (`.mp4`, `.mov`, `.mkv`, `.avi`, `.wmv`, `.flv`, `.webm`, `.mpeg`, `.mpg`, `.m4v`).
- **Skips Existing Transcripts:** Detects matching `.txt` transcripts in folders to skip already-processed videos, preventing redundant processing.
- **Fault-Tolerant:** Gracefully continues processing the remaining queue if a single video file fails or is corrupted.
- **Smart Hardware advisory:** Checks your system hardware (CUDA GPU VRAM or System RAM) and prints the recommended Whisper model size (`tiny`, `base`, `small`, `medium`, or `large`) to optimize performance and prevent crashes.
- **Flexible Parameters:** Customize device routing, override languages, and force re-transcription.

---

## Requirements

1. **Python 3.9+**
2. **FFmpeg** (Required by Whisper to decode audio/video streams)
   - **Windows:** Install via winget (`winget install ffmpeg`) or Chocolatey (`choco install ffmpeg`). Ensure `ffmpeg` is on your PATH.
   - **macOS:** Install via Homebrew (`brew install ffmpeg`).
   - **Linux:** Install via your package manager (`sudo apt install ffmpeg`).

---

## Setup & Installation

It is recommended to run the script inside a Python virtual environment. Follow the steps for your specific platform below:

### Windows (with NVIDIA GPU CUDA Support)

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/atomikfactory/atomik-scripts.git
cd atomik-scripts/python/video/transcribe-folder

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch with CUDA support (CUDA 12.1 example)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -U openai-whisper psutil
```

### macOS / Linux (Apple Silicon MPS or CPU/CUDA)

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/atomikfactory/atomik-scripts.git
cd atomik-scripts/python/video/transcribe-folder

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch (macOS M-series chips auto-route GPU via MPS)
pip install torch

# 4. Install dependencies
pip install -U openai-whisper psutil
```

---

## How to Run

To transcribe a directory, simply execute the script and pass the folder path:

```bash
python transcribe_folder.py "D:\MyVideos"
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `input_folder` | The path to the folder to scan recursively. | *(Required)* |
| `--model` | Whisper model size to use (`tiny`, `base`, `small`, `medium`, `large`). | `medium` |
| `--language` | Force a specific language (e.g., `English`, `Spanish`). Autodetects if omitted. | `None` |
| `--device` | Device to run inference on (`cpu`, `cuda`). Autodetects CUDA GPU if available. | `None` |
| `--force` | Re-transcribe and overwrite existing `.txt` transcript files. | `False` |

### Command Examples

**1. Use the lightweight `small` model on CPU:**
```bash
python transcribe_folder.py "./videos" --model small --device cpu
```

**2. Force a specific transcription language and override existing files:**
```bash
python transcribe_folder.py "/user/files/movies" --language English --force
```

**3. Run the largest model on your GPU:**
```bash
python transcribe_folder.py "D:\lecture_recordings" --model large --device cuda
```

---

## Outputs

For each video successfully transcribed, a plain text UTF-8 file containing the transcribed text is created right next to the original video file (e.g., `lecture_01.mp4` ➔ `lecture_01.txt`).

At the end of the execution, the script displays a summary report:

```text
============================================================
SUMMARY
============================================================
Total videos found:      3
Successfully transcribed: 3
Skipped (already done):  0
Failed:                  0
Total execution time:    4m 12s
============================================================
```
