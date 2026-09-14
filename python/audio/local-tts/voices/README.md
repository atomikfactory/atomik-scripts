# voices/

Each sub-folder is a reusable **voice profile**:

```
voices/
└── my_voice/
    ├── reference.wav     # required - 10-20 s of clean speech (WAV/FLAC/MP3/OGG/M4A)
    ├── reference.txt     # optional - exact transcript of the recording (helps Qwen3-TTS)
    ├── profile.json      # optional - {"language": "en", "description": "...", "options": {"exaggeration": 0.6}}
    └── .cache/           # generated automatically (pre-processed audio, speaker embeddings)
```

Create one from the command line:

```
python tts.py --save-voice-profile my_voice --voice-sample path/to/recording.wav --voice-text "what is said in the recording"
```

Use it:

```
python tts.py --input input/script.txt --voice-profile my_voice
```

Everything in this folder except this README is ignored by Git. Reference
recordings are personal data: only clone voices you own or have explicit
permission to use.
