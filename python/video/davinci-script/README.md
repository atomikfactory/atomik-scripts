# generate_voice.py — long-form narration for DaVinci Resolve

DaVinci Resolve Studio 20.x/21 has a native AI Speech Generator, exposed to
scripting as `Project.GenerateSpeech({settings}, timecode)`. It is excellent —
and it accepts at most **350 characters** per call. A ten-minute narration is
tens of thousands of characters.

This tool bridges that gap. Give it a `.txt` script and it will:

1. split the text into natural-language segments that never exceed a safe
   limit (default 300 characters);
2. generate every segment with Resolve's own AI Speech Generator;
3. cache the results so an interrupted run resumes instead of restarting;
4. lay the clips back-to-back, in script order, on a dedicated **AI Voice**
   audio track, starting at the playhead.

It never deletes, moves or overwrites anything already on your timeline.

```
Loading script...
8,432 characters found.
Split into 29 speech segments.

Generating segment 1/29... done (11.6s)
Generating segment 2/29... done (2.4s)
Generating segment 3/29... (cached)
...
Importing audio...
Building timeline...
Complete.
```

The first segment of a Resolve session is slow — around 12 seconds while the
speech model warms up. After that a 300-character segment takes 1–4 seconds.

Everything in this README marked *verified* was exercised against **DaVinci
Resolve Studio 21.0.1.11 on Windows 11 with Python 3.14.3**. See
*[What is verified, and what is not](#what-is-verified-and-what-is-not)*.

---

## Requirements

| Requirement | Notes |
|---|---|
| **DaVinci Resolve Studio 20.x or 21.x** | Running. `GenerateSpeech` is a Studio-only API; the free edition returns `False`/`None`. The tool also refuses to run against a major version below 20. |
| **AI Speech Generator package** | Install it from *DaVinci Resolve Studio → Extras Download Manager*, then restart Resolve. Without it, `GenerateSpeech` returns `None`. |
| **External scripting enabled** | *DaVinci Resolve → Preferences → System → General → "External scripting using"* must be **Local**, **and Resolve must be restarted afterwards**. Verified: on 21.0.1 `scriptapp("Resolve")` returned `None` until both were done. The preference is stored as `System.Scripting.Mode` (0 = none, 1 = local) in `%APPDATA%\Blackmagic Design\DaVinci Resolve\Preferences\config.dat`. |
| **A project and a timeline open** | The tool places clips on the *current* timeline. Any page works — verified from the Edit page. |
| **Python 3.9+, 64-bit** | See the interpreter note below. |
| **Third-party packages** | **None.** `requirements.txt` deliberately installs nothing. |

### Which Python to use (important on this machine)

Resolve's `fusionscript` library links against `python3.dll` (the stable ABI)
and is fussy about which interpreter loads it.

On **this machine**, importing `fusionscript.dll` under **Python 3.11.9 crashes
with an access violation** — the process dies before any Python code runs, so
you get no traceback and no error message, just a silent exit. **Python 3.14.3
imports it fine.** The `python` command here resolves to a 3.11 virtual
environment, so use the launcher with an explicit version:

```powershell
py -3.14 generate_voice.py script.txt
```

…or an explicit interpreter path:

```powershell
& "C:\Users\<you>\AppData\Local\Programs\Python\Python314\python.exe" generate_voice.py script.txt
```

The tool prints the interpreter version and path at INFO level on startup, so
the log always records which Python actually connected. The unit tests are pure
Python and pass under 3.11 as well as 3.14 — only *connecting to Resolve* needs
the newer interpreter.

## Installation

There is nothing to install. Copy the folder somewhere and run it:

```
davinci-script/
  generate_voice.py
  resolve_tts/
  tests/
  example_script.txt
```

### Environment variables (optional)

The tool finds Resolve's scripting module automatically using the platform
defaults documented in Blackmagic's `README.txt`. Set these only if you have a
non-standard installation:

| Platform | Variable | Default |
|---|---|---|
| Windows | `RESOLVE_SCRIPT_API` | `%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting` |
| Windows | `RESOLVE_SCRIPT_LIB` | `C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll` |
| macOS | `RESOLVE_SCRIPT_API` | `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting` |
| macOS | `RESOLVE_SCRIPT_LIB` | `/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so` |
| Linux | `RESOLVE_SCRIPT_API` | `/opt/resolve/Developer/Scripting` (or `/home/resolve/...`) |
| Linux | `RESOLVE_SCRIPT_LIB` | `/opt/resolve/libs/Fusion/fusionscript.so` (or `/home/resolve/...`) |

On Windows neither variable is needed: the probe above found Resolve with both
unset. The macOS and Linux defaults have not been tested.

`PYTHONPATH` is tried first; add `$RESOLVE_SCRIPT_API/Modules` to it if you
prefer. Failure to import prints every path that was tried.

Convenience overrides, all optional: `RESOLVE_TTS_OUTPUT_DIR`,
`RESOLVE_TTS_VOICE`, `RESOLVE_TTS_TRACK_NAME`, `RESOLVE_TTS_MAX_CHARS`,
`RESOLVE_TTS_LOG_LEVEL`.

## Usage

```powershell
# Check the environment first — this is the fastest way to diagnose anything.
py -3.14 generate_voice.py --probe

# See how the script will be split, without contacting Resolve at all.
py -3.14 generate_voice.py example_script.txt --dry-run

# The normal run: generate everything and lay it at the playhead.
py -3.14 generate_voice.py script.txt

# Pick a voice and start at the very beginning of the timeline.
py -3.14 generate_voice.py script.txt --voice "Male 1" --start timeline-start

# Start at an explicit timecode, on a track you name yourself.
py -3.14 generate_voice.py script.txt --start 01:00:30:00 --track-name "VO"

# Try a voice setting on one segment before committing to a whole script.
py -3.14 generate_voice.py script.txt --only 1 --speed 90

# Resume: just run the same command again. Finished segments are reused.
py -3.14 generate_voice.py script.txt

# Re-attempt only the segments that failed, then place everything.
py -3.14 generate_voice.py script.txt --retry-failed
py -3.14 generate_voice.py script.txt --place-only

# Regenerate specific segments (1-based), e.g. because you disliked a reading.
py -3.14 generate_voice.py script.txt --only 5,7-9

# Generate now, place later.
py -3.14 generate_voice.py script.txt --skip-placement
py -3.14 generate_voice.py script.txt --place-only

# A quarter-second pause between segments instead of back-to-back.
py -3.14 generate_voice.py script.txt --gap-frames 6

# A custom voice.
py -3.14 generate_voice.py script.txt --custom-voice-file "D:\voices\me.wav"

# No file argument opens a file-picker dialog.
py -3.14 generate_voice.py
```

Run `--help` for the complete, grouped option list.

## Where files go

Everything for one script lives in one directory, `./output/<script-name>/` by
default (`--output-dir` moves it):

```
output/script/
  manifest.json          progress + everything learned about each segment
  generate_voice.log     full DEBUG log with timestamps
  segments/
    001.wav              our cached copy of Resolve's audio
    002.wav
```

**Resolve writes its own audio somewhere else, and we do not move it.**
Verified: the generated WAVs land in the project's working folder,
`<project media location>\Audio Files\SpeechGenerator\`. The `Filename` value
in the settings dictionary is a **prefix, not a path**; Resolve builds the
actual name as

```
<Filename>-<VoiceModel>-G<GenerationID>-<NNN>.wav
     e.g.  AIVoice_script_003-Male 1-G1-001.wav
```

where `<NNN>` auto-increments to avoid collisions and `GenerationID` defaults
to 1. Omitting `Filename` altogether gives every file the prefix `SpeechGen`,
so this tool always supplies one: `AIVoice_<script-stem>_<NNN>`.

The tool discovers the real location by reading the returned clip's
`"File Path"` property, records it in the manifest, and copies the file into
`segments/NNN.wav` as a cache. `segments/*.wav` are **copies**, kept as a
safety net: Resolve keeps referencing its own files where it wrote them, and
this tool never moves, renames or replaces Resolve's audio. The cached copy is
only used to re-import a clip if Resolve's original has since disappeared.

Generated clips are put in a Media Pool bin named **AI Voice** (`--bin-name`).
Verified: `GenerateSpeech` creates the new MediaPoolItem in whatever folder is
*current*, so the tool makes the bin current for the run and restores your
previous selection at the end.

## What Resolve's generated WAV files look like

This matters because it broke the obvious implementation. Verified layout of a
generated clip:

| | |
|---|---|
| Chunks, in order | `JUNK` (28 bytes), `fmt ` (40 bytes), `bext` (968 bytes), `data` |
| `fmt ` | WAVE_FORMAT_EXTENSIBLE, tag `0xFFFE`, SubFormat GUID `00000003-0000-0010-8000-00aa00389b71` (IEEE float) |
| Audio | 48,000 Hz, 1 channel (mono), 32-bit float, block align 4 |

Python's `wave` module **cannot open these files**:

```
wave.Error: unknown extended format: 00000003-0000-0010-8000-00aa00389b71
```

So `resolve_tts/audio_utils.py` walks the RIFF chunks itself with `struct`. It
handles chunk padding to even sizes, format tags 1 (PCM), 3 (IEEE float) and
`0xFFFE` (extensible — the real format is the first two bytes of the SubFormat
GUID), sample widths of 8/16/24/32-bit integer and 32/64-bit float, and any
channel count. Duration is `data_bytes / block_align / sample_rate`. A
truncated `data` chunk is used as far as it goes and flagged; anything
genuinely unparseable comes back as `unknown` with an explanation and never
raises into the pipeline. `audioop`, removed in Python 3.13, is not used.

## How voice selection works (and its limitation)

The scripting API takes a `VoiceModel` **string**, and there is **no API call
that enumerates the available voices**. Blackmagic's documentation gives only
three examples: `"Female 1"`, `"Male 1"` and `"Custom Voice"`.

Because of that:

* **`--voice` is omitted by default.** When you do not pass it, `VoiceModel`
  is left out of the settings dictionary entirely and Resolve uses whatever
  voice the Speech Generator is currently set to. This is the safest default:
  verified, a voice name that does not exist makes `GenerateSpeech` return
  `None` (in about 0.02 s).
* To pick a voice, open the Speech Generator in the Resolve UI, read the name
  exactly as displayed, and pass it: `--voice "Male 1"`.
* `--custom-voice-file <path>` sets `VoiceModel` to `"Custom Voice"` and
  `CustomVoiceFile` to your path, in one step.
* `--generation-id` sets `GenerationID`, which is **verified to be a seed**:
  the same id and the same text produce byte-identical audio. Leave it alone
  and Resolve uses 1 for every segment, which is what you want — one
  consistent voice character across the whole narration. Change it only to
  re-roll a reading you disliked (with `--only 5`, say).
* `--speed`, `--variation` and `--pitch` are passed through **only if you
  supply them, and completely unvalidated**. Verified: Resolve accepted Speed
  2, 50, 100 and 1000 and Pitch −50 without complaint and generated audio for
  all of them. The ranges are undocumented and this tool does not invent
  limits, so **try a value on one segment first**: `--only 1 --speed 90`.

## How chunking works

The splitter descends through natural boundaries, only going one level deeper
when a piece still does not fit:

1. **Paragraphs.** Blank lines separate them. Bullet and numbered list lines
   (`-`, `*`, `•`, `1.`, `2)`) each start their own paragraph, because a list
   item is a natural pause. Segments are never packed across a paragraph
   boundary unless you pass `--merge-paragraphs`.
2. **Sentences.** Robust against the things that break naive `.split(".")`:
   abbreviations (`Mr.`, `Dr.`, `St.`, `vs.`, `e.g.`, `i.e.`, `etc.`, `U.S.`,
   `a.m.`), initials (`J. R. R. Tolkien`), decimals (`3.14`, `61.4%`),
   ellipses (`...`, `…`), closing punctuation after a full stop (`."`, `.")`,
   `?'`), CJK terminators (`。`, `！`, `？`), and text with no terminal
   punctuation at all. URLs and e-mail addresses are protected: no break is
   ever placed inside one.
3. **Clauses**, for a sentence that is itself too long: after `,` `;` `:`
   `—` `–` `--` `)` `]`, after a spaced dash, and before an opening bracket.
4. **Words**, for a clause that is still too long.
5. **A hard cut inside a word** — and only when a single "word" (in practice, a
   very long URL) is longer than the limit on its own. This is the one case
   where a character boundary is chosen without regard for meaning.

Within a paragraph, line breaks and runs of whitespace collapse to single
spaces, so the engine reads it as one flow. Nothing else is touched: quotes,
apostrophes, dashes, parentheses, numbers, contractions and non-ASCII
characters are passed through verbatim. A UTF-8 BOM is stripped, CRLF is
normalised, and Unicode is normalised to NFC.

The guarantees, enforced structurally and asserted in the tests:

* every segment is at most `--max-chars` characters;
* no segment is empty, and every segment is stripped;
* order is preserved;
* **nothing is added, deleted, duplicated or rewritten** — concatenating the
  segments reproduces the source text exactly, ignoring whitespace;
* with `--merge-paragraphs` off, every segment is a contiguous substring of its
  whitespace-collapsed source paragraph.

Preview the split at any time with `--dry-run`; it never contacts Resolve.

> A note on the abbreviation list: it deliberately excludes abbreviations that
> are also common English words (`no.`, `Sat.`, `Sun.`, `Mar.`). Including them
> would swallow real sentence endings like "The answer is no." — a much worse
> failure than an occasional over-long sentence, which the clause splitter
> handles anyway.

## How resume works

`manifest.json` records the script's SHA-256, the chunking parameters, the
voice settings, and — per segment — its text, status
(`pending`/`generated`/`placed`/`failed`), attempt count, error, the file path
Resolve reported, our cached copy, the MediaPoolItem's identifiers, the audio
properties, the measured silence, the generation time and the timeline
placement.

It is rewritten **atomically after every single segment** (temp file plus
`os.replace`), so a crash, a power cut or a Ctrl-C never loses completed work.

On a re-run:

* If the script's SHA-256 **and** the chunking parameters match, the manifest
  is reused and every `generated`/`placed` segment is skipped and reported as
  `(cached)`. The audio file is verified to still exist; if Resolve's original
  is gone but our cached copy survives, the copy is re-imported with
  `MediaPool.ImportMedia`.
* If the script text or `--max-chars`/`--merge-paragraphs` changed, the tool
  **refuses** and explains why, because segment 12 of the old split is not
  segment 12 of the new one. Pass `--force-rechunk` to archive the old
  manifest as `manifest.<timestamp>.json` and start over, or point
  `--output-dir` somewhere else to keep both.

Related flags: `--retry-failed` (attempt only failed segments), `--only 5,7-9`
(regenerate specific segments; placement is skipped so existing clips are never
disturbed), `--max-attempts N` (attempts per segment per run, default 2),
`--stop-on-error`.

## Timeline behaviour and safety

* **Track.** An audio track named `AI Voice` (`--track-name`) is reused if it
  already exists, otherwise created and renamed. Generated clips are mono
  (verified), so a **mono** track is created; verified that
  `AddTrack("audio", "mono")` returns `True`, that `GetTrackCount("audio")`
  reflects the new track immediately, that `SetTrackName` works, and that
  `GetTrackSubType` then reports `mono`. `--track-index N` targets an existing
  track by number instead and never creates one. A locked track is refused.
* **Start.** `--start playhead` (default) reads `GetCurrentTimecode()`;
  `--start timeline-start` uses `GetStartFrame()`; or give an explicit
  `HH:MM:SS:FF` / `HH:MM:SS;FF` timecode. `GetSetting("timelineFrameRate")`
  returns a float (verified: `24.0`).
* **Frame counts come from the WAV, floored.** A clip's frame count is
  `floor(duration_seconds × fps)`. This is not cosmetic: a 4.40 s clip on a
  24 fps timeline is 105 frames, not 106 (4.40 × 24 = 105.6), and Resolve
  agrees — its `"Duration"` property reads `00:00:04:09`. Asking
  `AppendToTimeline` for 106 frames was *accepted* and produced a clip
  reaching past the end of the media, so the tool always floors.
  The clip property `"Frames"` is an **empty string** for these audio clips
  and is never passed to `int()`; if the WAV cannot be parsed the tool falls
  back to the `"Duration"` timecode, and only then to `"Frames"` if it happens
  to hold a number.
* **`endFrame` is exclusive.** In `MediaPool.AppendToTimeline([{clipInfo}])`
  the placed duration is `endFrame − startFrame` (verified: `startFrame 0,
  endFrame 105` → 105 frames; `startFrame 3, endFrame 103` → 100 frames, with
  `GetLeftOffset() == 3` and `GetRightOffset() == 2`). Omitting both places the
  whole clip.
* **Order.** Clips are placed one at a time, in script order, each starting
  where the previous one ended (plus `--gap-frames`, default 0). The next
  position is taken from the returned `TimelineItem.GetEnd()`, which is also
  exclusive — verified contiguous: 515 → 621 → 728 → 828 → 935. A warning is
  logged only if the *placed* length differs from the requested
  `endFrame − startFrame`, and the following clips then follow the actual
  length rather than drifting.
* **`mediaType: 2`, `trackIndex` (1-based) and `recordFrame` are all honoured**
  (verified), other tracks are left untouched, and the timeline does not need
  to be on any particular page.
* **Safety guarantees.** The tool never calls `DeleteClips`, never moves an
  existing clip, and never writes over Resolve's audio files. Before placing
  anything it checks the target track for clips overlapping the intended range
  and **aborts with exit code 4** if it finds any, telling you which clip is in
  the way. If any segment failed to generate, placement is skipped entirely
  rather than putting an incomplete narration on your timeline — the tool then
  prints how many segments have succeeded so far and the two commands to
  finish the job (`--retry-failed`, then `--place-only`).
* **Saving.** `ProjectManager.SaveProject()` is called at the end and its
  result is logged.

### Pauses between segments, and why nothing is trimmed

The measurements below are from real generated clips, at the default
`--silence-threshold-db -50`:

| | |
|---|---|
| Leading silence | **24–32 ms** |
| Trailing silence | **48–76 ms** |
| First non-zero sample | 0 ms — the padding is low-level noise, not digital silence |
| One frame at 24 fps | 41.7 ms |

Back-to-back placement therefore leaves only about **70–110 ms** between the
end of one spoken phrase and the start of the next, which is *shorter* than a
natural sentence pause. Nothing needs trimming, and this tool never inserts
artificial silence.

So **`--trim-silence` is off by default** and is an opt-in flag. Frame
quantisation means it could only ever remove 0–1 frame from the padding above,
which is not worth the risk of clipping a word. Silence is still measured on
every segment and recorded in the manifest, so you can see the real numbers in
`manifest.json` or in the DEBUG log.

If you want **more** space between segments, use `--gap-frames N`:
6–12 frames at 24 fps (250–500 ms) is a natural inter-sentence pause. The
default stays 0.

When you do pass `--trim-silence`, the trim uses the clip's in/out points only
— `startFrame` and `endFrame` in the `clipInfo`. The audio file itself is never
edited, so nothing is destroyed and you can drag the handles back out in
Resolve.

* A frame counts as silence when its **peak** across all channels is at or
  below `--silence-threshold-db` (default −50 dBFS; full scale is 1.0 for the
  float samples Resolve writes). Peak rather than RMS is deliberately
  conservative: it under-reports silence.
* `--keep-pad-ms` (default 40) of silence is kept at each end so speech never
  starts abruptly. With the measured padding this alone means no trim happens.
* Frame quantisation errs outwards: the leading trim is floored and the
  trailing keep is ceiled, so rounding can only ever leave silence in, never
  clip a word.
* Trimming does not apply to `--placement resolve`, which is logged as a
  warning if you combine them.

### Placement modes

* **`--placement append` (default, and the one that works).** `AddToTimeline`
  is `False`, and the tool places each returned MediaPoolItem itself with
  `MediaPool.AppendToTimeline([{clipInfo}])`. This gives exact ordering, exact
  positions and in/out trimming.
* **`--placement resolve` (experimental — observed not to work).**
  `AddToTimeline` is `True`, `AudioTrack` is the target track index, and each
  `GenerateSpeech` call is given the running timecode. **Verified failure on
  Resolve Studio 21.0.1 from the Edit page:** with
  `{"AddToTimeline": True, "AudioTrack": 2, ...}` and timecode `00:00:38:23`,
  the MediaPoolItem *was* created in the current bin and the playhead *did*
  move to the requested timecode, but **no clip was added to any audio track**.
  The mode is kept for the day Blackmagic fixes it, and the tool now compares
  `GetItemListInTrack` before and after every call; if no new clip appears it
  stops the run with:

  > Resolve created the clip but did not place it (observed on 21.0.1 from the
  > Edit page). Use the default `--placement append`, or `--place-only` to
  > place the already generated segments.

  Nothing is lost when this happens — the audio is generated and cached, so
  `--place-only` finishes the job.

## Limitations imposed by Resolve

* **350 characters per call.** Not negotiable; it is why this tool exists.
  Verified: 359 characters of `TextInput` returns `None` in 0.01 s. The tool
  asserts the limit immediately before every call regardless of `--max-chars`.
* **`GenerateSpeech` returns only a MediaPoolItem.** It is synchronous
  (verified) but there is no progress callback, no job id, no way to cancel,
  and no documented way to know where the WAV was written. The tool discovers
  the location by reading the returned clip's `"File Path"` property, logs it,
  and records it in the manifest. If `"File Path"` is empty, the whole property
  dictionary is dumped to the log at DEBUG level and the segment fails with a
  clear message.
* **A `None` return tells you nothing by itself** — so the tool times the call.
  Verified: an invalid `VoiceModel` returns `None` in 0.02 s and over-long text
  in 0.01 s, whereas a working call takes seconds. A `None` returned in under
  half a second is therefore reported as a *settings* problem (voice name,
  text length, custom voice path); a slower one as an *engine* problem
  (package not installed, free edition, unsupported machine).
* **The voice list is not exposed.** See *How voice selection works*.
* **`Speed`, `Variation` and `Pitch` have no documented ranges** and Resolve
  validates nothing. Passed through untouched only when you supply them.
* **The second positional timecode argument is irrelevant** when
  `AddToTimeline` is `False` (the tool passes the current timecode, which is
  harmless).
* **`Project.GenerateSpeech` presence cannot be probed properly.** On 21.0.1
  `hasattr(project, "GenerateSpeech")` reports a "Remote Function" object,
  which is a genuine presence check on that build — but the bridge is generous,
  so the tool also refuses to run against a Resolve major version below 20.
* **The clip properties are thin.** For a generated clip (verified):
  `"Frames"` is `""`, `"Duration"` is a floored timecode such as
  `00:00:04:09`, `"FPS"` is the float `24.0` (the *timeline* rate, not a media
  property), `"Sample Rate"` is `"48000"`, `"Audio Ch"` is `"1"`,
  `"Audio Bit Depth"` is `"32"`, `"Format"` is `"Wave"`, `"Audio Codec"` is
  `"Linear PCM"`, `"Type"` is `"Audio"`, and `"File Path"`, `"Clip Directory"`
  and `"File Name"` hold the absolute path, its folder and the bare name.

## What is verified, and what is not

**Verified on DaVinci Resolve Studio 21.0.1.11 / Windows 11 / Python 3.14.3:**

* connection, the scripting preference and the restart it needs;
* `GenerateSpeech` is synchronous, returns a MediaPoolItem, warms up on the
  first call, and creates the clip in the current Media Pool folder;
* the `Filename` prefix and the `<Filename>-<VoiceModel>-G<ID>-<NNN>.wav`
  naming, and the `Audio Files/SpeechGenerator` output folder;
* `GenerationID` behaves as a seed; `Speed`/`Variation`/`Pitch` are
  unvalidated; an invalid voice or over-long text returns `None` immediately;
* the WAV format (extensible IEEE float, 48 kHz mono 32-bit) and the fact that
  Python's `wave` module rejects it;
* the measured leading/trailing silence;
* the clip properties listed above, including the empty `"Frames"`;
* `AppendToTimeline` with `mediaType`, `trackIndex`, `recordFrame`, and an
  **exclusive** `endFrame`; `TimelineItem.GetStart()`/`GetEnd()` semantics and
  contiguous placement from `GetEnd()`;
* `AddTrack("audio", "mono")`, `GetTrackCount`, `SetTrackName`,
  `GetTrackSubType`, `GetSetting("timelineFrameRate")` returning `24.0`;
* that `--placement resolve` (`AddToTimeline: True`) **does not place a clip**.

**Not verified:**

* macOS and Linux module/library paths (the code follows Blackmagic's
  documented defaults);
* stereo generated audio and stereo `AI Voice` tracks — every generated clip
  observed was mono, so the stereo branch is untested;
* drop-frame timelines (29.97 DF / 59.94 DF) against a live Resolve; the
  timecode arithmetic itself is covered by dense round-trip unit tests;
* Resolve 20.x (this was 21.0.1.11) and the free edition's exact failure mode;
* frame rates other than 24 against a live timeline.

Run with `--log-level DEBUG` and read `generate_voice.log`; every call, every
argument and every return value is recorded there.

## Troubleshooting

**`Could not connect to DaVinci Resolve (scriptapp returned None).`**
Resolve is not running, or external scripting is off. Set *DaVinci Resolve →
Preferences → System → General → "External scripting using"* to **Local** and
then **restart Resolve** — verified on 21.0.1, the change does not take effect
until you do. This is by far the most common cause.

**Python exits silently with no output, no traceback, no error.**
`fusionscript` crashed the interpreter on import. Try a different Python — on
this machine 3.11.9 crashes and 3.14.3 works. Use `py -3.14 generate_voice.py`.
Make sure the interpreter is 64-bit.

**`Could not import the DaVinci Resolve scripting module.`**
The message lists every path that was tried. Set `RESOLVE_SCRIPT_API` and
`RESOLVE_SCRIPT_LIB` (see the table above) and try again.

**`GenerateSpeech returned None ... after 0.02s`.**
That speed means the settings were rejected, not the engine: the `--voice` name
does not exist (drop `--voice` to use Resolve's current voice), the text is
empty or over 350 characters, or `--custom-voice-file` cannot be read.

**`GenerateSpeech returned None ... after 6.10s`.**
That points at the engine: the AI Speech Generator package is not installed
(*Extras Download Manager* → install → **restart Resolve**), this is the free
edition rather than Studio, or the machine does not meet the feature's
requirements — run the Speech Generator once from the UI and read the error
dialog.

**`Resolve created the clip but did not place it ...`**
You used `--placement resolve`, which does not work on 21.0.1. The audio is
already generated and cached, so just run `--place-only`.

**`MediaPool.AppendToTimeline returned ...` / nothing gets placed.**
Check that the target track really is an audio track and is unlocked, that the
segment's audio file still exists on disk, and that the clip layout matches the
track (mono clip on a mono track). Do **not** reach for `--placement resolve`;
it places nothing at all on this build.

**`Audio track A2 already has N clip(s) between ...` (exit code 4).**
Something is already there and this tool will not touch it. Move the playhead
somewhere clear (`--start <timecode>`), choose another track
(`--track-name` / `--track-index`), or delete the old clips yourself.

**`The existing manifest cannot be reused because ...`**
You edited the script, or changed `--max-chars`/`--merge-paragraphs`. Use
`--force-rechunk` to archive the old manifest and start over, or a different
`--output-dir`.

**`Could not determine where Resolve wrote the audio for segment N.`**
Re-run with `--log-level DEBUG` and search `generate_voice.log` for
`Full clip property dict` — it contains every property key Resolve exposes for
that clip, which will show where the file actually is.

**The clips sound too tight together.**
Add `--gap-frames 6` (a quarter second at 24 fps) or `--gap-frames 12` (half a
second). Trimming is not the answer; there is only ~70–110 ms of padding
between phrases to begin with.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Usage or input error (bad arguments, missing/empty/non-UTF-8 script) |
| 2 | Resolve connection or validation failure |
| 3 | Some segments failed to generate |
| 4 | Timeline placement refused for safety |
| 130 | Interrupted (Ctrl-C); progress is saved |

## Running the tests

```powershell
py -3.14 -m unittest discover -s tests -v
```

The suite is pure Python — it never imports `fusionscript` and never contacts
Resolve — so it runs under any supported interpreter, including the ones that
cannot load Resolve's library:

```powershell
& "C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe" -m unittest discover -s tests -v
```

197 tests, green on 3.14.3 and 3.11.9, covering:

* the chunker, including a randomised property test asserting the invariants
  above over 200 generated documents;
* timecode conversion, including dense drop-frame round-trips;
* the RIFF walker and silence detection at every supported width and format —
  including a file that reproduces Resolve's real layout byte for byte
  (`JUNK` + extensible-float `fmt ` + `bext` + `data`), a test that asserts
  Python's own `wave` module *fails* on it, truncated and odd-sized chunks, and
  the measured 28 ms / 60 ms padding;
* the frame arithmetic (flooring, the empty `"Frames"` property, the exclusive
  `endFrame`) and timeline placement against a fake Resolve that implements the
  verified semantics, asserting the 515 → 621 → 728 → 828 → 935 contiguity,
  gaps, track creation, overlap refusal and the "place nothing if any segment
  failed" rule;
* manifest persistence and resume logic.

## Layout

```
generate_voice.py           CLI entry point
resolve_tts/
  __init__.py               version and the error hierarchy
  chunker.py                the text splitter (pure, no Resolve dependency)
  timecode.py               timecode <-> frames, drop-frame aware
  audio_utils.py            RIFF/WAVE inspection and silence detection
  manifest.py               JSON manifest persistence and resume logic
  resolve_connect.py        platform paths, connection, validation, --probe
  pipeline.py               orchestration: load, chunk, generate, place
tests/                      unittest suite
example_script.txt          a realistic ~2,200-character sample
```
