# Speak and select an object in a saved photo

This demo records an English request, transcribes it locally with faster-whisper,
removes phrases such as “please pick the,” and passes the remaining object
description directly to the existing SAM 3 selector. Ollama Qwen
checks candidates by default. The result is an annotated photo and `result.json`.
SELECTED means visual selection only. This program has no arm connection.

From this project directory:

```sh
.venv/bin/python voice_pick.py --image wrist1.jpeg --show
```

Press Enter, then say **“Please pick the ring”** during the six-second recording.
Repeat for another request; Ctrl+C quits. Grant microphone access to the terminal
or launching app when macOS asks. `--list-devices` lists inputs; `--device N`
chooses one. The first speech run downloads `base.en` into `.voice_models`.

Use `--vlm none` for SAM-only selection without Ollama. This skips semantic
verification. `--text "pick the white ring"` bypasses recording and transcription;
`--audio recording.wav` tests saved speech. `--once` ends after one recording.
Outputs default to `review_outputs/voice_demo/` and are replaced each request.

Open descriptions are the default: try “pick the headphone case,” “pick the can,”
or another object description in your photo. No catalog entry is required.
Attributes are preserved rather than silently mapped to a different object.
Recognition is not guaranteed: the system may return NOT_FOUND or AMBIGUOUS,
and models can still make mistakes. Relational descriptions and requests for
multiple objects are not reliably resolved; request one object at a time.
Negated requests are conservatively cancelled. This is not a medication or
dosing system. `--vocabulary catalog` restores the previous restricted mode
using names and aliases in `voice_items.json`.

For another computer, install `requirements-voice.txt` in its virtual environment
and provision SAM 3 weights plus Ollama with `qwen2.5vl:7b`. The requirements are
a snapshot; a clean installation on the friend's computer still needs testing.

The future Reachy integration can pass its transcript to `select_from_text`.
This milestone does not implement Reachy communication, physical localization,
grasp execution, or verification that an object was lifted.

Validation: 54 unit tests passed after adding open descriptions. The earlier
catalog demo selected the white ring in wrist1.jpeg from text input.
Synthetic speech “Please pick the ring” transcribed correctly; “white ring” was
misheard as “wide ring” and rejected. Real microphone accuracy is not yet tested.
PyAV is pinned because version 19 rejected faster-whisper's decoder arguments.
The current macOS environment also reports duplicate AVFoundation classes from
PyAV and OpenCV when both load; this remains a deployment compatibility concern.
