# video-edit-agent

An Arabic-first, model-agnostic AI video editing agent. It turns a raw talking-head
or ad take into a captioned, cut, motion-graphics-enhanced final video — entirely
offline if you want, or enhanced by cloud providers if you give it API keys.

CLI command: `videoedit`. See [README.ar.md](README.ar.md) for the Arabic version.

## Philosophy

- **Arabic-first, dialect-preserving.** The agent transcribes exactly what was
  said — Egyptian, Gulf, Saudi, MSA, code-switched Arabic/English — and never
  rewrites, formalizes, or "corrects" the dialect into another one. Source audio
  has final authority. See `language/dialect_guard.py`.
- **Works with zero API keys.** Every cloud-backed feature (Gemini transcription
  and vision, ElevenLabs, Veo image/video generation) has a graceful, automatic
  fallback. With no keys at all, `videoedit edit` still transcribes locally
  (Faster-Whisper), builds an EDL, captions, composites, and renders a final
  video — just without cloud-only enhancements.
- **Cloud-enhanced, not cloud-dependent.** Set `GEMINI_API_KEY` and/or
  `ELEVENLABS_API_KEY` as environment variables (never in a config file) to
  unlock better transcription, generated B-roll, and visual QA.
- **Modular and inspectable.** Every stage (transcript, EDL, captions, B-roll
  plan, motion plan, QA report) is a plain JSON/text artifact under
  `<video>/edit/`, and `project.md` keeps a human-readable memory of what the
  agent decided and why.

## Install

```bash
pip install video-edit-agent
```

Or, for local (offline) transcription support:

```bash
pip install "video-edit-agent[local]"
```

`ffmpeg`/`ffprobe` must be on `PATH`. Run `videoedit doctor` after installing to
see exactly what's available on your machine.

## Quickstart

```bash
videoedit setup          # one-time environment check + optional local-whisper install
videoedit doctor         # see the full capability matrix
videoedit edit my_take.mp4
```

Output lands in `my_take_edit/` (an `edit/` folder next to the source video):
`final.mp4`, `transcript_unified.json`, `edl.json`, `captions.ass`, `project.md`,
and QA/B-roll/motion plan JSON files.

## Key commands

| Command | Purpose |
|---|---|
| `videoedit edit <video>` | Run the full pipeline: transcribe -> cut -> caption -> B-roll -> motion -> render -> QA |
| `videoedit doctor` | Print the capability matrix (ffmpeg, Node, providers, engines) |
| `videoedit providers` | List transcription/motion providers and their availability |
| `videoedit setup` | Interactive first-run wizard |
| `videoedit config show / set` | Inspect or edit layered config (never secrets) |
| `videoedit brand init / validate` | Create or validate a Brand Profile |
| `videoedit project inspect <dir>` | Print a project's `project.md` memory |

## Capability status (as of v0.1.1 hardening)

Honest per-capability status, using six categories:
`VERIFIED LOCALLY` (a real, non-mocked local acceptance run passed),
`VERIFIED WITH CLOUD` (a real live API call succeeded), `OPTIONAL` (not
required for the core pipeline), `FALLBACK AVAILABLE` (degrades gracefully
to a working alternative), `NOT VERIFIED` (code exists but has no real
acceptance evidence), `REQUIRES API KEY` (blocked until you set one). Run
`videoedit doctor` for a live, machine-specific view of this same matrix.

| Capability | Status | Notes |
|---|---|---|
| Local transcription (Faster-Whisper) | VERIFIED LOCALLY | Full offline `videoedit edit` pipeline runs end-to-end without any API key. |
| Gemini transcription | REQUIRES API KEY | Adapter is implemented and unit-tested; live call is skipped automatically unless `GEMINI_API_KEY` is set. Not exercised against the real API in this environment. |
| ElevenLabs transcription | REQUIRES API KEY / OPTIONAL | Never required for the pipeline to complete; live call skipped automatically unless `ELEVENLABS_API_KEY` is set. |
| Arabic captions (RTL, shaping, karaoke highlight, mixed AR/EN, brand fonts) | VERIFIED LOCALLY | Real ffmpeg render + ffprobe + pixel-content acceptance test passing (13/13 checks; `scripts/arabic_acceptance.py`). |
| Arabic ASR on real spoken audio | NOT VERIFIED | No real Arabic speech fixture/voice was available in this environment to run a live ASR acceptance test; dialect-preservation *logic* is covered by automated regression tests instead (Egyptian/Gulf/mixed-dialect cases). |
| Remotion motion rendering | FALLBACK AVAILABLE | Node/npm detected and renderable via `npx`, but no real acceptance render has been run and recorded in this environment (`videoedit doctor` shows "Verified: not yet"). |
| HyperFrames motion engine | NOT VERIFIED | No real HyperFrames SDK exists to install: the only same-named PyPI package is an unrelated N-dimensional DataFrame library with no rendering API. The motion router correctly falls back to a working engine (Simple/PIL) when HyperFrames is requested but unavailable, and this fallback is automated-test-covered and always recorded in `MotionPlanItem.fallback_log` — never a silent substitution. |
| Motion engine fallback (HyperFrames/Remotion -> Simple) | VERIFIED LOCALLY | Automated test proves an unavailable engine falls through to a working one, the project still completes, and every rejected engine is recorded in `fallback_log` (`tests/test_motion_router.py`). |
| Behind-Subject mask detection + caching | VERIFIED LOCALLY | Real (non-mocked) mediapipe segmentation runs against a real video fixture, and a second identical call reuses the cached mask instead of recomputing (`scripts/behind_subject_acceptance.py`, `tests/test_behind_subject.py`). Requires `mediapipe<1.0` (e.g. `mediapipe==0.10.21`) — mediapipe 1.0+ removed the API this project uses. |
| Behind-Subject video compositing (element rendered visibly behind a subject in the final output) | NOT VERIFIED | This is a real architecture gap, not just an unverified feature: `render/composition.py` never reads `Overlay.behind_subject`, so no code path in this repository can produce that visual result yet. Only the detection/caching half above is real. A permanent characterization test locks this in so it can't be silently claimed "fixed" later. |
| Brand Profiles | VERIFIED LOCALLY | Brand-driven fonts/colors/caption presets confirmed to reach the actual render pipeline. |
| Offline mode (no API keys) | VERIFIED LOCALLY | `videoedit edit --offline` produces `final.mp4` end-to-end with zero network calls. |

## Secrets

`GEMINI_API_KEY` and `ELEVENLABS_API_KEY` are read from environment variables
only. They are never written to a config file, never logged, and never printed
by any command, including `doctor`.

## Brand Profiles

Brands live as plain folders under `brands/<name>/` (colors, fonts, avoid-list,
preferred motion engine, caption preset). No source changes needed to add one:

```bash
videoedit brand init my_brand
videoedit brand validate my_brand
videoedit edit my_take.mp4 --brand my_brand
```

## Attribution

Inspired by, not forked from, [majedphotos/video-ad-editor](https://github.com/majedphotos/video-ad-editor)
and [browser-use/video-use](https://github.com/browser-use/video-use). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exactly what was adapted
versus independently implemented.

## License

MIT — see [LICENSE](LICENSE).
