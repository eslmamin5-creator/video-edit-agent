# video-edit-agent

An Arabic-first, model-agnostic AI video editing agent. It turns a raw talking-head
or ad take into a captioned, cut, motion-graphics-enhanced final video — entirely
offline if you want, or enhanced by cloud providers if you give it API keys.

CLI command: `videoedit`. See [README.ar.md](README.ar.md) for the Arabic version.

Three agents cover three workflows, all offline-capable:

- **Editor** (`videoedit edit`) — turns one raw take into a captioned, cut final video.
- **Creator** (`videoedit create`) — turns a script into a full storyboard, asset plan, and rendered final video.
- **Assembler** (`videoedit assemble`) — turns a folder of pre-shot scenes into a rough cut (`--rough`) and then a finished film (`--finish`) with real transitions and loudness normalization between scenes.

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

### Option A — Claude Code Skill (recommended for most users)

1. Clone or install this repo into your Claude Code skills directory.
2. Open (or restart) Claude Code.
3. Say: **"Set up video-edit-agent and verify it."**

Claude bootstraps everything for you: detects your OS and a compatible
Python (prefers 3.11), builds a private runtime this project owns
(`.runtime/venv` — never your global/system Python), installs the offline
profile, checks `ffmpeg`/`ffprobe` and Node/npm, and reports readiness in
plain language. It never installs OS-level software silently — if
`ffmpeg` is missing it gives you the exact command for your OS
(`winget`/`choco`/`scoop` on Windows, Homebrew on macOS, your distro's
package manager on Linux) and waits for you to run it. No API key is ever
required or stored.

Once it reports ready, just talk to it naturally:

- "عدل الفيديو ده" → Editor
- "اعمل فيديو من السكربت ده" → Creator
- "اجمع المشاهد دي في فيلم" → Assembler

You never need to understand Python virtual environments, pip extras, or
provider routing to use this path.

### Option B — Technical CLI install (source, for power users)

```bash
git clone https://github.com/eslmamin5-creator/video-edit-agent
cd video-edit-agent
python3 -m pip install -e .          # base CLI only
videoedit setup --profile full-local # builds .runtime/venv, installs the offline profile
```

On Windows (PowerShell), no WSL required:

```powershell
git clone https://github.com/eslmamin5-creator/video-edit-agent
cd video-edit-agent
py -3.11 -m pip install -e .
videoedit setup --profile full-local
```

`videoedit setup` supports:

| Command | What it does |
|---|---|
| `videoedit setup` | Full bootstrap with the recommended `full-local` (offline) profile. |
| `videoedit setup --profile local\|subject\|motion\|core` | Bootstrap with a narrower install profile. |
| `videoedit setup --check` | Read-only: reports readiness, changes nothing. |
| `videoedit setup --repair` | Force-rebuilds the private runtime from scratch, even if it currently appears healthy; never touches anything outside `.runtime/`. |

Re-running `videoedit setup` is always safe — it's idempotent and reuses a
healthy runtime instead of reinstalling.

Installing from PyPI (`pip install video-edit-agent`) still works for the
base CLI and any individual extra (`pip install "video-edit-agent[local]"`),
but a GitHub/source install plus `videoedit setup` is the first-class,
fully-supported path — PyPI is not the only story.

**Supported OS:** Windows (PowerShell, no WSL needed), macOS, Linux.
**Python:** 3.10+ required; 3.11 preferred and auto-selected when available.
**ffmpeg/ffprobe:** required for any rendering; `videoedit setup`/`doctor`
detect it and give exact install guidance if missing — never installed for
you automatically.
**Node/npm:** optional — only needed for the Remotion motion engine; the
built-in simple motion engine works fully offline without it.
**API keys:** always optional. Zero keys is a fully supported, fully
functional setup.

Run `videoedit doctor` any time to see the full capability matrix and the
last `videoedit setup` run's summary.

## Quickstart

```bash
videoedit setup           # one-time bootstrap: private runtime + offline profile + capability check
videoedit doctor           # see the full capability matrix
videoedit edit my_take.mp4
```

Output lands in `my_take_edit/` (an `edit/` folder next to the source video):
`final.mp4`, `transcript_unified.json`, `edl.json`, `captions.ass`, `project.md`,
and QA/B-roll/motion plan JSON files.

## Key commands

| Command | Purpose |
|---|---|
| `videoedit edit <video>` | Run the full pipeline: transcribe -> cut -> caption -> B-roll -> motion -> render -> QA |
| `videoedit create <script>` | Turn a script into storyboard -> asset plan -> motion -> render -> QA |
| `videoedit assemble <scenes_dir> --rough` | Build a fast rough cut from a folder of pre-shot scenes |
| `videoedit assemble <scenes_dir> --finish` | Finish the film: real transitions + loudness normalization between scenes, reusing the rough cut's plan |
| `videoedit doctor` | Print the capability matrix (ffmpeg, Node, providers, engines) |
| `videoedit providers` | List transcription/motion providers and their availability |
| `videoedit setup [--profile ...] [--check] [--repair]` | Bootstrap the private runtime and check environment readiness |
| `videoedit config show / set` | Inspect or edit layered config (never secrets) |
| `videoedit brand init / validate` | Create or validate a Brand Profile |
| `videoedit project inspect <dir>` | Print a project's `project.md` memory |

## Capability status (as of v0.2.1)

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
| Behind-Subject mask detection + caching | VERIFIED LOCALLY | Real (non-mocked) mediapipe segmentation runs against a real video fixture, and a second identical call reuses the cached mask instead of recomputing (`scripts/behind_subject_acceptance.py`, `tests/test_behind_subject.py`). Requires exactly `mediapipe==0.10.21` — `mediapipe<1.0` alone is not a safe bound: 0.10.35 (still <1.0) already dropped the legacy `mediapipe.solutions.selfie_segmentation` API this project uses (confirmed via fresh-install testing), and mediapipe 1.0+ removes it entirely. `videoedit setup --profile subject|full-local` installs the verified-working pin automatically. |
| Behind-Subject video compositing (element rendered visibly behind a subject in the final output) | VERIFIED LOCALLY | Fully implemented: `core/pipeline.py` renders a real RGBA subject cutout (`subject/compositor.py::render_subject_cutout()`, real ffmpeg extraction + qtrle alpha encoding) and places it, in overlay list order, on top of the graphic overlay it's meant to appear in front of — `render/composition.py::build_filter_complex` composites `plan.overlays` sequentially via ffmpeg's ordinary alpha-aware `overlay` filter, so no special-casing is needed there. Falls back traceably (logged, never silent) to a plain foreground overlay when no usable mask exists. 8/8 tests in `tests/test_behind_subject.py` pass with zero mocks and zero skips (real mediapipe segmentation, real mask-cache reuse, real ffprobe-verified RGBA output, real end-to-end ffmpeg render). |
| Brand Profiles | VERIFIED LOCALLY | Brand-driven fonts/colors/caption presets confirmed to reach the actual render pipeline. |
| Offline mode (no API keys) | VERIFIED LOCALLY | `videoedit edit --offline`, `videoedit create --offline`, and `videoedit assemble --offline` each produce a valid `final.mp4` end-to-end with zero network calls. |
| Real video transitions (hard cut, short crossfade, dissolve) rendered by Assembler | VERIFIED LOCALLY | The Transition Director's plan is genuinely rendered via ffmpeg `xfade` (video) — a transition is only ever reported `applied=true` when the renderer actually produced that effect. Crossfade acceptance test verifies output duration reflects the overlap and that intermediate frames contain real blended content, not a hard cut (`tests/test_phase2_transitions_loudness.py`). Dissolve reuses the same `xfade` mechanics as short crossfade by design — there is no meaningfully different underlying effect to implement separately. |
| Audio crossfade accompanying a visual crossfade | VERIFIED LOCALLY | When both clips on either side of a visual crossfade have real audio, ffmpeg `acrossfade` is applied so there's no click or abrupt volume jump; if only one side has real audio, the audio falls back to a plain concat instead of being forced into a crossfade. |
| Loudness normalization (Assembler) | VERIFIED LOCALLY | Conservative `loudnorm`-based normalization: only scenes that deviate from the target level are normalized, true digital silence is left untouched (avoids a known ffmpeg `loudnorm` NaN failure mode on pure silence), and every decision records whether normalization was required/applied and at what target. Acceptance test verifies no clipping in the final render (`tests/test_phase2_transitions_loudness.py`). |
| J-Cut / L-Cut audio-lead/audio-trail edits | NOT IMPLEMENTED | Conceptual only — the Transition Director never proposes one. Intentionally deferred past v0.2.0; not a release blocker. |

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
