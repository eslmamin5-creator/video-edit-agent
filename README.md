# video-edit-agent

**Talk to it in plain language — Arabic or English — and it turns raw
footage, a script, or a folder of scenes into a finished, captioned video.**
Fully offline if you want it, enhanced by cloud providers if you give it API
keys. CLI command: `videoedit`.

**English | [العربية](README.ar.md)**

📘 **[User Guide (English) — PDF](docs/video-edit-agent-user-guide.pdf) · [Markdown](docs/USER_GUIDE.md)**
📗 **[دليل المستخدم (العربية) — PDF](docs/video-edit-agent-user-guide-ar.pdf) · [Markdown](docs/USER_GUIDE.ar.md)**

---

## What can video-edit-agent do?

Open Claude Code, install/open `video-edit-agent`, talk naturally, and get a
video — you don't need to read technical docs or memorize CLI commands
first. There are three workflows, and you say what you want in your own
words; the agent figures out which one to run:

| You have... | You say (example) | It runs |
|---|---|---|
| One raw take (talking-head, ad, interview) | "Edit this video" / "عدل الفيديو ده" | **Editor** |
| A script or an idea, no footage yet | "Create a video from this script" / "اعمل فيديو من السكربت ده" | **Creator** |
| Several pre-shot scenes/clips | "Assemble these scenes into one film" / "اجمع المشاهد دي" | **Assembler** |

If what you want is genuinely ambiguous, the agent asks one short
clarifying question instead of guessing.

## Editor / Creator / Assembler

### Editor — one raw take in, a finished video out
For a talking-head recording or an ad take that already exists.
Handles: transcription, silence/repetition/false-start cleanup, captions
(Arabic RTL shaping, word highlighting), motion graphics, B-roll planning,
behind-subject graphic placement where available, brand styling, QA, and
final render. Run with `videoedit edit <video>`.

### Creator — a script or idea in, a finished video out
For when you don't have footage yet, just an idea or written script.
Handles: script parsing, scene planning, analysis, storyboard, asset plan,
MasterTimeline, motion treatment, render, QA. Run with
`videoedit create <script>`.

### Assembler — many existing clips in, one film out
For a folder of scenes you've already shot. Handles: scene discovery, order
preservation, a fast rough cut, continuity planning, real transitions, audio
crossfade, loudness normalization, and finish/resume. Run with
`videoedit assemble <scenes_dir> --rough` then `--finish`.

## Key features

- **Editing intelligence** — transcript-driven editing, silence/repetition/
  false-start handling, an inspectable EDL, re-renderable edit decisions
  (change a decision, re-render — no need to redo the whole pipeline), and
  a persistent `project.md` memory of what the agent decided and why.
- **Arabic** — RTL captions, Arabic shaping, karaoke-style word highlighting,
  Egyptian/Gulf/Saudi/MSA preservation, mixed Arabic/English text, and no
  hidden dialect conversion, ever.
- **Motion** — a built-in simple motion engine that always works offline,
  optional Remotion rendering, automatic motion-engine fallback, behind-
  subject compositing status reporting, and brand-aware styling.
- **B-roll** — planning with provider fallback, a fully offline path, and
  optional cloud enhancement (Gemini image/video) when keys are set.
- **Creator pipeline** — script → analysis → scenes → storyboard → asset
  plan → timeline → render.
- **Assembler pipeline** — rough cut → finish, preserving scene order, with
  continuity planning, real transitions, audio crossfade, and loudness
  normalization.
- **Brand Profiles** — reusable colors, fonts, and caption presets per brand,
  no source changes needed to add one.
- **Inspectability** — every stage is a plain file you can open:
  `project.md`, transcript, EDL, MasterTimeline, motion/B-roll/QA plans, and
  the final video.

## Arabic-first, dialect-preserving

- The agent transcribes exactly what was said — Egyptian, Gulf, Saudi, MSA,
  or code-switched Arabic/English — and never rewrites, formalizes, "fixes,"
  or converts the dialect. Source audio has final authority
  (`language/dialect_guard.py`).
- **This is independent of what language you talk to the agent in.** If you
  write to Claude Code in Arabic, onboarding/status/questions/explanations
  come back in natural Arabic; if you write in English, they come back in
  English. Either way, the video's transcript and captions stay exactly as
  spoken — the agent never translates or "cleans up" the dialect just
  because you happened to ask in a different language.
- Arabic captions render RTL with correct shaping, support karaoke-style
  word highlighting, and can mix Arabic/English text on the same line.

## Offline vs. Cloud

| | Works with zero API keys | What it adds when configured |
|---|---|---|
| Transcription | Yes — local Faster-Whisper | `GEMINI_API_KEY` / `ELEVENLABS_API_KEY` unlock cloud transcription |
| Captions, RTL shaping, karaoke highlight | Yes, fully local | — |
| Editing (silence/repetition/false-start, EDL) | Yes, fully local | — |
| Motion graphics | Yes — built-in simple engine | Remotion (needs Node/npm, still no API key) for richer motion |
| B-roll | Local library/planning fallback | Gemini image/video generation for AI-generated B-roll |
| Visual QA | Basic local checks | Gemini vision for richer QA |
| Brand Profiles, EDL, render, Assembler rough/finish | Yes, fully local | — |

No API key is ever required for the core pipeline to complete. Keys are read
only from `GEMINI_API_KEY` / `ELEVENLABS_API_KEY` environment variables —
never written to a config file, logged, or printed by any command.

## Install with Claude Code

1. Clone or install this repo into your Claude Code skills directory.
2. Open (or restart) Claude Code.
3. Say: **"Set up video-edit-agent and verify it."**

Claude bootstraps everything for you: detects your OS and a compatible
Python (prefers 3.11), builds a private runtime this project owns
(`.runtime/venv` — never your global/system Python), installs the offline
profile, checks `ffmpeg`/`ffprobe` and Node/npm, and reports readiness in
plain language (in whichever language you're talking to it in). It never
installs OS-level software silently — if `ffmpeg` is missing it gives you
the exact command for your OS (`winget`/`choco`/`scoop` on Windows,
Homebrew on macOS, your distro's package manager on Linux) and waits for you
to run it. No API key is ever required or stored.

You never need to understand Python virtual environments, pip extras, or
provider routing to use this path.

## First conversation examples

Arabic:
- `عدل الفيديو ده وخليه أسرع وأنضف`
- `حط كابشن عربي وحافظ على اللهجة المصرية`
- `اعمل فيديو من السكربت ده`
- `اجمع المشاهد دي واعمل rough cut`
- `طبّق البراند ده على الكابشن والموتشن`
- `اشتغل Offline فقط`

English:
- "Edit this video and make it faster and cleaner"
- "Add Arabic captions and keep the Egyptian dialect"
- "Create a video from this script"
- "Assemble these scenes into a rough cut"
- "Apply this brand to the captions and motion"
- "Work offline only"

Once the setup reports ready, just talk to it:

> ارفع فيديو، أو ابعت سكربت، أو حدّد فولدر المشاهد وقلّي عايز تعمل إيه.
> *(Upload a video, send a script, or point me at a folder of scenes and
> tell me what you want.)*

## Technical CLI usage

### Source install (power users)

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

### Quickstart

```bash
videoedit setup           # one-time bootstrap: private runtime + offline profile + capability check
videoedit doctor           # see the full capability matrix
videoedit edit my_take.mp4
```

Output lands in `my_take_edit/` (an `edit/` folder next to the source video):
`final.mp4`, `transcript_unified.json`, `edl.json`, `captions.ass`, `project.md`,
and QA/B-roll/motion plan JSON files.

### Commands

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

## Capability status (as of v0.2.2)

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

## Outputs / project artifacts

Every stage of every pipeline is a plain, inspectable file next to your
video/project — nothing is hidden inside opaque state:

`project.md` (human-readable memory of decisions and why), `transcript_unified.json`,
`edl.json`, `captions.ass`, motion/B-roll/QA plan JSON files, and `final.mp4`.

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

## Known limitations

- Arabic ASR on real spoken audio and Remotion motion rendering are not yet
  verified against real acceptance runs in this environment (see the
  capability table above).
- J-Cut/L-Cut audio-lead edits are conceptual only, not implemented.
- HyperFrames is not a real installable engine; the router always falls
  back to a working motion engine and logs the fallback.

## Attribution

Inspired by, not forked from, [majedphotos/video-ad-editor](https://github.com/majedphotos/video-ad-editor)
and [browser-use/video-use](https://github.com/browser-use/video-use). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exactly what was adapted
versus independently implemented.

## License

MIT — see [LICENSE](LICENSE).
