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
