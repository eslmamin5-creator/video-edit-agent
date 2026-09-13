# Examples

Runnable usage examples for `videoedit`. None of these require an API key.

## 1. Basic offline edit

```bash
videoedit edit path/to/raw_take.mp4 --offline
```

Runs the full pipeline (transcription via local Faster-Whisper, silence /
repetition / false-start cleanup, Arabic-aware captions, default motion
fallback) with no network calls at all.

## 2. Edit with a Brand Profile

```bash
videoedit brand init acme
# edit brands/acme/brand.yaml, drop a logo into brands/acme/logos/, etc.
videoedit brand validate acme
videoedit edit path/to/raw_take.mp4 --brand acme
```

See [`brand_profile/`](brand_profile/) for a filled-in example `brand.yaml`.

## 3. Word-highlight Arabic captions only, no B-roll or motion graphics

```bash
videoedit edit path/to/raw_take.mp4 \
  --caption-style word-highlight \
  --no-broll --no-motion
```

## 4. Inspect what the agent decided

Every run writes a re-renderable `project.md` + `edl.json` next to the
output. Inspect it with:

```bash
videoedit project inspect <project_dir>
```

## 5. Check what's available on this machine

```bash
videoedit doctor      # environment / dependency health check
videoedit providers   # which transcription + motion providers are active
```
