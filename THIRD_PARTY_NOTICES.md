# Third-Party Notices

`video-edit-agent` is an independent project. It does not fork either upstream
repository below. Both upstreams are MIT-licensed; their required copyright
notices are preserved here, and this document records exactly what was
adapted, inspired by, rewritten, or independently implemented.

---

## 1. majedphotos/video-ad-editor

Repository: https://github.com/majedphotos/video-ad-editor
License: MIT

```
MIT License

Copyright (c) 2026 Majed Alzaabi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: ...
```

**What was inspected:** the repository's Claude Code skill layout (`SKILL.md`,
numbered pipeline scripts `00_setup.sh` .. `13_assets.py`), its cut-plan /
caption / behind-text / Remotion-render / safe-zone-check workflow, its
Remotion template component layout (`Ad.tsx`, `Captions.tsx`, `Chrome.tsx`,
`Guides.tsx`, `Outro.tsx`, `Scenes.tsx`, `theme.ts`), and `scripts/03_cut_zoom.py`
(the per-segment talking-head punch-in/zoom-cycling render script).

**Adapted (architectural inspiration, rewritten in this project's own code):**
- The general idea of a numbered, resumable editing pipeline (media -> cut
  plan -> captions -> render -> safe-zone QA) informed `editorial/planner.py`
  and the `edit/` output-directory convention (§14 of the build spec).
- The "behind-subject" text/graphic compositing concept informed
  `subject/compositor.py`.
- The Remotion-based render approach informed `motion/remotion/` and the
  reusable component list in `motion/remotion/components/`.

**Ported (Baseline Recovery Milestone item 6, talking-head punch-ins):**
- `scripts/03_cut_zoom.py`'s per-segment zoom-level cycling lists (`Z` for its
  default pace, and its `CALM`-mode `Z` list) and its vertical crop anchor
  (`ANCH=0.30`) are ported near-verbatim into `editorial/punch_in.py`'s
  `ZOOM_LEVELS_NORMAL` / `ZOOM_LEVELS_CALM` / `ZOOM_ANCHOR_Y`, and its
  per-segment `crop=cw:ch:x:y,scale=...:flags=lanczos` filter pattern is
  ported into `render/composition.py::build_filter_complex`'s per-clip video
  filter chain (applied when `EDLClip.zoom != 1.0`). The reference script's
  `theme.json`-driven `"pace":"calm"` switch has no equivalent config file in
  this project, so `plan_punch_ins()` keys the calm variant off this
  project's own `Brand.motion.energy == "low"` field instead -- the closest
  existing concept in the Brand Profile system (spec section 23). The
  reference's `MINHOLD`/`held` "don't change zoom before 4s on the same shot"
  timer was not ported: this project's EDL clips are already discrete kept
  takes (one per cut), so every clip boundary is already a legitimate point
  to vary the zoom, unlike the reference's raw sub-segment loop.

**Explicitly NOT carried over (per spec §46):**
- Any implicit uppercase-Latin caption styling assumptions.
- Any dialect-normalization or Arabic rewriting behavior — this project's
  `language/dialect_guard.py` is a strict "transcribe, do not rewrite" guard,
  which is a deliberate, independently-designed contrast.
- No files, scripts, or Remotion component source code were copied verbatim;
  every file under `src/video_edit_agent/` and `motion/remotion/` in this
  project was independently written.

---

## 2. browser-use/video-use

Repository: https://github.com/browser-use/video-use
License: MIT

```
MIT License

Copyright (c) 2026 Browser Use

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: ...
```

**What was inspected:** `helpers/transcribe.py`, `helpers/pack_transcripts.py`,
`helpers/timeline_view.py`, `helpers/render.py`, `helpers/grade.py`, and the
bundled `skills/manim-video` skill.

**Adapted (architectural inspiration, rewritten in this project's own code):**
- The "pack transcripts into a compact, LLM-reviewable form" idea informed
  `editorial/packer.py` (`takes_packed.md` generation, spec §11/§14).
- The "timeline/filmstrip view around ambiguous edit points" idea informed
  the timeline-view helper referenced in `editorial/pacing.py`.
- The idea of an automated grading/QA pass informed `qa/technical.py` and
  `qa/repair.py`.
- Manim as a motion-graphics engine option informed `motion/manim/` as one
  of four selectable engines in the motion router (spec §17).

**Independently implemented (no upstream equivalent):**
- The entire provider router and unified transcript schema (Gemini /
  ElevenLabs / Faster-Whisper / OpenAI Whisper / whisper.cpp), the Arabic
  dialect-preservation guard, the Arabic caption engine (RTL shaping, safe
  zones, presets), the Brand Profile system, the HyperFrames adapter, the
  behind-subject segmentation/compositor, the B-roll planner/provider router,
  and the multi-layer QA loop.

---

## 3. heygen-com/hyperframes

Repository: https://github.com/heygen-com/hyperframes
License: Apache License 2.0

Due-diligence note: an unrelated PyPI package also named `hyperframes` exists
(a pandas-like N-dimensional DataFrame library, unrelated author, no
rendering API). It was deliberately never installed or referenced. The real
`heygen-com/hyperframes` project is a Bun/Node monorepo; its only stable,
install-free integration surface is its published npm CLI, invoked here as
`npx hyperframes@0.8.46` (version pinned for deterministic renders --
`motion/hyperframes/adapter.py`'s module docstring has the full trail).

**What was inspected:** `skills/hyperframes-core/SKILL.md` and its
`references/minimal-composition.md` and `references/variables-and-media.md`
(the real HTML/CSS/JS composition contract: `data-composition-id`,
`data-composition-variables`, `data-var-text`/`data-var-src`, a single paused
`gsap.timeline` registered on `window.__timelines`), plus
`skills/embedded-captions/scripts/render-and-composite.sh` (the project's own
reference ffmpeg compositing invocation, which is what revealed that its
WebM/VP9 alpha output only decodes correctly when the input is forced to the
`libvpx-vp9` decoder -- informing this project's choice of MOV/ProRes output
instead, see below).

**Classification: WRAPPED.** No HyperFrames source code is vendored or
copied. This project ships one small, independently hand-authored HTML/CSS/
JS composition (`src/video_edit_agent/motion/hyperframes/template/`) built
from the contract documented above, and `motion/hyperframes/adapter.py`
shells out to the real, unmodified, published `hyperframes` npm CLI via
`npx` to render it -- mirroring the existing `motion/remotion/adapter.py`
pattern (per-project npm cache isolation, pinned version, offline/cached-only
guard for fast unit tests). The CLI is fetched at render time via `npx`, the
same way a normal `npm install` dependency would be, and is not
redistributed with this repository.

**Independently discovered (not documented upstream, verified empirically in
this project):** HyperFrames' WebM/VP9 renders report `needsAlpha:true` and
an `ALPHA_MODE` container tag, but decode as opaque `yuv420p` unless the
consuming ffmpeg command forces `-c:v libvpx-vp9` on that specific input --
otherwise the alpha plane is silently dropped. Rendering to `--format mov`
instead produces `yuva444p12le` (ProRes) output that composites correctly
through a plain `-i file.mov` with no special flags, which is what this
project's `render/composition.py` overlay filter graph does -- so the
adapter renders MOV, not WebM, despite WebM being HyperFrames' more commonly
documented output format.

---

## 4. Third-Party Python Dependencies

The following is a regular pip dependency of this project (declared in
`pyproject.toml`, installed from PyPI at install/setup time) -- it is not
vendored or copied into this repository, and no source code from it is
included here.

- **python-bidi** (https://github.com/MeirKriheli/python-bidi) -- used for
  Unicode bidirectional (BiDi) text algorithm support in the Arabic caption
  engine (`captions/`). Licensed under the GNU Lesser General Public License
  (LGPL), per its PyPI classifier. This project's own code remains MIT
  licensed; using an LGPL library as an external, unmodified dependency (via
  normal `pip install`, not static linking or source inclusion) does not
  itself relicense this project. Users who redistribute this project should
  independently review python-bidi's license terms for their own use case.

---

## Summary

No source files were copied verbatim from either upstream. Both projects
contributed architectural ideas, credited above, which were reimplemented
from scratch against this project's own schemas, CLI, and provider
abstractions. Both upstream copyright notices are preserved in full above as
required by their MIT licenses.
