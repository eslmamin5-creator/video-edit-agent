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
caption / behind-text / Remotion-render / safe-zone-check workflow, and its
Remotion template component layout (`Ad.tsx`, `Captions.tsx`, `Chrome.tsx`,
`Guides.tsx`, `Outro.tsx`, `Scenes.tsx`, `theme.ts`).

**Adapted (architectural inspiration, rewritten in this project's own code):**
- The general idea of a numbered, resumable editing pipeline (media -> cut
  plan -> captions -> render -> safe-zone QA) informed `editorial/planner.py`
  and the `edit/` output-directory convention (§14 of the build spec).
- The "behind-subject" text/graphic compositing concept informed
  `subject/compositor.py`.
- The Remotion-based render approach informed `motion/remotion/` and the
  reusable component list in `motion/remotion/components/`.

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

## Summary

No source files were copied verbatim from either upstream. Both projects
contributed architectural ideas, credited above, which were reimplemented
from scratch against this project's own schemas, CLI, and provider
abstractions. Both upstream copyright notices are preserved in full above as
required by their MIT licenses.
