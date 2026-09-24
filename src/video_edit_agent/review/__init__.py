"""Review-First Editing Workflow (spec: review_first_editing_workflow_prompt.md).

Everything the user needs to inspect and correct BEFORE an expensive final
render lives here: a transcript/caption review artifact, a caption-style
preview, a compact brand summary/validation, a human-readable timeline
summary, a B-roll review list, lightweight preview frames + contact sheet,
and the `READY_FOR_FINAL_RENDER` gate state.

Mirrors the proven UX pattern from the Arabic reference project
(`majedphotos/video-ad-editor`'s `_pkg/video-ad-editor/SKILL.md`): review
cheap artifacts first, only render once the user has approved them.
"""
