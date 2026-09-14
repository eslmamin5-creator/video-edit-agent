"""Scene breakdown (Creator spec section 5): turns the script into a list of
scenes whose durations add up to the estimated target duration. Purely
deterministic -- no cloud calls -- built directly from `ScriptAnalysis` plus
the raw script sentences so provenance is transparent and text is never
paraphrased or translated (`on_screen_text` is always the verbatim segment).
"""
from __future__ import annotations

import re

from video_edit_agent.agents.creator.analysis import _WORDS_PER_SECOND, _split_sentences
from video_edit_agent.agents.creator.schemas import Scene, ScenePurpose, ScriptAnalysis

_MIN_SCENE_DURATION = 2.0


def _classify(sentence: str, index: int, total: int, analysis: ScriptAnalysis) -> ScenePurpose:
    if sentence == analysis.hook:
        return ScenePurpose.HOOK
    if sentence == analysis.cta:
        return ScenePurpose.CTA
    if sentence in analysis.statistics:
        return ScenePurpose.STATISTIC
    position = index / max(total - 1, 1)
    if position < 0.35:
        return ScenePurpose.PROBLEM
    if position < 0.7:
        return ScenePurpose.EXPLANATION
    return ScenePurpose.SOLUTION


_VISUAL_TYPE_BY_PURPOSE = {
    ScenePurpose.HOOK: "typography",
    ScenePurpose.PROBLEM: "broll",
    ScenePurpose.EXPLANATION: "broll",
    ScenePurpose.STATISTIC: "data_viz",
    ScenePurpose.EXAMPLE: "broll",
    ScenePurpose.SOLUTION: "broll",
    ScenePurpose.CTA: "typography",
    ScenePurpose.OTHER: "typography",
}

_BROLL_PURPOSES = {ScenePurpose.PROBLEM, ScenePurpose.EXPLANATION, ScenePurpose.EXAMPLE, ScenePurpose.SOLUTION}
_MOTION_PURPOSES = {ScenePurpose.HOOK, ScenePurpose.STATISTIC, ScenePurpose.CTA}


def _concept_from_sentence(sentence: str) -> str:
    words = re.findall(r"[\w؀-ۿ]+", sentence.lower())
    return " ".join(words[:6])


def build_scenes(text: str, analysis: ScriptAnalysis) -> list[Scene]:
    sentences = _split_sentences(text)
    scenes: list[Scene] = []
    for i, sentence in enumerate(sentences):
        purpose = _classify(sentence, i, len(sentences), analysis)
        word_count = len(sentence.split())
        duration = max(round(word_count / _WORDS_PER_SECOND, 1), _MIN_SCENE_DURATION)
        concept = _concept_from_sentence(sentence)

        scenes.append(
            Scene(
                id=f"scene-{i + 1:02d}",
                script_segment=sentence,
                purpose=purpose,
                estimated_duration=duration,
                visual_type=_VISUAL_TYPE_BY_PURPOSE[purpose],
                foreground_concept=concept,
                background_concept=concept,
                on_screen_text=sentence,
                motion_need=purpose in _MOTION_PURPOSES,
                broll_need=purpose in _BROLL_PURPOSES,
                generated_asset_need=False,
                voice_over_text=sentence,
                transition_intent="crossfade" if i > 0 else "hard_cut",
                brand_constraints=[],
            )
        )
    return scenes
