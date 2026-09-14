"""Deterministic script analysis tests (Creator spec section 4)."""
from __future__ import annotations

from video_edit_agent.agents.creator.analysis import analyze_script
from video_edit_agent.agents.creator.schemas import ScriptAnalysis

_EN_SCRIPT = (
    "Ever feel overwhelmed by clutter? Too much stuff makes it hard to focus. "
    "A simple declutter routine clears mental space. Studies show people save "
    "30 percent more time when organized. Try a five minute daily tidy up. "
    "Subscribe for more simple productivity tips."
)

_AR_SCRIPT = (
    "هل تشعر بالإرهاق من الفوضى؟ الفوضى تجعل التركيز صعبًا. "
    "روتين بسيط للترتيب يوفر مساحة ذهنية. تعرف على المزيد اليوم."
)


def test_analyze_script_is_deterministic():
    a1 = analyze_script(_EN_SCRIPT, title="t")
    a2 = analyze_script(_EN_SCRIPT, title="t")
    assert a1 == a2


def test_analyze_script_returns_schema_instance():
    result = analyze_script(_EN_SCRIPT)
    assert isinstance(result, ScriptAnalysis)


def test_analyze_script_detects_english():
    assert analyze_script(_EN_SCRIPT).language == "en"


def test_analyze_script_detects_arabic():
    assert analyze_script(_AR_SCRIPT).language == "ar"


def test_analyze_script_does_not_invent_statistics():
    result = analyze_script(_EN_SCRIPT)
    for stat_sentence in result.statistics:
        assert stat_sentence in _EN_SCRIPT


def test_analyze_script_no_statistics_when_none_present():
    text = "This script has no numbers at all. It just talks about ideas."
    result = analyze_script(text)
    assert result.statistics == []


def test_analyze_script_hook_is_first_sentence():
    result = analyze_script(_EN_SCRIPT)
    assert result.hook.startswith("Ever feel overwhelmed")


def test_analyze_script_cta_detected_from_keywords():
    result = analyze_script(_EN_SCRIPT)
    assert "subscribe" in result.cta.lower()


def test_analyze_script_empty_text_does_not_crash():
    result = analyze_script("")
    assert result.hook == ""
    assert result.estimated_target_duration == 0.0
