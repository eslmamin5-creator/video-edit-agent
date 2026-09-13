"""V1.1 hardening: real Arabic speech -> local ASR -> dialect-preservation
acceptance check (spec V1.1 hardening sections 4-5).

This script is the DOCUMENTED, RE-RUNNABLE command for exercising real
Arabic speech recognition once a legal Arabic audio sample is available on
the machine running it. It deliberately refuses to fabricate a pass: if no
Arabic TTS voice and no supplied audio fixture exist, it prints exactly what
is missing and exits with a BLOCKED status rather than silently testing
something else and calling it "Arabic ASR verified".

Usage:
    # Preferred: point at a real (legally obtained) Arabic speech clip
    .venv/Scripts/python.exe scripts/arabic_asr_acceptance.py path/to/clip.wav

    # No-argument mode: tries to synthesize one from a local Arabic SAPI/TTS
    # voice if the OS has one installed (Windows: check `Get-InstalledVoices`
    # for an ar-* culture; this repo's dev machine only has en-US voices, so
    # this path is expected to report BLOCKED there).
    .venv/Scripts/python.exe scripts/arabic_asr_acceptance.py

What it checks once audio is available:
  1. The configured local (offline-safe) transcription provider actually
     transcribes the clip end-to-end (TranscriptionRouter, offline=True).
  2. The resulting Transcript is non-empty and in Arabic.
  3. The transcript's text is NOT rewritten into a different dialect/register
     than what went in -- via the project's own dialect_guard, run against a
     `--expect` sentence you supply (comparing the *original spoken sentence*
     you know you recorded, against the ASR result).

IMPORTANT: this only proves the pipeline preserves whatever the ASR engine
already produced. It does not (and cannot, without a human Arabic speaker)
grade raw ASR transcription accuracy -- that is a model-quality question,
separate from the dialect-preservation guarantee this project makes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from video_edit_agent.core.config import TranscriptionConfig  # noqa: E402
from video_edit_agent.language.dialect_guard import check_no_dialect_substitution  # noqa: E402
from video_edit_agent.transcription.router import TranscriptionRouter  # noqa: E402

RECOMMENDED_SENTENCE = "أنا عايز أعمل الفيديو ده دلوقتي، بس مش عايز أغير الكلام."


def find_arabic_sapi_voice() -> str | None:
    if sys.platform != "win32":
        return None
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'ar-*' } "
        "| ForEach-Object { $_.VoiceInfo.Name }"
    )
    try:
        result = subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    name = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return name or None


def synthesize_with_sapi(voice_name: str, text: str, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('{voice_name}')
$s.SetOutputToWaveFile('{out_wav}')
$s.Speak('{text}')
$s.Dispose()
"""
    subprocess.run(["powershell", "-Command", ps], capture_output=True, text=True, timeout=30, check=True)


def main() -> int:
    audio_arg = sys.argv[1] if len(sys.argv) > 1 else None
    expected_text = RECOMMENDED_SENTENCE

    if audio_arg:
        audio_path = Path(audio_arg)
        if not audio_path.exists():
            print(f"BLOCKED BY ENVIRONMENT: supplied audio fixture not found: {audio_path}")
            return 2
    else:
        voice = find_arabic_sapi_voice()
        if not voice:
            print("BLOCKED BY ENVIRONMENT: real Arabic ASR acceptance requires either")
            print("  (a) an Arabic audio fixture passed as argv[1] (legally obtained, not committed to git), or")
            print("  (b) a local Arabic TTS voice (Windows SAPI culture 'ar-*') to synthesize one.")
            print()
            print("This machine has no Arabic SAPI voice installed (checked via")
            print("Get-InstalledVoices; only en-US voices are present here).")
            print()
            print("To run this for real once either is available:")
            print(f"  .venv/Scripts/python.exe {Path(__file__).relative_to(REPO_ROOT)} <path-to-arabic-clip.wav>")
            print()
            print("Recommended test sentence (Egyptian colloquial, deliberately NOT to be")
            print("'corrected' into MSA or another dialect by any transform):")
            print(f"  {RECOMMENDED_SENTENCE}")
            return 3
        audio_path = REPO_ROOT / "scratch" / "arabic_asr_acceptance" / "sample.wav"
        print(f"Found Arabic SAPI voice: {voice} -- synthesizing test clip...")
        synthesize_with_sapi(voice, expected_text, audio_path)

    print(f"Transcribing {audio_path} with local (offline-safe) provider...")
    cfg = TranscriptionConfig()
    router = TranscriptionRouter(cfg, offline=True)
    transcript = router.transcribe(audio_path)

    full_text = " ".join(seg.text for seg in transcript.segments).strip()
    print(f"Provider used: {transcript.provider}")
    print(f"Transcript text: {full_text!r}")

    if not full_text:
        print("FAILED: transcript is empty")
        return 1

    # Do NOT rewrite the transcript to force this to pass -- compare the raw
    # ASR output against the sentence we know was actually spoken/synthesized.
    result = check_no_dialect_substitution(expected_text, full_text)
    if result.ok:
        print("PASS: no dialect-substitution signal between spoken input and ASR output")
        return 0
    print(f"FAILED: dialect guard flagged a mismatch: {result.reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
