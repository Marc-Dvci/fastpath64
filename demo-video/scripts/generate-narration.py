"""Synthesise the narration and record each clip's measured duration.

Scene lengths in the video are derived from these durations rather than guessed, so the picture
cannot drift out of sync with the voice: change a line of script, re-run this, and the scene
lengthens to fit.

Uses edge-tts (no API key, no account). Requires ffprobe on PATH.

    pip install edge-tts
    python scripts/generate-narration.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "narration-source.json"
OUTPUT = ROOT / "src" / "generated-narration.json"

VOICE = "en-US-AndrewMultilingualNeural"
RATE = "+3%"      # a touch brisk: this is a technical walkthrough, not a meditation
PITCH = "-2Hz"


def duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return round(float(out.stdout.strip()), 3)


async def generate(item: dict) -> dict:
    target = ROOT / "public" / item["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    await edge_tts.Communicate(item["script"], VOICE, rate=RATE, pitch=PITCH).save(str(target))
    d = duration_seconds(target)
    print(f"  {item['id']:<10} {d:>6.2f}s  {target.relative_to(ROOT)}")
    return {**item, "duration": d}


async def main() -> None:
    items = json.loads(SOURCE.read_text(encoding="utf-8"))
    print(f"synthesising {len(items)} clips with {VOICE}")
    clips = [await generate(it) for it in items]
    OUTPUT.write_text(json.dumps(clips, indent=2) + "\n", encoding="utf-8")
    total = sum(c["duration"] for c in clips)
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")
    print(f"narration total: {total:.1f}s")
    if total > 165:
        print("WARNING: with per-scene tails this may exceed the 3-minute submission limit")


if __name__ == "__main__":
    asyncio.run(main())
