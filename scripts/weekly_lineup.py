"""
Weekly workflow: refresh current squads/rosters/fixtures from FPL+Sleeper,
then compute the best formation (auto-detecting the gameweek), LOB bench
suggestions, and a last-call pass restricted to confirmed-available players.

Run early in the week for the initial lineup, then re-run close to kickoff
to catch late fitness news via the last-call pass (recommend_formation.py's
[last call] section).

Usage:
  python scripts/weekly_lineup.py [--event N]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)} ===")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True)


if __name__ == "__main__":
    run("update_current.py")
    run("recommend_formation.py", *sys.argv[1:])
