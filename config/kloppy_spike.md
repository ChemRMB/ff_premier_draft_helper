# kloppy feasibility spike (2026-07-23)

**Verdict: not worth pursuing right now.** Installed `kloppy==3.19.0` via `uv add kloppy`
to evaluate, then removed it (`uv remove kloppy`) once the verdict was clear -
not worth carrying as a dependency for a negative result.

## What kloppy actually is

kloppy is **not a data source** - it's a parser/standardizer that converts
event and tracking data files you already have (from a specific vendor) into
one common format. It ships adapters for: Opta, StatsPerform, StatsBomb,
Wyscout, SkillCorner, Sportec, Tracab, PFF, SecondSpectrum, Metrica,
DataFactory, Impect, Signality, Hawkeye. None of these are free public APIs
you can just query the way FBref/Understat are - each requires either a paid
vendor subscription/license, or a specific limited open-data release.

## The one free option, checked directly

StatsBomb publishes a free `open-data` GitHub repo (the only realistic free
path into kloppy). Checked `competitions.json` directly: it does contain
**England - Premier League**, but only for seasons **2003/04** and
**2015/16**. This project needs 2022/23 through 2025/26
(`config/seasons.yaml::historical_seasons` + current season) - the free data
doesn't cover any season we actually need, by a decade in the worse case.

Metrica's kloppy adapter ships a small free "sample" tracking dataset, but
it's a fixed demo (2-3 non-EPL matches) for exercising the API, not usable
season data.

Every other provider (Opta/StatsPerform - the same underlying data source
FPL/Sleeper's own stats likely use, per `company: "opta"` on Sleeper's stats
endpoint; Wyscout; SkillCorner; etc.) requires a commercial license. Not
viable for a personal project.

## Conclusion

Event/tracking-level data (individual passes, shots, pressures, tracking
frames) is a real, more granular tier below what FBref/Understat already
provide (both of which already aggregate this kind of data into per-90 stats
and xG). Getting there via kloppy specifically would mean paying for a
vendor license - there's no free path to current-season EPL event data
through any of kloppy's supported providers. Not pursued further; revisit
only if a free/cheap EPL event-data provider becomes available (or if this
project ever gets a paid Opta/StatsBomb subscription, at which point kloppy
would immediately be the right tool to parse it).
