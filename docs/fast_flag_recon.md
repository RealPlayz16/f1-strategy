# Fast Flag recon

Source: github.com/Pseudocoder28/Fast-Flag, `main` at b435c9d (read 30 Sept 2026).
Fast Flag is an AI race control assistant built for FormulaTech Hacks 2026 by team
We Are So Back: Naman (A, main builder: ingest, replay, detectors, models, race control
engine, dashboard) and Ishaan (B: bridge, firmware). Credit Naman in our README.

We only consume its output. We never edit Fast Flag code.

## What it emits

Recommendations (`rec`), one JSON object per flag decision (PROJECT_BRIEF Section 7.4):

```json
{"id": "rec-000045", "t": 3601.50, "msector": 7,
 "flag": "CLEAR|YELLOW|DOUBLE_YELLOW|VSC|SC|RED", "confidence": 0.75,
 "reason": "high severity impact, car 23 stopped, recovery needed",
 "message": "SAFETY CAR DEPLOYED", "source_detections": ["det-000001"]}
```

- `t` is FastF1 SessionTime in seconds, the same clock as our `session_time_s`,
  `pit_in_time_s` and `rcm.parquet` Time. Verify on one race in Session 5 by matching
  an official SC in their `<race>_official.json` to our `rcm.parquet`.
- Track-wide flags: `VSC` (message `VIRTUAL SAFETY CAR DEPLOYED`), `SC`
  (`SAFETY CAR DEPLOYED`), `RED` (`RED FLAG`). `msector` is the marshal sector that
  caused it.
- End of a track-wide flag: a `CLEAR` rec with message `TRACK CLEAR`. Other `CLEAR`
  recs clear one sector only.
- No "SC in this lap" or "VSC ending" message. SC and VSC duration must come from our
  own model (sc_rates, historical durations).
- `official` envelopes carry race control's real messages at their real time:
  `{t, category, message, flag, scope, msector, drivers}`, `msector` null when track-wide.

## Timing

- Replay runs on a 250 ms grid (4 ticks per second). Every envelope uses only data with
  SessionTime at or before its tick. No future leakage.
- Out of sample (20 check races): recommended 22 of 33 official VSC/SC/red escalations
  (66.7%), all 22 earlier than race control, median lead 32.6 s. 0.76 escalations per
  race hour that race control never made. Red flag calls are not reliable.
- Training races: 31 of 51 escalations (60.8%), median lead 20.1 s.
- Consequence for us: a Fast Flag SC call is an early, uncertain signal. Session 5
  should re-optimize on the rec and carry P(official SC | Fast Flag SC), estimated from
  their `docs/charts/escalation_check_ours.csv` (column `category`: matched, official
  yellows only, ...) and `escalation_check_official.csv`.

## Transport

Two ways to get the recs.

1. Offline, recommended for the backtest. The server precomputes the whole pipeline
   once per race and caches it:
   `data/timeline/<race>.json.gz` = `{"key", "race", "ticks": [[envelope, ...], ...]}`,
   one list per 250 ms tick, envelopes `{"kind": "detection|risk|rec", "data": {...}}`.
   Filter `kind == "rec"` and `flag in {SC, VSC, RED}` or message `TRACK CLEAR`.
2. Live. `python -m src.replay.server --race <race> --speed 0`, then WebSocket
   `ws://localhost:8000/stream`, envelope `{"kind": "tick|detection|risk|rec|official",
   "data": {...}}`. Control with `POST /replay {"speed": 1, "seek_t": 3500.0,
   "race": "<race>"}`. `GET /races`, `GET /status`, `GET /official`.
   `src/replay/client.py` is a minimal subscriber to copy from.

Race ids: `<season>_<EventName without "Grand Prix", spaces to underscores>`, e.g.
`2025_Abu_Dhabi`, `2024_United_States`.

## Producing recs for our races

The local clone at `~/fast-flag` (branch `b-work`) already has trained models in
`data/models` and a FastF1 cache. For a race Fast Flag has not built, from that repo:

```bash
python -m src.ingest.build --case 2025_Japanese
python -m src.replay.timeline 2025_Japanese
```

- `--case` writes to `data/case_studies/`, which Fast Flag never trains or tunes on.
- The build downloads race and qualifying telemetry (it takes the reference line from the
  same weekend's qualifying), so it is a large download per race.

## Overlap with our races

| Our race | Our split | Fast Flag |
|---|---|---|
| 2025 Dutch | holdout | **training race** (lead times in-sample, optimistic) |
| 2025 Japanese, Singapore, Abu Dhabi | holdout | not built, needs `--case` build |
| 2023 Japanese, 2024 Abu Dhabi, 2024 Azerbaijan | train | check race (out of sample) |
| other 19 train races | train | not built |

Fast Flag's own holdout is 2026 Azerbaijan and 2026 Spanish, both outside our 2023-2025
window.

## Session 5 checklist

- Build `--case` timelines for 2025 Japanese, Singapore, Abu Dhabi. Reuse 2025 Dutch
  from their training set, and report its results separately.
- Parse recs from `data/timeline/<race>.json.gz`, align `t` to our laps via
  `session_time_s`.
- Trigger re-optimization on SC/VSC recs. Weight by P(confirmed). Treat `TRACK CLEAR`
  as the end signal. Compare against the official SC time from `rcm.parquet`.
