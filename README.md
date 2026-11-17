# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/steffanomcrey/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/steffanomcrey/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Rolling-midnight comparables forecaster for GTA VI launch-day Twitch CCU: a scale-conditioned log-ratio day shape with calibrated spread, forecasting hourly viewers for hours 2-24 from the hour-1 print. Leave-one-out over the 14 rolling-midnight comparables in the kit scores mean pinball loss 0.229 on the big-launch subset and 0.350 overall, against 0.245 and 0.461 for the persistence baseline.

## Run it

```sh
python3 main.py
```

Reads one case from `$MAKEFUN_INPUT_DIR/<id>.in` (default `/input`) and writes `$MAKEFUN_OUTPUT_DIR/<id>.out` (default `/output`): five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's Twitch category viewer count for hours 2-24. Python 3 stdlib only, deterministic, no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=7nh48qqg&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
