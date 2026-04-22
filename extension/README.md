# Insider Polymarket Extension

Browser-only MV3 extension for live Polymarket surveillance.

## What It Does

- detects the active Polymarket market from the current page
- pulls live market data from Gamma, Data API, and the public CLOB endpoints
- computes weighted suspicion scores without any model inference
- switches automatically between `live` and `post-event` modes
- computes markout horizons on demand: `5m`, `30m`, `1h`, `4h`, and `to resolution`

## Scoring Model

The shared scorer carries over the explainable `sus` approach, but keeps the browser runtime self-contained:

- pre-event timing anomaly
- market-relative bet size anomaly
- wallet-relative bet size anomaly
- odds / depth / OI pressure
- order-flow pressure
- directional concentration
- spread / volatility regime
- new-wallet / first-trade shock
- linked-wallet coordination
- realized or resolved markout

Linked-wallet coordination now includes observable account-age closeness from earliest Polymarket activity, plus synchronized entry and exit timing.

## Commands

```bash
cd extension
npm install
npm run test
npm run build
```

Load `extension/dist` as an unpacked extension in Chromium/Chrome.
