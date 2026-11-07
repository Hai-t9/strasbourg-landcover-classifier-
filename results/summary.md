# Project 2 — Sentinel-2 land-cover tile classification

- labelled tiles: 7699 (train 6022 / test 1677)
- held-out test accuracy: **0.922**
- held-out test macro-F1: **0.841**

| class | row-recall | notes |
|---|---|---|
| water | 1.00 | spectrally distinct |
| forest | 0.99 | spectrally distinct |
| grass | 0.37 | weak class |
| farmland | 0.95 | weak class |
| built_up | 0.92 | spectrally distinct |

- baseline (majority-class dummy): acc 0.561, macro-F1 ≈ 0.040
- model: HistGradientBoosting; split: spatial blocks (14×14 tiles ≈ 3.5 km)