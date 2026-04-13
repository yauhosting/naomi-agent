---
name: fetch-crypto-price
description: Fetch cryptocurrency prices using pycoingecko
version: 1.0.0
author: NAOMI (auto-learned)
tags: ["crypto", "bitcoin", "price"]
prerequisites: ["pycoingecko"]
created: 2026-04-13 22:09:06
---

# Fetch cryptocurrency prices using pycoingecko

## When to Use

When user asks for crypto/Bitcoin prices

## Procedure

1. pip install pycoingecko
2. from pycoingecko import CoinGeckoAPI
3. cg.get_price(ids="bitcoin", vs_currencies="usd")

## Example Commands

```bash
pip3 install pycoingecko
```
```bash
python3 -c "from pycoingecko import CoinGeckoAPI; print(CoinGeckoAPI().get_price(ids="bitcoin", vs_currencies="usd"))"
```

## Lessons Learned

First attempt failed due to SSL warning, fixed by ignoring urllib3 warning
