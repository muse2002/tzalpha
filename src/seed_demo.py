"""
데모 데이터 생성기.

인터넷 연결이나 yfinance 없이도 전체 파이프라인이 도는지 확인하려고 만든 가짜 데이터입니다.
실제 수집(python src/collect.py)을 돌리면 이 파일이 만든 데이터는 덮어써집니다.

주의: 여기서 나온 승률/수익률은 전부 무의미한 난수입니다. 절대 판단 근거로 쓰지 마세요.

실행:  python src/seed_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def make(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    with open(ROOT / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    kr_days = pd.bdate_range("2023-01-02", "2026-07-27")
    us_days = pd.bdate_range("2022-12-30", "2026-07-27")

    # 공통 섹터 요인 하나를 만들고, 종목별로 베타를 다르게 태운다.
    factor = pd.Series(rng.standard_normal(len(us_days)) * 0.018, index=us_days)

    rows = []
    for item in cfg["universe"]:
        key, mkt = item["key"], item["market"]
        days = kr_days if mkt == "KR" else us_days
        if key == "SKHY":                      # ADR은 2026-07-10 상장
            days = days[days >= "2026-07-10"]
        if key == "FX":
            beta, vol, px = 0.0, 0.004, 1380.0
        else:
            beta = {"HYNIX": 1.20, "SAMSUNG": 0.85, "HANMI": 1.45, "LGIT": 0.55,
                    "SKHY": 1.20, "MU": 1.10, "SNDK": 1.30, "NVDA": 1.25,
                    "SOXX": 1.00, "AAPL": 0.60, "TSLA": 1.40}.get(key, 1.0)
            vol, px = 0.012, 100.0

        f = factor.reindex(days).fillna(0.0).values
        if mkt == "KR":                        # 국내는 전일 미국 요인을 일부 이어받는다
            f = np.roll(factor.reindex(us_days).fillna(0.0).values[:len(days)], 1)

        ret = beta * f + rng.standard_normal(len(days)) * vol
        closes = px * np.exp(np.cumsum(ret))
        # 시가: 종가 방향을 대부분 미리 반영 + 약간의 노이즈
        opens = np.r_[closes[0], closes[:-1]] * (1 + 0.7 * ret + rng.standard_normal(len(days)) * 0.004)

        rows.append(pd.DataFrame({
            "date": days, "open": opens, "high": np.maximum(opens, closes) * 1.005,
            "low": np.minimum(opens, closes) * 0.995, "close": closes,
            "volume": rng.integers(1e5, 1e7, len(days)).astype(float),
            "key": key, "ticker": item["ticker"], "market": mkt, "label": item["label"],
        }))

    df = pd.concat(rows, ignore_index=True)
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "prices.csv", index=False)
    print(f"[데모] {len(df):,}행 생성 -> {DATA/'prices.csv'}  (전부 가짜 데이터입니다)")
    return df


if __name__ == "__main__":
    make()
