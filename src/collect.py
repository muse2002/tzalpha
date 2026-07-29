"""
가격 수집기.

하는 일: config.json의 모든 종목 일봉을 받아서 data/prices.csv에 누적한다.
누적(append)이라서 매일 돌려도 안전하다. 같은 날짜/종목은 새 값으로 덮어쓴다.

실행:  python src/collect.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sources import fetch_daily, SOURCE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "prices.csv"


def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def collect(full_refresh: bool = False) -> pd.DataFrame:
    cfg = load_config()
    DATA.mkdir(exist_ok=True)

    old = pd.DataFrame()
    if PRICES.exists() and not full_refresh:
        old = pd.read_csv(PRICES, parse_dates=["date"])

    # 증분 수집: 이미 있는 데이터의 마지막 날 - 7일부터 다시 받는다.
    # 7일을 겹쳐 받는 이유 = 최근 값이 사후 수정되는 경우를 반영하기 위해.
    if len(old) and not full_refresh:
        start = (old["date"].max() - timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        start = cfg["start_date"]
    end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[수집] 소스={SOURCE}  기간={start} ~ {end}")

    chunks = []
    for item in cfg["universe"]:
        df = fetch_daily(item["ticker"], start, end)
        if len(df) == 0:
            print(f"  - {item['key']:<8} {item['label']:<16} 0건  (티커 확인 필요)")
            continue
        df["key"] = item["key"]
        df["ticker"] = item["ticker"]
        df["market"] = item["market"]
        df["label"] = item["label"]
        chunks.append(df)
        print(f"  - {item['key']:<8} {item['label']:<16} {len(df):>5}건  "
              f"~{df['date'].max().date()}")

    if not chunks:
        raise RuntimeError("수집된 데이터가 0건입니다. 인터넷 연결 또는 config.json의 티커를 확인하세요.")

    new = pd.concat(chunks, ignore_index=True)
    merged = pd.concat([old, new], ignore_index=True) if len(old) else new

    # 같은 (날짜, 종목)이면 나중에 받은 값을 남긴다.
    merged = (merged
              .sort_values(["key", "date"])
              .drop_duplicates(subset=["key", "date"], keep="last")
              .reset_index(drop=True))

    merged.to_csv(PRICES, index=False)
    print(f"[저장] {PRICES}  총 {len(merged):,}행  "
          f"({merged['date'].min().date()} ~ {merged['date'].max().date()})")
    return merged


if __name__ == "__main__":
    collect(full_refresh="--full" in sys.argv)
