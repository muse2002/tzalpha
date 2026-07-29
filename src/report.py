"""
뷰어 생성기.

엔진 결과를 JSON으로 묶어서 viewer_template.html의 __BUNDLE__ 자리에 밀어넣고
viewer.html 한 개 파일로 뽑는다. 그 파일 하나만 있으면 어디서든 열린다.

실행:  python src/report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "viewer_template.html"
OUTPUT = ROOT / "viewer.html"


def build() -> Path:
    st = engine.run()
    results, models, panels = st["results"], st["models"], st["panels"]

    # 모든 모델이 공유하는 날짜 사전 -> 용량을 크게 줄인다
    all_dates = sorted({d for ev in results.values() if len(ev) for d in ev["date"]})
    idx = {d: i for i, d in enumerate(all_dates)}

    series = {}
    for mid, ev in results.items():
        if not len(ev):
            series[mid] = []
            continue
        series[mid] = [
            [idx[r.date], round(float(r.signal), 6), round(float(r.unit_return), 6),
             1 if r.mode == "live" else 0]
            for r in ev.itertuples()
        ]

    ledger = st["ledger"]
    ledger_out = ledger.fillna("").astype(str).to_dict("records") if len(ledger) else []

    universe = []
    for k, df in panels.items():
        universe.append({
            "key": k, "n": int(len(df)),
            "first": str(df.index.min().date()), "last": str(df.index.max().date()),
            "last_close": round(float(df["close"].iloc[-1]), 4),
        })

    labels = {}
    with open(ROOT / "config.json", encoding="utf-8") as f:
        for it in json.load(f)["universe"]:
            labels[it["key"]] = it["label"]

    ratio = st["adr_ratio"]
    bundle = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_days": len(all_dates),
            "cost_bps": st["cfg"].get("cost_bps", 30),
            "adr_ratio": None if not np.isfinite(ratio) else round(float(ratio), 6),
            "github_repo": st["cfg"].get("github_repo", ""),
            "range": [all_dates[0].strftime("%Y-%m-%d"), all_dates[-1].strftime("%Y-%m-%d")] if all_dates else ["", ""],
        },
        "dates": [d.strftime("%Y-%m-%d") for d in all_dates],
        "labels": labels,
        "models": models,
        "series": series,
        "ledger": ledger_out,
        "universe": sorted(universe, key=lambda x: x["key"]),
        "diag": st.get("diag", {}),
        "events": st.get("events", []),
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__BUNDLE__", json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    OUTPUT.write_text(html, encoding="utf-8")
    size = OUTPUT.stat().st_size / 1024
    print(f"[뷰어] {OUTPUT}  ({size:,.0f} KB)  브라우저로 열어보세요.")
    return OUTPUT


if __name__ == "__main__":
    build()
