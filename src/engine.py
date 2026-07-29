"""
평가 엔진.

하는 일 3가지:
  1. 날짜 정렬  : 한국 D일 09:00에 '알 수 있었던' 미국 세션을 정확히 짝지음
  2. 신호 계산  : models.json의 명세대로 신호값과 수익률을 계산
  3. 사전 기록  : 오늘 각 모델이 뭐라고 예측했는지를 결과가 나오기 '전에' 기록

3번이 이 프로젝트의 핵심입니다.
결과를 본 뒤에 만든 성적은 증거가 아닙니다. 미리 적어둔 예측만이 증거입니다.

실행:  python src/engine.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "prices.csv"
PREDICTIONS = DATA / "predictions.csv"

PRED_COLS = [
    "model_id", "recorded_at", "signal_date", "signal_value",
    "position", "target_date", "unit_return", "net_return", "status",
]


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_models() -> list[dict]:
    with open(ROOT / "models.json", encoding="utf-8") as f:
        return json.load(f)["models"]


def load_panels() -> tuple[dict, dict]:
    """prices.csv -> {종목키: 날짜인덱스 DataFrame} 두 벌(국내/미국)로 쪼갠다."""
    df = pd.read_csv(PRICES, parse_dates=["date"])
    panels, markets = {}, {}
    for key, g in df.groupby("key"):
        g = g.sort_values("date").set_index("date")[["open", "high", "low", "close", "volume"]]
        panels[key] = g[~g.index.duplicated(keep="last")]
        markets[key] = df.loc[df["key"] == key, "market"].iloc[0]
    return panels, markets


# ---------------------------------------------------------------------------
# 1. 날짜 정렬 — 여기가 가장 틀리기 쉬운 부분
# ---------------------------------------------------------------------------
def align_us_to_kr(kr_dates: pd.DatetimeIndex, us_dates: pd.DatetimeIndex) -> pd.Series:
    """
    한국 D일 09:00 개장 시점에, 이미 끝나 있는 가장 최근 미국 세션을 찾는다.

      한국 화요일 09:00  <-  미국 월요일 종가 (한국시간 화요일 05:00 마감)
      한국 월요일 09:00  <-  미국 금요일 종가 (주말 건너뜀)

    구현: '한국 날짜 - 1일' 이하인 미국 날짜 중 가장 마지막 것.
    미래 정보를 당겨쓰는 실수(lookahead bias)를 막는 유일한 장치이므로 절대 건드리지 말 것.
    """
    left = pd.DataFrame({"key_date": kr_dates - pd.Timedelta(days=1), "kr_date": kr_dates})
    right = pd.DataFrame({"key_date": us_dates, "us_date": us_dates})
    merged = pd.merge_asof(
        left.sort_values("key_date"),
        right.sort_values("key_date"),
        on="key_date",
        direction="backward",
    )
    return merged.set_index("kr_date")["us_date"]


# ---------------------------------------------------------------------------
# 2. 신호 계산
# ---------------------------------------------------------------------------
def _adr_ratio(panels: dict) -> float:
    """
    ADR 1주가 본주 몇 주에 해당하는지를 데이터에서 직접 추정한다.
    공시된 비율을 찾아 넣어도 되지만, 자동 추정이 오타에 강하다.
    """
    skhy, fx, hy = panels.get("SKHY"), panels.get("FX"), panels.get("HYNIX")
    if skhy is None or fx is None or hy is None or len(skhy) < 3:
        return float("nan")
    j = pd.DataFrame({"adr_usd": skhy["close"]}).join(fx["close"].rename("fx"), how="inner")
    j = j.join(hy["close"].rename("hy"), how="inner").dropna()
    if len(j) < 3:
        return float("nan")
    return float(np.median(j["adr_usd"] * j["fx"] / j["hy"]))


def compute_signal(spec: dict, panels: dict, markets: dict, kr_dates: pd.DatetimeIndex) -> pd.Series:
    key, field = spec["signal"]["key"], spec["signal"]["field"]

    if field == "coinflip":
        # 날짜를 씨앗으로 한 난수. 매번 같은 값이 나오도록 고정.
        vals = [np.random.default_rng(int(d.strftime("%Y%m%d"))).standard_normal()
                for d in kr_dates]
        return pd.Series(vals, index=kr_dates)

    if key not in panels:
        return pd.Series(dtype="float64", index=kr_dates)

    src = panels[key]

    if field == "adr_premium":
        ratio = _adr_ratio(panels)
        if not np.isfinite(ratio):
            return pd.Series(np.nan, index=kr_dates)
        fx, hy = panels["FX"], panels["HYNIX"]
        implied = (src["close"] * fx["close"].reindex(src.index).ffill()) / ratio  # 원화 환산가
        raw = implied
    elif field == "ret_close":
        raw = src["close"] / src["close"].shift(1) - 1
    elif field == "ret_intraday":
        raw = src["close"] / src["open"] - 1
    elif field == "ret_gap":
        raw = src["open"] / src["close"].shift(1) - 1
    else:
        raise ValueError(f"모르는 signal.field: {field}")

    if markets.get(key) == "KR":
        # 국내 신호는 같은 날 09:00에 알 수 있는 값만 허용(ret_gap).
        return raw.reindex(kr_dates)

    # 미국 신호는 직전 세션으로 당겨온다.
    us_map = align_us_to_kr(kr_dates, src.index)
    vals = raw.reindex(us_map.values).values
    s = pd.Series(vals, index=kr_dates)

    if field == "adr_premium":
        # 미국이 매긴 원화 환산가 vs 국내 전일 종가의 괴리율
        hy_prev = panels["HYNIX"]["close"].shift(1).reindex(kr_dates)
        s = s / hy_prev - 1
    return s


# ---------------------------------------------------------------------------
# 3. 모델 평가
# ---------------------------------------------------------------------------
def _leg_return(trade_df: pd.DataFrame, entry: str, exit_: str) -> pd.Series:
    def px(which):
        if which == "prev_close":
            return trade_df["close"].shift(1)
        return trade_df[which]
    return px(exit_) / px(entry) - 1


def evaluate(spec: dict, panels: dict, markets: dict, cost_bps: float) -> pd.DataFrame:
    tkey = spec["trade"]["key"]
    if tkey not in panels:
        return pd.DataFrame()

    trade_df = panels[tkey]
    kr_dates = trade_df.index

    sig = compute_signal(spec, panels, markets, kr_dates)
    unit = _leg_return(trade_df, spec["trade"]["entry"], spec["trade"]["exit"])

    out = pd.DataFrame({"signal": sig, "unit_return": unit}).dropna()

    pos = np.where(out["signal"] >= spec["rule"]["long_above"], 1,
          np.where(out["signal"] <= spec["rule"]["short_below"], -1, 0))
    if spec["trade"]["direction"] == "fade":
        pos = -pos
    out["position"] = pos

    cost = cost_bps / 10000.0
    out["net_return"] = out["position"] * out["unit_return"] - np.where(pos != 0, cost, 0.0)

    created = pd.Timestamp(spec["created_at"])
    out["mode"] = np.where(out.index >= created, "live", "backtest")
    out["model_id"] = spec["id"]
    return out.reset_index().rename(columns={"index": "date", "date": "date"})


# ---------------------------------------------------------------------------
# 4. 예측 사전 기록 (append only, 절대 수정 금지)
# ---------------------------------------------------------------------------
def record_predictions(models: list[dict], panels: dict, markets: dict, cost_bps: float) -> pd.DataFrame:
    """
    아직 결과가 안 나온 '다음 거래일'에 대한 예측을 기록한다.
    이미 기록된 (모델, 신호일) 조합은 건드리지 않는다. 이게 무결성의 전부다.
    """
    DATA.mkdir(exist_ok=True)
    if PREDICTIONS.exists():
        ledger = pd.read_csv(PREDICTIONS)
    else:
        ledger = pd.DataFrame(columns=PRED_COLS)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = []

    for spec in models:
        skey = spec["signal"]["key"]
        tkey = spec["trade"]["key"]
        if tkey not in panels:
            continue

        # 신호가 나온 가장 최근 세션
        if spec["signal"]["field"] == "coinflip" or markets.get(skey) == "KR":
            sig_date = panels[tkey].index.max()
        else:
            if skey not in panels:
                continue
            sig_date = panels[skey].index.max()

        # 국내 거래일 캘린더에 이 신호로 매매할 날이 이미 있는지 확인
        kr_dates = panels[tkey].index
        future = kr_dates[kr_dates > sig_date]
        already_traded = len(future) > 0

        exists = ((ledger["model_id"] == spec["id"]) &
                  (ledger["signal_date"] == str(sig_date.date()))).any() if len(ledger) else False
        if exists:
            continue

        ev = evaluate(spec, panels, markets, cost_bps)
        # 최신 신호값 계산 (매매 결과는 아직 모름)
        tmp_dates = pd.DatetimeIndex([sig_date + pd.Timedelta(days=1)])
        sigval = compute_signal(spec, panels, markets, tmp_dates).iloc[0] if len(ev) else np.nan
        if not np.isfinite(sigval):
            continue

        p = 1 if sigval >= spec["rule"]["long_above"] else (-1 if sigval <= spec["rule"]["short_below"] else 0)
        if spec["trade"]["direction"] == "fade":
            p = -p

        new_rows.append({
            "model_id": spec["id"],
            "recorded_at": now,
            "signal_date": str(sig_date.date()),
            "signal_value": round(float(sigval), 6),
            "position": p,
            "target_date": str(future[0].date()) if already_traded else "",
            "unit_return": "",
            "net_return": "",
            "status": "pending",
        })

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)

    # 결과가 나온 pending 건들을 채점한다
    cost = cost_bps / 10000.0
    for i, row in ledger[ledger["status"] == "pending"].iterrows():
        spec = next((m for m in models if m["id"] == row["model_id"]), None)
        if spec is None:
            continue
        tkey = spec["trade"]["key"]
        if tkey not in panels:
            continue
        kr_dates = panels[tkey].index
        sd = pd.Timestamp(row["signal_date"])
        future = kr_dates[kr_dates > sd]
        if len(future) == 0:
            continue
        td = future[0]
        unit = _leg_return(panels[tkey], spec["trade"]["entry"], spec["trade"]["exit"])
        if td not in unit.index or not np.isfinite(unit.loc[td]):
            continue
        p = int(row["position"])
        ledger.at[i, "target_date"] = str(td.date())
        ledger.at[i, "unit_return"] = round(float(unit.loc[td]), 6)
        ledger.at[i, "net_return"] = round(float(p * unit.loc[td] - (cost if p != 0 else 0.0)), 6)
        ledger.at[i, "status"] = "scored"

    ledger[PRED_COLS].to_csv(PREDICTIONS, index=False)
    return ledger


# ---------------------------------------------------------------------------
# 5. 신호 해석 — "이 숫자가 왜 이렇게 나왔나"를 데이터에서 뽑아낸다
# ---------------------------------------------------------------------------
BENCH = "SOXX"   # 섹터 기준. 반도체 ETF.


def _beta(x: pd.Series, y: pd.Series, window: int = 252) -> float:
    """x가 y(섹터)에 얼마나 민감한지. 최근 window일 기준."""
    j = pd.DataFrame({"x": x, "y": y}).dropna().tail(window)
    if len(j) < 30 or j["y"].var() == 0:
        return float("nan")
    return float(j["x"].cov(j["y"]) / j["y"].var())


def diagnose(panels: dict, markets: dict) -> dict:
    """
    가장 최근 미국 세션에 대해:
      - 미국 종목별 등락률 (한눈에 보는 보드)
      - 각 종목의 움직임을 '섹터 몫'과 '개별 몫'으로 분해
      - 그 움직임이 과거 대비 얼마나 큰지 (백분위)
    """
    rets = {k: (df["close"] / df["close"].shift(1) - 1)
            for k, df in panels.items() if markets.get(k) == "US" and k != "FX"}
    if BENCH not in rets or len(rets[BENCH]) == 0:
        return {}

    bench = rets[BENCH]
    session = max(s.dropna().index.max() for s in rets.values() if len(s.dropna()))
    board = {}

    for k, r in rets.items():
        if session not in r.index or not np.isfinite(r.get(session, np.nan)):
            continue
        val = float(r.loc[session])
        b = _beta(r, bench)
        bval = float(bench.loc[session]) if session in bench.index else float("nan")
        sector = b * bval if np.isfinite(b) and np.isfinite(bval) else float("nan")
        idio = val - sector if np.isfinite(sector) else float("nan")

        hist = r.dropna().tail(504).abs()
        pctile = float((hist < abs(val)).mean()) if len(hist) > 30 else float("nan")

        if k == BENCH:
            driver = "sector"
        elif not np.isfinite(idio):
            driver = "unknown"     # 상장 초기 등으로 베타를 못 구한 경우
        elif abs(val) < 0.003:
            driver = "quiet"
        elif abs(idio) < 0.4 * abs(val):
            driver = "sector"          # 섹터가 끌고 감
        elif abs(sector) < 0.4 * abs(val):
            driver = "idio"            # 개별 이슈
        else:
            driver = "mixed"

        board[k] = {
            "ret": round(val, 5),
            "beta": None if not np.isfinite(b) else round(b, 2),
            "sector_part": None if not np.isfinite(sector) else round(sector, 5),
            "idio_part": None if not np.isfinite(idio) else round(idio, 5),
            "pctile": None if not np.isfinite(pctile) else round(pctile, 3),
            "driver": driver,
        }

    fx = panels.get("FX")
    fx_ret = None
    if fx is not None and len(fx) > 1:
        fr = fx["close"] / fx["close"].shift(1) - 1
        if session in fr.index and np.isfinite(fr.loc[session]):
            fx_ret = round(float(fr.loc[session]), 5)

    return {"session": str(session.date()), "bench": BENCH,
            "board": board, "fx_ret": fx_ret,
            "fx_level": None if fx is None else round(float(fx["close"].iloc[-1]), 2)}


def load_events() -> list[dict]:
    p = ROOT / "events.json"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("events", [])


# ---------------------------------------------------------------------------
def run() -> dict:
    cfg = load_config()
    models = load_models()
    panels, markets = load_panels()
    cost_bps = cfg.get("cost_bps", 30)

    results = {}
    for spec in models:
        ev = evaluate(spec, panels, markets, cost_bps)
        results[spec["id"]] = ev
        if len(ev):
            live = ev[ev["mode"] == "live"]
            traded = ev[ev["position"] != 0]
            wr = (traded["net_return"] > 0).mean() if len(traded) else float("nan")
            print(f"  {spec['id']}  {spec['name'][:28]:<30} "
                  f"표본{len(ev):>5}  진입{len(traded):>5}  승률{wr:>6.1%}  사전기록{len(live):>4}")
        else:
            print(f"  {spec['id']}  {spec['name'][:28]:<30} 데이터 없음")

    ledger = record_predictions(models, panels, markets, cost_bps)
    print(f"[예측장부] {PREDICTIONS}  총 {len(ledger)}건 "
          f"(대기 {int((ledger['status']=='pending').sum())}건)")

    diag = diagnose(panels, markets)
    if diag:
        n_idio = sum(1 for v in diag["board"].values() if v["driver"] == "idio")
        print(f"[해석] {diag['session']} 미국 세션 · 개별이슈 {n_idio}종목")

    return {"results": results, "ledger": ledger, "models": models,
            "panels": panels, "markets": markets, "cfg": cfg,
            "adr_ratio": _adr_ratio(panels),
            "diag": diag, "events": load_events()}


if __name__ == "__main__":
    run()
