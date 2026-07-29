"""
데이터 소스 어댑터 (adapter = 콘센트 변환기).

이 파일 하나만 바꾸면 yfinance -> 한국투자증권 API로 갈아끼울 수 있다.
바깥 코드는 아래 '계약'만 알면 된다. 그 외에는 아무것도 알 필요 없다.

    계약(contract):
        fetch_daily(ticker: str, start: str, end: str) -> pandas.DataFrame

        반환 컬럼 (이 이름과 순서를 반드시 지킬 것):
            date   : datetime64[ns]  거래소 현지 날짜. 시간대(timezone) 없음.
            open   : float           시가
            high   : float           고가
            low    : float           저가
            close  : float           종가
            volume : float           거래량

        - 액면분할/배당 조정이 반영된 값이어야 한다 (adjusted).
          조정 안 하면 배당락일에 가짜 갭이 생겨서 분석이 오염된다.
        - 데이터가 없으면 빈 DataFrame을 반환한다. 예외를 던지지 않는다.
"""

from __future__ import annotations

import time
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}).astype(
        {"date": "datetime64[ns]"}
    )


# ---------------------------------------------------------------------------
# 어댑터 1: yfinance (기본값)
# ---------------------------------------------------------------------------
def fetch_daily_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # 분할/배당 조정. 가짜 갭 방지용. 끄지 말 것.
                progress=False,
                threads=False,
            )
            break
        except Exception as exc:  # 일시적 네트워크 오류는 재시도
            if attempt == 2:
                print(f"  [실패] {ticker}: {exc}")
                return _empty()
            time.sleep(2 * (attempt + 1))

    if raw is None or len(raw) == 0:
        return _empty()

    # yfinance는 버전/티커수에 따라 컬럼이 2단(MultiIndex)으로 오기도 한다. 평탄화.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    # 시간대 정보 제거 -> 순수 '거래소 현지 날짜'만 남긴다.
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = float("nan")

    df = df[COLUMNS].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 어댑터 2: 한국투자증권 KIS API (자리만 만들어 둠)
# ---------------------------------------------------------------------------
def fetch_daily_kis(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    나중에 여기를 채운다. 채울 때 필요한 것:

      1) https://apiportal.koreainvestment.com 에서 앱키(appkey) / 앱시크릿 발급
      2) 접근토큰(access_token) 발급 -> 유효기간 24시간, 파일에 캐시할 것
      3) 국내주식 일봉:  /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice
         해외주식 일봉:  /uapi/overseas-price/v1/quotations/dailyprice
      4) 한 번에 100건 제한 -> start~end를 100일씩 잘라서 반복 호출 후 이어붙이기
      5) 반환 직전에 위 '계약'대로 컬럼명/타입을 맞출 것

    앱키는 절대 코드에 직접 쓰지 말 것. .env 파일이나 환경변수로 뺀다.
    """
    raise NotImplementedError("한투 어댑터는 아직 안 만들었습니다. SOURCE='yfinance' 유지.")


# ---------------------------------------------------------------------------
# 여기만 바꾸면 전체 소스가 교체된다
# ---------------------------------------------------------------------------
SOURCE = "yfinance"

_ADAPTERS = {
    "yfinance": fetch_daily_yfinance,
    "kis": fetch_daily_kis,
}


def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    return _ADAPTERS[SOURCE](ticker, start, end)
