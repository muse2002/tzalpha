"""
매일 한 번 실행하는 진입점.

    python run_daily.py

이거 하나면 수집 -> 평가 -> 예측 사전기록 -> 뷰어 생성까지 전부 끝납니다.
윈도우 작업 스케줄러에 등록할 때도 이 파일 하나만 걸면 됩니다.

권장 실행 시각: 매일 오전 6~8시 (미국장 마감 05:00 이후, 국장 개장 09:00 이전)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import collect   # noqa: E402
import report    # noqa: E402

if __name__ == "__main__":
    print("=" * 62)
    collect.collect(full_refresh="--full" in sys.argv)
    print("-" * 62)
    report.build()
    print("=" * 62)
