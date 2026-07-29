"""
로컬 뷰어 서버.

    python serve.py

브라우저에서 http://localhost:8777 이 자동으로 열립니다.
화면의 [데이터 갱신] 버튼을 누르면 실제로 수집 -> 분석 -> 뷰어 재생성까지 돕니다.

왜 서버가 필요한가:
  viewer.html 은 그냥 파일이라 스스로 인터넷에서 데이터를 못 가져옵니다.
  수집은 파이썬이 합니다. 그래서 브라우저와 파이썬을 이어줄 창구가 하나 필요합니다.
  그 창구가 이 파일입니다. 파이썬 기본 내장 기능만 써서 추가 설치는 없습니다.

이 서버는 내 PC 안에서만 돕니다. 외부에서 접속 못 합니다.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

PORT = 8777

# 갱신 상태를 담아두는 상자. 브라우저가 주기적으로 물어본다.
STATE = {"running": False, "ok": None, "message": "", "finished_at": ""}
LOCK = threading.Lock()


def do_refresh(full: bool = False) -> None:
    import collect
    import report
    try:
        collect.collect(full_refresh=full)
        report.build()
        with LOCK:
            STATE.update(ok=True, message="갱신 완료",
                         finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            STATE.update(ok=False, message=f"{type(exc).__name__}: {exc}",
                         finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    finally:
        with LOCK:
            STATE["running"] = False


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        if "/api/" not in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)

    def _json(self, payload: dict, code: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            with LOCK:
                return self._json(dict(STATE))
        if self.path in ("/", "/index.html"):
            self.path = "/viewer.html"
        # 뷰어는 매번 새로 읽어야 하므로 캐시 금지
        self.send_header_hook = True
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/refresh"):
            return self._json({"error": "unknown"}, 404)
        with LOCK:
            if STATE["running"]:
                return self._json({"started": False, "message": "이미 갱신 중입니다"})
            STATE.update(running=True, ok=None, message="수집 중...", finished_at="")
        full = "full=1" in self.path
        threading.Thread(target=do_refresh, args=(full,), daemon=True).start()
        return self._json({"started": True})

    def end_headers(self):
        if self.path.endswith(".html") or self.path == "/":
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def main():
    if not (ROOT / "viewer.html").exists():
        print("viewer.html이 없습니다. 먼저 'python run_daily.py' 를 한 번 실행하세요.")
        print("(인터넷 없이 화면만 보려면 'python src/seed_demo.py && python src/report.py')")
        return
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print("=" * 58)
    print(f"  뷰어 서버 실행 중  ->  {url}")
    print("  종료하려면 이 창에서 Ctrl+C")
    print("=" * 58)
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        srv.shutdown()


if __name__ == "__main__":
    main()
