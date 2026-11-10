"""Start/stop one Cube Core process per business and query its REST API."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import subprocess
import time
from pathlib import Path

import requests

from .config import Config

_SECRET = "dm-local-secret-not-a-real-secret"


def _jwt(secret: str) -> str:
    def b64(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps({"iat": int(time.time()), "exp": int(time.time()) + 86400}).encode())
    sig = hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}.{b64(sig)}"


class CubeServer:
    def __init__(self, cfg: Config, biz_dir: Path, db_path: Path):
        self.cfg, self.biz_dir, self.db_path = cfg, Path(biz_dir), Path(db_path)
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{cfg.cube_port}"
        self.log_path = self.biz_dir / "cube.log"
        self.error: str | None = None

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> bool:
        env = dict(os.environ)
        env.update({
            "CUBEJS_DB_TYPE": "duckdb",
            "CUBEJS_DB_NAME": str(self.db_path),
            "CUBEJS_DB_DUCKDB_DATABASE_PATH": str(self.db_path),
            "CUBEJS_API_SECRET": _SECRET,
            "CUBEJS_DEV_MODE": "false",
            "CUBEJS_TIMEZONE": "Asia/Ho_Chi_Minh",
            "CUBEJS_PORT": str(self.cfg.cube_port),
            "CUBEJS_TELEMETRY": "false",
            "CUBEJS_SCHEDULED_REFRESH_TIMER": "false",
            "CUBEJS_CACHE_AND_QUEUE_DRIVER": "memory",
            "CUBEJS_LOG_LEVEL": "warn",
            "CUBEJS_ROW_LIMIT": "5000",
        })
        log = open(self.log_path, "w")
        runner = self.cfg.project_root / "cube_runner.js"
        try:
            self.proc = subprocess.Popen([self.cfg.node_bin, str(runner)], cwd=self.biz_dir, env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
        except Exception as e:
            self.error = f"cannot launch node: {e}"
            return False
        deadline = time.time() + self.cfg.cube_start_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.error = "cube exited early: " + self.log_path.read_text(errors="ignore")[-600:]
                return False
            try:
                if self._smoke():
                    return True
                for path in ("/readyz", "/livez"):
                    r = requests.get(self.base + path, timeout=2)
                    if r.status_code == 200 and self._smoke():
                        return True
            except requests.RequestException:
                pass
            time.sleep(1.0)
        self.error = "cube start timeout"
        return False

    def _smoke(self) -> bool:
        try:
            r = requests.get(self.base + "/cubejs-api/v1/meta", headers={"Authorization": _jwt(_SECRET)}, timeout=20)
            if r.status_code != 200:
                self.error = f"meta status {r.status_code}: {r.text[:300]}"
                return False
            return True
        except requests.RequestException as e:
            self.error = f"meta failed: {e}"
            return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    # ---- querying -------------------------------------------------------
    def query(self, q: dict, retries: int = 30) -> dict:
        """Run a Cube JSON query. Returns {'data': [...]} or {'error': str}."""
        q = dict(q)
        q.setdefault("timezone", "Asia/Ho_Chi_Minh")
        hdr = {"Authorization": _jwt(_SECRET), "Content-Type": "application/json"}
        for _ in range(retries):
            try:
                r = requests.post(self.base + "/cubejs-api/v1/load", headers=hdr, json={"query": q}, timeout=120)
            except requests.RequestException as e:
                return {"error": f"request failed: {e}"}
            try:
                js = r.json()
            except ValueError:
                return {"error": f"bad response {r.status_code}: {r.text[:300]}"}
            if js.get("error") == "Continue wait":
                time.sleep(0.5)
                continue
            if "error" in js:
                return {"error": str(js["error"])[:500]}
            return {"data": js.get("data", [])}
        return {"error": "query timed out in Cube queue"}
