import os
import time
import requests


def _build_web_command(web_service_folder: str, listen_port: str) -> str:
    return f"cd {web_service_folder}; LISTEN_PORT={listen_port} npm start"


def wait_until_listening(web_ip: str, timeout_sec: int = 60) -> None:
    url = f"http://{web_ip}/healthz"
    start = time.time()
    last_err = None
    while time.time() - start < timeout_sec:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code in (200, 404):
                return  # server is up (healthz may not exist, 404 is fine)
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise RuntimeError(f"Web service at {web_ip} did not come up within {timeout_sec}s: {last_err}")


def test_webenv_connectivity():
    web_ip = os.environ.get("WEB_IP", "localhost:3000")
    web_service_folder = os.environ.get("WEB_SERVICE_FOLDER", "")
    listen_port = os.environ.get("LISTEN_PORT", web_ip.split(":")[-1])

    assert web_service_folder, "WEB_SERVICE_FOLDER must be set to the _web directory"

    web_command = _build_web_command(web_service_folder, listen_port)

    # Fire up the server one-shot to validate connectivity
    # Do not rely on internal start delay; actively wait until reachable
    import subprocess, signal
    popen = subprocess.Popen(web_command, shell=True, preexec_fn=os.setsid)
    try:
        wait_until_listening(web_ip, timeout_sec=90)
        # basic endpoint call used by WebEnv
        url = f"http://{web_ip}/getBrowser"
        resp = requests.post(url, json={"storageState": None, "geoLocation": None}, timeout=30)
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}, body={resp.text}"
    finally:
        try:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
        except Exception:
            pass


if __name__ == "__main__":
    test_webenv_connectivity()
    print("OK: web connectivity basic test passed")




