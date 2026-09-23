import base64
from pathlib import Path
from urllib.parse import urlparse
import threading
import time

import requests
from flask import Flask, jsonify, request


API_URL = "http://127.0.0.1:4076/api/v1/video_subtitle_remover"
CALLBACK_URL = "http://127.0.0.1:8003/api/v1/callbacks/subtitle-removal"
CALLBACK_HOST = "0.0.0.0"
CALLBACK_PORT = 8003
CALLBACK_DOWNLOAD_DIR = Path("./datasets/callback_downloads1")
LOCAL_VIDEO = Path(
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/video/婚礼当天/01.mp4"
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/video/余公公/01.mp4"
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/video/53.夫人又把先生哄成翘嘴了（60集）丁铭＆陈俊如/04.mp4"
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/20.mp4"
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/19s.mp4"
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/ted.mp4"
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/video/上司/02.mp4"
    # "/home/tianzhiuser/project/video_subtitles_remover/resources/video/pw6c6_1975847.mp4"
    "3.mp4"

)

app = Flask(__name__)


def download_callback_video(video_url, job_id):
    CALLBACK_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(video_url).path).suffix or ".mp4"
    filename = f"{job_id}_clear-video{suffix}"
    output_path = CALLBACK_DOWNLOAD_DIR / filename
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(
        video_url,
        headers={"User-Agent": "subtitle-remover-callback-client/1.0"},
        stream=True,
        timeout=600,
    ) as response:
        response.raise_for_status()
        with temp_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    temp_path.replace(output_path)
    return output_path


@app.route("/api/v1/callbacks/subtitle-removal", methods=["POST"])
def subtitle_removal_callback():
    body = request.get_json(silent=True) or {}
    print("callback received:")
    print(body)

    status = str(body.get("status", "")).strip().lower()
    video_url = body.get("videoUrl") or body.get("video_url")
    job_id = body.get("jobId") or body.get("job_id") or "job"

    if status == "success" and video_url:
        try:
            output_path = download_callback_video(video_url, job_id)
            print(f"callback video downloaded: {output_path}", flush=True)
        except requests.RequestException as exc:
            print(f"video download failed: {exc}")
            return jsonify({"ok": False, "message": str(exc)}), 502
        except OSError as exc:
            print(f"video save failed: {exc}")
            return jsonify({"ok": False, "message": str(exc)}), 500

    return jsonify({"ok": True})


def run_callback_server():
    app.run(host=CALLBACK_HOST, port=CALLBACK_PORT, debug=False, use_reloader=False)


def submit_job():
    video_base64 = base64.b64encode(LOCAL_VIDEO.read_bytes()).decode("ascii")
    payload = {
        "job_id": "b4fed56614524c26b492817cafbfdd01",
        "video_url": f"data:video/mp4;base64,{video_base64}",
        "callback_url": CALLBACK_URL,
        # "ocr_mode": "full",  # 可选："roi"（默认）或 "full"
        "ocr_mode": "full"
    }
    resp = requests.post(API_URL, json=payload, timeout=6000)
    print(resp.status_code)
    print(resp.text)


if __name__ == "__main__":
    threading.Thread(target=run_callback_server, daemon=True).start()
    time.sleep(1)
    submit_job()
