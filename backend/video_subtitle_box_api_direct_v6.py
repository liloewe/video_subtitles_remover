"""HTTP API that detects subtitles with PP-OCRv6 and draws boxes on video.

The request and callback format matches ``backend1/video_subtitle_box_api.py``.
Unlike the older API, this module uses PaddleOCR's PP-OCRv6 text detector
directly and keeps audio by remuxing the rendered video with ffmpeg.
"""

import base64
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import url2pathname

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory


BACKEND_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BACKEND_DIR.parent
for _path in (str(BACKEND_DIR), str(RESOURCES_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.config import config


FFMPEG_PATH = BACKEND_DIR / "ffmpeg" / "linux_x64" / "ffmpeg"
PPOCRV6_DET_MODEL_DIR = BACKEND_DIR / "models" / "V6" / "PP-OCRv6_medium_det"

DOWNLOAD_DIR = Path(
    os.environ.get(
        "API_DOWNLOAD_DIR",
        "/mnt/nas_share_woka/video_subtitle_remover/downloads",
    )
)
PROCESSED_DIR = Path(
    os.environ.get(
        "API_PROCESSED_DIR",
        "/mnt/nas_share_woka/video_subtitle_remover/processed",
    )
)
PUBLIC_BASE_URL = os.environ.get(
    "PUBLIC_BASE_URL",
    "http://39.170.82.211:8027",
).rstrip("/")
PROCESS_TIMEOUT_SECONDS = int(os.environ.get("PROCESS_TIMEOUT_SECONDS", "6000"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "300"))
OCR_DEVICE = os.environ.get("OCR_DEVICE", "gpu:0")
OCR_DET_DB_THRESH = float(os.environ.get("OCR_DET_DB_THRESH", "0.3"))
OCR_DET_DB_BOX_THRESH = float(os.environ.get("OCR_DET_DB_BOX_THRESH", "0.85"))
OCR_DET_DB_UNCLIP_RATIO = float(
    os.environ.get("OCR_DET_DB_UNCLIP_RATIO", "1.6")
)
DEFAULT_REGION_MIN_FRAME_RATIO = float(
    os.environ.get("OCR_REGION_MIN_FRAME_RATIO", "0.05")
)

app = Flask(__name__)
PROCESS_LOCK = threading.Lock()
MODEL_LOAD_LOCK = threading.Lock()
GLOBAL_DETECTOR = None


def log(message):
    print(f"[subtitle-box-v6] {message}", flush=True)


class PPOCRV6Detector:
    """Small adapter exposing the same ``(boxes, scores)`` shape as the old API."""

    def __init__(self):
        from paddleocr import TextDetection

        self.model = TextDetection(
            model_name="PP-OCRv6_medium_det",
            model_dir=str(PPOCRV6_DET_MODEL_DIR),
            device=OCR_DEVICE,
            enable_mkldnn=False,
            enable_cinn=False,
            thresh=OCR_DET_DB_THRESH,
            box_thresh=OCR_DET_DB_BOX_THRESH,
            unclip_ratio=OCR_DET_DB_UNCLIP_RATIO,
        )

    def __call__(self, image):
        results = self.model.predict(image)
        if not results:
            return np.empty((0, 4, 2), dtype=np.float32), None

        result = results[0]
        polygons = np.asarray(result.get("dt_polys", []), dtype=np.float32)
        if polygons.size == 0:
            polygons = np.empty((0, 4, 2), dtype=np.float32)
        return polygons, None


def load_model():
    global GLOBAL_DETECTOR
    if GLOBAL_DETECTOR is not None:
        return GLOBAL_DETECTOR

    with MODEL_LOAD_LOCK:
        if GLOBAL_DETECTOR is None:
            log(
                "loading PP-OCRv6 model: "
                f"model_dir={PPOCRV6_DET_MODEL_DIR}, device={OCR_DEVICE}, "
                f"db_thresh={OCR_DET_DB_THRESH}, "
                f"box_thresh={OCR_DET_DB_BOX_THRESH}, "
                f"unclip_ratio={OCR_DET_DB_UNCLIP_RATIO}"
            )
            GLOBAL_DETECTOR = PPOCRV6Detector()
            log("PP-OCRv6 model loaded")
    return GLOBAL_DETECTOR


def clean_url(value):
    if not value:
        return value
    value = str(value).strip().strip("`")
    markdown_match = re.match(r"^\[[^\]]+\]\((https?://[^)]+)\)$", value)
    return markdown_match.group(1) if markdown_match else value


def safe_job_id(job_id):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(job_id)).strip("._")
    return value or "job"


def get_input_suffix(video_url):
    suffix = Path(urlparse(video_url).path).suffix.lower()
    return suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix or "") else ".mp4"


def get_local_video_path(video_url):
    parsed = urlparse(video_url)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise RuntimeError(f"不支持远程 file URL: {video_url}")
        return Path(url2pathname(unquote(parsed.path)))
    if parsed.scheme == "":
        local_path = Path(video_url).expanduser()
        if local_path.is_absolute():
            return local_path
    return None


def download_video(video_url, job_id):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    input_path = DOWNLOAD_DIR / f"{safe_job_id(job_id)}{get_input_suffix(video_url)}"
    local_path = get_local_video_path(video_url)

    if local_path is not None:
        if not local_path.is_file():
            raise RuntimeError(f"本地视频文件不存在: {local_path}")
        shutil.copyfile(local_path, input_path)
    elif video_url.startswith("data:"):
        try:
            encoded = video_url.split(",", 1)[1]
            input_path.write_bytes(base64.b64decode(encoded, validate=True))
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Base64 视频数据格式无效") from exc
    elif video_url.startswith(("http://", "https://")):
        with requests.get(
            video_url,
            headers={"User-Agent": "subtitle-box-v6/1.0"},
            stream=True,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            with input_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
    else:
        try:
            input_path.write_bytes(base64.b64decode(video_url, validate=True))
        except ValueError as exc:
            raise RuntimeError(
                "video_url 必须是 HTTP(S) URL、file:// 本地文件 URI、"
                "本地绝对路径或有效的 Base64 视频数据"
            ) from exc

    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise RuntimeError("视频下载失败或文件为空")
    return input_path


def build_public_video_url(output_path):
    base_url = PUBLIC_BASE_URL or request.url_root.rstrip("/")
    return f"{base_url}/api/v1/processed/{quote(output_path.name)}"


def json_response(job_id, status, video_url=None, message=None, http_status=200):
    payload = {"jobId": job_id, "status": status}
    if video_url is not None:
        payload["videoUrl"] = video_url
    if message is not None:
        payload["message"] = message
    return jsonify(payload), http_status


def get_readable_path(path):
    return str(path)


def polygon_to_box(polygon, width, height):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return None
    xmin = max(0, min(width - 1, int(np.floor(points[:, 0].min()))))
    xmax = max(0, min(width - 1, int(np.ceil(points[:, 0].max()))))
    ymin = max(0, min(height - 1, int(np.floor(points[:, 1].min()))))
    ymax = max(0, min(height - 1, int(np.ceil(points[:, 1].max()))))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, xmax, ymin, ymax


def get_boxes_with_scores(polygons, scores, width, height):
    boxes = []
    polygons = np.asarray(polygons)
    score_values = (
        scores.tolist()
        if scores is not None and hasattr(scores, "tolist")
        else list(scores or [])
    )
    for index, polygon in enumerate(polygons):
        box = polygon_to_box(polygon, width, height)
        if box is None:
            continue
        score = float(score_values[index]) if index < len(score_values) else None
        boxes.append((*box, score))
    return boxes


def merge_two_boxes(first, second):
    return (
        min(first[0], second[0]),
        max(first[1], second[1]),
        min(first[2], second[2]),
        max(first[3], second[3]),
    )


def box_height(box):
    return max(1, box[3] - box[2])


def merge_boxes_by_rows(boxes, gap_ratio=0.35):
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda box: (box[2], box[0]))
    line_groups = [[sorted_boxes[0]]]
    for box in sorted_boxes[1:]:
        current = line_groups[-1]
        group_box = current[0]
        for item in current[1:]:
            group_box = merge_two_boxes(group_box, item)
        vertical_gap = box[2] - group_box[3]
        threshold = max(box_height(group_box), box_height(box)) * gap_ratio
        if vertical_gap <= threshold:
            current.append(box)
        else:
            line_groups.append([box])

    merged = []
    for group in line_groups:
        group_box = group[0]
        for item in group[1:]:
            group_box = merge_two_boxes(group_box, item)
        merged.append(group_box)
    return merged


def scan_video_boxes(video_path, detector):
    cap = cv2.VideoCapture(get_readable_path(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
    frame_boxes = {}
    log(
        f"box scan start: frames={frame_count}, size={width}x{height}, "
        f"fps={fps:.3f}"
    )

    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            polygons, scores = detector(frame)
            frame_boxes[frame_no] = get_boxes_with_scores(
                polygons,
                scores,
                width,
                height,
            )
            if frame_no % 100 == 0:
                log(f"box scan progress: {frame_no}/{frame_count}")
    finally:
        cap.release()

    log(
        f"box scan finished: frames={frame_no}, "
        f"frames_with_boxes={sum(bool(boxes) for boxes in frame_boxes.values())}"
    )
    return frame_boxes, (width, height, fps, frame_count)


def find_frequent_regions(frame_boxes, frame_count):
    line_regions = []
    for frame_no, boxes in frame_boxes.items():
        for line_xmin, line_xmax, line_ymin, line_ymax in merge_boxes_by_rows(
            [box[:4] for box in boxes]
        ):
            line_height = max(1, line_ymax - line_ymin)
            line_center = (line_ymin + line_ymax) / 2.0
            compatible = []
            for index, region in enumerate(line_regions):
                center_gap = abs(line_center - region["center"])
                height_gap = abs(line_height - region["height"])
                center_limit = max(10.0, 0.75 * max(line_height, region["height"]))
                height_limit = max(8.0, 0.60 * max(line_height, region["height"]))
                if center_gap <= center_limit and height_gap <= height_limit:
                    compatible.append((center_gap + height_gap, index))

            if compatible:
                _, region_index = min(compatible)
                region = line_regions[region_index]
            else:
                region = {
                    "center": line_center,
                    "height": float(line_height),
                    "frames": set(),
                    "observations": 0,
                }
                line_regions.append(region)

            region["frames"].add(frame_no)
            region["center"] = region["center"] * 0.8 + line_center * 0.2
            region["height"] = region["height"] * 0.8 + line_height * 0.2
            region["observations"] += 1

    min_frames = max(3, int(frame_count * DEFAULT_REGION_MIN_FRAME_RATIO + 0.5))
    regions = []
    for region in line_regions:
        if len(region["frames"]) < min_frames:
            continue
        half_height = region["height"] / 2.0
        regions.append(
            {
                "top": region["center"] - half_height,
                "bottom": region["center"] + half_height,
                "center": region["center"],
                "min_height": region["height"],
                "max_height": region["height"],
                "frames": set(region["frames"]),
                "observations": region["observations"],
            }
        )

    regions.sort(key=lambda region: region["top"])
    merged = []
    for region in regions:
        if not merged:
            merged.append(region)
            continue
        previous = merged[-1]
        gap = region["top"] - previous["bottom"]
        max_height = max(previous["max_height"], region["max_height"])
        shared_frames = len(previous["frames"] & region["frames"])
        shorter_count = max(
            1,
            min(len(previous["frames"]), len(region["frames"])),
        )
        overlap_ratio = shared_frames / shorter_count
        previous_last = max(previous["frames"])
        region_first = min(region["frames"])
        temporal_gap = abs(region_first - previous_last)
        if gap <= max(12.0, 1.5 * max_height) and (
            overlap_ratio >= 0.10
            or temporal_gap <= max(3, int(frame_count * 0.02 + 0.5))
        ):
            previous["top"] = min(previous["top"], region["top"])
            previous["bottom"] = max(previous["bottom"], region["bottom"])
            previous["center"] = (previous["top"] + previous["bottom"]) / 2.0
            previous["frames"].update(region["frames"])
            previous["observations"] += region["observations"]
            previous["min_height"] = min(
                previous["min_height"],
                region["min_height"],
            )
            previous["max_height"] = max(
                previous["max_height"],
                region["max_height"],
            )
        else:
            merged.append(region)

    log(
        f"frequent subtitle regions: line_regions={len(line_regions)}, "
        f"kept_blocks={len(merged)}, min_frames={min_frames}"
    )
    return merged


def filter_boxes_by_regions(frame_boxes, regions):
    filtered = {}
    for frame_no, boxes in frame_boxes.items():
        kept = []
        for box in boxes:
            xmin, xmax, ymin, ymax, score = box
            current_height = max(1, ymax - ymin)
            center = (ymin + ymax) / 2.0
            for region in regions:
                margin = max(
                    10.0,
                    0.60 * max(region["max_height"], current_height),
                )
                inside = region["top"] - margin <= center <= region["bottom"] + margin
                plausible_height = (
                    current_height >= max(4.0, region["min_height"] * 0.45)
                    and current_height <= region["max_height"] * 1.80
                )
                if inside and plausible_height:
                    kept.append(box)
                    break
        filtered[frame_no] = kept
    return filtered


def draw_boxes(frame, boxes):
    output = frame.copy()
    for xmin, xmax, ymin, ymax, score in boxes:
        cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        if score is None:
            continue
        label = f"{score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )
        label_top = max(0, ymin - label_height - baseline - 4)
        label_bottom = label_top + label_height + baseline + 4
        cv2.rectangle(
            output,
            (xmin, label_top),
            (xmin + label_width + 6, label_bottom),
            (0, 0, 255),
            thickness=-1,
        )
        cv2.putText(
            output,
            label,
            (xmin + 3, label_bottom - baseline - 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


def draw_boxes_on_video(video_path, output_path, frame_boxes, video_info):
    _, _, fps, frame_count = video_info
    cap = cv2.VideoCapture(get_readable_path(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        get_readable_path(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建视频输出: {output_path}")

    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            writer.write(draw_boxes(frame, frame_boxes.get(frame_no, [])))
            if frame_no % 100 == 0:
                log(f"box render progress: {frame_no}/{frame_count}")
    finally:
        cap.release()
        writer.release()
    log(f"box render finished: output={output_path}")


def browser_compatible_video(video_path):
    ffmpeg_path = Path(os.environ.get("FFMPEG_PATH", str(FFMPEG_PATH)))
    if not ffmpeg_path.is_file():
        raise RuntimeError(f"找不到 ffmpeg: {ffmpeg_path}")

    temp_path = video_path.with_name(f"{video_path.stem}.browser.tmp{video_path.suffix}")
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(RESOURCES_DIR),
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"浏览器兼容转码失败: {detail[-2000:]}")
        os.replace(temp_path, video_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def send_callback(callback_url, payload):
    if not callback_url:
        return
    try:
        response = requests.post(
            callback_url,
            json=payload,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        log(f"回调结果: status={response.status_code}, body={response.text[:300]}")
    except Exception as exc:
        log(f"回调失败: {exc}")


def normalize_ocr_mode(body):
    mode = body.get("ocr_mode") or body.get("ocrMode")
    if mode is None and "use_roi" in body:
        value = body.get("use_roi")
        mode = (
            "roi"
            if isinstance(value, bool) and value
            else "full"
            if isinstance(value, bool)
            else "roi"
            if str(value).strip().lower() in {"1", "true", "yes", "roi"}
            else "full"
        )

    mode = str(mode or "roi").strip().lower()
    mode = {
        "region": "roi",
        "full_screen": "full",
        "fullscreen": "full",
        "all": "full",
    }.get(mode, mode)
    if mode not in {"roi", "full"}:
        raise ValueError("ocr_mode must be either 'roi' or 'full'")
    return mode


def process_job(job_id, video_url, callback_url, ocr_mode):
    input_path = download_video(video_url, job_id)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{safe_job_id(job_id)}_boxed-video.mp4"

    detector = load_model()
    frame_boxes, video_info = scan_video_boxes(input_path, detector)
    regions = find_frequent_regions(frame_boxes, video_info[3])
    filtered_boxes = filter_boxes_by_regions(frame_boxes, regions)
    log(
        f"job_id={job_id}, ocr_mode={ocr_mode}, "
        f"raw_boxes={sum(len(boxes) for boxes in frame_boxes.values())}, "
        f"kept_boxes={sum(len(boxes) for boxes in filtered_boxes.values())}"
    )
    draw_boxes_on_video(input_path, output_path, filtered_boxes, video_info)
    browser_compatible_video(output_path)

    payload = {
        "jobId": job_id,
        "status": "success",
        "videoUrl": build_public_video_url(output_path),
    }
    send_callback(callback_url, payload)
    return payload


@app.route("/api/v1/processed/<path:filename>", methods=["GET"])
def processed_video(filename):
    return send_from_directory(
        PROCESSED_DIR,
        filename,
        as_attachment=False,
        conditional=True,
        max_age=0,
    )


@app.route("/api/v1/video_subtitle_box", methods=["POST"])
def video_subtitle_box():
    request_started_at = time.time()
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id") or body.get("jobId")
    video_url = clean_url(body.get("video_url") or body.get("videoUrl"))
    callback_url = clean_url(body.get("callback_url") or body.get("callbackUrl"))

    try:
        ocr_mode = normalize_ocr_mode(body)
    except ValueError as exc:
        return json_response(job_id, "failed", message=str(exc), http_status=400)

    if not job_id:
        return json_response(None, "failed", message="job_id is required", http_status=400)
    if not video_url:
        return json_response(
            job_id,
            "failed",
            message="video_url is required",
            http_status=400,
        )

    log(f"request received: job_id={job_id}, ocr_mode={ocr_mode}")
    try:
        with PROCESS_LOCK:
            payload = process_job(job_id, video_url, callback_url, ocr_mode)
        log(
            f"request finished: job_id={job_id}, "
            f"elapsed={time.time() - request_started_at:.1f}s"
        )
        return jsonify(payload)
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "jobId": job_id,
            "status": "failed",
            "message": str(exc),
        }
        send_callback(callback_url, payload)
        log(
            f"request failed: job_id={job_id}, "
            f"elapsed={time.time() - request_started_at:.1f}s"
        )
        return jsonify(payload), 500


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8028")))
