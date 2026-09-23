"""HTTP API wrapper using the same in-process logic as backend/main.py.

The API downloads the source video, runs SubtitleRemover directly with the
same settings as ``python backend/main.py --inpaint-mode sttn-det`` and exposes
the resulting MP4 through a browser-friendly download endpoint.
"""

import base64
import configparser
import json
import os
import re
import shutil
import subprocess
import tempfile
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
CONFIG_FILE = BACKEND_DIR / "video_subtitle_remover_api_direct_v6.conf"

_CONFIG = configparser.ConfigParser(interpolation=None)
_CONFIG.optionxform = str
_CONFIG.read(CONFIG_FILE, encoding="utf-8")


def setting(name, default):
    """Read an environment override, then the v6 config file, then default."""
    return os.environ.get(name, _CONFIG.defaults().get(name, default))


for _path in (str(BACKEND_DIR), str(RESOURCES_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.config import TRANSLATION_FILE, config, tr
from backend.main import SubtitleRemover
from backend.tools.common_tools import is_video_or_image
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.constant import InpaintMode
from backend.tools.subtitle_detect import SubtitleDetect
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
FFMPEG_PATH = BACKEND_DIR / "ffmpeg" / "linux_x64" / "ffmpeg"
PPOCRV6_DET_MODEL_DIR = BACKEND_DIR / "models" / "V6" / "PP-OCRv6_medium_det"
PPOCRV6_REC_MODEL_DIR = BACKEND_DIR / "models" / "V6" / "PP-OCRv6_medium_rec"
STTN_DET_MODEL_PATH = BACKEND_DIR / "models" / "sttn-det" / "sttn.pth"

DOWNLOAD_DIR = Path(
    setting("API_DOWNLOAD_DIR", "/mnt/nas_share_woka/video_subtitle_remover/downloads")
)
PROCESSED_DIR = Path(
    setting("API_PROCESSED_DIR", "/mnt/nas_share_woka/video_subtitle_remover/processed")
)
# PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
PUBLIC_BASE_URL = str(setting("PUBLIC_BASE_URL", "http://39.170.82.211:8027")).rstrip("/")
PROCESS_TIMEOUT_SECONDS = int(setting("PROCESS_TIMEOUT_SECONDS", "6000"))
DOWNLOAD_TIMEOUT_SECONDS = int(setting("DOWNLOAD_TIMEOUT_SECONDS", "300"))
OCR_DEVICE = setting("OCR_DEVICE", "gpu:0")
OCR_DET_DB_THRESH = float(setting("OCR_DET_DB_THRESH", "0.3"))
OCR_DET_DB_BOX_THRESH = float(setting("OCR_DET_DB_BOX_THRESH", "0.7"))
OCR_DET_DB_UNCLIP_RATIO = float(
    setting("OCR_DET_DB_UNCLIP_RATIO", "1.6")
)
OCR_FILTER_DETECTIONS = setting("OCR_FILTER_DETECTIONS", "1").lower() in {
    "1", "true", "yes", "on"
}
OCR_RECOGNIZE_TEXT = (
    OCR_FILTER_DETECTIONS
    or setting("OCR_RECOGNIZE_TEXT", "0").lower() in {"1", "true", "yes", "on"}
)
OCR_RECOGNIZE_EVERY_N_FRAMES = max(
    1, int(setting("OCR_RECOGNIZE_EVERY_N_FRAMES", "30"))
)
OCR_RECOGNITION_MIN_SCORE = float(
    setting("OCR_RECOGNITION_MIN_SCORE", "0.70")
)
OCR_FILTER_WHITE_TEXT = setting("OCR_FILTER_WHITE_TEXT", "1").lower() in {"1", "true", "yes", "on"}
OCR_WHITE_MIN_VALUE = int(setting("OCR_WHITE_MIN_VALUE", "180"))
OCR_WHITE_MAX_SATURATION = int(setting("OCR_WHITE_MAX_SATURATION", "55"))
OCR_WHITE_MIN_RATIO = float(setting("OCR_WHITE_MIN_RATIO", "0.025"))
OCR_WHITE_MIN_CONTRAST_RATIO = float(setting("OCR_WHITE_MIN_CONTRAST_RATIO", "0.012"))
OCR_MAX_TILT_DEGREES = float(setting("OCR_MAX_TILT_DEGREES", "15"))
# Boxes taller than they are wide are kept in the video (never inpainted).
OCR_KEEP_TALL_BOXES = setting("OCR_KEEP_TALL_BOXES", "1").lower() in {
    "1", "true", "yes", "on"
}
# Only boxes centred below this fraction of the frame height are inpainted;
# 0.5 means the lower half of the video, 0 disables the region filter.
OCR_REMOVE_MIN_Y_RATIO = float(setting("OCR_REMOVE_MIN_Y_RATIO", "0.5"))
# Reject unusually large regions (usually background/object false positives).
# Ratios are relative to the current video frame.
OCR_MAX_BOX_HEIGHT_RATIO = float(setting("OCR_MAX_BOX_HEIGHT_RATIO", "0.20"))
OCR_MAX_BOX_AREA_RATIO = float(setting("OCR_MAX_BOX_AREA_RATIO", "0.12"))
OCR_DEBUG_FILE = Path(
    setting("OCR_DEBUG_FILE", str(RESOURCES_DIR / "logs" / "ocr_debug.jsonl"))
)
OCR_DEBUG_LOCK = threading.Lock()
OCR_PORTRAIT_Y_MIN_RATIO = float(setting("OCR_PORTRAIT_Y_MIN_RATIO", "0.58"))
OCR_PORTRAIT_Y_MAX_RATIO = float(setting("OCR_PORTRAIT_Y_MAX_RATIO", "0.88"))
OCR_LANDSCAPE_Y_MIN_RATIO = float(setting("OCR_LANDSCAPE_Y_MIN_RATIO", "0.70"))
OCR_LANDSCAPE_Y_MAX_RATIO = float(setting("OCR_LANDSCAPE_Y_MAX_RATIO", "1.0"))
OCR_TOP_Y_MAX_RATIO = float(setting("OCR_TOP_Y_MAX_RATIO", "0.10"))
OCR_X_MIN_RATIO = float(setting("OCR_X_MIN_RATIO", "0.05"))
OCR_X_MAX_RATIO = float(setting("OCR_X_MAX_RATIO", "0.95"))
ANNOTATE_DETECTIONS = setting("ANNOTATE_DETECTIONS", "1").lower() in {"1", "true", "yes", "on"}
ANNOTATION_COLOR = (0, 0, 255)

app = Flask(__name__)
PROCESS_LOCK = threading.Lock()
MODEL_LOAD_LOCK = threading.Lock()
GLOBAL_PPOCR_DETECTOR = None
GLOBAL_STTN_DET_INPAINTER = None


def compute_subtitle_area(width: int, height: int):
    """
    返回坐标顺序: (y_min, y_max, x_min, x_max)
    """
    if height > width:
        y_min, y_max = int(height * OCR_PORTRAIT_Y_MIN_RATIO), int(height * OCR_PORTRAIT_Y_MAX_RATIO)
        x_min, x_max = int(width * OCR_X_MIN_RATIO), int(width * OCR_X_MAX_RATIO)
    else:
        y_min, y_max = int(height * OCR_LANDSCAPE_Y_MIN_RATIO), int(height * OCR_LANDSCAPE_Y_MAX_RATIO)
        x_min, x_max = int(width * OCR_X_MIN_RATIO), int(width * OCR_X_MAX_RATIO)
    if y_max <= y_min or x_max <= x_min:
        raise RuntimeError(
            f"字幕区域计算异常: width={width}, height={height}, "
            f"coords=({y_min}, {y_max}, {x_min}, {x_max})"
        )
    return y_min, y_max, x_min, x_max


class PPOCRV6Detector:
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
        self.recognizer = None
        self.frame_index = 0
        self.ocr_frame_index = 0
        self.last_ocr_results = []
        self.last_ocr_error = None
        self.frame_annotations = {}

        if OCR_RECOGNIZE_TEXT:
            from paddleocr import TextRecognition

            self.recognizer = TextRecognition(
                model_name="PP-OCRv6_medium_rec",
                model_dir=str(PPOCRV6_REC_MODEL_DIR),
                device=OCR_DEVICE,
            )
            log(
                "PP-OCRv6 text recognition enabled "
                f"(every_n_frames={OCR_RECOGNIZE_EVERY_N_FRAMES}, "
                f"min_score={OCR_RECOGNITION_MIN_SCORE})"
            )


    @staticmethod
    def _crop_polygon(image, polygon):
        points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
        top_width = np.linalg.norm(points[1] - points[0])
        bottom_width = np.linalg.norm(points[2] - points[3])
        left_height = np.linalg.norm(points[3] - points[0])
        right_height = np.linalg.norm(points[2] - points[1])
        crop_width = max(1, int(round(max(top_width, bottom_width))))
        crop_height = max(1, int(round(max(left_height, right_height))))
        target = np.array(
            [[0, 0], [crop_width - 1, 0],
             [crop_width - 1, crop_height - 1], [0, crop_height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(points, target)
        return cv2.warpPerspective(image, transform, (crop_width, crop_height))


    @staticmethod
    def _is_tall(polygon):
        """True when the box is taller than wide, i.e. text we keep on screen."""
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        width = float(points[:, 0].max() - points[:, 0].min())
        height = float(points[:, 1].max() - points[:, 1].min())
        return height > width

    @staticmethod
    def _tilt_degrees(polygon):
        """Return the absolute angle of the box's longest edge from horizontal."""
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 2:
            return 90.0
        edges = points - np.roll(points, 1, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        edge = edges[int(np.argmax(lengths))]
        angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
        # Box edges have no direction: map to [-90, 90].
        angle = (angle + 90.0) % 180.0 - 90.0
        return abs(angle)

    @staticmethod
    def _center_y(polygon):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        return float(points[:, 1].min() + points[:, 1].max()) / 2.0

    @staticmethod
    def _box_size(polygon):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        width = max(0.0, float(points[:, 0].max() - points[:, 0].min()))
        height = max(0.0, float(points[:, 1].max() - points[:, 1].min()))
        return width, height

    @staticmethod
    def _white_text_score(image, polygon):
        try:
            crop = PPOCRV6Detector._crop_polygon(image, polygon)
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                return 0.0, 0.0
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            white = (hsv[:, :, 2] >= OCR_WHITE_MIN_VALUE) & (hsv[:, :, 1] <= OCR_WHITE_MAX_SATURATION)
            dark = hsv[:, :, 2] <= max(0, OCR_WHITE_MIN_VALUE - 35)
            dark_nearby = cv2.dilate(dark.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            area = float(white.size)
            return float(white.sum()) / area, float((white & dark_nearby).sum()) / area
        except (TypeError, ValueError, cv2.error):
            return 0.0, 0.0

    @staticmethod
    def _get_result_value(result, key, default=None):
        if isinstance(result, dict):
            value = result.get(key, default)
        else:
            try:
                value = result[key]
            except (KeyError, IndexError, TypeError):
                value = getattr(result, key, default)
        if value is not None:
            return value
        payload = getattr(result, "json", None)
        if payload is not None:
            payload = payload() if callable(payload) else payload
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                payload = payload.get("res", payload)
                return payload.get(key, default)
        return default

    def recognize(self, image, polygons):
        if self.recognizer is None or len(polygons) == 0:
            return []
        crops = []
        valid_polygons = []
        for polygon_index, polygon in enumerate(polygons):
            try:
                crop = self._crop_polygon(image, polygon)
            except (TypeError, ValueError, cv2.error):
                continue
            if crop.size:
                crops.append(crop)
                valid_polygons.append((polygon_index, polygon))
        if not crops:
            return []
        self.last_ocr_error = None
        try:
            recognition_results = self.recognizer.predict(
                input=crops, batch_size=len(crops)
            )
            recognition_results = list(recognition_results or [])
        except Exception as exc:
            self.last_ocr_error = str(exc)
            log(f"OCR text recognition failed: {exc}")
            return []
        output = []
        for (polygon_index, polygon), result in zip(valid_polygons, recognition_results):
            text = self._get_result_value(result, "rec_text", "")
            score = self._get_result_value(result, "rec_score", 0.0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            points = np.asarray(polygon, dtype=np.float32)
            output.append({
                "index": polygon_index,
                "text": str(text or ""),
                "score": score,
                "text_score": score,
                "polygon": points.astype(int).tolist(),
                "box": [
                    int(points[:, 0].min()), int(points[:, 0].max()),
                    int(points[:, 1].min()), int(points[:, 1].max()),
                ],
            })
        return output

    def __call__(self, image, *args, **kwargs):
        started_at = time.perf_counter()
        self.frame_index += 1
        results = self.model.predict(image)
        detection_result = results[0] if results else None
        if detection_result is not None:
            polygons = self._get_result_value(detection_result, "dt_polys", [])
            boxes = np.asarray([] if polygons is None else polygons, dtype=np.float32)
            detection_scores = self._get_result_value(detection_result, "dt_scores", None)
            if detection_scores is None:
                detection_scores = self._get_result_value(detection_result, "scores", None)
            try:
                detection_scores = [float(value) for value in ([] if detection_scores is None else detection_scores)]
            except (TypeError, ValueError):
                detection_scores = []
        else:
            boxes = np.empty((0, 4, 2), dtype=np.float32)
            detection_scores = []
        if boxes.size == 0:
            boxes = np.empty((0, 4, 2), dtype=np.float32)
        detected_polygons = boxes.astype(int).tolist()
        self.last_ocr_results = []
        self.last_ocr_error = None
        all_boxes = boxes
        tall_indexes = (
            [index for index, box in enumerate(all_boxes) if self._is_tall(box)]
            if OCR_KEEP_TALL_BOXES
            else []
        )
        # Subtitles above the cut-off line (default: upper half) stay on screen.
        frame_height = float(np.asarray(image).shape[0])
        min_center_y = frame_height * OCR_REMOVE_MIN_Y_RATIO
        upper_indexes = (
            [
                index
                for index, box in enumerate(all_boxes)
                if self._center_y(box) < min_center_y
            ]
            if OCR_REMOVE_MIN_Y_RATIO > 0
            else []
        )
        oversized_indexes = []
        tilted_indexes = []
        for index, box in enumerate(all_boxes):
            if self._tilt_degrees(box) > OCR_MAX_TILT_DEGREES:
                tilted_indexes.append(index)
        frame_area = max(1.0, float(np.asarray(image).shape[0] * np.asarray(image).shape[1]))
        for index, box in enumerate(all_boxes):
            box_width, box_height = self._box_size(box)
            if (box_height / max(1.0, frame_height) > OCR_MAX_BOX_HEIGHT_RATIO
                    or box_width * box_height / frame_area > OCR_MAX_BOX_AREA_RATIO):
                oversized_indexes.append(index)
        skipped_indexes = (
            set(tall_indexes) | set(upper_indexes) | set(oversized_indexes)
            | set(tilted_indexes)
        )
        kept_indexes = [
            index for index in range(len(all_boxes)) if index not in skipped_indexes
        ]
        if self.recognizer is not None and kept_indexes:
            should_recognize = (
                OCR_FILTER_DETECTIONS
                or self.frame_index % OCR_RECOGNIZE_EVERY_N_FRAMES == 1
            )
            if should_recognize:
                self.ocr_frame_index += 1
                candidate_boxes = [all_boxes[index] for index in kept_indexes]
                results = self.recognize(image, candidate_boxes)
                # Map recognition output back onto the original detection indexes.
                for item in results:
                    item["index"] = kept_indexes[item["index"]]
                self.last_ocr_results = results
                if OCR_FILTER_DETECTIONS:
                    recognized_indexes = {
                        item["index"]
                        for item in results
                        if item["text"].strip()
                        and item["score"] >= OCR_RECOGNITION_MIN_SCORE
                    }
                    kept_indexes = [
                        index for index in kept_indexes if index in recognized_indexes
                    ]
        # Run color analysis only after OCR text/score filtering.
        white_scores = {}
        non_white_indexes = []
        if OCR_FILTER_WHITE_TEXT and kept_indexes:
            for index in kept_indexes:
                white_ratio, contrast_ratio = self._white_text_score(image, all_boxes[index])
                white_scores[index] = {"white_ratio": round(white_ratio, 5), "contrast_ratio": round(contrast_ratio, 5)}
                if white_ratio < OCR_WHITE_MIN_RATIO or contrast_ratio < OCR_WHITE_MIN_CONTRAST_RATIO:
                    non_white_indexes.append(index)
            kept_indexes = [index for index in kept_indexes if index not in non_white_indexes]

        boxes = np.asarray(
            [all_boxes[index] for index in kept_indexes], dtype=np.float32
        )
        if boxes.size == 0:
            boxes = np.empty((0, 4, 2), dtype=np.float32)
        recognized_by_index = {item.get("index"): item for item in self.last_ocr_results}
        self.frame_annotations[self.frame_index] = [{
            "polygon": np.asarray(all_boxes[index], dtype=np.int32).tolist(),
            "score": recognized_by_index.get(index, {}).get("score"),
            "detection_score": (detection_scores[index] if index < len(detection_scores) else None),
            "text": recognized_by_index.get(index, {}).get("text", ""),
            "text_score": recognized_by_index.get(index, {}).get("score"),
        } for index in kept_indexes]
        write_ocr_debug({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame": self.frame_index,
            "recognizer_enabled": self.recognizer is not None,
            "filter_enabled": OCR_FILTER_DETECTIONS,
            "keep_tall_boxes": OCR_KEEP_TALL_BOXES,
            "tall_indexes": tall_indexes,
            "remove_min_y_ratio": OCR_REMOVE_MIN_Y_RATIO,
            "remove_min_center_y": min_center_y,
            "upper_indexes": upper_indexes,
            "oversized_indexes": oversized_indexes,
            "tilted_indexes": tilted_indexes,
            "max_tilt_degrees": OCR_MAX_TILT_DEGREES,
            "max_box_height_ratio": OCR_MAX_BOX_HEIGHT_RATIO,
            "max_box_area_ratio": OCR_MAX_BOX_AREA_RATIO,
            "filter_white_text": OCR_FILTER_WHITE_TEXT,
            "white_min_value": OCR_WHITE_MIN_VALUE,
            "white_max_saturation": OCR_WHITE_MAX_SATURATION,
            "white_min_ratio": OCR_WHITE_MIN_RATIO,
            "white_min_contrast_ratio": OCR_WHITE_MIN_CONTRAST_RATIO,
            "white_scores": white_scores,
            "non_white_indexes": non_white_indexes,
            "min_score": OCR_RECOGNITION_MIN_SCORE,
            "detected_box_count": len(detected_polygons),
            "detected_polygons": detected_polygons,
            "detection_scores": detection_scores,
            "recognized_results": self.last_ocr_results,
            "recognition_error": self.last_ocr_error,
            "kept_indexes": kept_indexes,
            "kept_polygons": boxes.astype(int).tolist(),
            "discarded_indexes": [
                index for index in range(len(detected_polygons))
                if index not in kept_indexes
            ],
        })
        return boxes, time.perf_counter() - started_at


class OverlayVideoWriter:
    def __init__(self, writer, detector):
        self._writer = writer; self._detector = detector; self._frame_index = 0
    @staticmethod
    def _draw(frame, annotations):
        output = frame.copy()
        for item in annotations or []:
            points = np.asarray(item.get("polygon", []), dtype=np.int32)
            if points.shape != (4, 2): continue
            cv2.polylines(output, [points.reshape((-1, 1, 2))], True, ANNOTATION_COLOR, 2, cv2.LINE_AA)
            x = max(0, int(points[:, 0].min())); y = max(18, int(points[:, 1].min()) - 4)
            text_score = item.get("text_score")
            label = (f"text_score={text_score:.3f}"
                     if isinstance(text_score, (int, float)) else "text_score=?")
            text = str(item.get("text") or "").strip()
            if text: label += f" {text}"
            (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            top = max(0, y - height - baseline - 4)
            cv2.rectangle(output, (x, top), (x + width + 6, y), ANNOTATION_COLOR, -1)
            cv2.putText(output, label, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return output
    def write(self, frame):
        self._frame_index += 1
        self._writer.write(self._draw(frame, self._detector.frame_annotations.get(self._frame_index, [])))
    def release(self): return self._writer.release()
    def __getattr__(self, name): return getattr(self._writer, name)


def load_global_models():
    global GLOBAL_PPOCR_DETECTOR
    global GLOBAL_STTN_DET_INPAINTER

    if (
        GLOBAL_PPOCR_DETECTOR is not None
        and GLOBAL_STTN_DET_INPAINTER is not None
    ):
        return

    with MODEL_LOAD_LOCK:
        if (
            GLOBAL_PPOCR_DETECTOR is not None
            and GLOBAL_STTN_DET_INPAINTER is not None
        ):
            return

        log("loading PP-OCR and STTN models")
        accelerator = HardwareAccelerator.instance()
        accelerator.set_enabled(config.hardwareAcceleration.value)
        GLOBAL_PPOCR_DETECTOR = PPOCRV6Detector()
        SubtitleDetect.text_detector = property(
            lambda _detector: GLOBAL_PPOCR_DETECTOR
        )
        GLOBAL_STTN_DET_INPAINTER = STTNDetInpaint(
            accelerator.device,
            str(STTN_DET_MODEL_PATH),
        )
        log(
            "PP-OCR and STTN models loaded "
            f"(db_thresh={OCR_DET_DB_THRESH}, "
            f"box_thresh={OCR_DET_DB_BOX_THRESH}, "
            f"unclip_ratio={OCR_DET_DB_UNCLIP_RATIO})"
        )


def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [subtitle-remover-api] {message}", flush=True)


def write_ocr_debug(record):
    """Append one complete OCR decision record as JSONL for offline analysis."""
    try:
        OCR_DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OCR_DEBUG_LOCK:
            with OCR_DEBUG_FILE.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log(f"OCR debug file write failed: {exc}")


def summarize_log_value(value, limit=500):
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    else:
        value = str(value)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


def log_parameters(message, params):
    safe_params = {key: summarize_log_value(value) for key, value in params.items()}
    log(f"{message}: {json.dumps(safe_params, ensure_ascii=False)}")


def clean_url(value):
    if not value:
        return value
    value = str(value).strip().strip("`")
    markdown_match = re.match(r"^\[[^\]]+\]\((https?://[^)]+)\)$", value)
    return markdown_match.group(1) if markdown_match else value


def json_response(job_id, status, video_url=None, message=None, http_status=200):
    payload = {
        "jobId": job_id,
        "status": status,
    }
    if video_url is not None:
        payload["videoUrl"] = video_url
    if message is not None:
        payload["message"] = message
    return jsonify(payload), http_status


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
    local_path = DOWNLOAD_DIR / f"{safe_job_id(job_id)}{get_input_suffix(video_url)}"
    source_path = get_local_video_path(video_url)

    if source_path is not None:
        if not source_path.is_file():
            raise RuntimeError(f"本地视频文件不存在: {source_path}")
        shutil.copyfile(source_path, local_path)
    elif video_url.startswith("data:"):
        try:
            encoded = video_url.split(",", 1)[1]
            local_path.write_bytes(base64.b64decode(encoded, validate=True))
        except (IndexError, ValueError) as exc:
            raise RuntimeError("Base64 视频数据格式无效") from exc
    elif video_url.startswith(("http://", "https://")):
        with requests.get(
            video_url,
            headers={"User-Agent": "subtitle-remover-api/1.0"},
            stream=True,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            with local_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
    else:
        try:
            local_path.write_bytes(base64.b64decode(video_url, validate=True))
        except ValueError as exc:
            raise RuntimeError(
                "video_url 必须是 HTTP(S) URL、file:// 本地文件 URI、本地绝对路径或有效的 Base64 视频数据"
            ) from exc

    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise RuntimeError("视频下载失败或文件为空")
    return local_path


def build_public_video_url(output_path):
    base_url = PUBLIC_BASE_URL or request.url_root.rstrip("/")
    return f"{base_url}/api/v1/processed/{quote(output_path.name)}"


def browser_compatible_video(video_path):
    """Remux/re-encode output for browser playback while retaining audio."""
    ffmpeg_path = Path(setting("FFMPEG_PATH", str(FFMPEG_PATH)))
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


def process_video_direct(input_path, output_path):
    if not is_video_or_image(str(input_path)):
        raise RuntimeError(f"不支持或已损坏的视频文件: {input_path}")

    previous_mode = config.inpaintMode.value
    try:
        config.set(config.interface, "en")
        tr.read(TRANSLATION_FILE, encoding="utf-8")
        load_global_models()

        remover = SubtitleRemover(str(input_path))
        remover.show_processing_details = False
        if not is_video_or_image(str(input_path)):
            raise RuntimeError(f"不支持或已损坏的视频文件: {input_path}")
        remover.sub_areas = [
            compute_subtitle_area(remover.frame_width, remover.frame_height),
            (0, int(remover.frame_height * OCR_TOP_Y_MAX_RATIO),
             int(remover.frame_width * OCR_X_MIN_RATIO), int(remover.frame_width * OCR_X_MAX_RATIO)),
        ]
        remover.video_out_path = str(output_path)
        # Bypass SubtitleRemover's per-instance cached_property and reuse the
        # process-wide STTN model loaded by load_global_models().
        remover.sttn_det_inpaint = GLOBAL_STTN_DET_INPAINTER
        if ANNOTATE_DETECTIONS:
            remover.video_writer = OverlayVideoWriter(remover.video_writer, GLOBAL_PPOCR_DETECTOR)
        config.inpaintMode.value = InpaintMode.STTN_DET
        remover.run()
    finally:
        config.inpaintMode.value = previous_mode


def send_callback(callback_url, payload):
    if not callback_url:
        log_parameters("callback skipped", {"callback_url": callback_url, "payload": payload})
        return
    log_parameters("callback sending", {"callback_url": callback_url, "payload": payload})
    try:
        response = requests.post(
            callback_url,
            json=payload,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        log(f"回调结果: status={response.status_code}, body={response.text[:300]}")
    except Exception as exc:
        log(f"回调失败: {exc}")


def publish_video(local_path, output_path):
    """Publish a completed local video to NAS, tolerating NAS rename limits."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(local_path, output_path)
    except PermissionError as copy_error:
        try:
            if output_path.exists():
                output_path.unlink()
            shutil.copyfile(local_path, output_path)
        except Exception as retry_error:
            raise RuntimeError(
                "无法发布视频到 NAS；覆盖和删除后重试都失败: "
                f"copy={copy_error}; retry={retry_error}"
            ) from retry_error


def process_job(job_id, video_url, callback_url):
    input_path = download_video(video_url, job_id)
    output_path = PROCESSED_DIR / f"{safe_job_id(job_id)}_clear-video.mp4"
    staging_fd, staging_name = tempfile.mkstemp(
        prefix=f"{safe_job_id(job_id)}_", suffix=".mp4"
    )
    os.close(staging_fd)
    staging_path = Path(staging_name)
    staging_path.unlink()

    try:
        # Keep SubtitleRemover and ffmpeg audio merge off the NAS mount.
        process_video_direct(input_path, staging_path)
        if not staging_path.is_file() or staging_path.stat().st_size == 0:
            raise RuntimeError("直接处理未生成有效输出视频")

        browser_compatible_video(staging_path)
        if not staging_path.is_file() or staging_path.stat().st_size == 0:
            raise RuntimeError("浏览器兼容转码后未生成有效视频")

        publish_video(staging_path, output_path)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("视频发布到 NAS 后文件为空")

        payload = {
            "jobId": job_id,
            "status": "success",
            "videoUrl": build_public_video_url(output_path),
        }
        send_callback(callback_url, payload)
        return payload
    finally:
        if staging_path.exists():
            staging_path.unlink()


@app.route("/api/v1/processed/<path:filename>", methods=["GET"])
def processed_video(filename):
    # conditional=True enables Range/conditional responses used by HTML5 video.
    return send_from_directory(
        PROCESSED_DIR,
        filename,
        as_attachment=False,
        conditional=True,
        max_age=0,
    )


@app.route("/api/v1/video_subtitle_remover", methods=["POST"])
def video_subtitle_remover():
    request_started_at = time.time()
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id") or body.get("jobId")
    video_url = clean_url(body.get("video_url") or body.get("videoUrl"))
    callback_url = clean_url(body.get("callback_url") or body.get("callbackUrl"))
    log_parameters(
        "request received",
        {
            "job_id": job_id,
            "video_url": video_url,
            "callback_url": callback_url,
            "body": body,
        },
    )
    if not job_id:
        return json_response(None, "failed", message="job_id is required", http_status=400)
    if not video_url:
        return json_response(
            job_id,
            "failed",
            message="video_url is required",
            http_status=400,
        )

    try:
        # The in-process lock avoids concurrent GPU and global config contention.
        with PROCESS_LOCK:
            payload = process_job(job_id, video_url, callback_url)
        log(f"request finished: job_id={job_id}, elapsed={time.time() - request_started_at:.1f}s")
        return jsonify(payload)
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "jobId": job_id,
            "status": "failed",
            "message": str(exc),
        }
        send_callback(callback_url, payload)
        log(f"request failed: job_id={job_id}, elapsed={time.time() - request_started_at:.1f}s")
        return jsonify(payload), 500


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(setting("PORT", "8029")))
