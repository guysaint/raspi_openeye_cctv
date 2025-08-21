#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다중 ROI 템플릿 캡처 도구
- 프레임을 정지시킨 뒤, 여러 개 ROI(관심 영역)를 한 번에 선택하고
  자동으로 template_img1.jpg, template_img2.jpg ... 형식으로 저장.

키 조작
  m : 현재 프레임을 정지시키고 다중 ROI 선택 창 열기
  r : 정지된 화면에서 다시 실시간 프리뷰로 돌아가기
  q/ESC : 프로그램 종료

설정
  - SOURCE: 카메라 인덱스 또는 동영상 파일 경로
  - OUTPUT_DIR: 잘라낸 이미지를 저장할 폴더
  - PREFIX: 저장될 파일 이름 앞부분 (예: 'template_img')
  - START_IDX: 시작 번호 (None이면 기존 파일을 보고 자동으로 다음 번호부터 저장)
"""
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"

from picamera2 import Picamera2
import cv2
import numpy as np
import os
import re
from pathlib import Path
from typing import Optional, List, Tuple

# ====== Config ======
SOURCE = 0
OUTPUT_DIR = "../assets"
PREFIX = "template_img"
START_IDX = None     # None => auto-detect next index by scanning OUTPUT_DIR
EXT = ".jpg"         # ".jpg" or ".png"
JPEG_QUALITY = 95    # only for .jpg
FRAME_MAX_W = 960   # scale down preview for speed; saved crops are from frozen frame (same size)


def is_valid_frame(f) -> bool:
    return f is not None and getattr(f, "size", 0) > 0 and f.shape[1] > 0 and f.shape[0] > 0


def resize_keep_w(img, max_w: int):
    if not is_valid_frame(img):
        return None, 1.0
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    s = max_w / float(w)
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return out, s


def next_index(output_dir: str, prefix: str, ext: str, start_idx: Optional[int]) -> int:
    if start_idx is not None:
        return int(start_idx)
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\\d+){re.escape(ext)}$")
    max_n = 0
    for name in os.listdir(p):
        m = pattern.match(name)
        if m:
            try:
                n = int(m.group(1))
                max_n = max(max_n, n)
            except ValueError:
                pass
    return max_n + 1


def save_crop(img_bgr, roi: Tuple[int,int,int,int], out_path: Path, ext: str) -> bool:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return False
    H, W = img_bgr.shape[:2]
    x2, y2 = min(W, x + w), min(H, y + h)
    crop = img_bgr[y:y2, x:x2]
    if crop.size == 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ext.lower() == ".jpg":
        return bool(cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]))
    else:
        return bool(cv2.imwrite(str(out_path), crop))

def open_picam2(size=(1280, 720)):
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": size, "format": "RGB888"})
    picam2.configure(cfg)
    picam2.start()
    picam2.set_controls({
        "AeEnable": False, "AwbEnable": False,
        "ExposureTime": 10000,  # 10ms 예시 (환경에 맞게)
        "AnalogueGain": 1.5
    })
    
    return picam2

def read_frame_bgr(picam2):
    frame = picam2.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def main():
    USE_PICAM2 = True

    if USE_PICAM2:
        picam2 = open_picam2(size=(1280, 720))
        def grab():
            return True, read_frame_bgr(picam2)
    else:
        cap = cv2.VideoCapture(0)
        def grab():
            ok, f = cap.read()
            return ok, f

    

    # 첫 프레임
    ok, first = grab()
    if not ok or first is None:
        raise SystemExit("Camera not ready")
    
    
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {SOURCE}")

    idx = next_index(OUTPUT_DIR, PREFIX, EXT, START_IDX)
    frozen = None   # original-size frozen frame for saving
    preview = None  # scaled preview for display
    scale = 1.0

    print("[INFO] Press 'm' to freeze and select multiple ROIs; 'r' to resume; 'q'/'ESC' to quit.")
    
    while True:
        if frozen is None:
            ok, frame = cap.read()
            if not ok or not is_valid_frame(frame):
                cv2.waitKey(10)
                continue
            # show scaled preview
            preview, scale = resize_keep_w(frame, FRAME_MAX_W)
            if preview is None:
                continue
            cv2.putText(preview, "LIVE (press 'm' to select multiple ROIs)", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("template_capturer", preview)
        else:
            disp, _ = resize_keep_w(frozen, FRAME_MAX_W)
            cv2.putText(disp, "FROZEN (press 'm' again to re-select; 'r' to resume)", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
            cv2.imshow("template_capturer", disp)

        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord('q')):  # ESC or q
            break

        if k == ord('m'):
            if frozen is None:
                # Grab a few frames to avoid empties
                valid = None
                for _ in range(50):
                    ok, f = cap.read()
                    if ok and is_valid_frame(f):
                        valid = f
                if valid is None:
                    print("[WARN] Could not get a valid frame to freeze.")
                    continue
                frozen = valid.copy()

            disp, disp_scale = resize_keep_w(frozen, FRAME_MAX_W)
            rois = cv2.selectROIs("template_capturer", disp, showCrosshair=True, fromCenter=False)
            if rois is None or len(rois) == 0:
                print("[INFO] No ROI selected.")
                continue

            saved = 0
            inv_scale = 1.0 / disp_scale if disp_scale > 0 else 1.0
            for (x, y, w, h) in rois:
                x0 = int(round(x * inv_scale))
                y0 = int(round(y * inv_scale))
                w0 = int(round(w * inv_scale))
                h0 = int(round(h * inv_scale))
                out_path = Path(OUTPUT_DIR) / f"{PREFIX}{idx}{EXT}"
                ok = save_crop(frozen, (x0, y0, w0, h0), out_path, EXT)
                if ok:
                    print(f"[SAVED] {out_path}")
                    idx += 1
                    saved += 1
                else:
                    print(f"[FAIL ] Could not save ROI at index {idx}")

            print(f"[INFO] Saved {saved} template(s). Press 'm' to re-select, or 'r' to resume live.")

        if k == ord('r'):
            frozen = None

    try:
        picam2.stop()
    except Exception:
        pass
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()