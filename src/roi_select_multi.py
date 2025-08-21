#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다중 ROI 템플릿 캡처 도구 (Raspberry Pi 5 + PiCamera2 + VNC 친화 설정)
- 프레임을 정지시킨 뒤, 여러 개 ROI(관심 영역)를 한 번에 선택하고
  자동으로 template_img1.jpg, template_img2.jpg ... 형식으로 저장.

키 조작
  m : 현재 프레임을 정지시키고 다중 ROI 선택 창 열기
  r : 정지된 화면에서 다시 실시간 프리뷰로 돌아가기
  q/ESC : 프로그램 종료
"""

# === VNC/Wayland 환경에서 OpenCV/Qt 창이 잘 뜨도록 환경 변수 먼저 설정 ===
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # OpenCV가 GStreamer로 카메라 열지 않게
# VNC에서는 보통 X11이므로 xcb가 안전. (Wayland 직접 세션이면 xcb도 동작)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple

# ====== 설정 ======
OUTPUT_DIR = "../assets"        # 잘라낸 이미지를 저장할 폴더
PREFIX = "template_img"      # 저장될 파일 이름 앞부분
START_IDX = None             # None이면 기존 파일을 보고 자동으로 다음 번호부터 저장
EXT = ".jpg"                 # ".jpg" 또는 ".png"
JPEG_QUALITY = 95            # JPG 저장 시 화질
FRAME_MAX_W = 1280           # 미리보기 창 가로폭 (리소스 절약용, 저장은 원본 크기에서 자름)
PICAM_SIZE = (1280, 960)     # PiCam 캡처 해상도 (필요시 조절)

# ====== Picamera2 사용 ======
from picamera2 import Picamera2

def open_picam2(size=(1280, 960)):
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": size, "format": "BGR888"})
    picam2.configure(cfg)
    picam2.start()
    # 노출/화이트밸런스 고정이 필요하면 아래 주석 해제해서 값 조절
    # picam2.set_controls({"AeEnable": False, "AwbEnable": False, "ExposureTime": 10000, "AnalogueGain": 1.5})
    return picam2

def read_frame_bgr(picam2):
    rgb = picam2.capture_array()   # RGB888 ndarray
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# ====== 유틸 ======
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
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
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

def main():
    # 창을 먼저 만들어 두면 VNC에서 안정적
    cv2.namedWindow("template_capturer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("template_capturer", 960, 540)

    picam2 = open_picam2(size=PICAM_SIZE)

    # 첫 프레임 확인
    first = read_frame_bgr(picam2)
    if not is_valid_frame(first):
        raise SystemExit("Camera not ready")

    idx = next_index(OUTPUT_DIR, PREFIX, EXT, START_IDX)
    frozen = None    # 저장용 정지 프레임(원본 크기)
    preview = None   # 화면 표시용 축소 프레임
    print("[INFO] Press 'm' to freeze and select multiple ROIs; 'r' to resume; 'q'/'ESC' to quit.")

    while True:
        if frozen is None:
            # 라이브 프리뷰
            frame = read_frame_bgr(picam2)
            if not is_valid_frame(frame):
                cv2.waitKey(10)
                continue
            preview, scale = resize_keep_w(frame, FRAME_MAX_W)
            if preview is None:
                continue
            cv2.putText(preview, "LIVE (press 'm' to select multiple ROIs)", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("template_capturer", preview)
        else:
            # 정지 화면
            disp, _ = resize_keep_w(frozen, FRAME_MAX_W)
            cv2.putText(disp, "FROZEN (press 'm' again to re-select; 'r' to resume)", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
            cv2.imshow("template_capturer", disp)

        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord('q')):  # ESC 또는 q
            break

        if k == ord('m'):
            # 정지 프레임 캡처
            if frozen is None:
                # 연속으로 몇 장 읽어 안전 프레임 확보
                valid = None
                for _ in range(10):
                    f = read_frame_bgr(picam2)
                    if is_valid_frame(f):
                        valid = f
                if valid is None:
                    print("[WARN] Could not get a valid frame to freeze.")
                    continue
                frozen = valid.copy()

            # 정지 화면에서 다중 ROI 선택
            disp, disp_scale = resize_keep_w(frozen, FRAME_MAX_W)
            rois = cv2.selectROIs("template_capturer", disp, showCrosshair=True, fromCenter=False)
            if rois is None or len(rois) == 0:
                print("[INFO] No ROI selected.")
                continue

            # 선택된 ROI들을 원본 좌표로 환산하여 저장
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

    # 종료 처리
    try:
        picam2.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()