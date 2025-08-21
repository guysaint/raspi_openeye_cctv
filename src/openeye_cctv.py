
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
멀티 객체 로컬-템플릿 모니터
	- 단일 카메라 스트림에서 여러 템플릿(USB, 그립톡, 기타 등)을 동시에 매칭합니다.
	- 위치나 외형이 허용 범위를 벗어나면, 히스테리시스와 유예(grace) 조건을 적용해 **“패턴 이탈”**로 표시합니다.
	- 확장하기 쉽게 설계되었습니다: 아래 OBJECTS만 수정하거나, 이미지를 AUTOLOAD_DIR에 넣으면 됩니다.

조작키:
	- ESC : 종료
"""

import cv2
import numpy as np
import time
import glob
import os
from typing import Tuple, Dict, Any, List

# Picamera2 helpers (for Raspberry Pi 5)
from picamera2 import Picamera2

def open_picam2(size=(1280, 720)):
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": size, "format": "BGR888"})
    picam2.configure(cfg)
    picam2.start()
    return picam2

def read_frame_bgr(picam2):
    frame = picam2.capture_array()  # RGB
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# =========================
# 사용자 설정
# =========================
SOURCE = Picamera2  # 내부 카메라(0), 외부 카메라(1), 카메라 경로

# 옵션 1) 추적할 객체를 직접 지정 (명확성을 위해 권장)
OBJECTS: List[Dict[str, Any]] = [
    {"name": "valve", "template_path": "../assets/template_img1.jpg", "color": (0, 255, 0)},
    {"name": "led", "template_path": "../assets/template_img2.jpg", "color": (255, 128, 0)},
]

# 옵션 2) 폴더 내 모든 템플릿 자동 불러오기 (png/jpg). 사용하지 않으려면 "" 로 설정.
# 파일은 확장자를 제외한 파일 이름으로 추적됨.
AUTOLOAD_DIR = ""  # e.g., "assets/templates_autoload"

# 영상 프레임 & 전처리 단계
FRAME_MAX_W = 0
STAB_SCALE = 0.0  # 0 = 꺼짐; 카메라가 많이 흔들리면 0.5~0.7 시도.
BLUR_K = 3

# 템플릿을 회전시켜 비교할 각도를 도(degree) 단위로 지정
# 예: [0, 90, 180, 270] → 0도, 90도, 180도, 270도 회전된 템플릿과 비교
# 만약 템플릿 회전이 필요 없는 경우에는 [0] 으로 설정.
ROT_DEGS = [0, -10, +10, -20, +20]

# Matching
METHOD = cv2.TM_CCOEFF_NORMED
MIN_TRUST_SCORE = 0.35  # 새로운 매칭 결과를 받아들이기 위한 최소 템플릿 신뢰도(confidence) 기준값.
                        # 예: 0.8 → 매칭 신뢰도가 0.8 이상일 때만 새로운 매칭으로 인정.

# 윈도우(Window) 또는 패치(Patch) 단위 설정
ALLOWED_BOX_SCALE = 0.2   # HOME 주변에서 허용되는 위치 편차 박스 크기 (tw0/th0 × scale)
SEARCH_WIN_SCALE = 2.2    # 로컬 검색 윈도우 크기 배율
PATCH_SCALE = 1.2         # 템플릿 가로/세로 대비 외형 패치 크기 배율

# 모션 게이트(옵션): HOME/허용 박스 내 큰 움직임 시 위치 편차 무시
MOTION_GATE = True
MORPH_K = 5
MOTION_AREA_THRESH = 0.05  # 허용 박스 내 전경 픽셀 비율이 임계값 이상이면 '가려짐(occluded)'으로 간주

# 히스테리시스 및 타이밍 설정
EMA_ALPHA = 0.35           # 중심 좌표 보정을 위한 지수 이동 평균 가중치
DEVIATE_FRAMES_REQ = 3     # strong_deviate 발생 조건: 연속 편차 프레임 수
APPEAR_FRAMES_REQ = 3
GRACE_SECS = 10.0           # 알림 발생 전 복귀 허용 시간

ANALYZE_EVERY = 2          # CPU 절약을 위해 N번째 프레임마다 처리

WINDOW_NAME = "multi_template_monitor"


# =========================
# Utilities
# =========================
def resize_keep_w(img, max_w: int) -> Tuple[np.ndarray, float]:
    if img is None:
        return None, 1.0
    if getattr(img, 'size', 0) == 0:
        return None, 1.0
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return None, 1.0
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w) if w else 1.0
    if not np.isfinite(scale) or scale <= 0:
        return img, 1.0
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out, scale


def preprocess_for_matching(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    if BLUR_K and BLUR_K >= 3:
        gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    return gray


def rotate_img(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def best_match_global(search_img_gray: np.ndarray,
                      tgray_rots: List[Tuple[float, np.ndarray]]) -> Dict[str, Any]:
    best = {"score": -1.0, "top_left": (0, 0), "size": (None, None), "deg": 0}
    H, W = search_img_gray.shape[:2]
    for deg, tgray in tgray_rots:
        th, tw = tgray.shape[:2]
        if tw is None or th is None or tw < 8 or th < 8 or tw > W or th > H:
            continue
        res = cv2.matchTemplate(search_img_gray, tgray, METHOD)
        minv, maxv, minl, maxl = cv2.minMaxLoc(res)
        score = maxv if METHOD in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else (1.0 - minv)
        if score > best["score"]:
            best.update({"score": score, "top_left": maxl, "size": (tw, th), "deg": deg})
    return best


def center_from(top_left: Tuple[int, int], size: Tuple[int, int]) -> Tuple[int, int]:
    x, y = top_left
    tw, th = size
    return (int(x + tw / 2), int(y + th / 2))


def clamp_box(cx: int, cy: int, hw: int, hh: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, cx - hw); y1 = max(0, cy - hh)
    x2 = min(W, cx + hw); y2 = min(H, cy + hh)
    return x1, y1, x2, y2


def crop_patch(gray: np.ndarray, center: Tuple[int, int], size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    cx, cy = center
    w, h = size
    W, H = gray.shape[1], gray.shape[0]
    x1, y1, x2, y2 = clamp_box(cx, cy, w // 2, h // 2, W, H)
    patch = gray[y1:y2, x1:x2]
    return patch, (x1, y1, x2, y2)


def stabilize_ecc(prev_small: np.ndarray, curr_small: np.ndarray) -> np.ndarray:
    # ECC 알고리즘을 사용하여 강체 변환(회전 + 평행이동)을 근사적으로 계산.
    # 즉, 템플릿과 영상이 회전되거나 위치가 이동한 경우에도 일치하도록 맞춤.
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        cc, warp_matrix = cv2.findTransformECC(prev_small, curr_small, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error:
        pass
    return warp_matrix




def is_valid_frame(f):
    return f is not None and getattr(f, 'size', 0) > 0 and f.shape[1] > 0 and f.shape[0] > 0
# =========================
# # 객체 로더(Object loader): 템플릿 파일이나 폴더에서 객체 이미지를 불러옮.
# 추적할 객체를 자동 또는 수동으로 등록하는 데 사용.
# =========================
def load_objects() -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []

    # A) 명시적 목록: 추적할 객체를 코드에서 직접 지정하는 방식.
    # 예) OBJECTS = ["usb.png", "griptok.png"]
    for cfg in OBJECTS:
        path = cfg["template_path"]
        timg = cv2.imread(path)
        if timg is None:
            print(f"[WARN] Cannot read template: {path}")
            continue
        timg, _ = resize_keep_w(timg, 300)
        tgray_rots = [(deg, preprocess_for_matching(rotate_img(timg, deg))) for deg in ROT_DEGS]
        objs.append({
            "name": cfg["name"],
            "color": cfg.get("color", (0, 255, 0)),
            "template_img": timg,
            "templates_gray": tgray_rots,
            "home": None,
            "home_patch": None,
            "tw0": None,
            "th0": None,
            "ema_center": None,
            "deviate_since": None,
            "deviate_run": 0,
            "appear_run": 0
        })

    # B) 자동 불러오기 폴더(옵션): 지정된 폴더에서 템플릿을 자동으로 불러옮.
    # 파일 이름이 같은 경우(중복)는 건너뜀.
    if AUTOLOAD_DIR and os.path.isdir(AUTOLOAD_DIR):
        for fp in sorted(glob.glob(os.path.join(AUTOLOAD_DIR, "*.*"))):
            if not fp.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                continue
            name = os.path.splitext(os.path.basename(fp))[0]
            if any(o["name"] == name for o in objs):
                continue
            timg = cv2.imread(fp)
            if timg is None:
                print(f"[WARN] Cannot read template: {fp}")
                continue
            timg, _ = resize_keep_w(timg, 300)
            tgray_rots = [(deg, preprocess_for_matching(rotate_img(timg, deg))) for deg in ROT_DEGS]
            objs.append({
                "name": name,
                "color": (128, 255, 0),
                "template_img": timg,
                "templates_gray": tgray_rots,
                "home": None,
                "home_patch": None,
                "tw0": None,
                "th0": None,
                "ema_center": None,
                "deviate_since": None,
                "deviate_run": 0,
                "appear_run": 0
            })

    return objs


# =========================
# Main
# =========================
def main():
    USE_PICAM2 = True

    if USE_PICAM2:
        picam2 = open_picam2(size=(1280, 960))
        def grab():
            return True, read_frame_bgr(picam2)
    else:
        cap = cv2.VideoCapture(SOURCE)
        def grab():
            ok, f = cap.read()
            return ok, f

    objects = load_objects()
    if not objects:
        raise SystemExit("No templates loaded. Check OBJECTS or AUTOLOAD_DIR.")

    # 첫 프레임
    ok, first = grab()
    if not ok or first is None:
        raise SystemExit("Camera not ready")

    # 첫 번째 유효 프레임: macOS/Continuity Camera의 경우 초기에는 빈 프레임이 나올 수 있으므로
    # 실제 영상 데이터가 들어오기 시작한 시점을 기준으로 처리합니다.
    first = None
    for _ in range(100):
        ok, f = grab()
        if not ok or not is_valid_frame(f):
            time.sleep(0.02)
            continue
        first, _ = resize_keep_w(f, FRAME_MAX_W)
        if first is not None:
            break
    if first is None:
        raise SystemExit("Camera opened but returned empty/invalid frames. Try a different SOURCE index or ensure permissions.")
    first_fg = preprocess_for_matching(first)

    # 객체별 기준 위치(Home) 초기화:
    # 각 객체마다 추적의 기준점(Home 위치)을 설정하여 이후 위치 편차를 계산할 수 있도록 함.
    for obj in objects:
        bm = best_match_global(first_fg, obj["templates_gray"])
        top_left, score, (tw, th) = bm["top_left"], bm["score"], bm["size"]
        if tw is None or th is None or tw < 8 or th < 8 or score < 0.1:
            raise SystemExit(f"[{obj['name']}] template not confidently found in first frame. Score={score:.2f}")
        home = center_from(top_left, (tw, th))
        obj["home"] = home
        obj["ema_center"] = home
        obj["tw0"], obj["th0"] = tw, th

        # 외형/홈 패치(appearance/home patch):
        # 객체의 외형(appearance)과 기준 위치(Home) 주변을 잘라낸 작은 영역(patch).
        # 이 패치는 객체의 변화나 위치 이탈을 감지하는 데 활용.
        home_patch, _ = crop_patch(first_fg, home, (int(PATCH_SCALE * tw), int(PATCH_SCALE * th)))
        if home_patch.size == 0:
            raise SystemExit(f"[{obj['name']}] home_patch crop failed — reduce PATCH_SCALE.")
        obj["home_patch"] = home_patch

    # 배경 제거기 & 형태학적 처리(공용):
    # 영상에서 움직이는 객체를 분리하기 위해 배경을 제거(Background Subtraction)하고,
    # 노이즈 제거 및 영역 보정을 위해 형태학적 연산(Morphology)을 적용.
    # 이 설정은 모든 객체 추적에 공통으로 사용.
    bg = None
    kernel = None
    if MOTION_GATE:
        bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))

    prev_small = None
    fcount = 0

    while True:
        ok, frame = grab()
        if not ok or not is_valid_frame(frame):
            time.sleep(0.01)
            continue
        fcount += 1
        if fcount % max(1, ANALYZE_EVERY) != 0:
            continue

        frame, _ = resize_keep_w(frame, FRAME_MAX_W)
        if frame is None:
            continue
        fg = preprocess_for_matching(frame)

        # 선택적 ECC 기반 프레임 안정화
        if STAB_SCALE and STAB_SCALE > 0:
            small = cv2.resize(fg, None, fx=STAB_SCALE, fy=STAB_SCALE, interpolation=cv2.INTER_AREA)
            if prev_small is not None:
                warp = stabilize_ecc(prev_small, small)
                fg = cv2.warpAffine(fg, warp, (fg.shape[1], fg.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                frame = cv2.warpAffine(frame, warp, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            prev_small = small

        # 모션 게이트 마스크(Motion gate mask, 공용):
        # 영상에서 객체가 움직일 수 있는 영역을 제한하는 마스크.
        # 이 마스크를 벗어난 움직임은 무시되거나 편차로 간주되며,
        # 모든 객체 추적에 공통으로 적용.
        m = None
        if MOTION_GATE and bg is not None:
            m = bg.apply(frame)
            _, m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

        vis = frame.copy()
        global_alert = False

        H, W = fg.shape[:2]

        for idx, obj in enumerate(objects):
            name = obj["name"]
            color = obj["color"]
            home = obj["home"]
            ema_center = obj["ema_center"]
            tw0, th0 = obj["tw0"], obj["th0"]
            home_patch = obj["home_patch"]

            # EMA(지수 이동 평균, Exponential Moving Average) 위치를 중심으로 한 로컬 검색 윈도우:
            # 객체의 예상 중심 좌표(EMA)를 기준으로 주변 영역을 탐색 창(window)으로 설정하여,
            # 불필요한 전체 탐색을 줄이고 효율적으로 객체를 추적.
            allowed_half_w = int(ALLOWED_BOX_SCALE * tw0)
            allowed_half_h = int(ALLOWED_BOX_SCALE * th0)
            search_half_w = int(SEARCH_WIN_SCALE * tw0)
            search_half_h = int(SEARCH_WIN_SCALE * th0)

            sx1, sy1, sx2, sy2 = clamp_box(ema_center[0], ema_center[1], search_half_w, search_half_h, W, H)
            local = fg[sy1:sy2, sx1:sx2]

            bm = best_match_global(local, obj["templates_gray"])
            (lx, ly) = bm["top_left"]
            top_left = (sx1 + lx, sy1 + ly)
            score = bm["score"]
            tw, th = bm["size"]

            if score < MIN_TRUST_SCORE or tw is None or th is None:
                match_center = ema_center
            else:
                match_center = center_from(top_left, (tw, th))

            # EMA 업데이트(Exponential Moving Average update):
            # 객체 중심 좌표를 부드럽게 추적하기 위해, 이전 값과 새 값을 지수 이동 평균 방식으로 갱신.
            ema_center = (
                int(EMA_ALPHA * match_center[0] + (1 - EMA_ALPHA) * ema_center[0]),
                int(EMA_ALPHA * match_center[1] + (1 - EMA_ALPHA) * ema_center[1])
            )
            obj["ema_center"] = ema_center

            # 가려짐(occlusion) 게이트: HOME 주변 허용 박스 내부의 모션 비율로 판단.
            # 허용 박스 안에서 전경(움직임) 픽셀이 차지하는 비율이 임계값을 넘으면
            # 해당 프레임을 '가려짐' 상태로 간주하고 위치 편차 판단을 일시적으로 보류.
            # 예) motion_ratio > 0.4 → occluded = True
            occluded = False
            x1a, y1a, x2a, y2a = clamp_box(home[0], home[1], allowed_half_w, allowed_half_h, W, H)
            if MOTION_GATE and m is not None:
                m_crop = m[y1a:y2a, x1a:x2a]
                move_ratio = float(m_crop.sum()) / (255.0 * max(1, m_crop.size))
                occluded = (move_ratio >= MOTION_AREA_THRESH)

            # 위치 편차(Position deviation): 
            # 객체가 가려지지(occluded) 않았을 때만 계산.
            # 현재 객체 중심과 기준 위치(Home) 사이의 거리를 비교하여, 허용 범위를 벗어나면 편차로 판정.
            if occluded:
                pos_deviated = False
            else:
                pos_deviated = not (x1a <= ema_center[0] <= x2a and y1a <= ema_center[1] <= y2a)

            # 외형 편차(Appearance deviation):
            # 객체의 현재 외형(appearance patch)과 기준 외형(Home patch)을 비교하여 유사도가 일정 기준 이하로 떨어지면 '외형 편차'로 판정.
            # 즉, 모양·패턴 변화까지 감지하여 단순 위치 이탈 외에도 이상 여부를 판단.
            curr_patch, curr_box = crop_patch(fg, ema_center, (int(PATCH_SCALE * tw0), int(PATCH_SCALE * th0)))
            app_score = 1.0
            app_deviated = False
            if curr_patch.size and home_patch.size and (curr_patch.shape[0] >= 8 and curr_patch.shape[1] >= 8):
                res = cv2.matchTemplate(curr_patch, home_patch, METHOD)
                minv, maxv, _, _ = cv2.minMaxLoc(res)
                app_score = maxv if METHOD in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED) else (1.0 - minv)
                app_deviated = (app_score < 0.55)  # default threshold for appearance difference

            # 히스테리시스 카운터(Hysteresis counters):
            # 객체 상태(정상 ↔ 편차)를 판정할 때 즉시 바뀌지 않도록, 연속된 프레임 수를 카운트하여 안정성을 유지.
            # 예: 편차가 몇 프레임 이상 지속될 때만 '편차 발생'으로 확정.
            deviated_now = pos_deviated or app_deviated
            obj["deviate_run"] = obj["deviate_run"] + 1 if deviated_now else 0
            obj["appear_run"] = obj["appear_run"] + 1 if app_deviated else 0

            strong_deviate = (obj["deviate_run"] >= DEVIATE_FRAMES_REQ) or (obj["appear_run"] >= APPEAR_FRAMES_REQ)

            alert = False
            now = time.time()
            if strong_deviate:
                if obj["deviate_since"] is None:
                    obj["deviate_since"] = now
            else:
                obj["deviate_since"] = None

            remain = None
            if obj["deviate_since"] is not None:
                elapsed = now - obj["deviate_since"]
                remain = max(0, int(GRACE_SECS - elapsed))
                if elapsed >= GRACE_SECS:
                    alert = True
                    global_alert = True

            # 시각화(객체별, Visuals per object):
            # 각 객체에 대해 추적 상태를 화면에 표시하는 기능.
            # 예) 객체의 위치 박스, 중심 좌표, 편차 여부, 가려짐 상태 등을 색상이나 도형으로 표시.
            # 이렇게 하면 객체별 상태를 직관적으로 모니터링할 수 있음.
            if tw and th and tw > 0 and th > 0:
                cv2.rectangle(vis, top_left, (top_left[0] + tw, top_left[1] + th), color, 2)
            cv2.rectangle(vis, (x1a, y1a), (x2a, y2a), (0, 255, 255), 2)
            cv2.circle(vis, home, 5, (255, 255, 0), -1)
            cv2.circle(vis, ema_center, 5, (0, 0, 255) if deviated_now else (0, 255, 0), -1)
            cv2.line(vis, home, ema_center, (0, 0, 255) if deviated_now else (0, 255, 0), 2)

            ybase = 30 + idx * 50
            cv2.putText(vis, f"[{name}] score={score:.2f} app={app_score:.2f} occ={int(occluded)}",
                        (20, ybase), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if obj["deviate_since"] is not None and not alert and remain is not None:
                cv2.putText(vis, f"[{name}] Return within {remain}s",
                            (20, ybase + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if alert:
                cv2.putText(vis, f"[{name}] ALERT: Pattern deviated!", (20, ybase + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #if global_alert:
        #    cv2.putText(vis, "GLOBAL ALERT: Some pattern(s) deviated!",
        #                (20, 15 + len(objects)*50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imshow(WINDOW_NAME, vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

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
