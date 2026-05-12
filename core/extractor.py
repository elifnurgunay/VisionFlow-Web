import random
from pathlib import Path

import cv2


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = (total / fps) if fps > 0 else 0
    return {'total_frames': total, 'fps': fps, 'width': w, 'height': h, 'duration': duration}


def get_preview_frames(video_path, n=6):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = sorted(random.sample(range(total), min(n, total)))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def run_extraction(cfg, log_cb, progress_cb, stop_flag):
    cap = cv2.VideoCapture(str(cfg['video_path']))
    if not cap.isOpened():
        log_cb(f"[Hata] Video açılamadı: {cfg['video_path']}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    out   = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)

    mode     = cfg['mode']
    n        = cfg.get('nth', 10)
    target   = cfg.get('target_fps', 2.0)
    thresh   = cfg.get('motion_thresh', 15.0)
    interval = max(1, int(fps / target)) if mode == 'fps' else n

    prev_gray = None
    saved = 0
    frame_idx = 0

    log_cb(f"[Basliyor] mod={mode} | toplam={total} frame | {fps:.1f} FPS")

    while True:
        if stop_flag.is_set():
            log_cb("[Durduruldu] Kullanici tarafindan durduruldu.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        if mode == 'motion':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                extract = True
            else:
                score = cv2.absdiff(gray, prev_gray).mean()
                extract = score >= thresh
            if extract:
                prev_gray = gray
        else:
            extract = (frame_idx % interval == 0)

        if extract:
            fname = out / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if saved % 20 == 1:
                log_cb(f"  frame_{frame_idx:06d}.jpg — {saved}. kayit")

        progress_cb(frame_idx + 1, total)
        frame_idx += 1

    cap.release()
    log_cb(f"[Tamamlandi] {saved} frame kaydedildi → {out}")
    return saved
