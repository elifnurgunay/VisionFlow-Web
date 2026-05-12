import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_dirs(cfg):
    for key in ['labels_txt_dir', 'labels_json_dir', 'annotated_dir', 'report_dir']:
        Path(cfg['output'][key]).mkdir(parents=True, exist_ok=True)


def get_images(cfg):
    d = Path(cfg['input']['images_dir'])
    exts = cfg['input']['extensions']
    imgs = [p for p in d.iterdir() if p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def draw_bbox(img, det, class_colors, font_scale=0.55, thickness=2):
    px = det['bbox_pixels']
    x1, y1, x2, y2 = px['x1'], px['y1'], px['x2'], px['y2']
    color = tuple(class_colors.get(det['class_name'], [255, 255, 255]))
    label = f"{det['class_name']} {det['confidence']:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_txt(path, detections):
    lines = []
    for d in detections:
        n = d['bbox_normalized']
        lines.append(f"{d['class_id']} {n['cx']:.6f} {n['cy']:.6f} {n['w']:.6f} {n['h']:.6f}")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def save_json_label(path, img_name, w, h, detections):
    data = {"image": img_name, "width": w, "height": h, "detections": detections}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_report(stats, cfg):
    report_dir = Path(cfg['output']['report_dir'])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(report_dir / f"report_{ts}.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    classes = list(stats['detections'].items())
    if classes:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [c[0] for c in classes]
        counts = [c[1] for c in classes]
        colors_plt = ['#e24b4a', '#1d9e75', '#ef9f27', '#7f77dd']
        bars = ax.bar(names, counts, color=colors_plt[:len(names)], edgecolor='none')
        ax.set_title(f"Sinif Bazli Tespit Dagilimi — {stats['total_images']} gorsel", fontsize=13)
        ax.set_ylabel("Tespit Sayisi")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(cnt), ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.savefig(report_dir / f"report_{ts}.png", dpi=150, bbox_inches='tight')
        plt.close()

    total_det = sum(stats['detections'].values())
    rows = ''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats['detections'].items())
    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>IYHSS Auto-Label Rapor {ts}</title>
<style>
body{{font-family:sans-serif;max-width:860px;margin:2rem auto;color:#222;padding:0 1rem}}
h1{{font-size:20px;margin-bottom:1rem}}
.metrics{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:1.5rem}}
.m{{background:#f5f5f5;border-radius:8px;padding:12px 20px;text-align:center;min-width:100px}}
.m strong{{display:block;font-size:26px;font-weight:600}}
.m span{{font-size:12px;color:#666}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:8px 14px;border:1px solid #e0e0e0;font-size:13px}}
th{{background:#f5f5f5;font-weight:500}}
img{{max-width:100%;margin-top:1.5rem;border-radius:8px}}
</style></head><body>
<h1>IYHSS Auto-Label Rapor <small style="font-size:13px;color:#888">{ts}</small></h1>
<div class="metrics">
  <div class="m"><strong>{stats['total_images']}</strong><span>Gorsel</span></div>
  <div class="m"><strong>{total_det}</strong><span>Toplam Tespit</span></div>
  <div class="m"><strong>{stats.get('fps', 0):.1f}</strong><span>FPS</span></div>
  <div class="m"><strong>{stats.get('avg_ms', 0):.1f}ms</strong><span>Ort. Sure</span></div>
  <div class="m"><strong>{stats['empty_images']}</strong><span>Bos Gorsel</span></div>
</div>
<table><tr><th>Sinif</th><th>Tespit Sayisi</th></tr>{rows}</table>
<img src="report_{ts}.png" alt="Sinif dagilimi grafigi">
</body></html>"""
    with open(report_dir / f"report_{ts}.html", 'w', encoding='utf-8') as f:
        f.write(html)

    return str(report_dir / f"report_{ts}")


def run_pipeline(cfg, log_cb, progress_cb, stop_flag):
    cfg['model']['path'] = str(Path(cfg['model']['path']))
    setup_dirs(cfg)

    log_cb(f"[GPU] CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_cb(f"[GPU] {torch.cuda.get_device_name(0)}")

    model = YOLO(cfg['model']['path'])
    model.to(cfg['device'])

    class_names = cfg['classes']
    class_colors = {k: v for k, v in cfg['class_colors'].items()}

    images = get_images(cfg)
    if not images:
        log_cb(f"[Hata] Gorsel bulunamadi: {cfg['input']['images_dir']}")
        return None

    total = len(images)
    log_cb(f"[Basliyor] {total} gorsel isleniyor...")
    warmup_n = cfg['options'].get('warmup_images', 5)

    stats = {
        'total_images': total,
        'detections': {c: 0 for c in class_names},
        'empty_images': 0,
        'fps': 0.0,
        'avg_ms': 0.0,
    }
    inf_times = []

    for idx, img_path in enumerate(images):
        if stop_flag.is_set():
            log_cb("[Durduruldu] Kullanici tarafindan durduruldu.")
            break

        img = cv2.imread(str(img_path))
        if img is None:
            log_cb(f"  [Uyari] Okunamadi: {img_path.name}")
            progress_cb(idx + 1, total)
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        results = model(
            img,
            imgsz=cfg['model']['img_size'],
            conf=cfg['model']['conf_threshold'],
            iou=cfg['model']['iou_threshold'],
            verbose=False
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if idx >= warmup_n:
            inf_times.append(elapsed_ms)

        detections = []
        ann_img = img.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            det = {
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_normalized": {
                    "cx": round(cx, 6), "cy": round(cy, 6),
                    "w": round(bw, 6), "h": round(bh, 6),
                },
                "bbox_pixels": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
            detections.append(det)
            stats['detections'][cls_name] = stats['detections'].get(cls_name, 0) + 1

            if cfg['options']['save_annotated']:
                draw_bbox(ann_img, det, class_colors)

        stem = img_path.stem
        has_det = len(detections) > 0
        write = has_det or not cfg['options']['skip_empty']

        if cfg['options']['save_txt'] and write:
            save_txt(Path(cfg['output']['labels_txt_dir']) / f"{stem}.txt", detections)

        if cfg['options']['save_json'] and write:
            save_json_label(
                Path(cfg['output']['labels_json_dir']) / f"{stem}.json",
                img_path.name, w, h, detections,
            )

        if cfg['options']['save_annotated']:
            cv2.imwrite(str(Path(cfg['output']['annotated_dir']) / img_path.name), ann_img)

        if not has_det:
            stats['empty_images'] += 1

        if (idx + 1) % 10 == 0 or idx == total - 1:
            log_cb(f"  [{idx + 1}/{total}] {img_path.name} — {len(detections)} tespit, {elapsed_ms:.1f}ms")

        progress_cb(idx + 1, total)

    if inf_times:
        avg = sum(inf_times) / len(inf_times)
        stats['avg_ms'] = round(avg, 2)
        stats['fps'] = round(1000 / avg, 2)

    report_path = generate_report(stats, cfg)
    log_cb(f"[Rapor] {report_path}.*")
    log_cb(f"[Tamamlandi] Gorsel: {stats['total_images']} | Bos: {stats['empty_images']}")
    log_cb(f"  FPS: {stats['fps']} | Ort. sure: {stats['avg_ms']}ms")
    log_cb(f"  Tespit: {stats['detections']}")
    return stats
