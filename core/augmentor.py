from pathlib import Path

import cv2
import albumentations as A


def parse_yolo_txt(txt_path):
    bboxes, class_ids = [], []
    if not Path(txt_path).exists():
        return bboxes, class_ids
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                bboxes.append([cx, cy, w, h])
                class_ids.append(cls_id)
    return bboxes, class_ids


def get_image_label_pairs(input_dir):
    d = Path(input_dir)
    exts = {'.jpg', '.jpeg', '.png'}
    pairs = []
    for img_path in sorted(d.iterdir()):
        if img_path.suffix.lower() in exts:
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                pairs.append((img_path, txt_path))
    return pairs


def build_pipeline(aug_cfg):
    transforms = []

    geo = aug_cfg.get('geometric', {})
    if geo.get('enabled', False):
        hf = geo.get('horizontal_flip', {})
        if hf.get('enabled', False):
            transforms.append(A.HorizontalFlip(p=hf.get('p', 0.5)))
        af = geo.get('affine', {})
        if af.get('enabled', False):
            transforms.append(A.Affine(
                scale=(af.get('scale_min', 0.9), af.get('scale_max', 1.1)),
                rotate=(af.get('rotate_min', -15), af.get('rotate_max', 15)),
                shear=(af.get('shear_min', -5), af.get('shear_max', 5)),
                p=af.get('p', 0.5),
            ))

    pl = aug_cfg.get('photometric_linear', {})
    if pl.get('enabled', False):
        bc = pl.get('brightness_contrast', {})
        if bc.get('enabled', False):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=bc.get('brightness_limit', 0.2),
                contrast_limit=bc.get('contrast_limit', 0.2),
                p=bc.get('p', 0.5),
            ))

    pg = aug_cfg.get('photometric_gamma', {})
    if pg.get('enabled', False):
        rg = pg.get('random_gamma', {})
        if rg.get('enabled', False):
            transforms.append(A.RandomGamma(
                gamma_limit=(rg.get('gamma_min', 80), rg.get('gamma_max', 120)),
                p=rg.get('p', 0.3),
            ))
        cl = pg.get('clahe', {})
        if cl.get('enabled', False):
            tgs = cl.get('tile_grid_size', 8)
            transforms.append(A.CLAHE(
                clip_limit=cl.get('clip_limit', 4.0),
                tile_grid_size=(tgs, tgs),
                p=cl.get('p', 0.3),
            ))

    pc = aug_cfg.get('photometric_color', {})
    if pc.get('enabled', False):
        hsv = pc.get('hue_saturation', {})
        if hsv.get('enabled', False):
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=hsv.get('hue_shift', 20),
                sat_shift_limit=hsv.get('sat_shift', 30),
                val_shift_limit=hsv.get('val_shift', 20),
                p=hsv.get('p', 0.3),
            ))
        cj = pc.get('color_jitter', {})
        if cj.get('enabled', False):
            transforms.append(A.ColorJitter(
                brightness=cj.get('brightness', 0.2),
                contrast=cj.get('contrast', 0.2),
                saturation=cj.get('saturation', 0.2),
                hue=cj.get('hue', 0.1),
                p=cj.get('p', 0.3),
            ))

    bl = aug_cfg.get('blur', {})
    if bl.get('enabled', False):
        mb = bl.get('motion_blur', {})
        if mb.get('enabled', False):
            transforms.append(A.MotionBlur(blur_limit=mb.get('blur_limit', 7), p=mb.get('p', 0.2)))
        gb = bl.get('gaussian_blur', {})
        if gb.get('enabled', False):
            transforms.append(A.GaussianBlur(blur_limit=gb.get('blur_limit', 7), p=gb.get('p', 0.2)))
        df = bl.get('defocus', {})
        if df.get('enabled', False):
            transforms.append(A.Defocus(
                radius=(df.get('radius_min', 3), df.get('radius_max', 10)),
                alias_blur=df.get('alias_blur', 0.1),
                p=df.get('p', 0.2),
            ))

    ns = aug_cfg.get('noise', {})
    if ns.get('enabled', False):
        gn = ns.get('gauss_noise', {})
        if gn.get('enabled', False):
            try:
                transforms.append(A.GaussNoise(
                    var_limit=(gn.get('var_min', 10.0), gn.get('var_max', 50.0)),
                    p=gn.get('p', 0.2),
                ))
            except TypeError:
                import math
                std_min = math.sqrt(gn.get('var_min', 10.0))
                std_max = math.sqrt(gn.get('var_max', 50.0))
                transforms.append(A.GaussNoise(std_range=(std_min, std_max), p=gn.get('p', 0.2)))
        ic = ns.get('image_compression', {})
        if ic.get('enabled', False):
            transforms.append(A.ImageCompression(
                quality_lower=ic.get('quality_lower', 70),
                quality_upper=ic.get('quality_upper', 100),
                p=ic.get('p', 0.2),
            ))

    if not transforms:
        return None

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3),
    )


def _save_augmented(result, img_path, output_dir, aug_idx):
    out_images = Path(output_dir) / 'images'
    out_labels = Path(output_dir) / 'labels'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    out_name = f"{img_path.stem}_aug{aug_idx:04d}"
    cv2.imwrite(str(out_images / f"{out_name}{img_path.suffix}"), result['image'])

    with open(out_labels / f"{out_name}.txt", 'w') as f:
        for cls_id, bbox in zip(result.get('class_labels', []), result.get('bboxes', [])):
            cx, cy, w, h = bbox
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def run_augmentation(cfg, log_cb, progress_cb, stop_flag):
    pipeline = build_pipeline(cfg)
    if pipeline is None:
        log_cb("[Hata] Hicbir augmentation secilmedi.")
        return

    pairs = get_image_label_pairs(cfg['input_dir'])
    if not pairs:
        log_cb(f"[Hata] Gorsel-etiket cifti bulunamadi: {cfg['input_dir']}")
        return

    multiplier = cfg.get('multiplier', 5)
    total = len(pairs) * multiplier
    done = 0

    log_cb(f"[Basliyor] {len(pairs)} gorsel x {multiplier} = {total} augmented kopya")

    for img_path, txt_path in pairs:
        if stop_flag.is_set():
            log_cb("[Durduruldu] Kullanici tarafindan durduruldu.")
            break

        img = cv2.imread(str(img_path))
        if img is None:
            log_cb(f"  [Uyari] Okunamadi: {img_path.name}")
            done += multiplier
            progress_cb(done, total)
            continue

        bboxes, class_ids = parse_yolo_txt(txt_path)

        for i in range(multiplier):
            if stop_flag.is_set():
                break
            try:
                result = pipeline(image=img, bboxes=bboxes, class_labels=class_ids)
                _save_augmented(result, img_path, cfg['output_dir'], done)
            except Exception as e:
                log_cb(f"  [Uyari] {img_path.name} aug {i}: {e}")
            done += 1
            progress_cb(done, total)

        log_cb(f"  {img_path.name} — {multiplier} kopya")

    log_cb(f"[Tamamlandi] {done}/{total} augmented gorsel uretildi")
    log_cb(f"  Cikti: {cfg['output_dir']}")
