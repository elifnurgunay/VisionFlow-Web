import os
import shutil
import tempfile
import base64
import cv2
import threading
from flask import Flask, render_template, request, send_file, jsonify
from core.augmentor import run_augmentation

# Orijinal core fonksiyonlarını içe aktarıyoruz
from core.extractor import run_extraction, get_video_info, get_preview_frames

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Klasörleri oluştur
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_video_data', methods=['POST'])
def get_video_data():
    video = request.files['video']
    
    # tempfile YERİNE doğrudan uploads klasörüne kaydediyoruz
    safe_filename = "temp_preview_" + video.filename
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    video.save(video_path)
    
    # Core modülleriyle bilgileri ve önizlemeyi al
    info = get_video_info(video_path)
    preview_frames = get_preview_frames(video_path, n=6)
    
    # Önizleme karelerini base64'e çevir
    encoded_frames = []
    for frame in preview_frames:
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frames.append(base64.b64encode(buffer).decode('utf-8'))
        
    # Windows'un dosyayı bırakması için çok kısa bir an bekleyip siliyoruz
    try:
        os.remove(video_path) 
    except Exception as e:
        print(f"Uyarı: Dosya hemen silinemedi, bir sonraki sefer temizlenecek: {e}")
    
    return jsonify({
        "status": "success",
        "info": info,
        "previews": encoded_frames
    })

@app.route('/extract', methods=['POST'])
def extract():
    video = request.files['video']
    mode = request.form.get('mode', 'nth')
    
    # Geçici dosyaya kaydet
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'frames')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # extractor.py'nin beklediği config'i oluştur
    cfg = {
        'video_path': video_path,
        'output_dir': output_dir,
        'mode': mode,
    }
    
    # Seçilen moda göre parametreleri ekle
    if mode == 'nth':
        cfg['nth'] = int(request.form.get('interval', 10))
    elif mode == 'fps':
        cfg['target_fps'] = float(request.form.get('target_fps', 2.0))
    elif mode == 'motion':
        cfg['motion_thresh'] = float(request.form.get('motion_thresh', 15.0))
    
    # PyQt5 arayüzünün beklediği log_cb ve progress_cb fonksiyonlarını taklit ediyoruz
    def log_cb(msg):
        print(msg) # VS Code terminalinden canlı canlı izleyebileceğiz
        
    def progress_cb(current, total):
        pass # Web tarafında anlık progress bar şimdilik beklemede kalacak
        
    stop_flag = threading.Event()
    
    # Senin mükemmel çalışan orijinal fonksiyonunu çağırıyoruz!
    count = run_extraction(cfg, log_cb, progress_cb, stop_flag)
    
    # ZIP Hazırla (Senin imza özelliğin)
    zip_base = os.path.join(app.config['OUTPUT_FOLDER'], 'dataset_frames')
    shutil.make_archive(zip_base, 'zip', output_dir)
    
    return jsonify({
        "status": "success", 
        "count": count, 
        "download_url": "/download/dataset_frames.zip"
    })

# labeler'dan doğru fonksiyonu alıyoruz
from core.labeler import run_pipeline 

@app.route('/autolabel', methods=['POST'])
def autolabel():
    # Model yolu ve Cihaz tespiti
    model_path = os.path.join(os.getcwd(), 'model', 'best.pt')
    if not os.path.exists(model_path):
        return jsonify({"status": "error", "message": "Model bulunamadı! Lütfen best.pt dosyasını 'model' klasörüne ekleyin."})

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Arayüzden gelen değerleri alıyoruz
    conf = float(request.form.get('conf_thresh', 0.5))
    iou = float(request.form.get('iou_thresh', 0.45))
    img_size = int(request.form.get('img_size', 640))
    
    # 1. Aşama (Extractor'dan) elde edilen resimler nerede?
    input_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'frames')
    if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
         return jsonify({"status": "error", "message": "Önce 'Frame Extract' sekmesinden video ayıklamanız gerekiyor (veya frames klasörü boş)."})

    # Labeler'ın çıktı vereceği ana klasör
    base_out = os.path.join(app.config['OUTPUT_FOLDER'], 'label_results')
    
    # Senin labeler.py içindeki run_pipeline fonksiyonunun BİREBİR beklediği dev sözlük (cfg)
    cfg = {
        'model': {
            'path': model_path,
            'img_size': img_size,
            'conf_threshold': conf,
            'iou_threshold': iou
        },
        'device': device,
        'input': {
            'images_dir': input_dir,
            'extensions': ['.jpg', '.jpeg', '.png']
        },
        'output': {
            # Tüm çıktıları zipleyeceğimiz için aynı ana klasörün içine koyuyoruz
            'labels_txt_dir': os.path.join(base_out, 'labels_txt'),
            'labels_json_dir': os.path.join(base_out, 'labels_json'),
            'annotated_dir': os.path.join(base_out, 'images_annotated'),
            'report_dir': os.path.join(base_out, 'reports')
        },
        'options': {
            'warmup_images': 1,
            'save_annotated': True,  # Kutulu resimleri kaydetsin
            'skip_empty': False,     # Boş olanları atlasın mı?
            'save_txt': True,        # TXT YOLO formatı
            'save_json': False       # JSON formata şimdilik gerek yok
        },
        # Eğer YOLO modelin içinde kendi sınıf isimleri yoksa varsayılanlar
        'classes': ["nesne1", "nesne2", "nesne3"], 
        'class_colors': {
             "nesne1": [255, 0, 0],
             "nesne2": [0, 255, 0],
             "nesne3": [0, 0, 255]
        }
    }
    
    def log_cb(msg):
        print(f"[YOLO]: {msg}")
        
    def progress_cb(current, total):
        pass 
        
    stop_flag = threading.Event()
    
    try:
        # Senin harika fonksiyonunu çağırıyoruz!
        stats = run_pipeline(cfg, log_cb, progress_cb, stop_flag)
        
        if stats is None:
             return jsonify({"status": "error", "message": "Resimler işlenemedi."})
             
        # Tüm sonuçları (TXT'ler, kutulu resimler ve senin HTML raporun) tek bir ZIP yap
        zip_base = os.path.join(app.config['OUTPUT_FOLDER'], 'dataset_labeled_results')
        shutil.make_archive(zip_base, 'zip', base_out)
        
        total_işlenen = stats.get('total_images', 0)
        
        return jsonify({
            "status": "success", 
            "count": total_işlenen, 
            "download_url": "/download/dataset_labeled_results.zip"
        })
    except Exception as e:
         import traceback
         print(traceback.format_exc()) # Hatayı terminalde detaylı görmek için
         return jsonify({"status": "error", "message": str(e)})
@app.route('/augment', methods=['POST'])
def augment():
    multiplier = int(request.form.get('multiplier', 5))
    
    # --- FORM VERİLERİNİ ÇEKME VE cfg OLUŞTURMA ---
    
    # 1. Geometrik
    geo_hf_enabled = request.form.get('geo_hf') == 'on'
    geo_affine_enabled = request.form.get('geo_affine') == 'on'
    
    # 2. Fotometrik
    photo_bc_enabled = request.form.get('photo_bc') == 'on'
    photo_clahe_enabled = request.form.get('photo_clahe') == 'on'
    
    # 3. Bulanıklık ve Gürültü
    blur_gb_enabled = request.form.get('blur_gauss') == 'on'
    noise_gn_enabled = request.form.get('noise_gauss') == 'on'

    cfg = {
        'input_dir': '', # Aşağıda doldurulacak
        'output_dir': '', # Aşağıda doldurulacak
        'multiplier': multiplier,
        
        'geometric': {
            'enabled': geo_hf_enabled or geo_affine_enabled,
            'horizontal_flip': {
                'enabled': geo_hf_enabled, 
                'p': float(request.form.get('hf_p', 0.5))
            },
            'affine': {
                'enabled': geo_affine_enabled,
                'rotate_min': int(request.form.get('af_rot_min', -15)),
                'rotate_max': int(request.form.get('af_rot_max', 15)),
                'scale_min': 0.9, 'scale_max': 1.1, # Karmaşayı önlemek için bazılarını sabitledik
                'shear_min': -5, 'shear_max': 5,
                'p': float(request.form.get('af_p', 0.5))
            }
        },
        'photometric_linear': {
            'enabled': photo_bc_enabled,
            'brightness_contrast': {
                'enabled': photo_bc_enabled,
                'brightness_limit': float(request.form.get('bc_limit', 0.2)),
                'contrast_limit': float(request.form.get('bc_limit', 0.2)),
                'p': float(request.form.get('bc_p', 0.45))
            }
        },
        'photometric_gamma': {
            'enabled': photo_clahe_enabled,
            'clahe': {
                'enabled': photo_clahe_enabled,
                'clip_limit': float(request.form.get('cl_limit', 4.0)),
                'tile_grid_size': 8,
                'p': float(request.form.get('cl_p', 0.3))
            }
        },
        'blur': {
            'enabled': blur_gb_enabled,
            'gaussian_blur': {
                'enabled': blur_gb_enabled,
                'blur_limit': int(request.form.get('gb_limit', 7)),
                'p': float(request.form.get('gb_p', 0.2))
            }
        },
        'noise': {
            'enabled': noise_gn_enabled,
            'gauss_noise': {
                'enabled': noise_gn_enabled,
                'var_min': float(request.form.get('gn_var_min', 10.0)),
                'var_max': float(request.form.get('gn_var_max', 50.0)),
                'p': float(request.form.get('gn_p', 0.2))
            }
        }
    }

    # Dosya yolları işlemleri (Öncekiyle Aynı)
    frames_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'frames')
    labels_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'label_results', 'labels_txt')
    
    aug_input = os.path.join(app.config['OUTPUT_FOLDER'], 'aug_input')
    aug_output = os.path.join(app.config['OUTPUT_FOLDER'], 'aug_output')
    
    os.makedirs(aug_input, exist_ok=True)
    if os.path.exists(aug_output):
        shutil.rmtree(aug_output)
    os.makedirs(aug_output, exist_ok=True)

    if not os.path.exists(labels_dir) or len(os.listdir(labels_dir)) == 0:
        return jsonify({"status": "error", "message": "Önce Auto Label sekmesinden etiketleme yapmalısınız!"})

    import glob
    for f in glob.glob(os.path.join(aug_input, '*')):
        os.remove(f)

    for txt_name in os.listdir(labels_dir):
        if txt_name.endswith('.txt'):
            base_name = txt_name[:-4]
            img_path = os.path.join(frames_dir, base_name + '.jpg')
            txt_path = os.path.join(labels_dir, txt_name)
            
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(aug_input, base_name + '.jpg'))
                shutil.copy(txt_path, os.path.join(aug_input, txt_name))

    cfg['input_dir'] = aug_input
    cfg['output_dir'] = aug_output

    def log_cb(msg):
        print(f"[Augmentor]: {msg}")
        
    def progress_cb(current, total):
        pass 
        
    stop_flag = threading.Event()

    try:
        run_augmentation(cfg, log_cb, progress_cb, stop_flag)
        
        zip_base = os.path.join(app.config['OUTPUT_FOLDER'], 'dataset_augmented')
        shutil.make_archive(zip_base, 'zip', aug_output)
        
        return jsonify({"status": "success", "download_url": "/download/dataset_augmented.zip"})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

    
if __name__ == '__main__':
    app.run(debug=True)