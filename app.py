import os
import shutil
import tempfile
import base64
import cv2
import threading
from flask import Flask, render_template, request, send_file, jsonify

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

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)