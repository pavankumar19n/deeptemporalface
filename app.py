from flask import Flask, render_template, request, redirect, url_for
import os, cv2, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input, Xception
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = r"D:\mini2\website\final_model.h5"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model(MODEL_PATH)
proxy_cnn = Xception(weights='imagenet', include_top=False, input_shape=(71,71,3))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4','avi','mov','mkv'}

def extract_frames(video_path, frame_count=20, size=(71,71)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    interval = max(1, total // frame_count)
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames)

def gradcam_on_frame(frame_rgb):
    x = np.expand_dims(preprocess_input((frame_rgb * 255).astype(np.float32)), axis=0)
    grad_model = tf.keras.models.Model([proxy_cnn.inputs], [proxy_cnn.get_layer('block14_sepconv2_act').output, proxy_cnn.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (frame_rgb.shape[1], frame_rgb.shape[0]))
    heat_uint8 = np.uint8(255 * heatmap)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor((frame_rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR), 0.6, heat_color, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay)

def interpret_heatmap_colors():
    return (
        "🟥 <strong>Red</strong> areas show regions most responsible for model's decision, "
        "often revealing subtle blending artifacts or pixel inconsistencies.<br>"
        "🟨 <strong>Yellow</strong> indicates moderately influential facial regions where texture varies abnormally.<br>"
        "🟩 <strong>Green</strong> marks stable, realistic features — smoother skin or consistent motion patterns.<br>"
        "🟦 <strong>Blue</strong> represents low-importance zones that did not influence prediction much."
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        frames = extract_frames(path)
        seq = np.expand_dims(frames, axis=0)
        preds = model.predict(seq)[0][0]
        label = "Real Video" if preds < 0.5 else "AI-Generated (Deepfake)"
        bg_color = "#2ecc71" if label == "Real Video" else "#e74c3c"

        # Grad-CAM on mid-frame
        frame = frames[len(frames)//2]
        overlay = gradcam_on_frame(frame)
        heatmap_name = f"xai_{np.random.randint(1e9)}.png"
        overlay.save(os.path.join(app.config['UPLOAD_FOLDER'], heatmap_name))

        explanation = (
            f"The model predicts this video is <b>{label}</b>. "
            f"{'Authentic face details are smooth and consistent.' if label == 'Real Video' else 'Detected signs of synthetic blending and irregular patterns.'}<br>"
            f"{interpret_heatmap_colors()}<br><br>"
            "⚠️ Note: The colors highlight which facial regions most influenced the AI's judgment."
        )

        return render_template('result_xai.html',
                               label=label,
                               color=bg_color,
                               video_filename=filename,
                               xai_image=f'uploads/{heatmap_name}',
                               explanation=explanation)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
