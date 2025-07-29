
from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
import librosa
import io
import cv2 
from flask import Flask, request, jsonify
#from flask import Flask

app = Flask(__name__)






multiclass_pred_model = keras.models.load_model('backend/models/multiclass_pred_model.keras')


def preprocess_audio(file):
    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes) #read audio


    y, sr = librosa.load(audio_buffer, sr=None)

    S = librosa.stft(y, n_fft=1024, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    S_normalized = 255 * (S_db - S_db.min()) / (S_db.max() - S_db.min())
    S_normalized = S_normalized.astype(np.uint8)


    S_color = cv2.applyColorMap(S_normalized, cv2.COLORMAP_VIRIDIS)  # Shape: (freq, time, 3), dtype: uint8


    S_resized = cv2.resize(S_color, (128, 128), interpolation=cv2.INTER_LINEAR)


    S_resized = S_resized.astype(np.float32) / 255.0

    S_final = np.expand_dims(  S_resized, axis=0)

    return S_final


@app.route('/')
def home():
    return "Flask backend for audio prediction running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Preprocess audio to get features for model input

    features = preprocess_audio(audio_file)
    #features = np.expand_dims(features, axis=0)
    
    # Run prediction
    preds = multiclass_pred_model.predict(features)
    
    # Convert prediction output to list for JSON
    preds_list = preds.tolist()
    
    return jsonify({'prediction': preds_list})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)


