
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
import librosa
import io
import cv2 
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import base64
import tempfile
import torch

from src.unet import UNet  # Adjust the import based on your project structure  
from src.binary_preprocess import binary_preprocess_audio
from src.multiclass_preprocess import multiclass_preprocess_audio
from src.russian_preprocess import russian_preprocess_audio, mel_to_audio

app = Flask(__name__, template_folder='../frontend')

# Load models with error handling
try:
    russian_pred_model = UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    russian_pred_model.load_state_dict(torch.load('models/unet_russian_pretrained.pth', map_location=torch.device('cpu')))
    russian_pred_model.eval()
    print("Russian model loaded successfully")
except Exception as e:
    print(f"Error loading Russian model: {e}")
    russian_pred_model = None

try:
    multiclass_pred_model = keras.models.load_model('models/multiclass_pred_model.keras')
    print("Multiclass model loaded successfully")
except Exception as e:
    print(f"Error loading multiclass model: {e}")
    multiclass_pred_model = None

try:
    binaryclass_pred_model = keras.models.load_model('models/dysarthria_model_eng.keras')
    print("Binary model loaded successfully")
except Exception as e:
    print(f"Error loading binary model: {e}")
    binaryclass_pred_model = None

'''
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model="./whisper-finetuned",        # Local folder path
    tokenizer="./whisper-finetuned",    # Same folder
    feature_extractor="./whisper-finetuned",  # Important for audio
    framework="pt"
)
'''

@app.route('/')
def home():
    #return "Flask backend for audio prediction running."
    return render_template("index.html")

@app.route('/multiclass')
def multiclass():
    return render_template("multiclass.html")

@app.route('/classification')
def classification():
    return render_template("binary.html")

@app.route('/english')
def english():
    return render_template("english.html")
@app.route('/russian')
def russian():
    return render_template("russian.html")

'''
HOME: description of app + menu : app route = / (SOFIE)
BINARY CLASSIFICATION PAGE: app route = /classification (ananya)
SEVERITY PREDICTION PAGE: app route = /multiclass_classification (SOFIE)
ENGLISH app route = /english
Speech to text english (nithika)
Voice cloning (needs normal audio as well) (nithika)
Clean speech generation english - output spectogram (ananya)
Emotion detection english (nithika)


RUSSIAN app route = /russian
Clean speech generation Russian (ANISHA)

'''



@app.route('/multiclasspredict', methods=['POST'])
def multiclass_predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Preprocess audio to get features for model input

    features = multiclass_preprocess_audio(audio_file)
    #features = np.expand_dims(features, axis=0)
    
    # Run prediction
    preds = multiclass_pred_model.predict(features)
    
    # Convert prediction output to list for JSON
    preds_list = preds.tolist()
    
    highest_class = int(preds.argmax(axis=1)[0])
    output = ""
    if(highest_class == 0):
        highest_class = "High Dysarthria"
    elif(highest_class == 1):
        highest_class = "Low Dysarthria"
    elif(highest_class == 2):
        highest_class = "Medium Dysarthria"
    elif(highest_class == 3):
        highest_class = "Very Low Dysarthria"
    
    return jsonify({'prediction': highest_class})
@app.route('/binarypredict', methods=['POST'])
def binary_predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']

    # Preprocess audio
    features = binary_preprocess_audio(audio_file)  # shape: (1, 16, 8, 1)

    # Predict
    pred = binaryclass_pred_model.predict(features)
    prob = float(pred[0][0])  # Get the raw sigmoid output

    if prob >= 0.5:
        label = "Dysarthria Present"
    else:
        label = "No Dysarthria Present"

    return jsonify({
        'prediction': label,
        'confidence': round(prob, 3)
    })



@app.route('/emotionenglishpredict', methods=['POST'])
def transcribe_audio(audio_path):
    result = asr_pipeline(audio_path)
    # 2. TEXT → EMOTION (Zero-Shot Classification)
    emotion_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
    emotions = ["happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised"]
    result = emotion_classifier(result["text"], candidate_labels=emotions)
    return result["labels"][0], result  # top emotion & confidence

 
@app.route('/russianpredict', methods=['POST'])
def predict_russian():
    try:
        if russian_pred_model is None:
            return jsonify({'error': 'Russian model not loaded'}), 500
            
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']

        # === Step 1: Preprocess
        spectrogram = russian_preprocess_audio(file)  # shape (1, 1, 128, 128)
        
        # Create visualization for original spectrogram
        spectrogram_viz = np.squeeze(spectrogram)  # (128, 128)
        spectrogram_img = (spectrogram_viz * 255).astype(np.uint8)
        spectrogram_rgb = cv2.cvtColor(spectrogram_img, cv2.COLOR_BGR2RGB)
        _, spectro_buf = cv2.imencode('.png', spectrogram_rgb)
        spectro_base64 = base64.b64encode(spectro_buf).decode('utf-8')

        # Convert to tensor for model input
        spectrogram_tensor = torch.tensor(spectrogram).float()




        # === Step 2: Predict clean spectrogram
        pred_spectrogram = russian_pred_model(spectrogram)  # shape (1, 128, 128)
        pred_spectrogram = np.squeeze(pred_spectrogram)  # shape (128, 128)

        # === Step 3: Convert predicted spectrogram to waveform
        clean_audio = mel_to_audio(pred_spectrogram)
        audio_buf = io.BytesIO()
        sf.write(audio_buf, clean_audio, 16000, format='WAV')
        audio_buf.seek(0)
        audio_base64 = base64.b64encode(audio_buf.read()).decode('utf-8')

        # === Step 4: Plot predicted spectrogram
        pred_spectrogram_db = pred_spectrogram * 80 - 80
        fig, ax = plt.subplots()
        img = librosa.display.specshow(pred_spectrogram_db, sr=16000, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title('Predicted Clean Spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        pred_spectro_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # === Step 5: Return everything as base64
        '''
        return jsonify({
            'spectrogram_image': spectro_base64,
            'predicted_spectrogram_image': pred_spectro_base64,
            'clean_audio': audio_base64
        })'''
        # === Step 5: Return everything as base64
   
        return jsonify({
        'spectrogram_image': spectro_base64,
        'predicted_spectrogram_image': pred_spectro_base64,
        'clean_audio': audio_base64
        })
    except Exception as e:
        print("Internal server error:", str(e))
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)


