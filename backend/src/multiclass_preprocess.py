
import io
import cv2 
import librosa
import numpy as np

import numpy as np
import librosa
import io
import cv2
import tensorflow as tf  # OpenCV for color mapping and resizing


def multiclass_preprocess_audio(file):

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
'''
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
   # _ = model(img_array)  # call the model on the input to build it
  

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
'''

