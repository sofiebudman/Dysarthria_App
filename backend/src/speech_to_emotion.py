from transformers import pipeline
import torchaudio

def text_to_speech_emotion(audio,asr_pipeline):
    # 1. ASR Pipeline
    
    result = asr_pipeline(audio, return_timestamps=True)
    # 2. TEXT → EMOTION (Zero-Shot Classification)
    emotion_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
    emotions = ["happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised"]
    result = emotion_classifier(result["text"], candidate_labels=emotions)
    return result["labels"][0], result  # top emotion & confidence

