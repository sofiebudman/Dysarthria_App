from transformers import pipeline
import torchaudio
from transformers import pipeline

pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
import os
from huggingface_hub import InferenceClient
os.environ['HF_TOKEN'] = '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'
client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

def text_to_speech_emotion(audio,asr_pipeline):
    # 1. ASR Pipeline
    
    text = asr_pipeline(audio, return_timestamps=True)
    # 2. TEXT → EMOTION (Zero-Shot Classification)
    #emotion_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
    ''''
    emotions = ["happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised"]
    emotion = emotion_classifier(text["text"], candidate_labels=emotions)
    '''
    emotion = pipe(text["text"])
    emotion = emotion[0]['label']
    return emotion, text  # top emotion & confidence

