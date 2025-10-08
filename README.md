# A Multilingual Framework for Dysarthria: Detection, Severity Classification, Speech-to-Text, and Clean Speech Generation
Ananya Raghu, Anisha Raghu, Nithika Vivek, Sofie Budman, Omar Mansour

[https://arxiv.org/abs/2510.03986](Paper)

Dysarthria is a motor speech disorder that results in slow and often incomprehensible speech. Speech intelligibility significantly impacts communication, leading to barriers in social interactions. Dysarthria is often a characteristic of neurological diseases including Parkinson's and ALS, yet current tools lack generalizability across languages and levels of severity. In this study, we present a unified AI-based multilingual framework that addresses six key components: (1) binary dysarthria detection, (2) severity classification, (3) clean speech generation, (4) speech-to-text conversion, (5) emotion detection, and (6) voice cloning. We analyze datasets in English, Russian, and German, using spectrogram-based visualizations and acoustic feature extraction to inform model training. Our binary detection model achieved 97% accuracy across all three languages, demonstrating strong generalization across languages. The severity classification model also reached 97% test accuracy, with interpretable results showing model attention focused on lower harmonics. Our translation pipeline, trained on paired Russian dysarthric and clean speech, reconstructed intelligible outputs with low training (0.03) and test (0.06) L1 losses. Given the limited availability of English dysarthric-clean pairs, we fine-tuned the Russian model on English data and achieved improved losses of 0.02 (train) and 0.03 (test), highlighting the promise of cross-lingual transfer learning for low-resource settings. Our speech-to-text pipeline achieved a Word Error Rate of 0.1367 after three epochs, indicating accurate transcription on dysarthric speech and enabling downstream emotion recognition and voice cloning from transcribed speech. Overall, the results and products of this study can be used to diagnose dysarthria and improve communication and understanding for patients across different languages.




Flask for backend

### UV for package management

download uv
uv init - intializes uv

uv sync - syncs new downloaded packages

uv run backend/app.py to run site on http://127.0.0.1:5000/

## track files with git lfs


trained models in /backend/models


## LFS

brew install lfs

git lfs track "*.pth"

git add models/unet_pretrained_english.pth

git commit -m "Add English UNet model with LFS"

git push origin main
