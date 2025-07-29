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