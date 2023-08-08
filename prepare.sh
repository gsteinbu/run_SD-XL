
# Set up python virtual enviroment
python -m venv ./pyvenv
pip install diffusers 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install invisible_watermark transformers accelerate safetensors

# Download weights to disk (optional). WIll require lots of disk space (~150GB)
# Requires git lfs 'git lfs install'
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0