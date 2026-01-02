# Clean up any existing broken installations first
pip uninstall -y torch torchvision torchaudio

# Install the verified XPU versions
# 1. Remove all Intel and broken torch versions
pip uninstall -y torch torchvision torchaudio intel-extension-for-pytorch bitsandbytes

# 2. Install Torch with CUDA 12.4 support (Standard for A10 in 2025)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install the standard NVIDIA version of bitsandbytes
pip install bitsandbytes accelerate
pip install diffusers==0.36.0
pip install transformers==4.51.3
pip install bitsandbytes==0.46.1
pip install accelerate 
pip install openai
pip install dotenv
pip install sentencepiece
pip uninstall apex -y
pip install peft==0.17.0
pip install timm==0.9.16

export HF_TOKEN="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"