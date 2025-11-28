pip install openai
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install accelerate
pip uninstall -y torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U diffusers
pip install transformers==4.44.2
pip install numpy==1.24.4