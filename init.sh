pip install openai
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install accelerate
pip uninstall -y torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U diffusers==0.30.3
pip install peft==0.17.0
pip install transformers==4.37.2
