import os
os.environ["HF_HUB_DISABLE_TLS_VERIFY"] = "1"

from diffusers import DiffusionPipeline
import torch

class StableDiffusionImageGenerator:
    def __init__(self):
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", device_map="cuda")
        self.pipe = pipe


    def generate_image(self, prompt, negative_prompt = "", width=512, height=512):
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=50
        ).images[0]
        # image.save("example.png")
        return image 

if __name__ == "__main__":
    print("Generating...")
    sd = StableDiffusionImageGenerator() 
    image = sd.generate_image("generate a image of a dog with its owner")
    image.save("example_image.png")
