import os
os.environ["HF_HUB_DISABLE_TLS_VERIFY"] = "1"

from diffusers import StableDiffusionPipeline
import torch

# model_name = "Qwen/Qwen-Image"

class StableDiffusionImageGenerator:
    def __init__(self):
        # Load the pipeline
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
        else:
            torch_dtype = torch.float32
            device = "cpu"

        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", dtype=torch_dtype).to(device)

    def generate_image(self, prompt, negative_prompt = "", width=512, height=512):
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda")
        ).images[0]
        # image.save("example.png")
        return image 

if __name__ == "__main__":
    print("Generating...")
    sd = StableDiffusionImageGenerator() 
    image = sd.generate_image("generate a image of a dog with its owner")
    image.save("example_image.png")
