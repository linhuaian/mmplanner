import os
os.environ["HF_HUB_DISABLE_TLS_VERIFY"] = "1"

from diffusers import DiffusionPipeline
import torch

class StableDiffusionImageGenerator:
    def __init__(self):
        # Pick a sensible device across CUDA / Apple Silicon / CPU-only environments.
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=dtype,
        )
        # Reduce peak memory where possible.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        self.pipe = pipe.to(device)


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
