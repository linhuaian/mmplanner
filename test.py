from diffusers import DiffusionPipeline
import torch

# Load the pipeline
# Use float16 for faster inference and less VRAM usage if you have a compatible GPU (NVIDIA)
pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
)
# Move the pipeline to the GPU
pipeline.to("cuda")

# Define your prompt
prompt = "An image of a squirrel in Picasso style, vibrant colors"

# Generate the image
image = pipeline(prompt).images[0]

# Save the image
image.save("squirrel_picasso.png")
