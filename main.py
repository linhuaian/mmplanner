from instruction_planner import InstructionPlanner
from stable_diffusion_generator import StableDiffusionImageGenerator
from step_image_selector import StepImageSelector
from PIL import Image
import random 


if __name__ == "__main__":
    k = 2

    ip = InstructionPlanner("xxx")
    plan = ip.generate_text_plan("How to cook a fried egg?")

    sd = StableDiffusionImageGenerator() 

    ss = StepImageSelector()
    
    prev_img = None 
    selected_img = []

    for i, p in enumerate(plan): 
        images = []
        print(f"Generating image for step: {p}")
        for _ in range(k):
            image = sd.generate_image(p).convert('RGB') 
            images.append(image)
        if prev_img == None: 
            # for first image generation 
            selected = random.choice(images)
            selected_img.append(selected)
        else: 
            selected = ss.select_best_image(prev_img, images, step=i)
            selected_img.append(selected)
        prev_img = selected

    for i, img in enumerate(selected_img):
        img.save(f"./output/step_{i}.png")

                
