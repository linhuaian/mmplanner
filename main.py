from instruction_planner import InstructionPlanner
from stable_diffusion_generator import StableDiffusionImageGenerator
from step_image_selector import StepImageSelector
from PIL import Image
import random 

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_TOKEN = os.getenv("COMPASS_API_KEY")

print("OPENAI_TOKEN =", OPENAI_TOKEN)


if __name__ == "__main__":
    k = 10

    how_to_tasks = [
     "how to build a PC?"]
    for task in how_to_tasks:
        
        out_folder = f"./output/{task.replace('?', '').replace(' ', '_').lower()}"
        intermediate_output_folder = f"{out_folder}/intermediate/"

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        if not os.path.exists(intermediate_output_folder):
            os.mkdir(intermediate_output_folder)

        print("Planning task....")

        ip = InstructionPlanner(OPENAI_TOKEN)

        if not os.path.exists(os.path.join(out_folder, "text_plans.csv")):
            plan = ip.generate_text_plan(task, output_folder=out_folder)

        sd = StableDiffusionImageGenerator() 

        ss = StepImageSelector()
        
        prev_img = None 
        selected_img = []

        for i, p in enumerate(plan): 
            images = []
            print(f"Generating image for step: {p}")
            for _ in range(k):
                image = sd.generate_image(p).convert('RGB') 
                image.save(f"{intermediate_output_folder}/step_{i}_{_}.png")
                images.append(image)
            if prev_img == None: 
                # for first image generation 
                selected = random.choice(images)
                selected_img.append(selected)
            else: 
                selected = ss.select_best_image(prev_img, images, step=i)
                selected_img.append(selected)
            prev_img = selected
            selected.save(f"{out_folder}/step_{i}.png")

        # for i, img in enumerate(selected_img):
        #     img.save(f"./output/step_{i}.png")

                    
