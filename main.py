from instruction_planner import InstructionPlanner
from stable_diffusion_generator import StableDiffusionImageGenerator
from step_image_selector import StepImageSelector
from PIL import Image
import random 
import gc

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_TOKEN = os.getenv("COMPASS_API_KEY")

print("OPENAI_TOKEN =", OPENAI_TOKEN)


if __name__ == "__main__":
    k = 10

    how_to_tasks = [
     "how to build a PC?", "how to do basketball layup?"]

    # Heavy models (Stable Diffusion + selector backbone) should be constructed ONCE
    # and reused across tasks to avoid GPU/CPU memory fragmentation and OOM.
    ip = InstructionPlanner(OPENAI_TOKEN)
    sd = StableDiffusionImageGenerator()
    ss = StepImageSelector()

    for task in how_to_tasks:
        
        out_folder = f"./output/{task.replace('?', '').replace(' ', '_').lower()}"
        intermediate_output_folder = f"{out_folder}/intermediate/"

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        if not os.path.exists(intermediate_output_folder):
            os.mkdir(intermediate_output_folder)

        print("Planning task....")
        csv_path = os.path.join(out_folder, "text_plans.csv")
        if not os.path.exists(csv_path):
            plan = ip.generate_text_plan(task, output_folder=out_folder)
        else:
            # Reuse existing planned image descriptions (so we don't re-run planning unnecessarily)
            import pandas as pd

            df = pd.read_csv(csv_path)
            col = "image descriptions" if "image descriptions" in df.columns else df.columns[-1]
            plan = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
        
        prev_img = None 

        for i, p in enumerate(plan): 
            images = []
            print(f"Generating image for step: {p}")
            for _ in range(k):
                image = sd.generate_image(p).convert('RGB') 
                image.save(f"{intermediate_output_folder}/step_{i}_{_}.png")
                images.append(image)
            if prev_img is None:
                # for first image generation
                selected = random.choice(images)
            else:
                selected = ss.select_best_image(prev_img, images, step=i)

            # Save selected image, then release unneeded candidates to keep memory bounded.
            selected_path = f"{out_folder}/step_{i}.png"
            selected.save(selected_path)

            # Close previous image if it's no longer used.
            if prev_img is not None and prev_img is not selected:
                try:
                    prev_img.close()
                except Exception:
                    pass

            # Close all non-selected candidate images.
            for img in images:
                if img is selected:
                    continue
                try:
                    img.close()
                except Exception:
                    pass

            prev_img = selected
            images.clear()

        # Task-level cleanup (best-effort): release last image and clear caches.
        if prev_img is not None:
            try:
                prev_img.close()
            except Exception:
                pass
            prev_img = None
        del plan
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

                    
