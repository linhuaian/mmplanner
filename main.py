from instruction_planner import InstructionPlanner
from stable_diffusion_generator import StableDiffusionImageGenerator
from step_image_selector import StepImageSelector
from PIL import Image
import random 
import gc
import json
from pathlib import Path

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_TOKEN = os.getenv("COMPASS_API_KEY")

print("OPENAI_TOKEN =", OPENAI_TOKEN)


if __name__ == "__main__":
    # Number of independent candidates per step (kept small because we do iterative critique/regeneration).
    k = 3
    max_versions = 3

    how_to_tasks = [
     "how to build a PC?", "how to do basketball layup?"]

    # Heavy models (Stable Diffusion + selector backbone) should be constructed ONCE
    # and reused across tasks to avoid GPU/CPU memory fragmentation and OOM.
    ip = InstructionPlanner(OPENAI_TOKEN)
    sd = StableDiffusionImageGenerator()
    ss = StepImageSelector()
    try:
        from prompt_critique_agent import ImagePromptCritiqueAgent

        critique_agent = ImagePromptCritiqueAgent(api_key=OPENAI_TOKEN)
    except Exception as e:
        print(f"[warn] ImagePromptCritiqueAgent disabled: {e}")
        critique_agent = None

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
        
        prev_img_path: str | None = None

        for i, p in enumerate(plan): 
            print(f"Generating images for step {i}: {p}")

            # Generate k candidates, each with up to v1/v2/v3 prompt refinements.
            all_version_paths: list[str] = []
            for cand_idx in range(k):
                cur_prompt = p
                for v in range(1, max_versions + 1):
                    img = sd.generate_image(cur_prompt).convert("RGB")
                    try:
                        out_path = f"{intermediate_output_folder}/step_{i}_{cand_idx}_v{v}.png"
                        img.save(out_path)
                        all_version_paths.append(out_path)

                        # Save the prompt used for this version for debugging/repro.
                        Path(f"{intermediate_output_folder}/step_{i}_{cand_idx}_v{v}.txt").write_text(cur_prompt, encoding="utf-8")

                        # Critique image vs prompt and optionally rewrite prompt for next version.
                        if critique_agent is not None and v < max_versions:
                            try:
                                critique = critique_agent.critique_and_rewrite_prompt(image=img, current_prompt=cur_prompt)
                                Path(f"{intermediate_output_folder}/step_{i}_{cand_idx}_v{v}_critique.json").write_text(
                                    json.dumps(critique, indent=2),
                                    encoding="utf-8",
                                )
                                if critique.get("ok") is True:
                                    break
                                revised = (critique.get("revised_prompt") or "").strip()
                                if revised:
                                    cur_prompt = revised
                            except Exception as e:
                                Path(f"{intermediate_output_folder}/step_{i}_{cand_idx}_v{v}_critique_error.txt").write_text(
                                    f"{type(e).__name__}: {e}",
                                    encoding="utf-8",
                                )
                                break
                    finally:
                        # Release image object (it's on disk already).
                        try:
                            img.close()
                        except Exception:
                            pass

            # Load all generated versions and let the selector pick the best across v1/v2/v3.
            candidates: list[Image.Image] = []
            for pth in all_version_paths:
                try:
                    im = Image.open(pth).convert("RGB")
                    candidates.append(im)
                except Exception:
                    continue

            prev_img = Image.open(prev_img_path).convert("RGB") if prev_img_path else None
            if prev_img is None:
                selected = random.choice(candidates) if candidates else None
            else:
                selected = ss.select_best_image(prev_img, candidates, step=i) if candidates else None

            # Save selected (if any)
            if selected is not None:
                selected_path = f"{out_folder}/step_{i}.png"
                selected.save(selected_path)
                prev_img_path = selected_path

            # Cleanup PIL objects
            if prev_img is not None:
                try:
                    prev_img.close()
                except Exception:
                    pass
            for im in candidates:
                try:
                    im.close()
                except Exception:
                    pass
            candidates.clear()

        # Task-level cleanup (best-effort): release last image and clear caches.
        prev_img_path = None
        del plan
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

                    
