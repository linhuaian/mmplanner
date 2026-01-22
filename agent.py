"""
ReAct Agent with Memory for Multi-Modal Planning
"""

import os
import json
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from instruction_planner import InstructionPlanner

load_dotenv()
COMPASS_API_KEY = os.getenv("COMPASS_API_KEY")


class Agent:
    
    def __init__(self, api_key=None):
        self.api_key = api_key or COMPASS_API_KEY
        self.client = OpenAI(
            api_key=self.api_key,
            base_url='https://compass.llm.shopee.io/compass-api/v1',
        )
        self.instruction_planner = InstructionPlanner(self.api_key)
        self.memory = {
            "task": "",
            "current_step": 0,
            "objects": {},
            "step_history": [],
        }
        self.image_generator = None
        self.image_selector = None
        self.text_object_state_agent = None

    def reset(self, task: str):
        self.memory = {
            "task": task,
            "current_step": 0,
            "objects": {},
            "step_history": [],
        }

    # ==================== TOOLS ====================

    def update_object_state(self, name: str, state: str):
        step = self.memory["current_step"]
        if name not in self.memory["objects"]:
            self.memory["objects"][name] = {"state": state, "history": [(step, state)]}
        else:
            self.memory["objects"][name]["state"] = state
            self.memory["objects"][name]["history"].append((step, state))

    # NOTE: Object/state ground-truth for image descriptions is now produced by TextObjectStateAgent
    # inside InstructionPlanner.generate_text_plan(), and saved under the task output folder.

    def generate_images(self, prompt: str, k: int = 2) -> list:
        if self.image_generator is None:
            from stable_diffusion_generator import StableDiffusionImageGenerator
            self.image_generator = StableDiffusionImageGenerator()
        
        images = []
        for _ in range(k):
            img = self.image_generator.generate_image(prompt).convert('RGB')
            images.append(img)
        return images

    def select_best_image(self, prev_image, candidates: list, step: int):
        if prev_image is None:
            return random.choice(candidates)
        
        if self.image_selector is None:
            from step_image_selector import StepImageSelector
            self.image_selector = StepImageSelector()
        
        return self.image_selector.select_best_image(prev_image, candidates, step=step)

    # ==================== LLM CALLS ====================

    def generate_image_prompt(self, step_num: int, step_desc: str, expected_states: dict) -> str:
        prompt = f"""
Generate an image prompt for step {step_num} of "{self.memory['task']}".

Step: {step_desc}

Objects and their REQUIRED states:
{json.dumps(expected_states, indent=2)}

Return JSON: {{"image_prompt": "detailed prompt showing objects in correct states"}}
"""
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.3,
            extra_headers={"Provider": "OpenAI"},
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("image_prompt", "")

    # ==================== MAIN PIPELINE ====================

    def run(self, task: str, output_folder: str, k: int = 2):
        print(f"Task: {task}")
        self.reset(task)
        
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(f"{output_folder}/intermediate", exist_ok=True)
        
        # Generate steps using instruction planner
        print("\n[1] Generating step plan...")
        # InstructionPlanner now uses TextObjectStateAgent internally to:
        # - save ground_truth_object_phrases_text.json / ground_truth_object_states_text.json
        # - produce SD-friendly image descriptions grounded in those phrases
        image_prompts = self.instruction_planner.generate_text_plan(task, output_folder=output_folder)
        
        # Process each step
        print("\n[2] Generating images...")
        prev_image = None
        
        for i, prompt in enumerate(image_prompts):
            step_num = i + 1
            self.memory["current_step"] = step_num
            print(f"\nStep {step_num}: {prompt}...")
            
            # Generate and select image
            candidates = self.generate_images(prompt, k=k)
            for j, img in enumerate(candidates):
                img.save(f"{output_folder}/intermediate/step_{i}_{j}.png")
            
            selected = self.select_best_image(prev_image, candidates, step=step_num)
            selected.save(f"{output_folder}/step_{i}.png")
            prev_image = selected
        
        print(f"\nDone! Saved to {output_folder}")


if __name__ == "__main__":
    agent = Agent()
    
    task = "How to make cheesy garlic pull-apart bread?"
    output_folder = f"./output/{task.replace('?', '').replace(' ', '_').lower()}"
    
    # Run full pipeline (requires GPU)
    agent.run(task, output_folder, k=20)
