"""
GPT-based Image Sequence Evaluator

Evaluates generated image sequences on 4 dimensions:
1. Style Consistency (20%) - Visual style, color palette, lighting consistency
2. Object State Consistency (40%) - Whether objects change state correctly between steps  
3. Logical Flow (20%) - Whether the sequence makes logical sense
4. Task Completion (20%) - How well the final result matches the goal
"""

import os
import base64
import json
import glob
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
COMPASS_API_KEY = os.getenv("COMPASS_API_KEY")


class GPTEvaluator:

    def __init__(self, api_key: str = None):
        self.api_key = api_key or COMPASS_API_KEY
        self.client = OpenAI(
            api_key=self.api_key,
            base_url='https://compass.llm.shopee.io/compass-api/v1',
        )
        
        self.weights = {
            "style_consistency": 0.20,
            "object_state_consistency": 0.40,
            "logical_flow": 0.20,
            "task_completion": 0.20
        }

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    def get_image_media_type(self, image_path: str) -> str:
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(ext, "image/png")

    def load_task_images(self, task_folder: str) -> list[tuple[str, str]]:
        pattern = os.path.join(task_folder, "step_*.png")
        image_files = glob.glob(pattern)
        image_files = [f for f in image_files if "intermediate" not in f]
        
        def get_step_num(path):
            filename = os.path.basename(path)
            try:
                return int(filename.replace("step_", "").replace(".png", ""))
            except:
                return 0
        
        image_files.sort(key=get_step_num)
        
        images = []
        for img_path in image_files:
            base64_data = self.encode_image_to_base64(img_path)
            images.append((img_path, base64_data))
        
        return images

    def load_text_plans(self, task_folder: str):
        import pandas as pd
        csv_path = os.path.join(task_folder, "text_plans.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def build_evaluation_prompt(self, task_name: str, text_plans=None) -> str:
        task_description = task_name.replace("_", " ").title()
        
        steps_context = ""
        if text_plans is not None:
            steps = text_plans["text_plans"].tolist() if "text_plans" in text_plans.columns else []
            if steps:
                steps_context = "\n\nThe intended steps are:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
        
        prompt = f"""
You are an expert evaluator for AI-generated image sequences. You are evaluating a series of images that depict the steps of: "{task_description}".
{steps_context}

Please evaluate these images as a SEQUENCE (viewing them in order from first to last) on the following 4 dimensions. For each dimension, provide:
1. A score from 0-100
2. A detailed explanation (3-5 sentences) justifying why you gave this specific score. Reference specific images and step numbers where relevant.

## Evaluation Dimensions:

### 1. Style Consistency (Weight: 20%)
Evaluate how consistent the visual style is across ALL images:
- Color palette consistency
- Lighting and atmosphere consistency  
- Art style / rendering quality consistency
- Camera angle and composition consistency
- Same "world" feel across images

### 2. Object State Consistency (Weight: 40%)
THIS IS THE MOST CRITICAL DIMENSION. You must carefully track the state of each object across ALL steps.

Key principle: Objects must be in the CORRECT STATE for each step in the sequence.
- Example: If an egg is cracked in step 3, it should appear as a whole/raw egg in steps 1-2, and as a cracked/cooking egg from step 3 onwards.
- Example: If pasta is added to boiling water in step 2, the pot should show empty water in step 1, pasta in water from step 2 onwards.
- Example: If bread is toasted in step 4, it should look like regular bread in steps 1-3, and toasted from step 4 onwards.

Evaluate strictly:
- Does each object appear in the CORRECT state for that specific step? (not too early, not too late)
- Do objects maintain their identity across steps (same pan, same ingredients)?
- Are state transitions shown at the RIGHT step (transformation happens when it should)?
- Is an object that was transformed still showing the transformed state in later steps?
- Any anachronistic states? (e.g., cooked egg appearing before the cooking step)

### 3. Logical Flow (Weight: 20%)
Evaluate the narrative coherence:
- Does the sequence tell a coherent story?
- Are the steps in the right order?
- Does each image logically follow from the previous?
- Are there any jarring discontinuities?

### 4. Task Completion (Weight: 20%)
Evaluate how well the sequence achieves its goal:
- Does the final image show a completed result?
- Would someone following these images achieve the task?
- Is the end result recognizable as "{task_description}"?
- Overall quality of instruction visualization

## Output Format:
Return ONLY a JSON object with this exact structure:
{{
    "style_consistency": {{
        "score": <0-100>,
        "explanation": "<detailed explanation with specific examples from the images>"
    }},
    "object_state_consistency": {{
        "score": <0-100>,
        "explanation": "<MUST reference specific step numbers and object states. Example: 'In step 2, the egg appears already cooked when it should still be raw. The pan changes from non-stick to cast iron between steps 3 and 4.'>"
    }},
    "logical_flow": {{
        "score": <0-100>,
        "explanation": "<detailed explanation with specific examples from the images>"
    }},
    "task_completion": {{
        "score": <0-100>,
        "explanation": "<detailed explanation with specific examples from the images>"
    }},
    "overall_comments": "<2-3 sentences summarizing the main strengths and weaknesses of this image sequence>",
    "weighted_total": <calculated weighted score 0-100>
}}
"""
        return prompt

    def evaluate_task(self, task_folder: str) -> dict:
        task_name = os.path.basename(task_folder)
        print(f"\nEvaluating: {task_name}")
        
        images = self.load_task_images(task_folder)
        if not images:
            print(f"  Warning: No step images found in {task_folder}")
            return None
        
        print(f"  Found {len(images)} step images")
        
        text_plans = self.load_text_plans(task_folder)
        prompt = self.build_evaluation_prompt(task_name, text_plans)
        
        content = [{"type": "text", "text": prompt}]
        
        for i, (img_path, base64_data) in enumerate(images):
            media_type = self.get_image_media_type(img_path)
            content.append({
                "type": "text",
                "text": f"\n--- Step {i+1} Image ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_data}",
                    "detail": "high"
                }
            })
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": content}],
                model="gpt-4o",
                temperature=0.1,
                max_tokens=2000,
                extra_headers={"Provider": "OpenAI"},
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            if "weighted_total" not in result or result["weighted_total"] is None:
                weighted_total = 0
                for dim, weight in self.weights.items():
                    if dim in result and "score" in result[dim]:
                        weighted_total += result[dim]["score"] * weight
                result["weighted_total"] = round(weighted_total, 2)
            
            result["task_name"] = task_name
            result["num_steps"] = len(images)
            
            print(f"  Done - Weighted Score: {result['weighted_total']:.1f}/100")
            return result
            
        except Exception as e:
            print(f"  Error evaluating {task_name}: {e}")
            return {
                "task_name": task_name,
                "error": str(e),
                "weighted_total": 0
            }

    def generate_report(self, task_folder: str, result: dict) -> str:
        task_title = result.get('task_name', 'Unknown').replace('_', ' ').title()
        
        report = f"""# GPT Evaluation Report

## Task: {task_title}

### Overall Score: {result.get('weighted_total', 0):.1f} / 100

---

## Dimension Scores

| Dimension | Weight | Score |
|-----------|--------|-------|
| Style Consistency | 20% | {result.get('style_consistency', {}).get('score', 0)} |
| Object State Consistency | 40% | {result.get('object_state_consistency', {}).get('score', 0)} |
| Logical Flow | 20% | {result.get('logical_flow', {}).get('score', 0)} |
| Task Completion | 20% | {result.get('task_completion', {}).get('score', 0)} |

---

## Detailed Feedback

### 1. Style Consistency (Score: {result.get('style_consistency', {}).get('score', 0)}/100)

{result.get('style_consistency', {}).get('explanation', 'N/A')}

### 2. Object State Consistency (Score: {result.get('object_state_consistency', {}).get('score', 0)}/100)

{result.get('object_state_consistency', {}).get('explanation', 'N/A')}

### 3. Logical Flow (Score: {result.get('logical_flow', {}).get('score', 0)}/100)

{result.get('logical_flow', {}).get('explanation', 'N/A')}

### 4. Task Completion (Score: {result.get('task_completion', {}).get('score', 0)}/100)

{result.get('task_completion', {}).get('explanation', 'N/A')}

---

## Overall Comments

{result.get('overall_comments', 'N/A')}

---

Evaluated using GPT-4o Vision | Number of steps: {result.get('num_steps', 0)}
"""
        return report

    def evaluate_all_tasks(self, output_folder: str):
        print("=" * 60)
        print("Starting GPT Evaluation")
        print("=" * 60)
        
        task_folders = []
        for item in os.listdir(output_folder):
            item_path = os.path.join(output_folder, item)
            if os.path.isdir(item_path) and not item.startswith("."):
                if glob.glob(os.path.join(item_path, "step_*.png")):
                    task_folders.append(item_path)
        
        if not task_folders:
            print("No task folders with step images found.")
            return None
        
        print(f"\nFound {len(task_folders)} tasks to evaluate:")
        for tf in task_folders:
            print(f"   - {os.path.basename(tf)}")
        
        all_results = []
        for task_folder in task_folders:
            result = self.evaluate_task(task_folder)
            if result:
                all_results.append(result)
                report = self.generate_report(task_folder, result)
                report_path = os.path.join(task_folder, "gpt_evaluation_report.md")
                with open(report_path, "w") as f:
                    f.write(report)
                print(f"  Report saved: {report_path}")
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        for r in all_results:
            print(f"{r['task_name']}: {r['weighted_total']:.1f}/100")
        
        if all_results:
            avg = sum(r['weighted_total'] for r in all_results) / len(all_results)
            print(f"\nAverage Score: {avg:.1f}/100")
        
        return all_results

    def evaluate_single_task(self, task_folder: str) -> dict:
        print("=" * 60)
        print(f"Evaluating: {os.path.basename(task_folder)}")
        print("=" * 60)
        
        result = self.evaluate_task(task_folder)
        if result:
            report = self.generate_report(task_folder, result)
            report_path = os.path.join(task_folder, "gpt_evaluation_report.md")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"Report saved: {report_path}")
        
        return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_folder = os.path.join(project_root, "output")
    
    print(f"Output folder: {output_folder}")
    
    evaluator = GPTEvaluator()
    evaluator.evaluate_all_tasks(output_folder)


if __name__ == "__main__":
    main()
