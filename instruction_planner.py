from openai import OpenAI
import pandas as pd 
import json 
import os
from dotenv import load_dotenv

load_dotenv() 
COMPASS_API_KEY = os.getenv("COMPASS_API_KEY")

def _strip_leading_article(s: str) -> str:
    s = (s or "").strip()
    for art in ("a ", "an ", "the "):
        if s.lower().startswith(art):
            return s[len(art):].strip()
    return s


def _join_natural(items: list[str]) -> str:
    items = [x for x in (items or []) if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _task_to_context(task: str) -> str:
    """
    Convert an instruction like "how to build a PC?" into a short context phrase like "Building a PC".
    Heuristic only; meant to add scene context for SD prompts.
    """
    t = (task or "").strip()
    t = t.strip().rstrip("?!.")
    low = t.lower()
    if low.startswith("how to "):
        t = t[7:].strip()
    # Title-case acronyms like "pc" -> "PC" when original contains uppercase.
    # Keep the remainder as-is to preserve casing like "PC".
    words = t.split()
    if not words:
        return ""
    verb = words[0]
    rest = " ".join(words[1:]).strip()

    # Very small set of English -ing conversions; fallback to "<verb>ing".
    vlow = verb.lower()
    if vlow.endswith("e") and vlow not in {"see", "be"}:
        gerund = verb[:-1] + "ing"
    elif vlow.endswith("ie"):
        gerund = verb[:-2] + "ying"
    elif vlow in {"run"}:
        gerund = verb + "ning"
    else:
        gerund = verb + "ing"
    phrase = (gerund + (" " + rest if rest else "")).strip()
    if not phrase:
        return ""
    # Capitalize first letter for sentence start.
    return phrase[0].upper() + phrase[1:]


def _compose_sd_prompt(task: str, step_text: str, object_phrases: list[str], *, max_chars: int = 180) -> str:
    """
    Turn object-state phrases into one SHORT, continuous SD prompt with task context.
    Deterministic (no extra LLM call), broadly applicable, and length-capped for SD.
    """
    context = _task_to_context(task)
    objs = [_strip_leading_article(p) for p in (object_phrases or [])]
    objs = [o for o in objs if o]
    objs_clause = _join_natural(objs[:4])  # keep it tight

    # Short, sweet, task-grounded, and no camera/style terms.
    if objs_clause:
        if context:
            prompt = f"{context} with {objs_clause}."
        else:
            prompt = f"{objs_clause}."
    else:
        # If no phrases, fall back to step text but still add task context.
        step = (step_text or "").strip()
        if step[:3].strip().endswith(".") and step[:2].strip(" .").isdigit():
            step = step.split(".", 1)[-1].strip()
        step = step.rstrip(".")
        if context:
            prompt = f"{context}: {step}."
        else:
            prompt = f"{step}."

    prompt = " ".join(prompt.split())  # normalize whitespace
    if len(prompt) <= max_chars:
        return prompt

    # Trim gracefully to max_chars.
    trimmed = prompt[: max_chars - 1].rstrip(" ,.;:-")
    return trimmed + "…"


class InstructionPlanner:
    def __init__(self, api_key):
        self.api_key = api_key

        self.client = OpenAI(
            api_key=api_key,
            base_url='https://compass.llm.shopee.io/compass-api/v1',
        )

    def generate_image_plan(self, text_plans):
# --- Chain of Thought Prompt for Image Generation ---
        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(text_plans)])
        
        prompt = f"""
        You are an expert image prompt engineer for a text-to-image generator (like Stable Diffusion). 
        
        Your task is to convert a sequence of step-by-step instructions into a list of highly detailed, single-sentence visual descriptions.
        These descriptions must capture the state of the objects and the action being performed for each step.
        
        The overall plan is:
        ---
        {steps_text}
        ---
        
        For EACH step in the plan, follow this Chain of Thought process:
        1. **ANALYZE:** Identify the main action, the key subject(s) (e.g., egg, pan, spatula), and the resulting state of the subject after the action (e.g., "The egg is now cracked and frying").
        2. **VISUALIZE:** Formulate a single, creative, high-fidelity prompt that clearly describes the scene. Focus on lighting, composition (close-up/wide shot), and the visual state of the main objects to make the image photorealistic and engaging.
        3. **OUTPUT:** Provide ONLY the final, detailed image description.
        
        **Constraint:** The final output MUST be a JSON list of strings, where each string is the image description for the corresponding step. Do not include any other text or reasoning in the final output.

        Example of a good description: "A close-up, photorealistic shot showing a cracked egg dropping onto a hot, buttered non-stick pan, with steam gently rising from the surface."
        """


        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",
            temperature=0.3, # Higher temp for more creative descriptions
            extra_headers={
                        "Provider": "OpenAI"
                    },
            # Force JSON output for reliable parsing
            response_format={"type": "json_object"} 
        )
        
        # 1. Parse the content (which is guaranteed to be a JSON string)
        content = chat_completion.choices[0].message.content
        json_data = json.loads(content)
        
        # 2. Extract the list of descriptions. We assume the model provides a key like "descriptions" or "image_prompts"
        # Since we constrained the output to a JSON *list*, we can try to find the list.
        if isinstance(json_data, dict):
                # Try to find the list value if the model wrapped it in an object
                descriptions = next((v for v in json_data.values() if isinstance(v, list)), [])
        elif isinstance(json_data, list):
                descriptions = json_data
        else:
                descriptions = []

        return descriptions
            
        

    def generate_text_plan(self, instruction, output_folder=None, min_steps=5, max_steps=10, *, verbose: bool = False):
        prompt = (
            f"Break down the instruction \"{instruction}\" into a step-by-step plan. "
            f"Generate between {min_steps} and {max_steps} clear, concise steps, generate only the step plan without other words."
        )
        chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="gpt-5.1",
                    temperature=0.01,
                    extra_headers={
                        "Provider": "OpenAI"
                    }
                )
        content = chat_completion.choices[0].message.content
        steps = [line.strip() for line in content.strip().split('\n') if line.strip()]

        # Image descriptions should be grounded in TEXT-derived ground-truth object phrases.
        # We use the TextObjectStateAgent to extract visually-observable objects+states from each step,
        # then convert those phrases into an SD-friendly image prompt per step.
        image_descriptions = []
        if output_folder is None:
            raise ValueError("output_folder is required so we can save text_plans.csv and ground-truth JSON outputs.")

        try:
            from object_state_agent import TextObjectStateAgent

            state_agent = TextObjectStateAgent(api_key=self.api_key)
            for i, step_text in enumerate(steps):
                step_state = state_agent.analyze_step_text(
                    task=instruction,
                    step_index=i,
                    step_text=step_text,
                    expected_states=None,
                    num_phrases=5,
                    verbose=bool(verbose),
                    store_trace=True,
                )
                if step_state.phrases:
                    # Compose a single continuous SD prompt from the phrases.
                    image_descriptions.append(_compose_sd_prompt(instruction, step_text, step_state.phrases))
                else:
                    # Fallback: if no phrases, use the step text so SD still has something to render.
                    image_descriptions.append(_compose_sd_prompt(instruction, step_text, []))

            state_agent.save_task_text(output_folder, num_phrases=5)
        except Exception as e:
            # If the object-state agent cannot run, fall back to the old LLM image-plan behavior.
            # (Still save a CSV so downstream pipelines work.)
            print(f"[warn] TextObjectStateAgent unavailable; falling back to generate_image_plan: {type(e).__name__}: {e}")
            image_descriptions = self.generate_image_plan(steps)

        # Ensure same length (truncate to shorter)
        min_len = min(len(steps), len(image_descriptions))
        steps = steps[:min_len]
        image_descriptions = image_descriptions[:min_len]

        print(f"Generated {len(steps)} steps")
        pd.DataFrame({"text_plans": steps, "image descriptions": image_descriptions}).to_csv(f"{output_folder}/text_plans.csv", index=False)
        return image_descriptions

if __name__ == "__main__":
    ip = InstructionPlanner(COMPASS_API_KEY)
    plan = ip.generate_text_plan("How to cook a fried egg?", output_folder="./output/how_to_cook_a_fried_egg")
    print(plan)

# 2. Generate Image Plan
    image_prompts = ip.generate_image_plan(plan)
    print("\n--- Image Prompts ---")
    for i, prompt in enumerate(image_prompts):
        print(f"PROMPT {i+1}: {prompt}")

