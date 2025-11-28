from openai import OpenAI
import pandas as pd 
import json 

class InstructionPlanner:
    def __init__(self, api_key):

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
            
        

    def generate_text_plan(self, instruction, min_steps=5, max_steps=10):
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
                    model="gpt-4o",
                    temperature=0.01,
                    extra_headers={
                        "Provider": "OpenAI"
                    }
                )
        content = chat_completion.choices[0].message.content
        steps = [line.strip() for line in content.strip().split('\n') if line.strip()]
        image_prompts = self.generate_image_plan(steps)
        pd.DataFrame({"text_plans": steps, "image descriptions": image_prompts}).to_csv("output/text_plans.csv")
        return image_prompts

if __name__ == "__main__":
    ip = InstructionPlanner("56d5b8ebf4d5ec1fd1731af1cfa971be9cf711c0a0a366b01b42971937e6bf69")
    plan = ip.generate_text_plan("How to cook a fried egg?")
    print(plan)

# 2. Generate Image Plan
    image_prompts = ip.generate_image_plan(plan)
    print("\n--- Image Prompts ---")
    for i, prompt in enumerate(image_prompts):
        print(f"PROMPT {i+1}: {prompt}")

