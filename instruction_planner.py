import openai

from step_image_selector import StepImageSelector


class InstructionPlanner:
    def __init__(self, api_key, sd_api_url="http://localhost:7860/sdapi/v1/txt2img"):
        openai.api_key = api_key
        self.sd_api_url = sd_api_url

    def generate_text_plan(self, instruction, min_steps=5, max_steps=10):
        prompt = (
            f"Break down the instruction \"{instruction}\" into a step-by-step plan. "
            f"Generate between {min_steps} and {max_steps} clear, concise steps."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5
        )
        content = response['choices'][0]['message']['content']
        steps = [line.strip() for line in content.strip().split('\n') if line.strip()]

        image_selector = StepImageSelector(self.sd_api_url)
        step_images = []
        prev_pil_image = None
        k = 3  # number of images to generate per step
        for step in steps:
            best_img_b64 = image_selector.select_best_image(step, prev_pil_image, k=k)
            if best_img_b64:
                best_img_pil = image_selector.b64_to_pil(best_img_b64)
            else:
                best_img_pil = None
            step_images.append((step, best_img_b64))
            prev_pil_image = best_img_pil

        return step_images

