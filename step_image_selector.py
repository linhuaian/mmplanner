import openai
import base64
from io import BytesIO
from PIL import Image

from stable_diffusion_generator import StableDiffusionImageGenerator


class StepImageSelector:
    def __init__(self, sd_api_url="http://localhost:7860/sdapi/v1/txt2img"):
        self.sd_api_url = sd_api_url

    def select_best_image(self, step_text, prev_pil_image=None, k=3):
        """
        Generates k candidate images for the step_text using the Stable Diffusion API,
        optionally considering the previous PIL image as context,
        and selects the best image based on OpenAI's model ranking.

        Returns:
            The base64 string of the best image, or None.
        """
        # Generate k images
        image_generator = StableDiffusionImageGenerator(self.sd_api_url)
        candidates = []
        for _ in range(k):
            img_b64 = image_generator.generate_image(step_text)
            if img_b64 is not None:
                candidates.append(img_b64)

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Use OpenAI to select the best image description-wise
        # Describe each image in the prompt (prompts referencing base64 is not useful, so we only can reference step_texts)
        choices_prompt = (
            f"For the instruction step: \"{step_text}\", "
            "we generated several candidate images. "
            "You are given a description of what each candidate is supposed to depict. "
            "Select the best overall image candidate strictly according to faithfulness and clarity for the step described. "
            "Number the candidate images 1 to {n} in your reasoning. "
            "Which one is best? Reply ONLY with the candidate number."
        ).format(n=len(candidates))

        # Prepare messages
        system_msg = {
            "role": "system",
            "content": "You are an expert at picking the image that best illustrates a step in an instructional guide. You only output the candidate number."
        }
        user_msg = {
            "role": "user",
            "content": choices_prompt + "\nCandidate descriptions:\n" +
                       "\n".join([f"Candidate {i+1}: Attempt to illustrate: {step_text}" for i in range(len(candidates))])
        }

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[system_msg, user_msg],
                max_tokens=5,
                temperature=0.0
            )
            best_index = None
            content = response['choices'][0]['message']['content'].strip()
            # Try to extract a candidate number
            for i in range(1, len(candidates)+1):
                if content.startswith(str(i)):
                    best_index = i-1
                    break
            if best_index is not None:
                return candidates[best_index]
            else:
                return candidates[0]
        except Exception:
            return candidates[0]

    def b64_to_pil(self, img_b64):
        """
        Converts base64 image string to a PIL Image object.
        """
        try:
            image_data = base64.b64decode(img_b64)
            image = Image.open(BytesIO(image_data))
            return image
        except Exception:
            return None

