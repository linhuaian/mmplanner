import requests


class StableDiffusionImageGenerator:
    def __init__(self, sd_api_url="http://localhost:7860/sdapi/v1/txt2img"):
        self.sd_api_url = sd_api_url

    def generate_image(self, prompt, steps=30, width=512, height=512):
        sd_payload = {
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height
        }
        try:
            sd_response = requests.post(self.sd_api_url, json=sd_payload)
            sd_response.raise_for_status()
            img_data = sd_response.json().get("images", [None])[0]
            return img_data
        except Exception:
            return None

