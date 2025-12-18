import openai
import base64
from io import BytesIO
from PIL import Image
from gme_inference import GmeQwen2VL
import torch 

def cosine_similarity(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    return torch.nn.functional.cosine_similarity(a, b, dim=1).item()



class StepImageSelector:
    def __init__(self):
        self.qwen_model = GmeQwen2VL('Alibaba-NLP/gme-Qwen2-VL-2B-Instruct')
        self.prompt = "describe this image in terms of visual appearance, functions of the items in the image and general colour scheme of the image."

    def select_best_image(self, prev_img, images):
        """
        Generates k candidate images for the step_text using the Stable Diffusion API,
        optionally considering the previous PIL image as context,
        and selects the best image based on OpenAI's model ranking.

        Returns:
            The base64 string of the best image, or None.
        """
        # Generate k images
        prev_embedding = self.qwen_model.embed([self.prompt], [prev_img])
        cur_embedding = self.qwen_model.embed([self.prompt] * len(images), images)

        best_score = 0
        best_img = images[0]
        for cur in cur_embedding: 
            score = cosine_similarity(prev_embedding[0], cur_embedding)
            if score >= best_score: 
                best_score = score 
                best_img = cur 
        return best_img 
            
       

