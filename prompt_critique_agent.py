from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, Optional

from PIL import Image
from dotenv import load_dotenv

load_dotenv()

COMPASS_API_KEY = os.getenv("COMPASS_API_KEY")
COMPASS_BASE_URL = os.getenv("COMPASS_BASE_URL", "https://compass.llm.shopee.io/compass-api/v1")
DEFAULT_VISION_MODEL = os.getenv("GT_GPT_MODEL", "gpt-4o")


class ImagePromptCritiqueAgent:
    """
    Feedback-loop agent: critique whether an image matches the prompt, then suggest a revised prompt.
    Intended for iterative SD generation: v1 -> critique -> v2 -> critique -> v3.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_VISION_MODEL):
        self.api_key = api_key or COMPASS_API_KEY
        if not self.api_key:
            raise RuntimeError("Missing COMPASS_API_KEY; required for ImagePromptCritiqueAgent.")
        self.model = model

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=COMPASS_BASE_URL)

    @staticmethod
    def _image_to_data_url(img: Image.Image, *, format: str = "PNG") -> str:
        buf = BytesIO()
        img.save(buf, format=format)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64}"

    def critique_and_rewrite_prompt(
        self,
        *,
        image: Image.Image,
        current_prompt: str,
        max_words: int = 20,
    ) -> Dict[str, Any]:
        """
        Returns dict:
          {
            "ok": bool,                     # True if prompt already matches well enough
            "issues": ["..."],              # short list
            "revised_prompt": "..."         # short, visual, noun-phrase-ish prompt
          }
        """
        data_url = self._image_to_data_url(image)

        prompt = f"""
You are a strict image-prompt critic for a text-to-image model.

Given:
- an image
- the current prompt (which should describe ONLY what is visible)

Task:
1) Decide if the image matches the prompt well enough (OK=true/false).
2) If not OK, rewrite the prompt to better match what is ACTUALLY visible in the image.

Rules for revised_prompt:
- MUST be short (<= {max_words} words).
- MUST describe only visible items/states (no camera/style words).
- Prefer a compact comma-separated list of visual nouns/adjectives.

Return JSON ONLY:
{{
  "ok": true/false,
  "issues": ["short issue", "..."],
  "revised_prompt": "..."
}}
Current prompt:
{current_prompt}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            extra_headers={"Provider": "OpenAI"},
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=250,
        )

        return json.loads(resp.choices[0].message.content or "{}")

