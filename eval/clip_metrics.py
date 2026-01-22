"""
CLIP-based State Consistency Evaluator

Goal
----
Evaluate whether each generated step image matches the expected *semantic object states*
for that step (e.g., doneness/color/shape/props present), and therefore whether the
sequence is state-consistent across steps.

Approach
--------
1) Generate or reuse per-task ground-truth state descriptions with GPT, primarily from
   `text_plans.csv` (when present). The ground-truth is stored in:
     output/<task_name>/ground_truth.txt
   Format:
     step1: ...
     step2: ...
2) Embed ground-truth text using CLIP text encoder.
3) Embed generated step images using the same CLIP image encoder.
4) Compute cosine similarity per step (text vs image). Higher is better.

Notes
-----
- Uses the same COMPASS/OpenAI client style as the rest of this repo (see `agent.py` and
  `eval/gpt_metrics.py`).
- "Medium" CLIP model default: `openai/clip-vit-base-patch16`.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


# =========================
# Config
# =========================

DEFAULT_CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch16")
DEFAULT_GPT_MODEL = os.getenv("GT_GPT_MODEL", "gpt-4o")
COMPASS_API_KEY = os.getenv("COMPASS_API_KEY")
COMPASS_BASE_URL = os.getenv("COMPASS_BASE_URL", "https://compass.llm.shopee.io/compass-api/v1")


# =========================
# Utilities
# =========================

_STEP_LINE_RE = re.compile(r"^\s*step_?\s*(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def _task_title_from_folder(task_folder: str) -> str:
    return Path(task_folder).name.replace("_", " ").strip()


def _find_step_images(task_folder: str) -> List[Path]:
    """Returns step_0.png, step_1.png ... (excluding intermediate/)."""
    task_path = Path(task_folder)
    images = [p for p in task_path.glob("step_*.png") if "intermediate" not in str(p)]

    def _step_idx(p: Path) -> int:
        m = re.search(r"step_(\d+)\.png$", p.name)
        return int(m.group(1)) if m else 0

    return sorted(images, key=_step_idx)


def _find_intermediate_candidates(task_folder: str, step_idx_zero_based: int) -> List[Path]:
    """
    Returns intermediate candidates like:
      output/<task>/intermediate/step_{i}_{j}.png
    for a given step index i (0-based).
    """
    inter = Path(task_folder) / "intermediate"
    if not inter.exists():
        return []

    pattern = f"step_{step_idx_zero_based}_*.png"
    candidates = list(inter.glob(pattern))

    def _cand_j(p: Path) -> int:
        m = re.search(rf"step_{step_idx_zero_based}_(\d+)\.png$", p.name)
        return int(m.group(1)) if m else 0

    return sorted(candidates, key=_cand_j)


def _safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _cosine_sim(a, b) -> float:
    # a,b: torch tensors already normalized to unit norm
    return float((a * b).sum().item())


def _pixel_hash(image_path: Path, *, size: int = 64) -> str:
    """
    Robust image-content hash (ignores PNG metadata) by hashing resized RGB pixels.
    Used to match `step_i.png` back to the exact intermediate candidate file.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((size, size))
    return hashlib.sha1(img.tobytes()).hexdigest()


# =========================
# Text plan loading
# =========================

def load_text_plans(task_folder: str) -> Optional[List[str]]:
    """
    Load steps from `text_plans.csv` if present.

    The repo has CSVs with a leading unnamed index column, e.g.:
      ,text_plans,image descriptions
      0,"1. ...","..."
    We prefer a column whose header contains "text" and "plan".
    """
    csv_path = Path(task_folder) / "text_plans.csv"
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None

        # Choose the best column for step text
        fieldnames = [fn or "" for fn in reader.fieldnames]
        preferred = None
        for fn in fieldnames:
            low = fn.lower()
            if "text" in low and "plan" in low:
                preferred = fn
                break

        if preferred is None:
            # fall back to first non-empty, non-index-ish column
            candidates = [fn for fn in fieldnames if fn.strip() and fn.strip().lower() not in {"", "index"}]
            preferred = candidates[0] if candidates else fieldnames[0]

        steps: List[str] = []
        for row in reader:
            val = (row.get(preferred) or "").strip()
            if val:
                steps.append(val)

        return steps or None


# =========================
# Ground-truth generation / reuse
# =========================

@dataclass(frozen=True)
class GroundTruth:
    # step index (base depends on file) -> text (no "stepN:" prefix)
    by_step: Dict[int, str]

    def to_lines(self, num_steps: int, *, index_base: int = 1) -> List[str]:
        lines = []
        for i in range(num_steps):
            step = index_base + i
            txt = self.by_step.get(step, "").strip()
            lines.append(f"step{step}: {txt}")
        return lines


def parse_ground_truth(text: str) -> GroundTruth:
    by_step: Dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _STEP_LINE_RE.match(line)
        if not m:
            continue
        step = int(m.group(1))
        by_step[step] = m.group(2).strip()
    return GroundTruth(by_step=by_step)


class GroundTruthManager:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_GPT_MODEL):
        self.api_key = api_key or COMPASS_API_KEY
        self.model = model

        if not self.api_key:
            raise RuntimeError(
                "Missing COMPASS_API_KEY in environment. "
                "Set it in your .env or shell to enable GPT ground-truth generation."
            )

        # Import here so users can run purely-from-existing-ground-truth without openai installed.
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=COMPASS_BASE_URL)

    def ensure_ground_truth(
        self,
        task_folder: str,
        num_steps: int,
        text_plans: Optional[List[str]],
        force_regen: bool = False,
        *,
        index_base: int = 1,
    ) -> GroundTruth:
        gt_path = Path(task_folder) / "ground_truth.txt"
        if not force_regen:
            existing = _safe_read_text(gt_path)
            if existing:
                parsed = parse_ground_truth(existing)
                normalized = self._normalize_ground_truth(parsed, num_steps=num_steps, index_base=index_base)
                if normalized.by_step:
                    return normalized

        gt = self._generate_ground_truth(task_folder, num_steps, text_plans, index_base=index_base)
        normalized = self._normalize_ground_truth(gt, num_steps=num_steps, index_base=index_base)
        gt_path.write_text("\n".join(normalized.to_lines(num_steps, index_base=index_base)) + "\n", encoding="utf-8")
        return normalized

    def _generate_ground_truth(
        self,
        task_folder: str,
        num_steps: int,
        text_plans: Optional[List[str]],
        *,
        index_base: int = 1,
    ) -> GroundTruth:
        task_title = _task_title_from_folder(task_folder)
        start = index_base
        end = index_base + num_steps - 1

        step_mapping_note = (
            f"IMPORTANT: step{start} corresponds to image step_0.png, step{start+1} to step_1.png, etc."
            if index_base in (0, 1)
            else "IMPORTANT: step indices correspond in order to step_0.png, step_1.png, ..."
        )

        steps_context = ""
        if text_plans:
            steps_context = "\n".join([f"{i+1}. {s}" for i, s in enumerate(text_plans)])
        else:
            steps_context = "(No text_plans.csv found. Infer from the task name and typical procedure.)"

        prompt = f"""
You are generating GROUND TRUTH semantic state descriptions for a sequence of step images.

Task: "{task_title}"
Number of steps/images: {num_steps}
{step_mapping_note}

Reference text plans (if present):
{steps_context}

Requirements:
- Output EXACTLY {num_steps} lines.
- Each line MUST follow this exact format: "stepN: <description>"
  where N is {start}..{end}.
- The description should be visual and state-based: include object identity and state
  attributes such as color, doneness, shape, position, and key props present.
- Track persistence: if an object is changed in step k (e.g., cooked/broken/installed),
  later steps MUST preserve that changed state unless the plan explicitly changes it again.
- Keep each line 1-2 sentences. Avoid fluff. Be concrete.

Example format:
step1: the egg is raw in a bowl; a nonstick pan is empty on the stove.
step2: the egg is cracked into the pan; the yolk is intact; the pan is on medium heat.
""".strip()

        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.1,
            max_tokens=1200,
            extra_headers={"Provider": "OpenAI"},
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = parse_ground_truth(text)

        # If GPT didn't follow the format perfectly, patch it up minimally by re-prompting once.
        if len(parsed.by_step) != num_steps:
            reprompt = f"""
Your previous output did not match the required format.
Return EXACTLY {num_steps} lines, numbered step{start}..step{end}, following the exact "stepN: ..." format.

Here was your previous attempt:
{text}
""".strip()
            resp2 = self.client.chat.completions.create(
                messages=[{"role": "user", "content": reprompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=1200,
                extra_headers={"Provider": "OpenAI"},
            )
            text2 = (resp2.choices[0].message.content or "").strip()
            parsed = parse_ground_truth(text2)

        return parsed

    def _normalize_ground_truth(self, gt: GroundTruth, *, num_steps: int, index_base: int) -> GroundTruth:
        """
        Ensure GT keys match the requested index base.
        Supports reusing existing files that are 0-based or 1-based by shifting.
        """
        expected = [index_base + i for i in range(num_steps)]
        if all(k in gt.by_step for k in expected):
            by_step = {k: (gt.by_step.get(k, "") or "").strip() for k in expected}
            return GroundTruth(by_step=by_step)

        # Try the opposite base (common case)
        alt_base = 0 if index_base == 1 else 1
        alt_expected = [alt_base + i for i in range(num_steps)]
        if all(k in gt.by_step for k in alt_expected):
            shift = index_base - alt_base
            by_step = {}
            for k in alt_expected:
                by_step[k + shift] = (gt.by_step.get(k, "") or "").strip()
            return GroundTruth(by_step=by_step)

        # Fallback: keep what we can, fill missing.
        by_step = {}
        for k in expected:
            by_step[k] = (gt.by_step.get(k, "") or "").strip()
        return GroundTruth(by_step=by_step)


class ObjectPhraseManager:
    """
    Extract per-step object-state phrases from ground-truth lines using GPT.

    Stored per task at:
      output/<task>/ground_truth_object_phrases_text.json
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_GPT_MODEL):
        self.api_key = api_key or COMPASS_API_KEY
        self.model = model
        if not self.api_key:
            raise RuntimeError(
                "Missing COMPASS_API_KEY in environment. "
                "Set it in your .env or shell to enable GPT phrase extraction."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=COMPASS_BASE_URL)

    def ensure_phrases(
        self,
        task_folder: str,
        *,
        gt_lines: List[str],
        step_numbers: List[int],
        num_phrases: int = 5,
        force_regen: bool = False,
        cache_filename: str = "ground_truth_object_phrases_text.json",
    ) -> Dict[int, List[str]]:
        """
        Returns mapping: step_number -> list[str] (length == num_phrases).
        """
        cache_path = Path(task_folder) / cache_filename
        if not force_regen and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                by_step_raw = cached.get("by_step", {})
                by_step: Dict[int, List[str]] = {}
                for k, v in by_step_raw.items():
                    phrases_norm = [self._normalize_phrase(str(x)) for x in (v or [])]
                    phrases_norm = [p for p in phrases_norm if p and self._is_valid_phrase(p)]
                    by_step[int(k)] = phrases_norm[:num_phrases]
                # Accept cache only if every step has >=1 and <=N valid normalized phrases
                if all(
                    (s in by_step and isinstance(by_step[s], list) and 1 <= len(by_step[s]) <= num_phrases)
                    for s in step_numbers
                ):
                    # Rewrite normalized cache to disk (idempotent)
                    cache_path.write_text(
                        json.dumps({"num_phrases": num_phrases, "by_step": {str(k): v for k, v in by_step.items()}}, indent=2),
                        encoding="utf-8",
                    )
                    return by_step
            except Exception:
                pass

        by_step: Dict[int, List[str]] = {}
        for step, gt in zip(step_numbers, gt_lines, strict=False):
            phrases = self._extract_phrases_for_step(gt, num_phrases=num_phrases)
            phrases_norm = [self._normalize_phrase(p) for p in phrases]
            phrases_norm = [p for p in phrases_norm if p and self._is_valid_phrase(p)]
            # Keep at most N; allow fewer than N.
            by_step[int(step)] = phrases_norm[:num_phrases]

        cache_path.write_text(
            json.dumps(
                {
                    "num_phrases": num_phrases,
                    "by_step": {str(k): v for k, v in by_step.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return by_step


def _load_text_derived_phrases(task_folder: str) -> Optional[Dict[int, List[str]]]:
    """
    Prefer text-derived expected phrases saved by TextObjectStateAgent at:
      output/<task>/ground_truth_object_phrases_text.json (source=text)
    """
    path = Path(task_folder) / "ground_truth_object_phrases_text.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(data.get("source", "")).strip().lower() not in {"text", ""}:
            return None
        by_step_raw = data.get("by_step", {}) or {}
        by_step: Dict[int, List[str]] = {}
        for k, v in by_step_raw.items():
            by_step[int(k)] = [str(x).strip() for x in (v or []) if str(x).strip()]
        return by_step
    except Exception:
        return None

    def _extract_phrases_for_step(self, ground_truth_line: str, *, num_phrases: int) -> List[str]:
        prompt = f"""
You are extracting object-state phrases for CLIP scoring.

Given this step description:
{ground_truth_line}

Task:
- Extract EXACTLY {num_phrases} short phrases.
- Each phrase MUST contain:
  (1) at least ONE object noun (e.g., egg, pan, stove, spatula, bowl), AND
  (2) at least ONE descriptive/state word (e.g., raw/cooked/intact/shimmering/foaming/hot/cold/empty/full/metal/wooden).
- Prefer the pattern: "<descriptor> <object>" (e.g., "shimmering oil", "intact yolk", "empty nonstick pan").
- Keep each phrase 2-8 words, visually grounded, and non-overlapping.
- Output JSON ONLY with this exact schema:
  {{"phrases": ["a raw egg", "a nonstick pan on a stove", "..."]}}
""".strip()

        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.0,
            max_tokens=400,
            extra_headers={"Provider": "OpenAI"},
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        phrases = data.get("phrases", [])
        if not isinstance(phrases, list):
            return []
        raw = [str(p).strip() for p in phrases if str(p).strip()]
        normalized = [self._normalize_phrase(p) for p in raw]
        normalized = [p for p in normalized if p]
        if len(normalized) >= num_phrases:
            return normalized[:num_phrases]

        # One retry with stricter formatting if needed
        reprompt = f"""
Your output must contain EXACTLY {num_phrases} phrases. Each phrase MUST contain at least one descriptive/state word and at least one object noun.
Return JSON ONLY: {{"phrases": ["<descriptor> <object>", "..."]}}

Step description:
{ground_truth_line}
""".strip()
        resp2 = self.client.chat.completions.create(
            messages=[{"role": "user", "content": reprompt}],
            model=self.model,
            temperature=0.0,
            max_tokens=400,
            extra_headers={"Provider": "OpenAI"},
            response_format={"type": "json_object"},
        )
        data2 = json.loads(resp2.choices[0].message.content or "{}")
        phrases2 = data2.get("phrases", [])
        if not isinstance(phrases2, list):
            return normalized
        raw2 = [str(p).strip() for p in phrases2 if str(p).strip()]
        norm2 = [self._normalize_phrase(p) for p in raw2]
        norm2 = [p for p in norm2 if p]
        return (norm2 + normalized)[:num_phrases]

    def _normalize_phrase(self, phrase: str) -> str:
        p = (phrase or "").strip().strip(".;,:")
        if not p:
            return ""
        # Remove duplicate spaces
        p = re.sub(r"\s+", " ", p)
        # Ensure at least 2 content tokens (excluding articles)
        tokens = p.split()
        content = [t.lower() for t in tokens if t.lower() not in {"a", "an", "the"}]
        if len(content) < 2:
            return ""
        # If no obvious descriptor word, prefix a safe descriptor
        if not any(t in self._descriptor_tokens() for t in content):
            p = f"visible {p}"
        return p

    def _is_valid_phrase(self, phrase: str) -> bool:
        p = (phrase or "").strip()
        if not p:
            return False
        tokens = [t.lower() for t in re.split(r"\s+", p) if t]
        content = [t for t in tokens if t not in {"a", "an", "the"}]
        if len(content) < 2:
            return False
        desc = self._descriptor_tokens()
        has_desc = any(t in desc for t in content)
        has_obj = any(t not in desc for t in content)
        return has_desc and has_obj

    @staticmethod
    def _descriptor_tokens() -> set[str]:
        # Heuristic list of state/visual descriptor tokens.
        return {
            # cooking/state
            "raw","cooked","uncooked","runny","intact","broken","cracked","boiling","simmering","shimmering","foaming",
            "hot","warm","cold","cool","steaming","smoking","melted","burnt","crispy","browned","golden","set","fully",
            # presence/quantity
            "empty","full","filled","open","closed","covered","uncovered","nearby","visible","missing",
            "small","large","tiny","big","thick","thin",
            # materials/colors
            "metal","wooden","plastic","glass","nonstick","stainless","black","white","red","green","blue","yellow","brown",
            # positioning (often helps as descriptors)
            "on","in","inside","over","under","above","below","next","beside","around","with","without",
        }


# =========================
# CLIP evaluator
# =========================

class CLIPStateConsistencyEvaluator:
    def __init__(
        self,
        model_id: str = DEFAULT_CLIP_MODEL_ID,
        device: Optional[str] = None,
        *,
        local_files_only: bool = False,
        step_index_base: int = 1,
    ):
        try:
            import torch
        except Exception as e:
            raise RuntimeError("PyTorch is required for CLIP evaluation. Install torch first.") from e

        # Avoid TensorFlow import (some TF builds crash/print AVX warnings on older CPUs).
        # Must be set before importing `transformers`.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

        # Environment patch:
        # Some dependency stacks (accelerate/safetensors/torchvision/transformers) reference unsigned
        # dtypes (uint16/uint32/uint64), while many torch builds only expose uint8. Safetensors uses
        # these as dtype keys; mapping to closest signed integer dtypes is sufficient to avoid import
        # failures in this repo's evaluation use-cases.
        _unsigned_dtype_fallbacks = {
            "uint16": torch.int16,
            "uint32": torch.int32,
            "uint64": torch.int64,
        }
        for _name, _fallback in _unsigned_dtype_fallbacks.items():
            if not hasattr(torch, _name):
                setattr(torch, _name, _fallback)  # type: ignore[attr-defined]

        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            raise RuntimeError("transformers is required for CLIP evaluation. Install transformers first.") from e

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.step_index_base = step_index_base

        print(f"[CLIP] Loading processor/model: {model_id} (device={self.device}, local_files_only={local_files_only})")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_id, local_files_only=local_files_only)
            self.model = CLIPModel.from_pretrained(model_id, local_files_only=local_files_only).to(self.device)
        except Exception as e:
            if local_files_only:
                raise RuntimeError(
                    "CLIP model files not found in local HuggingFace cache, but --local_files_only was set. "
                    "Re-run without --local_files_only once to download the model."
                ) from e
            raise
        self.model.eval()

    def embed_text(self, texts: List[str]):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats

    def embed_images(self, image_paths: List[Path]):
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats

    def score_task(
        self,
        task_folder: str,
        gt: GroundTruth,
        *,
        phrases_by_step: Optional[Dict[int, List[str]]] = None,
    ) -> Dict:
        image_paths = _find_step_images(task_folder)
        if not image_paths:
            raise FileNotFoundError(f"No step_*.png images found in {task_folder}")

        num_steps = len(image_paths)
        img_emb = self.embed_images(image_paths)  # [N, D]

        scores: List[float] = []
        per_step_phrase_scores: List[List[float]] = []
        scoring_mode = "gt_line"

        if phrases_by_step is None:
            gt_texts = [gt.by_step.get(self.step_index_base + i, "") for i in range(num_steps)]
            text_emb = self.embed_text(gt_texts)  # [N, D]
            for i in range(num_steps):
                scores.append(_cosine_sim(text_emb[i], img_emb[i]))
            per_step_phrase_scores = [[] for _ in range(num_steps)]
        else:
            scoring_mode = "gt_object_phrases"
            for i in range(num_steps):
                step_num = self.step_index_base + i
                phrases = [(p or "").strip() for p in (phrases_by_step.get(step_num, []) or []) if (p or "").strip()]
                phrases = phrases[:]
                # Average over available phrases (can be < 5)
                emb_scores: List[float] = []
                if phrases:
                    phrase_emb = self.embed_text(phrases)
                    emb_scores = [_cosine_sim(e, img_emb[i]) for e in phrase_emb]
                step_score = sum(emb_scores) / max(len(emb_scores), 1)
                scores.append(step_score)
                per_step_phrase_scores.append(emb_scores)

        avg = sum(scores) / max(len(scores), 1)
        return {
            "task_name": Path(task_folder).name,
            "task_folder": str(Path(task_folder).resolve()),
            "clip_model_id": self.model_id,
            "device": self.device,
            "step_index_base": self.step_index_base,
            "scoring_mode": scoring_mode,
            "num_steps": num_steps,
            "per_step_scores": scores,
            "per_step_phrase_scores": per_step_phrase_scores,
            "average_score": avg,
            "ground_truth_lines": gt.to_lines(num_steps, index_base=self.step_index_base),
            "image_files": [p.name for p in image_paths],
        }

    def score_intermediate_candidates(
        self,
        task_folder: str,
        gt: GroundTruth,
        *,
        phrases_by_step: Optional[Dict[int, List[str]]] = None,
    ) -> Dict:
        """
        For each step i, score every intermediate candidate image `intermediate/step_i_*.png`
        against the step's ground-truth text. Also record the chosen final `step_i.png` score.
        """
        final_images = _find_step_images(task_folder)
        if not final_images:
            raise FileNotFoundError(f"No step_*.png images found in {task_folder}")

        num_steps = len(final_images)
        scoring_mode = "gt_line" if phrases_by_step is None else "gt_object_phrases"
        text_emb = None
        if phrases_by_step is None:
            gt_texts = [gt.by_step.get(self.step_index_base + i, "") for i in range(num_steps)]
            text_emb = self.embed_text(gt_texts)  # [N, D]

        per_step: List[Dict] = []
        for i in range(num_steps):
            candidates = _find_intermediate_candidates(task_folder, i)
            chosen_path = final_images[i]

            # Try to recover which exact intermediate image was selected as the final `step_i.png`
            chosen_candidate: Optional[str] = None
            try:
                chosen_h = _pixel_hash(chosen_path)
                for p in candidates:
                    if _pixel_hash(p) == chosen_h:
                        chosen_candidate = p.name
                        break
            except Exception:
                chosen_candidate = None

            chosen_emb = self.embed_images([chosen_path])[0]
            step_num = self.step_index_base + i
            if phrases_by_step is None:
                chosen_score = _cosine_sim(text_emb[i], chosen_emb)  # type: ignore[index]
                chosen_phrase_scores: List[float] = []
                phrases: List[str] = []
            else:
                phrases = [(p or "").strip() for p in (phrases_by_step.get(step_num, []) or []) if (p or "").strip()]
                if phrases:
                    emb = self.embed_text(phrases)
                    chosen_phrase_scores = [_cosine_sim(e, chosen_emb) for e in emb]
                    chosen_score = sum(chosen_phrase_scores) / len(chosen_phrase_scores)
                else:
                    chosen_phrase_scores = []
                    chosen_score = 0.0

            cand_scores: List[Tuple[str, float]] = []
            cand_phrase_scores: Dict[str, List[float]] = {}
            best = None
            if candidates:
                cand_embs = self.embed_images(candidates)  # [K, D]
                for j, p in enumerate(candidates):
                    if phrases_by_step is None:
                        s = _cosine_sim(text_emb[i], cand_embs[j])  # type: ignore[index]
                        cand_phrase_scores[p.name] = []
                    else:
                        if phrases:
                            # Reuse the same phrase embeddings for all candidates within this step
                            emb = self.embed_text(phrases)
                            per_phrase = [_cosine_sim(e, cand_embs[j]) for e in emb]
                            cand_phrase_scores[p.name] = per_phrase
                            s = sum(per_phrase) / len(per_phrase)
                        else:
                            cand_phrase_scores[p.name] = []
                            s = 0.0
                    cand_scores.append((p.name, s))
                best = max(cand_scores, key=lambda x: x[1])

            per_step.append(
                {
                    "step": self.step_index_base + i,
                    "image_file": chosen_path.name,
                    "ground_truth": gt.to_lines(num_steps, index_base=self.step_index_base)[i],
                    "phrases": phrases,
                    "chosen_image": chosen_path.name,
                    "chosen_candidate": chosen_candidate,
                    "chosen_score": chosen_score,
                    "chosen_phrase_scores": chosen_phrase_scores,
                    "num_candidates": len(candidates),
                    "best_candidate": best[0] if best else None,
                    "best_score": best[1] if best else None,
                    "candidate_scores": [
                        {"image": n, "score": s, "phrase_scores": cand_phrase_scores.get(n, [])} for (n, s) in cand_scores
                    ],
                }
            )

        return {
            "task_name": Path(task_folder).name,
            "task_folder": str(Path(task_folder).resolve()),
            "clip_model_id": self.model_id,
            "device": self.device,
            "step_index_base": self.step_index_base,
            "scoring_mode": scoring_mode,
            "num_steps": num_steps,
            "per_step": per_step,
        }


# =========================
# Reporting
# =========================

def _write_task_report(task_folder: str, result: Dict) -> None:
    task_dir = Path(task_folder)
    md_path = task_dir / "clip_state_consistency_report.md"
    csv_path = task_dir / "clip_state_consistency_scores.csv"
    json_path = task_dir / "clip_state_consistency_result.json"

    # CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "image_file", "cosine_similarity", "ground_truth"])
        for i, s in enumerate(result["per_step_scores"], start=1):
            w.writerow([i, result["image_files"][i - 1], f"{s:.6f}", result["ground_truth_lines"][i - 1]])

    # JSON (for programmatic use)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Markdown
    title = result["task_name"].replace("_", " ").title()
    lines = []
    lines.append(f"# CLIP State Consistency Report\n")
    lines.append(f"## Task: {title}\n")
    lines.append(f"- **CLIP model**: `{result['clip_model_id']}`")
    lines.append(f"- **Device**: `{result['device']}`")
    lines.append(f"- **Average similarity**: **{result['average_score']:.4f}**")
    lines.append("")
    lines.append("## Per-step scores\n")
    lines.append("| Step | Image | Cosine similarity | Ground truth |")
    lines.append("|------|-------|-------------------|--------------|")
    base = int(result.get("step_index_base", 1))
    for i, s in enumerate(result["per_step_scores"]):
        step_label = base + i
        img = result["image_files"][i]
        gt = result["ground_truth_lines"][i].replace("|", "\\|")
        lines.append(f"| {step_label} (image `{img}`) | `{img}` | {s:.4f} | {gt} |")
    lines.append("")
    lines.append(f"- Saved files: `{md_path.name}`, `{csv_path.name}`, `{json_path.name}`")
    # Defensive cleanup: we've observed rare stray trailing fragments like "json`" in some environments.
    lines = [ln for ln in lines if ln.strip() != "json`"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_intermediate_report(task_folder: str, result: Dict) -> None:
    task_dir = Path(task_folder)
    md_path = task_dir / "clip_intermediate_candidate_report.md"
    csv_path = task_dir / "clip_intermediate_candidate_scores.csv"
    all_csv_path = task_dir / "clip_intermediate_candidate_all_scores.csv"
    json_path = task_dir / "clip_intermediate_candidate_result.json"

    # JSON
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # CSV (one row per step)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "chosen_image",
                "chosen_score",
                "num_candidates",
                "best_candidate",
                "best_score",
                "delta_best_minus_chosen",
                "ground_truth",
            ]
        )
        for row in result["per_step"]:
            best_score = row["best_score"]
            chosen_score = row["chosen_score"]
            delta = (best_score - chosen_score) if (best_score is not None) else ""
            w.writerow(
                [
                    row["step"],
                    row["chosen_image"],
                    f"{chosen_score:.6f}",
                    row["num_candidates"],
                    row["best_candidate"] or "",
                    f"{best_score:.6f}" if best_score is not None else "",
                    f"{delta:.6f}" if isinstance(delta, float) else "",
                    row["ground_truth"],
                ]
            )

    # CSV (one row per candidate image)
    with all_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "candidate_image", "score", "is_chosen", "is_best"])
        for row in result["per_step"]:
            step = row["step"]
            chosen = row["chosen_image"]
            best = row["best_candidate"]
            for cand in row.get("candidate_scores", []):
                img = cand["image"]
                score = cand["score"]
                w.writerow([step, img, f"{score:.6f}", str(img == chosen), str(best is not None and img == best)])

    # Markdown summary
    title = result["task_name"].replace("_", " ").title()
    lines = []
    lines.append("# CLIP Intermediate Candidate Report\n")
    lines.append(f"## Task: {title}\n")
    lines.append(f"- **CLIP model**: `{result['clip_model_id']}`")
    lines.append(f"- **Device**: `{result['device']}`")
    lines.append("")
    lines.append("## Best intermediate candidate per step\n")
    lines.append("| Step | Chosen | Chosen score | #cands | Best cand | Best score | Δ(best-chosen) |")
    lines.append("|------|--------|-------------|--------|----------|-----------|----------------|")
    for row in result["per_step"]:
        best_score = row["best_score"]
        chosen_score = row["chosen_score"]
        delta = (best_score - chosen_score) if (best_score is not None) else None
        step_display = f"{row['step']} (image {row.get('image_file', '')})"
        lines.append(
            "| {step} | `{chosen}` | {cs:.4f} | {k} | `{best}` | {bs} | {d} |".format(
                step=step_display,
                chosen=row["chosen_image"],
                cs=chosen_score,
                k=row["num_candidates"],
                best=row["best_candidate"] or "",
                bs=f"{best_score:.4f}" if best_score is not None else "",
                d=f"{delta:.4f}" if delta is not None else "",
            )
        )
    lines.append("")
    lines.append("## All candidate scores per step\n")
    lines.append("")
    for row in result["per_step"]:
        step = row["step"]
        chosen = row["chosen_image"]
        best = row["best_candidate"]
        lines.append(f"### Step {step}\n")
        lines.append(f"- Ground truth: {row['ground_truth']}")
        lines.append("")
        if not row.get("candidate_scores"):
            lines.append("- No intermediate candidates found for this step.")
            lines.append("")
            continue
        lines.append("| Candidate image | Score | Chosen? | Best? |")
        lines.append("|-----------------|-------|--------|-------|")
        for cand in row["candidate_scores"]:
            img = cand["image"]
            score = cand["score"]
            lines.append(
                "| `{img}` | {s:.4f} | {c} | {b} |".format(
                    img=img,
                    s=score,
                    c="yes" if img == chosen else "",
                    b="yes" if best is not None and img == best else "",
                )
            )
        lines.append("")

    lines.append(f"- Saved files: `{md_path.name}`, `{csv_path.name}`, `{all_csv_path.name}`, `{json_path.name}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_task_folders(output_folder: str) -> List[Path]:
    out = Path(output_folder)
    if not out.exists():
        return []
    task_folders = []
    for p in out.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            if list(p.glob("step_*.png")):
                task_folders.append(p)
    return sorted(task_folders, key=lambda x: x.name)


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Evaluate state consistency via GPT-ground-truth + CLIP similarity.")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "output"))
    parser.add_argument("--task_dir", type=str, default=None, help="Optional: evaluate a single task folder.")
    parser.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL_ID)
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda (auto if omitted)")
    parser.add_argument("--force_regen_gt", action="store_true", help="Regenerate ground_truth.txt even if it exists.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not download models; only use locally cached HuggingFace files.",
    )
    parser.add_argument(
        "--step_index_base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Step index base for ground truth lines. 1 means step1 corresponds to step_0.png. 0 means step0 corresponds to step_0.png.",
    )
    parser.add_argument(
        "--score_intermediate",
        action="store_true",
        help="Also score `intermediate/step_i_j.png` candidates vs ground truth and write a candidate report.",
    )
    parser.add_argument(
        "--object_phrase_scoring",
        action="store_true",
        help="Use GPT to extract 5 main object-state phrases per step, score each phrase vs image, and average.",
    )
    parser.add_argument("--num_object_phrases", type=int, default=5, help="Number of object-state phrases per step.")
    parser.add_argument(
        "--force_regen_phrases",
        action="store_true",
        help="Regenerate cached ground-truth object phrases even if they exist.",
    )
    args = parser.parse_args()

    if args.task_dir:
        task_folders = [Path(args.task_dir)]
    else:
        task_folders = _discover_task_folders(args.output_dir)

    if not task_folders:
        print("No task folders found with step_*.png images.")
        return

    print("=" * 60)
    print("Starting CLIP state consistency evaluation")
    print("=" * 60)
    print(f"CLIP model: {args.clip_model} | device: {args.device or 'auto'} | local_files_only: {args.local_files_only}")

    evaluator = CLIPStateConsistencyEvaluator(
        model_id=args.clip_model,
        device=args.device,
        local_files_only=args.local_files_only,
        step_index_base=args.step_index_base,
    )
    gt_mgr = GroundTruthManager()
    phrase_mgr = ObjectPhraseManager() if args.object_phrase_scoring else None
    print(f"[CLIP] Ready (device={evaluator.device})")

    for task in task_folders:
        image_paths = _find_step_images(str(task))
        num_steps = len(image_paths)
        text_plans = load_text_plans(str(task))

        print(f"\nEvaluating: {task.name} ({num_steps} steps)")
        if text_plans is None:
            print("  Note: no text_plans.csv found; GPT will infer ground truth from task name + step count.")

        print("  Preparing ground truth (ground_truth.txt)...")
        gt = gt_mgr.ensure_ground_truth(
            task_folder=str(task),
            num_steps=num_steps,
            text_plans=text_plans,
            force_regen=args.force_regen_gt,
            index_base=args.step_index_base,
        )

        phrases_by_step = None
        if args.object_phrase_scoring:
            phrases_by_step = _load_text_derived_phrases(str(task))
            if phrases_by_step is not None:
                print("  Using text-derived object/state ground truth (ground_truth_object_phrases_text.json) ...")
            else:
                print(f"  Extracting {args.num_object_phrases} object-state phrases per step from text (cached) ...")
                step_numbers = [args.step_index_base + i for i in range(num_steps)]
                gt_lines = gt.to_lines(num_steps, index_base=args.step_index_base)
                phrases_by_step = phrase_mgr.ensure_phrases(  # type: ignore[union-attr]
                    task_folder=str(task),
                    gt_lines=gt_lines,
                    step_numbers=step_numbers,
                    num_phrases=args.num_object_phrases,
                    force_regen=args.force_regen_phrases,
                    cache_filename="ground_truth_object_phrases_text.json",
                )

        print("  Computing CLIP similarity (text vs image) ...")
        result = evaluator.score_task(str(task), gt, phrases_by_step=phrases_by_step)
        _write_task_report(str(task), result)
        print(f"  Average similarity: {result['average_score']:.4f}")
        print(f"  Saved: {task / 'clip_state_consistency_report.md'}")

        if args.score_intermediate:
            print("  Scoring intermediate candidates ...")
            cand_result = evaluator.score_intermediate_candidates(str(task), gt, phrases_by_step=phrases_by_step)
            _write_intermediate_report(str(task), cand_result)
            print(f"  Saved: {task / 'clip_intermediate_candidate_report.md'}")


if __name__ == "__main__":
    main()


