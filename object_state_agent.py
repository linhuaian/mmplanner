"""
Text-derived Expected Object/State Ground Truth
=============================================

We intentionally do NOT inspect generated images for "what is visible".
Ground truth here means: expected object states derived from the step text plan
with a persistent memory across steps.

Primary outputs per task (saved under output/<task>/):
  - ground_truth_object_phrases_text.json   # source=text, step -> list of object/state phrases (<= N)
  - ground_truth_object_states_text.json    # source=text, step -> structured objects/states
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load .env from the repository directory (next to this file) to avoid cwd-dependent behavior.
# Note: by default, dotenv will NOT override already-exported environment variables.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

COMPASS_API_KEY = os.getenv("COMPASS_API_KEY")
COMPASS_BASE_URL = os.getenv("COMPASS_BASE_URL", "https://compass.llm.shopee.io/compass-api/v1")
DEFAULT_VISION_MODEL = os.getenv("GT_GPT_MODEL", "gpt-4o")


def _safe_print(*args: Any, **kwargs: Any) -> None:
    """
    Print but exit cleanly if stdout is closed (common when piping to `head`).
    """
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        # Use an immediate exit to avoid Python trying to flush a broken stdout at shutdown.
        os._exit(0)


@dataclass
class StepState:
    step: int
    phrases: List[str]
    objects: List[Dict[str, str]]
    notes: str = ""


class _BaseLLMAgent:
    """
    Shared utilities for LLM-backed step-state agents.
    Keeps the OpenAI/Compass client wiring + payload normalization in one place.
    """

    def __init__(self, *, api_key: Optional[str], model: str):
        self.api_key = api_key or COMPASS_API_KEY
        if not self.api_key:
            raise RuntimeError("Missing COMPASS_API_KEY; required for TextObjectStateAgent.")
        self.model = model

        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=COMPASS_BASE_URL)

    def _chat_json(self, prompt: str, *, max_tokens: int) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_headers={"Provider": "OpenAI"},
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")

    @staticmethod
    def _normalize_step_payload(data: Dict[str, Any], *, num_phrases: int) -> Tuple[List[str], List[Dict[str, str]], str]:
        phrases = [str(p).strip() for p in (data.get("phrases", []) or []) if str(p).strip()]
        phrases = phrases[:num_phrases]
        objects = data.get("objects", []) or []
        if not isinstance(objects, list):
            objects = []
        objects_norm: List[Dict[str, str]] = []
        for o in objects:
            if not isinstance(o, dict):
                continue
            name = str(o.get("name", "")).strip()
            state = str(o.get("state", "")).strip()
            if name and state:
                objects_norm.append({"name": name, "state": state})
        notes = str(data.get("notes", "") or "").strip()
        return phrases, objects_norm, notes

    @staticmethod
    def _fallback_from_text(
        *,
        step_text: str,
        expected_states: Dict[str, str],
        prev_objects: List[Dict[str, str]],
        num_phrases: int,
        err: Exception,
    ) -> Tuple[List[str], List[Dict[str, str]], str]:
        """
        Best-effort deterministic fallback when GPT is unavailable.
        Uses (1) expected_states (object->state), (2) simple keyword spotting in step_text,
        and (3) persistence from prev_objects.
        """
        text = (step_text or "").strip()
        objects: List[Dict[str, str]] = []

        # 1) From expected_states (already structured)
        for name, st in (expected_states or {}).items():
            name_s = str(name).strip()
            st_s = str(st).strip()
            if name_s and st_s:
                objects.append({"name": name_s, "state": st_s})

        # 2) Persist prior objects unless explicitly removed (very naive; keep limited)
        for o in prev_objects[:10]:
            try:
                n = str(o.get("name", "")).strip()
                s = str(o.get("state", "")).strip()
                if n and s and all(x.get("name") != n for x in objects):
                    objects.append({"name": n, "state": s})
            except Exception:
                continue

        # Build phrases: "<descriptor> <object>" (derived only from structured/persisted objects)
        phrases: List[str] = []
        for o in objects:
            n = o.get("name", "").strip()
            s = o.get("state", "").strip()
            if n and s:
                # ensure article + descriptor + object
                article = "an" if (s[:1].lower() in {"a", "e", "i", "o", "u"}) else "a"
                phrases.append(f"{article} {s} {n}".strip())
            if len(phrases) >= num_phrases:
                break

        notes = f"fallback_used: {type(err).__name__}"
        if "403" in str(err):
            notes += " (403_forbidden)"
        return phrases, objects[: max(num_phrases, 8)], notes


class StepSuggestionAgent(_BaseLLMAgent):
    """
    Produces an initial suggestion/extraction of expected states from step text (+ persistence).
    """

    def suggest(
        self,
        *,
        task: str,
        step_index: int,
        step_text: str,
        expected_states: Dict[str, str],
        prev_phrases: List[str],
        prev_objects: List[Dict[str, str]],
        num_phrases: int,
    ) -> Tuple[List[str], List[Dict[str, str]], str]:
        extract_prompt = f"""
You are a state-tracking agent for procedural instructions (TEXT ONLY; do not assume any image input).

Task: {task}
Step index (0-based): {step_index}
Step text:
{step_text}

Previous step memory (expected/persistent states):
- phrases: {prev_phrases}
- objects: {prev_objects}

Planner expected states (may be incomplete):
{json.dumps(expected_states, indent=2)}

Output requirements:
- Produce 1..{num_phrases} phrases (prefer {num_phrases} when possible).
- Focus on objects that are useful for IMAGE generation (e.g., Stable Diffusion): choose ONLY PHOTOGRAPHABLE PHYSICAL OBJECTS.
- "Visual object" means: something you can literally SEE in a photo (tools, parts, materials, containers, food items, etc.).
- Do NOT output abstract concepts (e.g., goals, plans, budgets, “intended use”, configurations, systems, software/settings).
- Each phrase MUST contain at least ONE concrete object noun and at least ONE descriptive/state word.
- Prefer phrases that START with an article and follow: "a/an <descriptor> <object>".
  Good: "a connected cable", "an open container", "a tightened fastener", "a dirty surface", "a powered-on device"
  Bad: "a defined budget", "an intended use", "a good plan", "a compatible part", "an optimal configuration"
- Also output structured objects with states (name + state string). The object name must be a physical noun phrase.
- The "state" MUST be VISUALLY SPECIFIC / image-observable. Good state examples:
  - position/location: "on the table", "inside a container", "under/over/next to <object>"
  - attachment/assembly: "installed", "removed", "mounted", "seated", "tightened", "loosened"
  - connection: "connected", "disconnected", "plugged in", "unplugged"
  - open/closed + on/off: "open", "closed", "powered on", "powered off"
  - material/condition: "wet", "dirty", "clean", "broken", "cut", "cracked"
  - BAD state examples (too abstract / not directly visible): "compatible", "matching", "suitable", "powerful", "spacious", "high quality".
- If a visual state cannot be inferred from text, set state to "unspecified" and explain in notes.
- Only infer what is reasonable from the step text + persistence; mark uncertain assumptions in notes.

Return JSON ONLY:
{{
  "phrases": ["<descriptor> <object>", ...],
  "objects": [{{"name": "<object>", "state": "<state/attributes>"}}],
  "notes": "<short notes / assumptions>"
}}
""".strip()

        try:
            extract = self._chat_json(extract_prompt, max_tokens=500)
            return self._normalize_step_payload(extract, num_phrases=num_phrases)
        except Exception as e:
            return self._fallback_from_text(
                step_text=step_text,
                expected_states=expected_states,
                prev_objects=prev_objects,
                num_phrases=num_phrases,
                err=e,
            )


class StepCritiqueAgent(_BaseLLMAgent):
    """
    Critiques and revises a proposed set of states for consistency/persistence.
    """

    def critique(
        self,
        *,
        task: str,
        step_index: int,
        step_text: str,
        expected_states: Dict[str, str],
        prev_phrases: List[str],
        prev_objects: List[Dict[str, str]],
        cur_phrases: List[str],
        cur_objects: List[Dict[str, str]],
        cur_notes: str,
        num_phrases: int,
    ) -> Tuple[List[str], List[Dict[str, str]], str]:
        critique_prompt_tmpl = """
You are a consistency critic for procedural state tracking (TEXT ONLY).
Your goal is to keep outputs useful for IMAGE generation: only keep PHOTOGRAPHABLE PHYSICAL OBJECTS.

Task: {task}
Step index: {step_index}
Step text:
{step_text}

Previous step memory:
- phrases: {prev_phrases}
- objects: {prev_objects}

Planner expected states:
{expected_states_json}

Proposed states to critique:
- phrases: {cur_phrases}
- objects: {cur_objects}
- notes: {cur_notes}

Check:
- Missing prerequisites (e.g., flipping requires egg + pan).
- Persistence: if an object exists previously, it should remain unless removed by the text.
- Contradictions with the step text or expected states.
- Phrase format: every phrase must contain BOTH descriptor + object noun.
- Visuality: remove any non-visual/abstract concepts and replace with concrete physical objects that can be seen in a photo.
- Visual specificity: every object "state" must be directly image-observable (position, connection, installed/seated, open/closed, on/off, etc.).
  Replace vague/abstract adjectives like "compatible/matching/suitable/powerful/spacious" with a visual state, or set state="unspecified" and explain in notes.

If issues exist, revise the phrases/objects accordingly.
If everything is already correct, return the same content unchanged.
Return JSON ONLY with the same schema:
{{
  "phrases": [...],
  "objects": [...],
  "notes": "<include any detected issues and assumptions>"
}}
""".strip()

        expected_states_json = json.dumps(expected_states, indent=2)
        critique_prompt = critique_prompt_tmpl.format(
            task=task,
            step_index=step_index,
            step_text=step_text,
            prev_phrases=prev_phrases,
            prev_objects=prev_objects,
            expected_states_json=expected_states_json,
            cur_phrases=cur_phrases,
            cur_objects=cur_objects,
            cur_notes=cur_notes,
        )

        revised = self._chat_json(critique_prompt, max_tokens=600)
        return self._normalize_step_payload(revised, num_phrases=num_phrases)


class TextObjectStateAgent:
    """
    Text-only "expected state" reasoning agent.

    Uses step text + prior memory + (optional) expected_states to infer what objects should exist
    and their expected states, without looking at images.

    Saved to:
      - ground_truth_object_phrases_text.json  (source=text)
      - ground_truth_object_states_text.json  (source=text)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_VISION_MODEL):
        self.api_key = api_key or COMPASS_API_KEY
        self.model = model
        # Split responsibilities into two sub-agents:
        # - suggestion/extraction
        # - critique/revision
        self.suggester = StepSuggestionAgent(api_key=self.api_key, model=self.model)
        self.critic = StepCritiqueAgent(api_key=self.api_key, model=self.model)
        self.memory: Dict[str, Any] = {"by_step": {}}

    def analyze_step_text(
        self,
        *,
        task: str,
        step_index: int,
        step_text: str,
        expected_states: Optional[Dict[str, str]] = None,
        num_phrases: int = 5,
        critique_rounds: int = 3,
        verbose: bool = False,
    ) -> StepState:
        """
        Produce expected object/state phrases purely from step text (no image).
        Implements a feedback loop: Extract -> Critique/Revise (multi-round).
        """
        expected_states = expected_states or {}
        prev = self.memory["by_step"].get(step_index - 1)
        prev_objects = prev.get("objects", []) if isinstance(prev, dict) else []
        prev_phrases = prev.get("phrases", []) if isinstance(prev, dict) else []

        # ---- Pass 1: Extract expected states from text ----
        phrases_1, objects_1, notes_1 = self.suggester.suggest(
            task=task,
            step_index=step_index,
            step_text=step_text,
            expected_states=expected_states,
            prev_phrases=prev_phrases,
            prev_objects=prev_objects,
            num_phrases=num_phrases,
        )
        if verbose:
            _safe_print("\n--- SuggestionAgent output ---")
            _safe_print(json.dumps({"phrases": phrases_1, "objects": objects_1, "notes": notes_1}, indent=2))

        # ---- Pass 2+: Critique + Revise for consistency/persistence (repeat >=3 rounds) ----
        critique_rounds = int(critique_rounds or 0)
        if critique_rounds < 3:
            critique_rounds = 3

        cur_phrases, cur_objects, cur_notes = phrases_1, objects_1, notes_1

        for r in range(critique_rounds):
            try:
                cur_phrases, cur_objects, cur_notes = self.critic.critique(
                    task=task,
                    step_index=step_index,
                    step_text=step_text,
                    expected_states=expected_states,
                    prev_phrases=prev_phrases,
                    prev_objects=prev_objects,
                    cur_phrases=cur_phrases,
                    cur_objects=cur_objects,
                    cur_notes=cur_notes,
                    num_phrases=num_phrases,
                )
                if verbose:
                    _safe_print(f"\n--- CritiqueAgent output (round {r+1}/{critique_rounds}) ---")
                    _safe_print(json.dumps({"phrases": cur_phrases, "objects": cur_objects, "notes": cur_notes}, indent=2))
            except Exception as e:
                # If critique fails, keep the latest and annotate.
                cur_notes = (cur_notes + f" | critique_round_{r+1}_failed: {type(e).__name__}").strip()
                if verbose:
                    _safe_print(f"\n--- CritiqueAgent failed (round {r+1}/{critique_rounds}): {type(e).__name__} ---")
                break

        step_state = StepState(step=step_index, phrases=cur_phrases, objects=cur_objects, notes=cur_notes)
        self.memory["by_step"][step_index] = {
            "step": step_index,
            "phrases": step_state.phrases,
            "objects": step_state.objects,
            "notes": step_state.notes,
        }
        return step_state

    def save_task_text(self, task_folder: str, *, num_phrases: int = 5) -> None:
        folder = Path(task_folder)
        folder.mkdir(parents=True, exist_ok=True)

        by_step = self.memory.get("by_step", {})
        phrases_by_step = {str(k): (v.get("phrases", []) if isinstance(v, dict) else []) for k, v in by_step.items()}
        objects_by_step = {str(k): (v.get("objects", []) if isinstance(v, dict) else []) for k, v in by_step.items()}

        (folder / "ground_truth_object_phrases_text.json").write_text(
            json.dumps({"source": "text", "num_phrases": num_phrases, "by_step": phrases_by_step}, indent=2),
            encoding="utf-8",
        )
        (folder / "ground_truth_object_states_text.json").write_text(
            json.dumps({"source": "text", "by_step": objects_by_step}, indent=2),
            encoding="utf-8",
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backfill TEXT-derived expected object/state ground truth for an existing task folder.")
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, default=None, help="Optional human-readable task name.")
    parser.add_argument("--num_phrases", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="Print Suggestion/Critique agent intermediate JSON per step.")
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["text"],
        help="text: infer expected states from step text only.",
    )
    args = parser.parse_args()

    task_dir = Path(args.task_dir).resolve()
    task_name = args.task_name or task_dir.name.replace("_", " ")

    # Load text plan if present (best-effort) for step_text context
    step_texts: List[str] = []
    csv_path = task_dir / "text_plans.csv"
    if csv_path.exists():
        import csv

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                preferred = None
                for fn in reader.fieldnames:
                    low = (fn or "").lower()
                    if "text" in low and "plan" in low:
                        preferred = fn
                        break
                preferred = preferred or reader.fieldnames[0]
                for row in reader:
                    v = (row.get(preferred) or "").strip()
                    if v:
                        step_texts.append(v)

    agent = TextObjectStateAgent()
    if not step_texts:
        raise SystemExit("No text_plans.csv found or empty; cannot run --mode text.")
    for idx, step_text in enumerate(step_texts):
        _safe_print(f"Analyzing text step {idx}: {step_text[:80]}...")
        agent.analyze_step_text(
            task=task_name,
            step_index=idx,
            step_text=step_text,
            num_phrases=args.num_phrases,
            verbose=bool(args.verbose),
        )
    agent.save_task_text(str(task_dir), num_phrases=args.num_phrases)
    _safe_print(f"Saved TEXT-derived expected states to: {task_dir}")


if __name__ == "__main__":
    main()


