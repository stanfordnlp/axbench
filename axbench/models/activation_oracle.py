"""
Activation Oracle integration for AxBench.

Implements concept detection using activation oracles — LoRA-finetuned LLMs
that interpret activations from a target model by receiving them as additive
steering vectors injected at an early layer.

Based on: https://github.com/adamkarvonen/activation_oracles
Paper: https://arxiv.org/abs/2512.15674

Requires the activation_oracles repo:
  pip install activation_oracles
  Or: pip install 'axbench[activation_oracles]'
"""
import os
import re
import sys

import torch
from tqdm.auto import tqdm

from .model import BaseModel

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


# ── Optional dependency: activation_oracles (nl_probes) ──────────────────────
# Add local clone to sys.path if present: axbench/models/_activation_oracles/
_ao_local = os.path.join(os.path.dirname(__file__), "_activation_oracles")
if os.path.isdir(_ao_local) and _ao_local not in sys.path:
    sys.path.insert(0, _ao_local)

try:
    from nl_probes.utils.activation_utils import collect_activations, get_hf_submodule
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook
    from nl_probes.utils.dataset_utils import (
        SPECIAL_TOKEN, get_introspection_prefix, find_pattern_in_tokens,
    )
    from nl_probes.utils.common import layer_percent_to_layer
    _HAS_AO = True
except ImportError:
    _HAS_AO = False


def _require_ao():
    if not _HAS_AO:
        raise ImportError(
            "activation_oracles is not installed. Clone into axbench/models/:\n"
            "  cd axbench/models && git clone https://github.com/adamkarvonen/activation_oracles.git _activation_oracles\n"
            "Or add the repo to PYTHONPATH."
        )


# ── Prompt templates ─────────────────────────────────────────────────────────
CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING = (
    "Is the following concept present in the text? Concept: '{concept}'. "
    "Answer with a rating from 0 to 2, where 0 means the concept is not present at all, "
    "1 means the concept is somewhat present, and 2 means the concept is strongly present. "
    "Provide your rating using this exact format: Rating: [[score]]."
)

CONCEPT_DETECTION_QUESTION_TEMPLATE = (
    "Is the assistant thinking about the concept '{concept}'?"
)


# ── Model class ──────────────────────────────────────────────────────────────

class ActivationOracleReading(BaseModel):
    """Activation Oracle concept-detection for AxBench.

    **No training required.**  Uses a pre-trained oracle LoRA adapter
    (e.g. ``adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B``)
    loaded onto the same base model as the target.

    Flow for each example:

    1. Run the *target model* (base weights, oracle LoRA disabled) on the
       input text and extract residual-stream activations at
       ``layer_percent`` depth.
    2. Build an introspection prompt whose ``" ?"`` placeholder tokens mark
       where activations will be injected.
    3. Switch to the *oracle adapter*, register a steering hook at
       ``injection_layer`` (default 1), and generate a response.
    4. Parse the response for a 0-2 concept-presence rating.

    Constructor kwargs (beyond the standard ``model, tokenizer, layer``):

    ==================== ====================================================
    ``oracle_lora_path``  HuggingFace repo or local path to the oracle LoRA.
    ``target_model_name`` Model name (for ``apply_chat_template``).
    ``layer_percent``     Activation extraction depth as % (default 50).
    ``injection_layer``   Layer in the oracle to inject at (default 1).
    ``steering_coefficient``  Injection strength (default 1.0).
    ``max_new_tokens``    Max tokens to generate per question (default 50).
    ==================== ====================================================
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        _require_ao()
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.device = kwargs.get("device", "cuda:0")
        self.seed = kwargs.get("seed", 42)

        # Oracle config — auto-detect LoRA path from model name if not provided
        _ORACLE_LORA_MAP = {
            "meta-llama/Llama-3.1-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
            "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        }
        model_name_str = getattr(tokenizer, "name_or_path", "")
        default_lora = _ORACLE_LORA_MAP.get(
            model_name_str,
            "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        )
        self.oracle_lora_path = kwargs.get("oracle_lora_path", default_lora)
        self.target_model_name = kwargs.get(
            "target_model_name", model_name_str or "Qwen/Qwen3-8B")
        self.layer_percent = kwargs.get("layer_percent", 50)
        self.injection_layer = kwargs.get("injection_layer", 1)
        self.steering_coefficient = kwargs.get("steering_coefficient", 1.0)
        self.max_new_tokens = kwargs.get("max_new_tokens", 50)

        # Internal state (populated by load())
        self._extraction_layer = None
        self._oracle_adapter_name = None
        self._original_model = model

    # ── BaseModel interface ──────────────────────────────────────────────

    def __str__(self):
        return "ActivationOracleReading"

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass

    def train(self, examples, **kwargs):
        pass

    def load(self, dump_dir=None, **kwargs):
        """Load the oracle LoRA adapter onto the target model."""
        if self._oracle_adapter_name is not None:
            return  # already loaded

        from peft import PeftModel, LoraConfig

        # Compute absolute extraction layer
        self._extraction_layer = layer_percent_to_layer(
            self.target_model_name, self.layer_percent)
        logger.warning(
            f"Extraction layer: {self._extraction_layer} "
            f"({self.layer_percent}% of model depth)")

        # Make model a PeftModel if it isn't already
        if not isinstance(self.model, PeftModel):
            dummy_config = LoraConfig(
                r=1, lora_alpha=1, target_modules=["q_proj"], bias="none",
            )
            self.model = PeftModel(
                self.model, dummy_config, adapter_name="base_default")

        # Load the oracle LoRA adapter
        adapter_name = "activation_oracle"
        self.model.load_adapter(self.oracle_lora_path, adapter_name=adapter_name)
        self._oracle_adapter_name = adapter_name
        logger.warning(
            f"Loaded oracle adapter '{adapter_name}' from "
            f"{self.oracle_lora_path}")

        # Tokenizer setup for batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Pre-compute Yes/No token ids
        self._yes_token_id = self.tokenizer.encode(
            "Yes", add_special_tokens=False)[0]
        self._no_token_id = self.tokenizer.encode(
            "No", add_special_tokens=False)[0]
        logger.warning(
            f"Yes token: {self._yes_token_id}, No token: {self._no_token_id}")

    # ── Core inference ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        """Run activation-oracle concept detection.

        Returns ``{"max_act": [rating_per_example]}``.
        """
        self.model.eval()
        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE.format(
            concept=concept)

        all_max_act = []
        use_lora = isinstance(self.model, __import__('peft').PeftModel)

        for i in tqdm(
            range(0, len(examples), batch_size),
            desc="ActivationOracle Reading",
        ):
            batch_examples = examples.iloc[i : i + batch_size]

            # ── 1. Extract activations from target (base) model ──────
            texts = []
            for _, row in batch_examples.iterrows():
                texts.append(row.get("output", row.get("input", "")))

            self.model.disable_adapter_layers()

            target_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            extraction_submodule = get_hf_submodule(
                self.model, self._extraction_layer, use_lora=use_lora)
            activations = collect_activations(
                self.model, extraction_submodule, target_inputs)

            self.model.enable_adapter_layers()
            self.model.set_adapter(self._oracle_adapter_name)

            # ── 2. Build oracle prompts with introspection prefix ────
            oracle_prompts = []
            per_example_vecs = []

            for b in range(len(texts)):
                if target_inputs.attention_mask is not None:
                    num_valid = int(
                        target_inputs.attention_mask[b].sum().item())
                else:
                    num_valid = target_inputs.input_ids.shape[1]

                act_vecs = activations[b, -num_valid:]
                per_example_vecs.append(act_vecs)

                prefix = get_introspection_prefix(
                    self._extraction_layer, num_valid)
                user_msg = prefix + question_text

                prompt_str = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                oracle_prompts.append(prompt_str)

            oracle_inputs = self.tokenizer(
                oracle_prompts, return_tensors="pt",
                padding=True, truncation=True,
            ).to(self.device)

            # ── 3. Locate " ?" positions & build steering vectors ────
            all_positions = []
            all_vectors = []

            for b in range(oracle_inputs.input_ids.shape[0]):
                num_valid = per_example_vecs[b].shape[0]
                positions = find_pattern_in_tokens(
                    oracle_inputs.input_ids[b].tolist(),
                    SPECIAL_TOKEN, num_valid, self.tokenizer)
                vecs = per_example_vecs[b]
                n = min(len(positions), vecs.shape[0])
                all_positions.append(positions[:n])
                all_vectors.append(vecs[:n])

            # ── 4. Forward pass with steering hook at injection layer ──
            injection_submodule = get_hf_submodule(
                self.model, self.injection_layer, use_lora=use_lora)
            hook_fn = get_hf_activation_steering_hook(
                all_vectors, all_positions,
                self.steering_coefficient,
                device=self.device,
                dtype=activations.dtype,
            )

            with add_hook(injection_submodule, hook_fn):
                outputs = self.model(**oracle_inputs)

            self.model.set_adapter("base_default")

            # ── 5. Extract Yes/No logits at last position ─────────
            logits = outputs.logits
            attn_mask = oracle_inputs.attention_mask
            last_pos = attn_mask.sum(dim=1) - 1

            for b in range(logits.shape[0]):
                last_logits = logits[b, last_pos[b]]
                yes_logit = last_logits[self._yes_token_id].item()
                no_logit = last_logits[self._no_token_id].item()
                score = torch.softmax(
                    torch.tensor([yes_logit, no_logit]), dim=0)[0].item()
                all_max_act.append(score)

            torch.cuda.empty_cache()

        return {"max_act": all_max_act}

    def predict_latents(self, examples, **kwargs):
        return self.predict_latent(examples, **kwargs)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        return {}

    def to(self, device):
        self.device = device
        return self


class ActivationOracleReadingRating(ActivationOracleReading):
    """Activation Oracle using 0-2 rating generation instead of Yes/No logits."""

    def __str__(self):
        return "ActivationOracleReadingRating"

    @staticmethod
    def _get_rating_from_completion(completion):
        """Parse a 0-2 rating from the oracle's free-text completion."""
        try:
            if "Rating:" in completion:
                rating_text = completion.split("Rating:")[-1].strip()
                rating_text = rating_text.split("\n")[0].strip()
                rating_text = (
                    rating_text.replace("[", "")
                    .replace("]", "")
                    .strip('"')
                    .strip("'")
                    .strip("*")
                    .strip()
                )
                rating = float(rating_text)
                if 0 <= rating <= 2:
                    return rating
            numbers = re.findall(r"\b([012](?:\.\d+)?)\b", completion)
            if numbers:
                return float(numbers[-1])
            logger.warning(
                f"Cannot find rating in completion: {completion[:200]}")
            return -1
        except (ValueError, IndexError) as e:
            logger.error(
                f"Error parsing rating: {completion[:200]}. Error: {e}")
            return -1

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.model.eval()
        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(
            concept=concept)

        all_max_act = []
        use_lora = isinstance(self.model, __import__('peft').PeftModel)

        for i in tqdm(
            range(0, len(examples), batch_size),
            desc="ActivationOracle Rating",
        ):
            batch_examples = examples.iloc[i : i + batch_size]

            texts = []
            for _, row in batch_examples.iterrows():
                texts.append(row.get("output", row.get("input", "")))

            self.model.disable_adapter_layers()

            target_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            extraction_submodule = get_hf_submodule(
                self.model, self._extraction_layer, use_lora=use_lora)
            activations = collect_activations(
                self.model, extraction_submodule, target_inputs)

            self.model.enable_adapter_layers()
            self.model.set_adapter(self._oracle_adapter_name)

            oracle_prompts = []
            per_example_vecs = []

            for b in range(len(texts)):
                if target_inputs.attention_mask is not None:
                    num_valid = int(target_inputs.attention_mask[b].sum().item())
                else:
                    num_valid = target_inputs.input_ids.shape[1]

                act_vecs = activations[b, -num_valid:]
                per_example_vecs.append(act_vecs)

                prefix = get_introspection_prefix(
                    self._extraction_layer, num_valid)
                user_msg = prefix + question_text

                prompt_str = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                oracle_prompts.append(prompt_str)

            oracle_inputs = self.tokenizer(
                oracle_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            all_positions = []
            all_vectors = []

            for b in range(oracle_inputs.input_ids.shape[0]):
                num_valid = per_example_vecs[b].shape[0]
                positions = find_pattern_in_tokens(
                    oracle_inputs.input_ids[b].tolist(),
                    SPECIAL_TOKEN, num_valid, self.tokenizer)
                vecs = per_example_vecs[b]
                n = min(len(positions), vecs.shape[0])
                all_positions.append(positions[:n])
                all_vectors.append(vecs[:n])

            injection_submodule = get_hf_submodule(
                self.model, self.injection_layer, use_lora=use_lora)
            hook_fn = get_hf_activation_steering_hook(
                all_vectors, all_positions,
                self.steering_coefficient,
                device=self.device,
                dtype=activations.dtype,
            )

            with add_hook(injection_submodule, hook_fn):
                outputs = self.model.generate(
                    **oracle_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            self.model.set_adapter("base_default")

            prompt_len = oracle_inputs.input_ids.shape[1]
            for b in range(outputs.shape[0]):
                completion_ids = outputs[b, prompt_len:]
                completion = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True)
                rating = self._get_rating_from_completion(completion)
                if len(all_max_act) < 20:
                    logger.warning(
                        f"[AO Rating] text={texts[b][:80]}... "
                        f"completion={completion!r} rating={rating}")
                all_max_act.append(rating)

            torch.cuda.empty_cache()

        return {"max_act": all_max_act}


