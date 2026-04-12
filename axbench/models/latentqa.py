"""
LatentQA integration for AxBench (reading only).

Implements two model classes:
- LatentQAReading: Uses LatentQA's reading mode for concept detection (latent inference).
- LatentQAReadingRating: Variant that uses free-text rating generation instead of Yes/No logits.

Requires the LatentQA repo (https://github.com/aypan17/latentqa) to be installed.
"""
import json
import os
import sys
import re

import numpy as np
import torch
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM
from .model import Model, BaseModel

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


# Yes/No question template for concept detection.
# We measure logits for "Yes" vs "No" tokens from the decoder.
CONCEPT_DETECTION_QUESTION_TEMPLATE = (
    "Is the assistant thinking about the concept '{concept}'?"
)

CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING = (
    "Is the following concept present in the text? Concept: '{concept}'. "
    "Answer with a rating from 0 to 2, where 0 means the concept is not present at all, "
    "1 means the concept is somewhat present, and 2 means the concept is strongly present. "
    "Provide your rating using this exact format: Rating: [[score]]."
)


# Add local clone to sys.path if present: axbench/models/_latentqa/
_latentqa_local = os.path.join(os.path.dirname(__file__), "_latentqa")
if os.path.isdir(_latentqa_local) and _latentqa_local not in sys.path:
    sys.path.insert(0, _latentqa_local)

try:
    from lit.utils.activation_utils import latent_qa as _latent_qa
    from lit.utils.dataset_utils import BASE_DIALOG as _BASE_DIALOG, ENCODER_CHAT_TEMPLATES as _ENCODER_CHAT_TEMPLATES
    from lit.utils.infra_utils import get_tokenizer as _lqa_get_tokenizer
    try:
        from lit.utils.dataset_utils import lqa_tokenize as _lqa_tokenize
    except ImportError:
        from lit.utils.dataset_utils import tokenize as _lqa_tokenize
    _HAS_LATENTQA = True
except ImportError:
    _HAS_LATENTQA = False


def _require_latentqa():
    if not _HAS_LATENTQA:
        raise ImportError(
            "LatentQA is not installed. Clone into axbench/models/:\n"
            "  cd axbench/models && git clone https://github.com/aypan17/latentqa.git _latentqa\n"
            "Or add the repo to PYTHONPATH."
        )


def _get_model_layers_str(model):
    """Determine the correct attribute path to model layers."""
    for path in [
        "model.layers",
        "model.model.layers",
        "module.model.model.layers",
        "language_model.model.layers",
        "module.language_model.model.layers",
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return path
        except AttributeError:
            continue
    raise RuntimeError("Cannot find model layers. Unsupported model architecture.")


def _load_decoder_model(target_model_name, decoder_model_name, decoder_device):
    """Load the LatentQA decoder model (shared across model classes).

    Replicates the essential steps of latentqa's ``get_model`` but uses
    ``sdpa`` attention so we don't depend on ``flash-attn``.
    """
    _require_latentqa()
    from peft import PeftModel

    logger.warning(f"Loading LatentQA decoder from {decoder_model_name} to {decoder_device}")
    lqa_tokenizer = _lqa_get_tokenizer(target_model_name)

    # Load base model with sdpa (no flash-attn dependency)
    base_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="auto" if decoder_device == "auto" else None,
    )
    base_model.resize_token_embeddings(len(lqa_tokenizer))
    for p in base_model.parameters():
        p.requires_grad = False

    # Load LoRA decoder weights
    decoder_model = PeftModel.from_pretrained(base_model, decoder_model_name)
    if decoder_device is not None and decoder_device != "auto":
        decoder_model = decoder_model.to(decoder_device)
    decoder_model.eval()
    return decoder_model


def _get_modules(target_model, decoder_model, min_layer=15, max_layer=16,
                 layer_to_write=0, num_layers_to_read=1):
    """Get read/write module hooks for LatentQA.

    Returns List[List[Module]] for read and write modules.
    """
    target_path = _get_model_layers_str(target_model)
    decoder_path = _get_model_layers_str(decoder_model)

    def get_layer(model, path, idx):
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj[idx]

    module_read, module_write = [], []
    for i in range(min_layer, max_layer):
        module_read_i = [get_layer(target_model, target_path, j)
                         for j in range(i, i + num_layers_to_read)]
        module_write_i = [get_layer(decoder_model, decoder_path, j)
                          for j in range(layer_to_write, layer_to_write + num_layers_to_read)]
        module_read.append(module_read_i)
        module_write.append(module_write_i)
    return module_read, module_write


class LatentQAReading(BaseModel):
    """LatentQA Reading mode for concept detection.

    Uses LatentQA's decoder to read and interpret activations from the
    target model, then determines if a concept is present.

    This is similar to PromptDetection but reads from internal activations
    rather than prompting the model directly about its output.
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        self.model = model  # target model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.device = kwargs.get("device", "cuda:0")
        self.decoder_device = kwargs.get("decoder_device", "cuda:1")
        self.seed = kwargs.get("seed", 42)

        # LatentQA-specific config
        self.decoder_model_name = kwargs.get(
            "decoder_model_name", "aypan17/latentqa_llama-3-8b-instruct")
        self.target_model_name = kwargs.get(
            "target_model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.min_layer_to_read = kwargs.get("min_layer_to_read", 15)
        self.max_layer_to_read = kwargs.get("max_layer_to_read", 16)
        self.num_layers_to_read = kwargs.get("num_layers_to_read", 1)
        self.layer_to_write = kwargs.get("layer_to_write", 0)
        self.modify_chat_template = kwargs.get("modify_chat_template", True)
        self.max_new_tokens = kwargs.get("max_new_tokens", 100)

        self.decoder_model = None
        self.module_read = None
        self.module_write = None

    def __str__(self):
        return 'LatentQAReading'

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass  # no training needed

    def train(self, examples, **kwargs):
        pass  # no training needed

    def load(self, dump_dir=None, **kwargs):
        """Load the LatentQA decoder model."""
        if self.decoder_model is not None:
            return  # already loaded

        self.decoder_model = _load_decoder_model(
            self.target_model_name, self.decoder_model_name, self.decoder_device)

        # Ensure decoder vocab matches target model (e.g. if PAD token was added)
        target_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        decoder_vocab_size = self.decoder_model.get_input_embeddings().weight.shape[0]
        if target_vocab_size != decoder_vocab_size:
            logger.warning(
                f"Resizing decoder embeddings from {decoder_vocab_size} to {target_vocab_size}")
            self.decoder_model.resize_token_embeddings(target_vocab_size)

        # Set up read/write module hooks
        self.module_read, self.module_write = _get_modules(
            self.model, self.decoder_model,
            min_layer=self.min_layer_to_read,
            max_layer=self.max_layer_to_read,
            layer_to_write=self.layer_to_write,
            num_layers_to_read=self.num_layers_to_read,
        )

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        """Use LatentQA reading mode for concept detection.

        For each example, extracts activations from the target model,
        feeds them to the LatentQA decoder with a yes/no concept question,
        and uses P(Yes) - P(No) logit difference as the detection score.
        """
        _require_latentqa()

        self.model.eval()
        self.decoder_model.eval()

        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        # Get token IDs for "Yes" and "No"
        yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE.format(concept=concept)
        chat_template = _ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        all_max_act = []

        # LatentQA requires left padding
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        for i in tqdm(range(0, len(examples), batch_size), desc="LatentQA Reading"):
            batch_examples = examples.iloc[i:i + batch_size]

            probe_data = []
            for _, row in batch_examples.iterrows():
                user_text = row.get("input", "")
                assistant_text = row.get("output", "")
                read_prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template=chat_template,
                )
                dialog = _BASE_DIALOG + [
                    {"role": "user", "content": question_text},
                ]
                probe_data.append({
                    "read_prompt": read_prompt,
                    "dialog": dialog,
                })

            # Tokenize with generate=True to include the assistant header,
            # then do a forward pass to get logits (not model.generate).
            batch_tokenized = _lqa_tokenize(
                probe_data,
                self.tokenizer,
                name=self.target_model_name,
                generate=True,
                mask_type=None,
                mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            # Add dummy labels so latent_qa accepts generate=False
            input_ids = batch_tokenized["tokenized_write"]["input_ids"]
            batch_tokenized["tokenized_write"]["labels"] = input_ids.clone()

            # Forward pass to get logits
            out = _latent_qa(
                batch_tokenized,
                self.model,
                self.decoder_model,
                self.module_read[0],
                self.module_write[0],
                self.tokenizer,
                shift_position_ids=False,
                generate=False,
                no_grad=True,
            )

            # Extract logits at the last non-padding position for each example
            logits = out.logits  # (batch, seq_len, vocab_size)
            attention_mask = batch_tokenized["tokenized_write"]["attention_mask"].to(logits.device)
            # Last real token position per example
            last_pos = attention_mask.sum(dim=1) - 1  # (batch,)

            for j in range(logits.shape[0]):
                last_logits = logits[j, last_pos[j]]  # (vocab_size,)
                yes_logit = last_logits[yes_token_id].item()
                no_logit = last_logits[no_token_id].item()
                # Score: P(Yes) from softmax over Yes/No
                score = torch.softmax(
                    torch.tensor([yes_logit, no_logit]), dim=0
                )[0].item()
                all_max_act.append(score)

            torch.cuda.empty_cache()

        self.tokenizer.padding_side = orig_padding_side
        return {"max_act": all_max_act}

    def predict_latents(self, examples, **kwargs):
        return self.predict_latent(examples, **kwargs)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        return {}

    def to(self, device):
        self.device = device
        # Note: target model device is managed by AxBench infrastructure
        # decoder stays on its own device
        return self


class LatentQAReadingRating(LatentQAReading):
    """LatentQA Reading using 0-2 rating generation instead of Yes/No logits."""

    def __str__(self):
        return "LatentQAReadingRating"

    @staticmethod
    def _get_rating_from_completion(completion):
        """Parse a 0-2 rating from the decoder's free-text completion."""
        import re
        try:
            if "Rating:" in completion:
                rating_text = completion.split("Rating:")[-1].strip()
                rating_text = rating_text.split("\n")[0].strip()
                rating_text = (
                    rating_text.replace("[", "").replace("]", "")
                    .strip('"').strip("'").strip("*").strip()
                )
                rating = float(rating_text)
                if 0 <= rating <= 2:
                    return rating
            numbers = re.findall(r"\b([012](?:\.\d+)?)\b", completion)
            if numbers:
                return float(numbers[-1])
            return -1
        except (ValueError, IndexError):
            return -1

    def predict_latent(self, examples, **kwargs):
        _require_latentqa()

        self.model.eval()
        self.decoder_model.eval()

        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(concept=concept)
        chat_template = _ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        all_max_act = []

        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        for i in tqdm(range(0, len(examples), batch_size), desc="LatentQA Rating"):
            batch_examples = examples.iloc[i:i + batch_size]

            probe_data = []
            for _, row in batch_examples.iterrows():
                user_text = row.get("input", "")
                assistant_text = row.get("output", "")
                read_prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template=chat_template,
                )
                dialog = _BASE_DIALOG + [
                    {"role": "user", "content": question_text},
                ]
                probe_data.append({
                    "read_prompt": read_prompt,
                    "dialog": dialog,
                })

            batch_tokenized = _lqa_tokenize(
                probe_data, self.tokenizer, name=self.target_model_name,
                generate=True, mask_type=None, mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            with torch.no_grad():
                out = _latent_qa(
                    batch_tokenized, self.model, self.decoder_model,
                    self.module_read[0], self.module_write[0], self.tokenizer,
                    shift_position_ids=False, generate=True,
                    max_new_tokens=50, no_grad=True,
                )

            num_tokens = batch_tokenized["tokenized_write"]["input_ids"][0].shape[0]
            for j in range(len(batch_examples)):
                completion = self.tokenizer.decode(
                    out[j][num_tokens:], skip_special_tokens=True)
                rating = self._get_rating_from_completion(completion)
                if len(all_max_act) < 20:
                    text = batch_examples.iloc[j].get("output", "")[:80]
                    logger.warning(
                        f"[LQA Rating] text={text}... "
                        f"completion={completion!r} rating={rating}")
                all_max_act.append(rating)

            torch.cuda.empty_cache()

        self.tokenizer.padding_side = orig_padding_side
        return {"max_act": all_max_act}


