"""FLUX.3 text conditioning -- Qwen3-VL-4B hidden-state stack.

The DiT text context is the stack of a fixed set of Qwen3-VL hidden layers
along the channel dim: ``ctx = (B, L, len(SELECT_LAYERS) * 2560 = 20480)``.
"""

import math
from typing import List, Optional

import torch
from einops import rearrange
from transformers import AutoProcessor
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

SELECT_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
CONTEXT_DIM = 20480  # 8 * 2560

DEFAULT_PAD_MULTIPLE = 80
DEFAULT_MAX_TOKENS = 8192
QWEN3_VL_4B_REPO = "Qwen/Qwen3-VL-4B-Instruct"


def padded_token_length(
    tokenizer,
    texts: List[str],
    pad_multiple: int,
    max_tokens: int,
) -> int:
    longest = 0
    for text in texts:
        real = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_tokens
        )["input_ids"].shape[1]
        longest = max(longest, real)
    return min(math.ceil(longest / pad_multiple) * pad_multiple, max_tokens)


@torch.no_grad()
def encode_flux3_prompt(
    text_encoder,
    processor,
    prompts: List[str],
    max_length: int = DEFAULT_MAX_TOKENS,
    dtype: torch.dtype = torch.bfloat16,
    pad_multiple: Optional[int] = DEFAULT_PAD_MULTIPLE,
) -> torch.Tensor:
    """Prompts -> text context ``(B, L, 20480)``."""
    messages = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    tokenizer = processor.tokenizer

    if pad_multiple is not None:
        max_length = padded_token_length(tokenizer, texts, pad_multiple, max_length)

    input_ids_list, attn_list = [], []
    for text in texts:
        toks = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids_list.append(toks["input_ids"])
        attn_list.append(toks["attention_mask"])

    input_ids = torch.cat(input_ids_list, dim=0).to(text_encoder.device)
    attention_mask = torch.cat(attn_list, dim=0).to(text_encoder.device)

    out = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    stacked = torch.stack([out.hidden_states[k] for k in SELECT_LAYERS], dim=1)
    result = rearrange(stacked, "b c l d -> b l (c d)")
    return result.to(dtype)