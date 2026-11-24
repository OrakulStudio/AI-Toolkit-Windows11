import torch
from typing import List, Union

class CustomPromptProcessor:
    """
    Наша собственная логика обработки и батчинга эмбеддингов 
    под кастомные модели без зависимости от чужих громоздких обвязок.
    """
    def __init__(self, max_length: int = 4096):
        self.max_length = max_length

    @torch.no_grad()
    def encode_batch(self, text_encoder, tokenizer, prompts: Union[str, List[str]], device: torch.device) -> List[torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]

        # Токенизация с правосторонним паддингом под нашу логику
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=True,
            max_length=self.max_length,
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"]
        lengths = attention_mask.sum(dim=1).tolist()

        # Чистый прогон через энкодер
        outputs = text_encoder(input_ids=input_ids, use_cache=False)
        hidden_states = outputs.last_hidden_state

        # Нарезаем тензоры по реальной длине каждого промпта без лишнего хвоста
        return [hidden_states[i, :length].clone() for i, length in enumerate(lengths)]

    def build_embeds(self, text_encoder, tokenizer, prompts, device, target_dtype=torch.bfloat16):
        states = self.encode_batch(text_encoder, tokenizer, prompts, device)
        processed_embeds = [st.to(target_dtype) for st in states]
        return processed_embeds