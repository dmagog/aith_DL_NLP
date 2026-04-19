---
base_model: Qwen/Qwen2.5-1.5B
library_name: peft
pipeline_tag: text-generation
language:
- ru
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B
- lora
- qlora
- transformers
---

# QLoRA-адаптер: Qwen2.5-1.5B на Lenta.ru

LoRA-адаптер для дообучения `Qwen/Qwen2.5-1.5B` на новостных текстах
`Lenta.ru` (ITMO AITH DL+NLP, HW5).

## Базовая модель и дизайн

- Base model: `Qwen/Qwen2.5-1.5B` в 4-bit NF4 + double-quant, compute dtype `fp16`.
- LoRA: `r=16`, `α=32`, `dropout=0.05`, target modules — полный набор
  линейных слоёв (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
- Оптимизатор: `paged_adamw_8bit`, `lr=2e-4`, cosine schedule, `warmup_ratio=0.03`.
- `max_seq_length=384`, `per_device_batch=2`, `grad_accum=4` (effective
  batch 8), gradient checkpointing on.
- Обучение: 0.5 эпохи на 11 500 документах, eval — 500 документов,
  seed=42.
- Аппаратура: Colab T4 (Turing, sm75). Поэтому `fp16` compute, а не
  `bf16`.

## Данные

Сэмпл из `Lenta.Ru-News-Dataset` (~800k документов). Reservoir sampling
12 000 записей с фиксированным seed, фильтры по длине (`text ≥ 300`,
`title ≥ 10`), исключение рубрики «Библиотека», обрезка тел до 600
символов. Формат строки под causal LM: `"Заголовок: {title}\n\nТекст: {text}"`.

## Метрики до / после

| Метрика | before | after |
|---|---|---|
| Perplexity (200 eval) | 6.378 | 4.861 |
| `xnli_ru` acc (limit=200) | 0.445 | 0.465 |
| `xwinograd_ru` acc (limit=200) | 0.630 | 0.620 |

Главный эффект — доменная адаптация: связные новостные генерации в стиле
Lenta с корректной структурой «Заголовок → Текст».

## Использование

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", quantization_config=bnb, device_map="auto"
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", use_fast=True)
model = PeftModel.from_pretrained(base, "./lora_adapter")

prompt = "Заголовок: Минфин рассказал о новых правилах налогообложения для самозанятых\n\nТекст:"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs, max_new_tokens=160, do_sample=True,
    temperature=0.8, top_p=0.9, repetition_penalty=1.1,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

Полный раннер и ноутбук-отчёт — в репозитории ДЗ 5.

### Framework versions

- PEFT 0.18.1
- transformers ≥ 4.45
- bitsandbytes ≥ 0.43
