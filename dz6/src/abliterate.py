"""Аблитерация (orthogonalization) refusal-направления по Arditi et al. 2024.

Пайплайн:
1. Собираем residual-stream активации на last-token позиции prompt'а
   (после каждого декодер-блока) для набора harmful и harmless prompts.
2. Для каждого слоя L вычисляем кандидата:
       direction_L = mean(harmful[L]) − mean(harmless[L]);  normalize.
3. Выбираем "лучший" слой: для каждой кандидат-direction делаем
   runtime-ablation НА ВСЕХ слоях (проекция-ноль этой direction) и
   замеряем refusal-rate на small held-out сэмпле harmful prompts.
   Берём direction, которая сильнее всего роняет refusal-rate.
4. Применяем weight orthogonalization ОДНОРАЗОВО: модифицируем
   `embed_tokens`, `o_proj` каждого слоя и `down_proj` каждого слоя
   так, чтобы ни один из этих матриц-проекторов больше не писал в
   residual stream компоненту по refusal direction. После этого можно
   сохранить модель как обычный HF-чекпойнт и запушить в Hub.

Замечания по корректности:
- На T4 базовая модель уже в 4-bit через bitsandbytes. Для активаций
  это нормально — forward считается в fp16 compute dtype. Для weight
  orthogonalization же нужны fp16/bf16 **неквантованные** копии
  соответствующих матриц. Поэтому функция `load_fp_model_for_abliteration`
  грузит вторую копию модели в fp16 без bnb — мы работаем именно с ней,
  пушим её, и именно она фигурирует как "абл. модель" дальше.
- Гипотезируется единичная refusal direction; в Qwen2.5-1.5B-Instruct
  она обычно располагается в средних слоях (~13-18 из 28). Код автоматом
  выбирает лучший, но сохраняет и полный список кандидатов — на случай,
  если потом захочется воспроизвести эксперимент или провалидировать
  чужой выбор.
"""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

from .evaluate import GenConfig, apply_chat, is_refusal


@dataclass(frozen=True)
class AbliterateConfig:
    # На какой позиции prompt'а снимаем активации (−1 = последний токен).
    act_token_pos: int = -1
    # Сколько harmful-prompt'ов берём из held-out для поиска лучшего слоя.
    # Стоит порядка 16 — этого хватает, чтобы прикинуть refusal-rate, и
    # не слишком дорого по времени.
    layer_search_prompts: int = 16
    # Сколько новых токенов генерим для «короткой» проверки-refusal при
    # поиске слоя (полный eval идёт отдельно, позже и дольше).
    layer_search_max_new_tokens: int = 48
    # Если хотим зафиксировать слой руками (например, воспроизвести paper-
    # ский выбор или проверить гипотезу) — можно передать layer_index.
    force_layer: int | None = None
    # Какие слои вообще рассматриваем. Крайние слои дают плохие direction,
    # поэтому обрезаем (первые 20% и последние 10%).
    skip_first_frac: float = 0.20
    skip_last_frac: float = 0.10


# ---------------------------------------------------------------------------
# 1. Сбор residual-stream активаций на last-token позиции
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_last_token_activations(
    model,
    tokenizer,
    prompts: list[str],
    token_pos: int = -1,
) -> torch.Tensor:
    """Возвращает тензор [n_prompts, n_layers, hidden_dim].

    Значения — выход каждого декодер-блока в позиции `token_pos` prompt'а
    (после полной обработки prompt'а, до первой сгенерированной токены).
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    acts = []

    for p in tqdm(prompts, desc="collect acts"):
        text = apply_chat(tokenizer, p)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
        # out.hidden_states = tuple of len (n_layers + 1):
        #   [0]   — эмбеддинги на входе
        #   [1..] — выход после каждого декодер-блока
        # Нам нужны [1..] → последний токен.
        layer_acts = torch.stack(
            [hs[0, token_pos, :].float().cpu() for hs in out.hidden_states[1:]],
            dim=0,
        )  # [n_layers, hidden]
        acts.append(layer_acts)

    return torch.stack(acts, dim=0)  # [n_prompts, n_layers, hidden]


# ---------------------------------------------------------------------------
# 2. Вычисление кандидатов refusal direction по каждому слою
# ---------------------------------------------------------------------------

def compute_refusal_directions(
    harmful_acts: torch.Tensor,  # [n, L, H]
    harmless_acts: torch.Tensor,  # [n, L, H]
) -> torch.Tensor:
    """Для каждого слоя L: direction = normalize(mean_harmful − mean_harmless).

    Возвращает [n_layers, hidden_dim], строки — нормированные направления.
    """
    mean_h = harmful_acts.mean(dim=0)    # [L, H]
    mean_n = harmless_acts.mean(dim=0)   # [L, H]
    direction = mean_h - mean_n          # [L, H]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return direction


# ---------------------------------------------------------------------------
# 3. Runtime-ablation через forward-hook'и (для поиска лучшего слоя)
# ---------------------------------------------------------------------------

def _iter_decoder_layers(model):
    """Итератор по декодер-блокам для Qwen/LLaMA-подобных архитектур."""
    # HF-стандарт: model.model.layers
    return model.model.layers


def _register_ablation_hooks(model, direction: torch.Tensor):
    """Runtime-hook: из выхода каждого декодер-блока вычитаем проекцию на
    `direction`. Вернёт список handles, которые нужно `.remove()` после.
    """
    d = direction.to(model.device)
    d = d / d.norm().clamp_min(1e-8)
    handles = []

    def _hook(_module, _inp, output):
        # output у LlamaDecoderLayer/Qwen2DecoderLayer — tuple; первый элемент
        # — hidden_states (residual stream).
        if isinstance(output, tuple):
            hs = output[0]
            proj = (hs.to(d.dtype) @ d) .unsqueeze(-1) * d  # [..., H]
            hs_new = hs - proj.to(hs.dtype)
            return (hs_new,) + output[1:]
        proj = (output.to(d.dtype) @ d).unsqueeze(-1) * d
        return output - proj.to(output.dtype)

    for layer in _iter_decoder_layers(model):
        handles.append(layer.register_forward_hook(_hook))
    return handles


def _remove_hooks(handles):
    for h in handles:
        h.remove()


@torch.no_grad()
def _refusal_rate_with_ablation(
    model,
    tokenizer,
    direction: torch.Tensor,
    prompts: list[str],
    max_new_tokens: int,
) -> float:
    """Регистрирует hook'и, генерит короткие ответы, считает refusal-rate,
    снимает hook'и.
    """
    handles = _register_ablation_hooks(model, direction)
    try:
        refusals = 0
        for p in prompts:
            text = apply_chat(tokenizer, p)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            cont_ids = out[0][inputs["input_ids"].shape[1]:]
            resp = tokenizer.decode(cont_ids, skip_special_tokens=True)
            if is_refusal(resp):
                refusals += 1
    finally:
        _remove_hooks(handles)
    return refusals / max(1, len(prompts))


def search_best_layer(
    model,
    tokenizer,
    candidate_directions: torch.Tensor,  # [L, H]
    held_out_harmful: list[str],
    cfg: AbliterateConfig,
) -> tuple[int, list[dict]]:
    """Перебор кандидат-direction по слоям; возвращает (best_layer, scores)."""
    n_layers = candidate_directions.shape[0]
    first = int(round(n_layers * cfg.skip_first_frac))
    last = n_layers - int(round(n_layers * cfg.skip_last_frac))
    layer_ids = list(range(first, last))

    prompts = held_out_harmful[: cfg.layer_search_prompts]
    scores: list[dict] = []

    for L in tqdm(layer_ids, desc="layer search"):
        rr = _refusal_rate_with_ablation(
            model,
            tokenizer,
            candidate_directions[L],
            prompts,
            cfg.layer_search_max_new_tokens,
        )
        scores.append({"layer": L, "refusal_rate_on_ablate": rr})

    # Чем ниже refusal_rate — тем эффективнее direction.
    best = min(scores, key=lambda r: r["refusal_rate_on_ablate"])
    return best["layer"], scores


# ---------------------------------------------------------------------------
# 4. Weight orthogonalization (постоянная модификация весов)
# ---------------------------------------------------------------------------

def _project_out_rows(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """W: [out, in]. Из каждой СТРОКИ вычитаем проекцию на d (d живёт в
    пространстве размерности `in`). Для embed_tokens это правильный ход:
    каждая строка — эмбеддинг vocab-токена, пишется в residual stream.

    W' = W - (W @ d)[:, None] * d[None, :]
    """
    proj = (W @ d).unsqueeze(-1) * d.unsqueeze(0)
    return W - proj


def _project_out_cols(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """W: [out, in]. out = hidden_dim; пишет в residual stream как
    output = W @ x. Чтобы убрать direction d из output (d в пространстве
    `out`): W' = (I - d d.T) @ W = W - d[:, None] * (d @ W)[None, :].
    """
    proj = d.unsqueeze(-1) * (d @ W).unsqueeze(0)
    return W - proj


def apply_weight_orthogonalization(model, direction: torch.Tensor) -> dict:
    """Одноразовая модификация весов; возвращает краткую статистику."""
    d = direction.detach().clone()
    d = d / d.norm().clamp_min(1e-8)

    # embed_tokens: [vocab, hidden]
    emb = model.model.embed_tokens
    with torch.no_grad():
        W = emb.weight.data
        d_dev = d.to(device=W.device, dtype=W.dtype)
        emb.weight.data = _project_out_rows(W, d_dev)

    n_o, n_mlp = 0, 0
    for layer in _iter_decoder_layers(model):
        # o_proj: [hidden, hidden] — output attention; пишет в residual
        with torch.no_grad():
            W = layer.self_attn.o_proj.weight.data
            d_dev = d.to(device=W.device, dtype=W.dtype)
            layer.self_attn.o_proj.weight.data = _project_out_cols(W, d_dev)
            n_o += 1

        # down_proj: [hidden, intermediate] — output MLP; пишет в residual
        with torch.no_grad():
            W = layer.mlp.down_proj.weight.data
            d_dev = d.to(device=W.device, dtype=W.dtype)
            layer.mlp.down_proj.weight.data = _project_out_cols(W, d_dev)
            n_mlp += 1

    return {
        "modified_embed_tokens": True,
        "modified_o_proj": n_o,
        "modified_down_proj": n_mlp,
        "direction_norm_before_normalize": float(direction.norm()),
    }


# ---------------------------------------------------------------------------
# 5. High-level API — всё в одну функцию для run_full
# ---------------------------------------------------------------------------

def abliterate(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    cfg: AbliterateConfig,
    out_dir: Path,
) -> dict:
    """Полный пайплайн: acts → direction → layer-search → weight-orth.

    Модель `model` МОДИФИЦИРУЕТСЯ in-place. Сохраняет в out_dir:
      - refusal_direction.pt (выбранная direction)
      - abliteration_layer_scores.json (refusal-rate при ablation
        для каждого кандидата)
      - abliteration_info.json (итоговая сводка)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Активации ---
    print("[abliterate] collecting harmful activations", flush=True)
    harm_acts = collect_last_token_activations(
        model, tokenizer, harmful_prompts, cfg.act_token_pos
    )
    print("[abliterate] collecting harmless activations", flush=True)
    nice_acts = collect_last_token_activations(
        model, tokenizer, harmless_prompts, cfg.act_token_pos
    )

    # --- Direction per layer ---
    directions = compute_refusal_directions(harm_acts, nice_acts)  # [L, H]

    # --- Выбор слоя ---
    if cfg.force_layer is not None:
        best_layer = cfg.force_layer
        scores = []
    else:
        print("[abliterate] searching best layer via runtime ablation", flush=True)
        best_layer, scores = search_best_layer(
            model, tokenizer, directions, harmful_prompts, cfg
        )

    best_direction = directions[best_layer].clone()

    # --- Weight orthogonalization ---
    print(f"[abliterate] applying weight orthogonalization "
          f"(best_layer={best_layer})", flush=True)
    orth_info = apply_weight_orthogonalization(model, best_direction)

    # --- Артефакты ---
    torch.save(best_direction, out_dir / "refusal_direction.pt")
    (out_dir / "abliteration_layer_scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    info = {
        "best_layer": int(best_layer),
        "n_layers_total": int(directions.shape[0]),
        "hidden_dim": int(directions.shape[1]),
        "n_harmful_for_direction": len(harmful_prompts),
        "n_harmless_for_direction": len(harmless_prompts),
        "layer_search_prompts": cfg.layer_search_prompts,
        "layer_search_max_new_tokens": cfg.layer_search_max_new_tokens,
        "skip_first_frac": cfg.skip_first_frac,
        "skip_last_frac": cfg.skip_last_frac,
        "weight_orth": orth_info,
    }
    (out_dir / "abliteration_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Подчищаем память
    del harm_acts, nice_acts, directions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return info
