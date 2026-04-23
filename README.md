# ITMO AITH: DL+NLP HW6

Ветка `hw_6` содержит решение домашней работы 6 — аблитерация
(Arditi et al. 2024) + QLoRA-DPO модели `Qwen/Qwen2.5-1.5B-Instruct` на
русскоязычных harmful-промптах из `masterkristall/harmful_behaviors_ru`.

Основные материалы — в каталоге [dz6/README.md](dz6/README.md):

- [dz6/hw6.ipynb](dz6/hw6.ipynb) — отчётный ноутбук (читает артефакты);
- [dz6/colab_train.ipynb](dz6/colab_train.ipynb) — раннер для Colab T4;
- [dz6/src/](dz6/src/) — пайплайн (data, evaluate, abliterate, dpo_data,
  dpo_train, run_full);
- [dz6/artifacts_hw6/](dz6/artifacts_hw6/) — зафиксированные артефакты
  прогона, включая LoRA-адаптер DPO. Аблитерированная база детерминированно
  воспроизводима из `src/abliterate.py` при том же seed (`42`), поэтому в
  репо/на Hub она не хранится.
