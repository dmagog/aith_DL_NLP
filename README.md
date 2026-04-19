# ITMO AITH: DL+NLP HW5

Ветка `hw_5` содержит решение домашней работы 5 — QLoRA-дообучение
`Qwen/Qwen2.5-1.5B` на новостном корпусе `Lenta.ru` с профилированием
`torch.profiler` и замером до/после на русскоязычных метриках.

Основные материалы — в каталоге [dz5/README.md](dz5/README.md):

- [dz5/hw5.ipynb](dz5/hw5.ipynb) — отчётный ноутбук (читает артефакты);
- [dz5/colab_train.ipynb](dz5/colab_train.ipynb) — end-to-end раннер для
  Colab T4;
- [dz5/src/](dz5/src/) — пайплайн (data, basket, train, evaluate, run_full);
- [dz5/artifacts_hw5/](dz5/artifacts_hw5/) — зафиксированные артефакты
  прогона, включая LoRA-адаптер и `profiler_summary.md`.
