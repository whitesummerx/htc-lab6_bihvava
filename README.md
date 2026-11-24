# Лабораторна робота 6. Розподілене навчання нейромережі за даними на CPU (варіант 3)

Варіант 3: датасет **Д3 = OrganAMNIST** (K = 11, C = 1), ширина **Ш3** (c1/c2/h = 64/128/512),
**P_max = 3**, **B = 96** (b = 32), **K_c = 20**, **s_f = 50**. N = 7000, n_ep = 3, SGD (момент 0,9), η₀ = 0,02.
PyTorch (CPU), бекенд gloo, запуск через torchrun, цикл навчання власний.

## Файли
- `train.py` — модель, дані, ручна синхронізація, DDP, вимірювання, контрольні точки, імітація відмови.
- `run_all.py` — уся серія запусків, результати в `results/*.json` (можна перезапускати: готові пропускаються).
- `analyze.py` — таблиці 1–6 (`results/tables.md`) і рисунок `results/lab6_overview.png`.

## Запуск
```
pip install torch numpy scikit-learn matplotlib psutil
python run_all.py            # сам завантажить organamnist.npz з Zenodo у data/
python analyze.py results
```
Якщо кількість фізичних ядер не визначилась автоматично: `python run_all.py --cphys 6`.
Під час серії не запускайте інших важких програм.
