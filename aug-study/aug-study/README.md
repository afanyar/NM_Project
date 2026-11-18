# Вплив Data Augmentation на якість класифікації зображень (CIFAR-10)

Навчальний проєкт з дисципліни «Нейронні мережі». Модель, оптимізатор і гіперпараметри
навчання **однакові** для всіх запусків; змінюється лише аугментація (`configs/augmentations.yaml`).

## Протокол
- Дані: CIFAR-10, стратифікований спліт 45 000 train / 5 000 val (фіксований `split_seed=0`) / 10 000 test.
- Вибір гіперпараметрів аугментацій - за **val**; **test** рахується один раз після навчання (фінальна модель).
- Модель: CIFAR-ResNet (`resnet20` за замовчуванням, `resnet18` опційно), SGD+Nesterov, cosine LR з warmup, 30 епох, AMP.
- Кожна конфігурація: seed ∈ {0,1,2}; звітуються mean ± std.
- Метрики: test accuracy, macro-F1, per-class accuracy, confusion matrix, `train_clean_acc − val_acc`
  (train accuracy рахується на чистій, неаугментованій підвибірці train), час навчання,
  робастність на спотвореному тесті (noise, brightness, contrast, rotation, blur).
- `train_loss` у логах - це loss на **аугментованих** батчах (для Mixup/CutMix - змішаний loss), тому між пресетами він непорівнюваний.

## Структура
```
configs/   base.yaml (спільне) | augmentations.yaml (пресети) | plans.yaml (сітки запусків)
src/       augment.py, data.py, models.py, train.py, evaluate.py, utils.py
run_grid.py          запуск плану з plans.yaml (вже виконане пропускає)
visualize_augs.py    картинка з прикладами аугментацій для звіту
results/runs/<run_id>/   config.json, epochs.csv, metrics.json, confusion_matrix.csv
```

## Швидкий старт
```bash
pip install -r requirements.txt
python -m src.train --aug baseline --smoke --epochs 2 --out /tmp/smoke   # перевірка коду на випадкових даних
python -m src.train --aug flip_crop --seed 0               # один справжній запуск
python run_grid.py --plan main                             # повне порівняння (20 конфігурацій x 3 seed)
python run_grid.py --plan sample_efficiency                # частки даних 10/25/50%
python visualize_augs.py --augs flip_crop cutout_16 rotation_30 mixup_a1 cutmix_a1
```

## Kaggle
1. Notebook → Settings: Accelerator = GPU T4, Internet = On (для завантаження CIFAR-10 потрібна верифікація телефону).
2. Комірки:
```python
!git clone https://github.com/<user>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt
!python -m src.train --aug baseline --smoke --epochs 2 --out /tmp/smoke
!python run_grid.py --plan main --out /kaggle/working/results --data_root /kaggle/working/data
```
3. «Save Version → Save & Run All»: результати з `/kaggle/working/results` зберігаються як output ноутбука. Завантажте їх і закомітьте в `results/`.
   Якщо сесія обірвалась - запустіть ту саму команду ще раз, виконані запуски будуть пропущені.

## Відтворюваність
Фіксуються seed Python/NumPy/PyTorch, порядок батчів і аугментацій у DataLoader. Через AMP і `cudnn.benchmark`
GPU-запуски не гарантують побітової відтворюваності - саме тому використовується кілька seed.

## Джерела методів
Cutout: DeVries & Taylor, 2017. Mixup: Zhang et al., ICLR 2018. CutMix: Yun et al., ICCV 2019.
RandAugment: Cubuk et al., CVPRW 2020. Реалізації RandomCrop/Flip/Rotation/ColorJitter/RandAugment - `torchvision.transforms`.
Архітектура ResNet: He et al., CVPR 2016.
