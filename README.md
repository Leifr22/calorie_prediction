# 🍽️ Нейросеть для предсказания калорийности блюд

Мультимодальная DL-модель, предсказывающая калорийность блюда по фотографии, списку ингредиентов и массе порции. Проект выполнен в рамках учебного модуля по Deep Learning.

## 🎯 Задача

Построить и обучить нейросеть, которая по входным данным (фото блюда + ингредиенты + масса) предсказывает число калорий в порции.

**Целевая метрика:** `MAE < 50` ккал на тестовой выборке.

## 📊 Результаты

| Метрика | Значение |
|---|---|
| **Test MAE** | **36.53 ккал** ✅ |
| Test RMSE | 61.44 ккал |
| Test MAPE | 16.9 % |
| Val MAE (best checkpoint) | 33.43 ккал |
| Baseline MAE (mass × avg density) | 104.12 ккал |
| Naive MAE (const = avg calories) | 168.52 ккал |
| Улучшение vs baseline | **65%** (в 2.85× лучше) |
| Медиана \|error\| | 20.09 ккал |

Цель достигнута с запасом 27%. Модель учится ~47 минут на CPU (12th Gen i5-12600KF), на GPU должна уложиться в 5–10 минут.

## 🏗️ Архитектура

Мультимодальная модель, объединяющая три источника информации:

```
┌─────────────┐      ┌──────────────────────┐
│ Image 224×  │────▶ │ EfficientNet-B0      │──┐
│ 224 × 3     │      │ (pretrained ImageNet)│  │
└─────────────┘      └──────────────────────┘  │
                                               │
┌─────────────┐      ┌──────────────────────┐  │   ┌─────────┐   ┌─────────┐
│ Ingredients │────▶ │ nn.Embedding(556,64) │──┼──▶│ Concat  │──▶│ MLP     │──▶ density (ккал/г)
│ [max 40 ID] │      │ + masked mean pool   │  │   │         │   │ 256→128 │           │
└─────────────┘      └──────────────────────┘  │   └─────────┘   │ →1      │           │
                                               │                 └─────────┘           ▼
┌─────────────┐      ┌──────────────────────┐  │                                × mass = калории
│ Mass scalar │────▶ │ Linear(1,16) + ReLU  │──┘
└─────────────┘      └──────────────────────┘
```

**Ключевые решения:**

- **Предсказание плотности калорий (ккал/г)**, а не калорий напрямую. Масса подаётся как явный признак, а итоговое число получаем умножением: `calories = density × mass`. Это снимает с сети работу масштабирования и улучшает обобщение.
- **EfficientNet-B0** — компактный бэкбон (4.4M параметров, из них 4 в самом backbone), быстрый fine-tune. Проверено: на этом датасете (3262 блюда) более крупные модели не дают выигрыша из-за переобучения.
- **Embedding ингредиентов с masked mean pooling** — компактное представление переменной длины, padding-aware (`padding_idx=0` в `nn.Embedding` не обучается).
- **Два learning rate**: `lr_head=1e-3` для новых слоёв, `lr_backbone=1e-4` для fine-tune — стандартный приём transfer learning.
- **SmoothL1 loss** в пространстве density — устойчив к выбросам; **метрики MAE/RMSE считаются в исходных калориях** (для соответствия ТЗ).

## 📁 Структура репозитория

```
.
├── scripts/
│   ├── __init__.py
│   ├── dataset.py     # DishDataset, парсинг ингредиентов, трансформы, DataLoader-ы
│   ├── model.py       # CalorieModel: image + ingredients + mass → density
│   ├── utils.py       # train(), predict_loader(), set_seed(), load_model_from_checkpoint()
│   └── config.py      # Загрузка YAML-конфига
├── configs/
│   └── config.yaml    # Все гиперпараметры и пути в одном месте
├── data/              # .gitkeep; сам датасет не коммитим
├── notebook.ipynb     # EDA + обучение + инференс + анализ ошибок (этапы 1–4)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Установка и запуск

### 1. Окружение

```bash
git clone <this-repo-url>
cd calorie_prediction
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Для ускорения на NVIDIA GPU:
```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Датасет

Датасет доступен по [ссылке (1.3 ГБ)](https://disk.yandex.ru/d/kz9g5msVqtahiw). Распакуй его в папку `data/` в корне проекта так, чтобы получилась структура:

```
data/
├── dish.csv
├── ingredients.csv
└── images/
    ├── dish_1561662216/rgb.png
    ├── dish_1561662054/rgb.png
    └── ...
```

Если хочешь положить датасет в другое место — поменяй пути в `configs/config.yaml` (блок `paths`).

### 3. Обучение

**Вариант 1 — через ноутбук** (рекомендуется, так делал проект):

```bash
jupyter lab notebook.ipynb
```

Ноутбук разбит на 4 этапа:
- **Этап 1** — EDA
- **Этап 2** — smoke-test пайплайна (2 эпохи, проверка что всё работает)
- **Этап 3** — полное обучение (30 эпох, early stopping patience=8)
- **Этап 4** — инференс на test, анализ топ-5 худших примеров

**Вариант 2 — из кода:**

```python
from scripts.config import load_config
from scripts.utils import train

cfg = load_config("configs/config.yaml")
result = train(cfg)

print(f"Best val MAE: {result['best_val_mae']:.2f}")
print(f"Checkpoint:   {result['best_ckpt_path']}")
```

### 4. Инференс на своих данных

```python
import torch
from scripts.config import load_config
from scripts.utils import load_model_from_checkpoint, predict_loader
from scripts.dataset import build_dataloaders

cfg = load_config("configs/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, ckpt = load_model_from_checkpoint(
    f"{cfg.paths.checkpoints_dir}/{cfg.paths.best_model_name}", cfg, device
)

data = build_dataloaders(cfg, seed=cfg.seed)
dish_ids, preds, targets, masses = predict_loader(
    model, data["test_loader"], device, predict_density=cfg.train.predict_density
)
```

## ⚙️ Гиперпараметры (configs/config.yaml)

| Параметр | Значение | Примечание |
|---|---|---|
| `epochs` | 30 | + early stopping `patience=8` |
| `batch_size` | 64 | |
| `lr_head` | 1e-3 | новые слои (embedding, MLP-голова) |
| `lr_backbone` | 1e-4 | fine-tune EfficientNet-B0 |
| `weight_decay` | 1e-4 | L2-регуляризация |
| `scheduler` | cosine | по шагам, `T_max = epochs × len(loader)` |
| `loss` | SmoothL1 | в пространстве density |
| `dropout` | 0.3 | в MLP-голове |
| `image_size` | 224 | стандарт для ImageNet-бэкбонов |
| `max_ingredients` | 40 | обрезка длины списка (max в данных = 34) |
| `predict_density` | true | выход модели — ккал/г, затем ×mass |
| `seed` | 42 | |

## 🎲 Воспроизводимость

- `set_seed()` фиксирует `random`, `numpy`, `torch` (CPU+CUDA), `PYTHONHASHSEED`
- `torch.backends.cudnn.deterministic = True`
- DataLoader с фиксированным `torch.Generator`
- Статистики массы (`mass_mean`, `mass_std`) считаются **только на train** и сохраняются в чекпоинт — нет утечки в val/test

Запуск `train()` с одним и тем же конфигом даёт идентичный результат.

## 📈 Анализ слабых мест модели (Этап 4)

Топ-5 самых тяжёлых примеров из test показали **систематическую проблему**: модель занижает калорийность для блюд с аномально высокой плотностью калорий.

| # | Истина плотность | Предикт плотность | Ключевой компонент |
|---|---|---|---|
| 1 | 2.17 ккал/г | 1.22 ккал/г | goat cheese |
| 2 | 3.85 ккал/г | 2.25 ккал/г | almonds |
| 3 | 2.79 ккал/г | 1.62 ккал/г | almonds + apple |
| 4 | 1.90 ккал/г | 1.38 ккал/г | pizza + chicken |
| 5 | 2.18 ккал/г | 1.56 ккал/г | almonds + cauliflower |

**Причины:**
1. **Смещение в сторону средней плотности** — в датасете средняя плотность ~1.28 ккал/г, обучение с MAE-подобным лоссом минимизирует именно модуль ошибки в калориях, что толкает предсказания к центру распределения.
2. **Редкие высококалорийные ингредиенты** — миндаль встречается в 3 из 5 худших случаев. Таких блюд в train было мало, поэтому embedding для миндаля не сформировал сильный сигнал.
3. **Визуальная неочевидность плотности** — по фото сложно отличить обычный салат от салата с горстью миндаля или жирной заправкой.

**Возможные направления улучшения:**
- Weighted sampler для апсемплинга блюд с высокой плотностью калорий
- Более мощный бэкбон (ConvNeXt-Tiny, EfficientNet-B3)
- Логарифмирование таргета — чтобы штрафовать ошибки равномерно по шкале

## 📚 Данные

- **Всего блюд**: 3262 (train: 2755, test: 507)
- **После фильтрации нулевых**: train: 2754, test: 506 (удалены 2 блюда с `total_calories=0`)
- **Внутри train сделан val-сплит** 15% → train: 2341, val: 413
- **Уникальных ингредиентов**: 555 в справочнике, 200 реально встречаются в блюдах
- **Фото**: все 640×480 RGB, .png

## 🛠️ Стек

- Python 3.12
- PyTorch 2.7.1 + torchvision
- timm (EfficientNet-B0 с весами ImageNet)
- pandas, numpy, scikit-learn
- matplotlib, seaborn — визуализация
- PyYAML — конфиг
- Jupyter Lab — ноутбук
