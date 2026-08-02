# Kaggle T4 Setup — 20-Patient Fair Comparison

## المبدأ
نُدرّب RED-CNN, ResNet, وLocalResidual (نموذجنا) على **نفس 20 مريضاً بنفس pipeline**.  
المقارنة داخلية فقط — ليست مقارنة بالجدول المنشور.

---

## المرضى الـ20 (اختر ملفاتهم من البيانات الكاملة)

```
Train (10): C095 C261 C218 C224 C099  L081 L248 L203 L219 L210
Val   ( 5): C202 C219 C107             L033 L187
Test  ( 5): C121 C249 C170             L241 L107
```

---

## الخطوة 1 — تحضير Kaggle Dataset

في notebook محلي أو Lightning:

```bash
# ابضغط فقط المرضى الـ20 من dataset/ و test/
zip -r ldct_20p.zip \
  dataset/C095 dataset/C261 dataset/C218 dataset/C224 dataset/C099 \
  dataset/L081 dataset/L248 dataset/L203 dataset/L219 dataset/L210 \
  dataset/C202 dataset/C219 dataset/C107 dataset/L033 dataset/L187 \
  test/C121    test/C249    test/C170    test/L241    test/L107
```

ثم ارفع `ldct_20p.zip` إلى Kaggle Datasets وأضفه للنوتبوك.

---

## الخطوة 2 — إعداد النوتبوك على Kaggle

أول خلية:

```python
import os, subprocess

# Clone المستودع
subprocess.run(["git", "clone", "https://github.com/BrahimSoufghalem/pseudo3d-ldct-denoising.git"], check=True)
os.chdir("pseudo3d-ldct-denoising")
subprocess.run(["git", "checkout", "feat/physics-spectral-denoiser"], check=True)

# تثبيت المتطلبات
subprocess.run(["pip", "install", "-q",
    "monai", "pydicom", "scikit-image", "torchmetrics",
    "pandas", "tqdm", "tensorboard"], check=True)
```

الخلية الثانية — فك ضغط البيانات:

```python
DATASET_ZIP = "/kaggle/input/ldct-20p/ldct_20p.zip"
subprocess.run(["unzip", "-q", DATASET_ZIP, "-d", "."], check=True)
```

---

## الخطوة 3 — تدريب الثلاثة نماذج

```python
import os
os.environ["HU_RANGE_PRESET"] = "benchmark"

# RED-CNN (~60-70 دقيقة على T4)
subprocess.run(["python", "train_20p.py", "--arch", "redcnn",
    "--num-workers", "2", "--cache-rate", "1.0"], check=True)

# ResNet (~60-70 دقيقة على T4)
subprocess.run(["python", "train_20p.py", "--arch", "resnet",
    "--num-workers", "2", "--cache-rate", "1.0"], check=True)

# LocalResidual (نموذجنا) (~60-70 دقيقة على T4)
subprocess.run(["python", "train_20p.py", "--arch", "local_residual",
    "--num-workers", "2", "--cache-rate", "1.0"], check=True)
```

> المجموع: ~3-4 ساعات GPU من الـ12 المتاحة أسبوعياً.

---

## الخطوة 4 — التقييم والمقارنة

```python
subprocess.run(["python", "evaluate_20p.py"], check=True)
```

النتيجة المتوقعة:

```
  [CHEST]
  Model                    PSNR       SSIM   RMSE_HU       VIF
  ------------------------------------------------------------------
    RED-CNN               ??.???     0.????   ??.??       0.????  
    ResNet                ??.???     0.????   ??.??       0.????  
  * LocalResidual (Ours)  ??.???     0.????   ??.??       0.????  
```

---

## معيار النجاح (20 مريض)

| النتيجة | الحكم |
|---|---|
| VIF_ours > VIF_resnet في الصدر | **نجاح** — ننتقل لـ100 مريض |
| VIF_ours > VIF_redcnn فقط | **جزئي** — نحلل الفجوة مع ResNet |
| VIF_ours < VIF_redcnn | **فشل** — مراجعة التصميم |

---

## ملاحظات T4

- `--num-workers 2` ضروري (Kaggle يحدد عمليات DataLoader)
- `--cache-rate 1.0` مناسب مع 20 مريض (~3 GB RAM)
- لا حاجة لـ`--batch-size` أقل — 64 مناسب لـT4 16GB مع هذه المعماريات
- إذا نفد RAM: اخفض `--cache-rate 0.5`
- احفظ `runs_20p/` كـOutput في نهاية الجلسة
