# 📘 دليل تشغيل مشروع SentinelAI — خطوة بخطوة

---

## 🖥️ الجزء الأول: التحضير (مرة واحدة فقط)

### الخطوة 1: تثبيت Python
1. اذهب إلى **python.org/downloads** وحمّل أحدث إصدار (3.10 أو أحدث)
2. شغّل المثبّت و⚠️ **مهم جداً**: علّم على مربع
   **`Add Python to PATH`** قبل الضغط على Install Now
3. للتأكد من النجاح: افتح موجه الأوامر واكتب:
   ```
   python --version
   ```
   يجب أن يظهر رقم إصدار (مثل `Python 3.12.x`)

### الخطوة 2: تنزيل المشروع من GitHub
- الطريقة السهلة: افتح `github.com/ko456123mo-jpg/SentinelAI`
  → زر الأخضر **`<> Code`** → **`Download ZIP`** → فك الضغط
- أو بالأوامر (إن كان git مثبتاً):
  ```
  git clone https://github.com/ko456123mo-jpg/SentinelAI.git
  ```

### الخطوة 3: فتح موجه الأوامر داخل مجلد المشروع
- افتح مجلد `SentinelAI` في File Explorer
- اكتب في شريط العنوان أعلى النافذة: **`cmd`** ثم Enter
- (بديل للماك: Terminal | للينكس: Ctrl+Alt+T ثم cd)

### الخطوة 4: تثبيت المكتبات (3-5 دقائق)
```
pip install -r requirements.txt
```
> ⚠️ إن فشلت أمراض TensorFlow على جهاز قديم، جرّب:
> `pip install tensorflow-cpu==2.16.1`

### الخطوة 5 (اختيارية لكن موصى بها): بيانات السبام الحقيقية
النظام يعمل بدونها (يستخدم قائمة احتياطية)، لكن للحصول على النتائج
الحقيقية نفسها (5,572 رسالة):
1. حمّل: `archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip`
2. فك الضغط
3. ضع ملف `SMSSpamCollection` داخل مجلد **`data/raw/`**

---

## 🚀 الجزء الثاني: التشغيل

### التشغيل الكامل (الأمر الأساسي — احفظه!)
```
python -m src.main
```
**ماذا يحدث؟** المراحل الإحدى عشرة بالترتيب:
جمع البيانات → المعالجة → 6 نماذج خاضعة + CV + ضبط → غير خاضع (5 تقنيات)
→ شبكة MLP → NLP (TF-IDF + LSTM) → CNN → Q-Learning → الوكيل + التقرير

**المدة المتوقعة:** 3-6 دقائق (حسب قوة الجهاز)
**علامات النجاح:** سطور `[data]`, `acc=...`, `[figure]` ثم
`PIPELINE COMPLETE` في النهاية

### تشغيل مراحل محددة فقط
```
python -m src.main --stages collect,preprocess,supervised
python -m src.main --stages agent
```

### تخطي مراحل (مثلاً لتسريع العرض)
```
python -m src.main --skip cv,dl
```

### عرض قائمة أسماء المراحل
```
python -m src.main --list
```
الأسماء: `collect, preprocess, supervised, unsupervised, dl, nlp, cv, rl, agent`

### تشغيل الاختبارات (21 اختباراً)
```
python -m unittest discover -s tests
```
النتيجة المطلوبة: `OK` — تعني 21/21 ناجحة

### 🖥️ تشغيل الواجهة الرسومية (وحدة تحكم SOC)
```
pip install flask          # مرة واحدة فقط
python -m src.webapp       # ← ثم افتح المتصفح على: http://localhost:7860
```
- لوحة Dashboard بالإحصائيات والرسوم
- محلل تدفقات بأزرار هجمات جاهزة (DDoS / PortScan / BruteForce / Botnet)
- فاحص رسائل (سبام/تصيّد + فحص IOC ضد القائمة السوداء)
- مصنف صور البرمجيات (ارفع صورة أو استخدم العينات)
- مولد التقرير الأمني من الأحداث الحالية
- للإيقاف: Ctrl+C داخل الطرفية

---

## 📂 الجزء الثالث: أين أجد النتائج بعد التشغيل؟

| المجلد | ماذا يحتوي |
|---|---|
| `results/figures/` | 37 رسماً بيانياً (PNG) — جاهزة للعرض |
| `results/reports/security_report.md` | **التقرير الأمني الذي يكتبه الوكيل** — أهم مخرج حي |
| `results/reports/*.json` | كل المقاييس بالأرقام |
| `results/predictions/` | التنبؤات + الأخطاء المصنّفة (للمناقشة) |
| `models/` | 17 نموذجاً مدرّباً محفوظاً |

---

## 🔧 مشاكل شائعة وحلولها

| المشكلة | الحل |
|---|---|
| `'python' is not recognized` | أعد تثبيت Python مع `Add to PATH` أو استخدم `py` بدل `python` |
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow-cpu` |
| التشغيل بطيء جداً | طبيعي على CPU (خصوصاً LSTM/CNN) — انتظر 3-6 دقائق |
| نفاد الذاكرة | أغلق البرامج الأخرى، أو شغّل بـ `--skip cv,dl` |
| لا يوجد إنترنت | يعمل! يستخدم قائمة نطاقات احتياطية موثقة |
| لا تظهر الرسوم أثناء التشغيل | طبيعي — كلها تُحفظ في `results/figures/` (النظام headless) |

---

## 🎤 سيناريو مقترح للعرض أمام المهندسة (5 دقائق)

```
1) python -m src.main --list          ← اشرح بنية المراحل
2) python -m src.main                  ← شغّل كاملاً (اتركه يعمل واشرح أثناءها)
3) افتح results/reports/security_report.md   ← مخرج الوكيل الحي
4) افتح models/rl_policy.csv            ← سياسة RL المتعلمة
5) python -m unittest discover -s tests    ← اختم بـ 21/21 OK
```

> 💡 نصيحة: البذرة ثابتة (42) لذا النتائج ستتطابق تقريباً في كل تشغيل —
> يمكنك التدرب مسبقاً ومعرفة الأرقام التي ستظهر.
