# 🐉 دليل تشغيل SentinelAI على Kali Linux — خطوة بخطوة

> دليل كامل لتثبيت وتشغيل مشروع **SentinelAI** على نظام **Kali Linux**،
> من الصفر حتى فتح وحدة تحكم SOC في المتصفح — بشرح كل أمر وسببه.

---

## 📋 أولًا: متطلبات النظام (قبل أن تبدأ)

| المتطلب | الحد الأدنى | ملاحظات |
|---|---|---|
| نظام التشغيل | Kali Linux (أي إصدار حديث: 2023.1 فما فوق) | يعمل أيضًا على أي توزيعة Debian/Ubuntu |
| Python | **3.10 – 3.13** | ⚠️ كالي **2025.2+** يأتي بـ Python **3.14** — يجب استخدام miniforge بـ 3.12 (انظر «حل المشاكل» فقرة 3) |
| الذاكرة RAM | 2 جيجابايت (يفضل 4) | التشغيل العادي خفيف؛ التدريب الكامل يحتاج أكثر |
| مساحة القرص | **~2 جيجابايت** حرة | للتثبيت + تحميل البيانات (عند إعادة التدريب) |
| الإنترنت | مطلوب **مرة واحدة فقط** | لتثبيت المكتبات (وإن أعدت التدريب فتحتاج تحميل البيانات) |

> ⚠️ **مهم جدًا:** في Kali، نظام Python هو Python «مُدار من apt»، ويمنع تثبيت
> المكتبات عليه مباشرة (خطأ `externally-managed-environment`). لذلك **نستخدم دائمًا
> بيئة افتراضية `venv`** — وهذا ما يفعله ملف التشغيل الجاهز تلقائيًا.

---

## ⚡ الطريقة السريعة (سطر واحد)

افتح الطرفية (Terminal) ونفّذ:

```bash
git clone https://github.com/ko456123mo-jpg/SentinelAI.git
cd SentinelAI
bash run_kali.sh
```

`run_kali.sh` يقوم تلقائيًا بـ: كشف نسخة Python ← إنشاء بيئة `.venv` ← تثبيت
المتطلبات (أول مرة فقط) ← تشغيل الواجهة وفتح المتصفح على `http://localhost:7860`.

> إذا أردت **فهم ما يحدث** (الأفضل للعرض أمام المقيّمة)، اتبع الطريقة اليدوية أدناه.

---

## 🛠️ الطريقة اليدوية — خطوة بخطوة (موصى بها للفهم والمناقشة)

### الخطوة 1 — تحديث النظام وتثبيت الأدوات الأساسية

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

- `git` لنسخ المشروع، `python3` المفسر، `python3-venv` لإنشاء بيئة معزولة،
  `python3-pip` لمدير الحزم.
- ستُطلب كلمة مرور root (في كالي الافتراضية `kali` أو ما حددته أنت).

### الخطوة 2 — التحقق من نسخة Python

```bash
python3 --version
```

يجب أن ترى `Python 3.10` حتى `3.13`. لو ظهرت `3.14` أو أعلى، انتقل إلى قسم
«حل المشاكل» (الفقرة 3).

### الخطوة 3 — نسخ المشروع من GitHub

```bash
git clone https://github.com/ko456123mo-jpg/SentinelAI.git
cd SentinelAI
```

> بديل بدون Git: من صفحة المستودع اضغط **Code ← Download ZIP**، ثم فك الضغط
> وادخل المجلد: `cd ~/Downloads/SentinelAI-main`

### الخطوة 4 — إنشاء بيئة افتراضية معزولة

```bash
python3 -m venv .venv
```

- تُنشئ مجلد `.venv` يحوي نسخة Python مستقلة — فلا تتعارض مع نظام كالي.

### الخطوة 5 — تفعيل البيئة

```bash
source .venv/bin/activate
```

- ستلاحظ ظهور `(.venv)` في بداية سطر الأوامر — أي أنك داخل البيئة الآن.
- (لإلغاء التفعيل لاحقًا اكتب `deactivate`)

### الخطوة 6 — تحديث pip وتثبيت المتطلبات

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- يستغرق **5–10 دقائق** ويحمل ~600 ميجابايت (TensorFlow + scikit-learn وغيرها).
- هذا **أثقل جزء** ويحدث مرة واحدة فقط.

### الخطوة 7 — تشغيل وحدة التحكم

```bash
python -m src.webapp
```

سترى رسالة شبيهة بـ:

```
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:7860
```

- المتصفح **يفتح تلقائيًا** على `http://localhost:7860`.
- إن لم يُفتح تلقائيًا، افتحه يدويًا في Firefox/Chromium داخل كالي.

### الخطوة 8 — تجربة الواجهة (جولة سريعة)

| زر/تبويب | ماذا تفعل |
|---|---|
| **Dashboard** | إحصائيات حية: دقة RF 99.6%، CNN 100%، LSTM F1 0.944… |
| **Flow Analyzer** | اضغط زر **DDoS flood** → يصنّف الهجوم ويعزل المضيف فورًا |
| **Message Scanner** | جرّب رسالة التصيّد الجاهزة → يكشف النطاق الخبيث (IOC) |
| **Malware Images** | اختر صورة عائلة برمجية → يتعرف عليها بثقة عالية |
| **System** | زر **Run tests** يشغّل الاختبارات الـ21 داخل الواجهة |

### الخطوة 9 — إيقاف التشغيل

- اضغط **Ctrl+C** في الطرفية لإيقاف الخادم.
- ثم اكتب `deactivate` للخروج من البيئة.

---

## 🧪 أوامر إضافية مفيدة (ستحتاجها في العرض)

```bash
# شغّل الاختبارات الآلية (يجب أن تنتهي بـ 21/21 OK)
python -m unittest discover -s tests

# شغّل الوكيل الذكي على 27 حدثًا فقط (سريع)
python -m src.main --stages agent

# أعد تدريب كل شيء من الصفر (يتطلب إنترنت أول مرة لتحميل البيانات)
python -m src.main

# أعد توليد التقرير DOCX والعرض PPTX
python docs/generate_report.py
python docs/generate_presentation.py
```

---

## 🩹 حل المشاكل الشائعة على Kali

### 1) `error: externally-managed-environment` عند تثبيت المكتبات
**السبب:** حاولت التثبيت على Python النظام مباشرة.
**الحل:** لا تستخدم `pip install` خارج بيئة — فعّل `.venv` أولًا:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) `ensurepip` / `No module named venv` عند إنشاء البيئة
**السبب:** حزمة `python3-venv` غير مثبتة.
**الحل:**
```bash
sudo apt install -y python3-venv python3-pip
```

### 3) نسخة Python هي 3.14 أو أعلى (غير مدعومة من TensorFlow) ⚠️ الأكثر شيوعًا
**السبب:** كالي 2025.2 وما بعده يأتي بـ Python 3.14 افتراضيًا، وTensorFlow لا يدعمها بعد (حتى 3.13 فقط). عند التثبيت ستظهر رسالة:
`ERROR: Ignored the following versions that require a different Python version`
**الحل:** ثبّت miniforge وأنشئ بيئة بـ Python 3.12 (لا يحتاج sudo):
```bash
# احذف أي بيئة قديمة خاطئة
rm -rf .venv venv

# ثبّت miniforge (مرة واحدة)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b

# أنشئ بيئة sentinel بـ Python 3.12
~/miniforge/bin/conda create -n sentinel python=3.12 -y

# فعّلها وثبّت المتطلبات
source ~/miniforge/bin/activate sentinel
pip install -r requirements.txt
python -m src.webapp
```
> ملف `run_kali.sh` يكتشف بيئة conda باسم `sentinel` تلقائيًا في كل مرة لاحقة.

### 4) المنفذ 7860 مشغول بالفعل
**السبب:** نسخة أخرى من البرنامج تعمل.
**الحل:** أوقف العملية القديمة أولًا:
```bash
sudo fuser -k 7860/tcp      # أو: kill <رقم العملية>
python -m src.webapp
```
أو غيّر المنفذ من نهاية `src/webapp.py` (السطر `app.run(... port=7860 ...)`).

### 5) `Permission denied` عند تشغيل `run_kali.sh`
**الحل:** أعطه صلاحية التنفيذ ثم شغّله:
```bash
chmod +x run_kali.sh
./run_kali.sh
```
(أو ببساطة `bash run_kali.sh` — تعمل بدون صلاحية تنفيذ)

### 6) مساحة قرص غير كافية
**السبب:** تحميل TensorFlow + البيانات يحتاج ~2 جيجابايت.
**الحل:** تحقق ثم حرر مساحة:
```bash
df -h
sudo apt clean
```

### 7) المتصفح لا يفتح تلقائيًا (شائع عند تشغيل كالي كـ root أو عبر SSH)
**الحل:** افتح يدويًا داخل كالي: `firefox http://localhost:7860`
(الخدمة تعمل على 0.0.0.0، أي يمكن فتحها من أي متصفح على نفس الجهاز)

### 8) خطأ يتعلق بـ NLTK (stopwords) عند فحص رسالة لأول مرة
**الحل:** يتحمّل تلقائيًا عند أول استخدام (يتطلب إنترنت لحظيًا)،
ويوجد بديل مدمج يعمل دون إنترنت — لا حاجة لأي تدخل.

---

## 📌 ملاحظات ختامية

- **لا تحتاج كرت شاشة (GPU)** — المشروع يستخدم `tensorflow-cpu`.
- **كل شيء داخل المستودع:** البيانات + 17 نموذجًا مدرَّبًا + النتائج،
  لذا تعمل الواجهة **فورًا بعد التثبيت** بدون أي تدريب.
- **وقت التثبيت أول مرة:** ~10 دقائق (مرة واحدة). **وقت التشغيل بعدها:** ثوانٍ.
- للتوقف في أي لحظة: `Ctrl+C` ثم `deactivate`.
