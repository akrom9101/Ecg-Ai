# ECG Intelligence — Backend (hybrid: DSP + Gemini)

## Fayllar
- `ecg_core.py` — haqiqiy signal-processing (OpenCV + scipy). Rasmdan EKG chizig'ini
  ajratadi, R-peaklarni topadi, BPM va ritm muntazamligini matematik hisoblaydi.
  Hech qanday AI API'ga bog'liq emas.
- `main.py` — FastAPI ilova. `/analyze` endpoint: rasmni qabul qiladi → `ecg_core`
  bilan o'lchaydi → Gemini'ga (faqat interpretatsiya uchun) yuboradi → JSON qaytaradi.
- `requirements.txt` — kerakli kutubxonalar.

## 1. Gemini API key olish (bepul)
1. https://aistudio.google.com/apikey ga kiring (Google akkaunt bilan)
2. "Create API key" bosing, kalitni nusxalang

## 2. Render'da sozlash (mavjud ecg-ai-1.onrender.com xizmatingizga)
1. Bu 3 faylni GitHub repo'ingizga qo'shing (backend papkasiga)
2. Render dashboard → sizning service → **Environment** → yangi env var qo'shing:
   - Key: `GEMINI_API_KEY`
   - Value: (1-qadamda olgan kalit)
3. Render → **Settings** → Start Command:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. Deploy qiling (Render avtomatik push'dan keyin qayta deploy qiladi)

## 3. Tekshirish
Brauzerda `https://ecg-ai-1.onrender.com/` ochib `{"status":"ok",...}` ko'rinishini tekshiring.
Keyin frontend orqali (yoki curl bilan) haqiqiy rasm yuklab `/analyze`ni sinab ko'ring.

## Muhim: rasm talablari
Bu DSP pipeline **grid qog'ozli** EKG rasmlari uchun ishlaydi (fon to'r chiziqlari qizil/
pushti, chiziq qora/quyuq bo'lishi kerak). Juda xira, qiya burchakdan olingan yoki
to'r ko'rinmaydigan rasmlar uchun aniqlik pasayishi mumkin — demo uchun frontal, yorug'
rasmlardan foydalaning.

## Agar Gemini xato bersa / kalit hali yo'q bo'lsa
`main.py` ichida oddiy qoidaga asoslangan fallback bor (`_rule_based_label`) — kalit
qo'yilmagan bo'lsa ham `/analyze` ishlayveradi, faqat "rhythm" izohi soddaroq bo'ladi.
Bu demo hech qachon to'liq to'xtab qolmasligini ta'minlaydi.
