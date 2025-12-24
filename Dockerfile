# 1. صورة الأساس
FROM python:3.9-slim

# 2. تعيين مجلد العمل
WORKDIR /app

# 3. نسخ ملفات المشروع
COPY . /app

# 4. تثبيت المتطلبات
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. تعيين الأمر الافتراضي لتشغيل الـ API
CMD ["uvicorn", "src.04_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
