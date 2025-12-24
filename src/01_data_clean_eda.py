import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths relative to this script
# -----------------------------
current_dir = Path(__file__).parent       # src/
project_root = current_dir.parent        # CustomerChurnProject/
data_dir = project_root / "data"
data_path = data_dir / "customer_churn_mini.json"
output_path = data_dir / "cleaned_customer_churn.csv"
plots_dir = data_dir / "plots"
plots_dir.mkdir(exist_ok=True, parents=True)  # إنشاء فولدر للرسومات

# التأكد من وجود المجلد قبل الحفظ
output_path.parent.mkdir(parents=True, exist_ok=True)

# قراءة JSON مع كل صف JSON مستقل
df = pd.read_json(data_path, lines=True)
print(f"✅ Loaded data: {df.shape} rows, {df.shape[1]} columns")
print(df.head())

# معالجة القيم الفارغة
df.fillna({'level':'free','gender':'unknown','last_auth':'Logged Out'}, inplace=True)

# تحليل سريع: عدد المستخدمين
print(f"Total users: {df['userId'].nunique()}")

# حفظ نسخة منظفة
df.to_csv(output_path, index=False)
print(f"🎯 Cleaned dataset saved: {output_path}")

# -----------------------------
# رسومات تحليلية
# -----------------------------

# 1. توزيع المستخدمين حسب المستوى (level)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='level', palette='pastel')
plt.title('توزيع المستخدمين حسب المستوى')
plt.xlabel('المستوى')
plt.ylabel('عدد المستخدمين')
plt.tight_layout()
plt.savefig(plots_dir / "users_by_level.png")
plt.close()

# 2. توزيع المستخدمين حسب الجنس (gender)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='gender', palette='pastel')
plt.title('توزيع المستخدمين حسب الجنس')
plt.xlabel('الجنس')
plt.ylabel('عدد المستخدمين')
plt.tight_layout()
plt.savefig(plots_dir / "users_by_gender.png")
plt.close()

# 3. عدد الجلسات لكل مستخدم (top 20 مستخدم)
top_users = df['userId'].value_counts().head(20)
plt.figure(figsize=(8,5))
sns.barplot(x=top_users.index.astype(str), y=top_users.values, palette='pastel')
plt.title('أعلى 20 مستخدم بعدد الجلسات')
plt.xlabel('userId')
plt.ylabel('عدد الجلسات')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plots_dir / "top_users_sessions.png")
plt.close()

print("📊 الرسومات حفظت في:", plots_dir)
