# 02_feature_engineering.py
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths relative to this script
# -----------------------------
current_dir = Path(__file__).parent       # src/
project_root = current_dir.parent        # CustomerChurnProject/
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# ملفات البيانات
cleaned_path = data_dir / "cleaned_customer_churn.csv"
feature_path = data_dir / "customer_churn_with_features.csv"

# -----------------------------
# قراءة البيانات النظيفة
# -----------------------------
if not cleaned_path.exists():
    raise FileNotFoundError(f"❌ الملف غير موجود: {cleaned_path}")

df = pd.read_csv(cleaned_path)
print(f"✅ Loaded cleaned data: {df.shape} rows, {df.shape[1]} columns")

# -----------------------------
# تنظيف البيانات الأساسية
# -----------------------------
df = df[df['userId'].notna()]

# إنشاء last_auth: إذا الصفحة Logout → Logged Out، وإلا خذ auth
df['last_auth'] = df.apply(lambda row: 'Logged Out' if row['page']=='Logout' else row['auth'], axis=1)

# -----------------------------
# Feature Engineering
# -----------------------------
# عدد الجلسات، إجمالي وقت الاستماع، عدد الفنانين والأغاني المختلفة لكل مستخدم
df_sessions = df.groupby('userId').agg({
    'sessionId': 'count',
    'length': 'sum',
    'artist': 'nunique',
    'song': 'nunique'
}).rename(columns={
    'sessionId': 'total_sessions',
    'length': 'total_listen_time',
    'artist': 'unique_artists',
    'song': 'unique_songs'
})

# آخر حالة تسجيل دخول لكل مستخدم
df_last_auth = df.groupby('userId')['last_auth'].last().rename('last_auth_status')

# حساب PositivePage و NegativePage
positive_pages = ['NextSong', 'Home']       # ممكن تضيف صفحات تعتبر إيجابية
negative_pages = ['Logout', 'Cancel']       # ممكن تضيف صفحات تعتبر سلبية

df_positive = df[df['page'].isin(positive_pages)].groupby('userId')['page'].count().rename('PositivePage')
df_negative = df[df['page'].isin(negative_pages)].groupby('userId')['page'].count().rename('NegativePage')

# دمج كل المميزات في DataFrame واحد
df_features = df_sessions.join(df_last_auth)
df_features = df_features.join(df_positive)
df_features = df_features.join(df_negative)
df_features.reset_index(inplace=True)

# استبدال NaN في PositivePage/NegativePage بـ 0
df_features['PositivePage'] = df_features['PositivePage'].fillna(0).astype(int)
df_features['NegativePage'] = df_features['NegativePage'].fillna(0).astype(int)

# إنشاء عمود churn: إذا آخر تسجيل logout → churn=1
df_features['churn'] = (df_features['last_auth_status'] == 'Logged Out').astype(int)

# -----------------------------
# حفظ ملف المميزات النهائي
# -----------------------------
feature_path.parent.mkdir(parents=True, exist_ok=True)
df_features.to_csv(feature_path, index=False)
print(f"✅ Features dataset saved: {feature_path}")
print(df_features.head())
