import pandas as pd
from pathlib import Path
import numpy as np

# -----------------------------
# Paths
# -----------------------------
current_dir = Path(__file__).parent
project_root = current_dir.parent
data_dir = project_root / "data"
feature_path = data_dir / "customer_churn_with_features.csv"
cleaned_path = data_dir / "cleaned_customer_churn.csv"

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv(cleaned_path)
df['ts'] = pd.to_datetime(df['ts'])

# -----------------------------
# Feature Engineering
# -----------------------------
# Aggregate sessions per user
df_sessions = df.groupby('userId').agg({
    'sessionId':'count',
    'length':'sum',
    'artist':'nunique',
    'song':'nunique'
}).rename(columns={
    'sessionId':'total_sessions',
    'length':'total_listen_time',
    'artist':'unique_artists',
    'song':'unique_songs'
})

# Positive / Negative pages
positive_pages = ['NextSong', 'Home', 'ThumbsUp', 'AddToPlaylist']
negative_pages = ['Logout', 'Cancel', 'ThumbsDown']

df['is_positive'] = df['page'].isin(positive_pages).astype(int)
df['is_negative'] = df['page'].isin(negative_pages).astype(int)

# -----------------------------
# Churn definition
# -----------------------------
# آخر نشاط لكل مستخدم
last_activity = df.groupby('userId')['ts'].max()

# ترتيب البيانات لكل مستخدم حسب الوقت تنازلي
df_sorted = df.sort_values(['userId','ts'], ascending=[True, False])

# آخر 50 نشاط لكل مستخدم
last_50 = df_sorted.groupby('userId').head(50)
positive_count = last_50.groupby('userId')['is_positive'].sum()
total_events_50 = last_50.groupby('userId')['ts'].count()

# تعريف churn: إذا آخر 50 نشاط كلها سلبي أو أقل من 50 نشاط، أو آخر نشاط قبل أكثر من 10 أيام
df_churn = df_sessions.copy()
df_churn['churn'] = (
    ((positive_count == 0) & (total_events_50 == 50)) |  # آخر 50 سلبي كلها
    (total_events_50 < 50) |  # أقل من 50 نشاط
    (df['ts'].max() - last_activity > pd.Timedelta(days=10))  # آخر نشاط قبل أكثر من 10 أيام
).astype(int)

# Merge مع main features
df_features = df_sessions.join(df_churn[['churn']])
df_features = df_features.fillna(0).astype(int)

# Relative metrics
df_features['total_events'] = df_features['total_sessions']
df_features['positive_ratio'] = df_features['total_events'] / df_features['total_events']  # يبقى 1 لكل مستخدم
df_features['negative_ratio'] = 1 - df_features['positive_ratio']  # يبقى 0 لكل مستخدم
df_features['avg_listen_time'] = df_features['total_listen_time'] / df_features['total_sessions']

# -----------------------------
# Save features
# -----------------------------
feature_path.parent.mkdir(parents=True, exist_ok=True)
df_features.reset_index(inplace=True)
df_features.to_csv(feature_path, index=False)
print(f"✅ Features dataset saved: {feature_path}")
print("Class distribution:\n", df_features['churn'].value_counts())
