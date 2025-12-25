import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
current_dir = Path(__file__).parent  # src/
project_root = current_dir.parent  # CustomerChurnProject/
data_dir = project_root / "data"

# Original files
mini_path = data_dir / "customer_churn_mini.json"
full_path = data_dir / "customer_churn.json"

# Output cleaned & merged file
output_path = data_dir / "cleaned_customer_churn.csv"
missing_users_path = data_dir / "missing_userId_rows.csv"
plots_dir = data_dir / "plots"
plots_dir.mkdir(exist_ok=True, parents=True)


# -----------------------------
# Load data
# -----------------------------
df_mini = pd.read_json(mini_path, lines=True)
df_full = pd.read_json(full_path, lines=True)

print(f"✅ Loaded mini dataset: {df_mini.shape[0]} rows")
print(f"✅ Loaded full dataset: {df_full.shape[0]} rows")

# -----------------------------
# Merge datasets
# -----------------------------
df = pd.concat([df_mini, df_full], ignore_index=True)
print(f"🔗 Merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -----------------------------
# Handle rows without userId
# -----------------------------
missing_userid_df = df[
    df["userId"].isna() | df["userId"].astype(str).str.strip().eq("")
]
if not missing_userid_df.empty:
    print(
        f"⚠️ Found {missing_userid_df.shape[0]} rows without userId. Saving them separately."
    )
    missing_userid_df.to_csv(missing_users_path, index=False)

# Remove these rows from main dataset
df = df[~(df["userId"].isna() | df["userId"].astype(str).str.strip().eq(""))]

# Convert userId to string and strip whitespace
df["userId"] = df["userId"].astype(str).str.strip()

# -----------------------------
# Fill missing values for other columns
# -----------------------------
df.fillna(
    {
        "level": "free",
        "gender": "unknown",
        "last_auth": "Logged Out",
        "page": "unknown",
    },
    inplace=True,
)

# Drop duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"⚠️ Found {duplicates} duplicate rows. Dropping them")
    df.drop_duplicates(inplace=True)

print(f"Total unique users after cleaning: {df['userId'].nunique()}")

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(output_path, index=False)
print(f"🎯 Cleaned & merged dataset saved: {output_path}")

# -----------------------------
# Prepare for EDA (unique users)
# -----------------------------
df_users = df.drop_duplicates(
    subset="userId"
)  # each user counted once for level/page plots

# -----------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------

# 1. User distribution by level
plt.figure(figsize=(6, 4))
sns.countplot(data=df_users, x="level", palette="pastel")
plt.title("User distribution by level")
plt.xlabel("Level")
plt.ylabel("Number of unique users")
plt.tight_layout()
plt.savefig(plots_dir / "users_by_level.png")
plt.close()

# 2. Top 10 most visited pages as Pie Chart
top_pages = df["page"].value_counts().head(10)
plt.figure(figsize=(8, 8))
plt.pie(
    top_pages.values,
    labels=top_pages.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("pastel"),
)
plt.title("Top 10 most visited pages")
plt.tight_layout()
plt.savefig(plots_dir / "top_10_pages_pie.png")
plt.close()


# 3. Top 20 users by session count (activity)
top_users = df["userId"].value_counts().head(20)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_users.index, y=top_users.values, palette="pastel")
plt.title("Top 20 users by session count")
plt.xlabel("userId")
plt.ylabel("Number of sessions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plots_dir / "top_users_sessions.png")
plt.close()

print("📊 All plots saved in:", plots_dir)
print(f"⚠️ Rows without userId saved in: {missing_users_path}")
