import pandas as pd
from pathlib import Path

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
df["ts"] = pd.to_datetime(df["ts"])
df["date"] = df["ts"].dt.date

# -----------------------------
# Snapshot date (robust)
# -----------------------------
snapshot_ts = df["ts"].quantile(0.8)

# نستخدم فقط البيانات قبل الـ snapshot للـ features
df_features_base = df[df["ts"] <= snapshot_ts].copy()

# -----------------------------
# Feature Engineering
# -----------------------------

# Activity
activity = df_features_base.groupby("userId").agg(
    total_events=("ts", "count"),
    total_sessions=("sessionId", "nunique"),
    active_days=("date", "nunique"),
    first_seen=("ts", "min"),
    last_seen=("ts", "max"),
)

activity["tenure_days"] = (activity["last_seen"] - activity["first_seen"]).dt.days
activity["days_since_last_activity"] = (snapshot_ts - activity["last_seen"]).dt.days
activity["avg_events_per_session"] = (
    activity["total_events"] / activity["total_sessions"]
)
activity = activity.drop(columns=["first_seen", "last_seen"])

# Listening
songs = df_features_base[df_features_base["page"] == "NextSong"]
listening = songs.groupby("userId").agg(
    total_songs=("song", "count"),
    total_listen_time=("length", "sum"),
    avg_song_length=("length", "mean"),
    unique_artists=("artist", "nunique"),
    unique_songs=("song", "nunique"),
)

# Engagement
engagement = df_features_base.groupby("userId").agg(
    thumbs_up_count=("page", lambda x: (x == "ThumbsUp").sum()),
    thumbs_down_count=("page", lambda x: (x == "ThumbsDown").sum()),
    add_to_playlist_count=("page", lambda x: (x == "AddToPlaylist").sum()),
    add_friend_count=("page", lambda x: (x == "AddFriend").sum()),
)

# Friction
friction = df_features_base.groupby("userId").agg(
    logout_count=("page", lambda x: (x == "Logout").sum()),
    help_page_views=("page", lambda x: (x == "Help").sum()),
    error_rate=("status", lambda x: (x != 200).mean()),
)

# Plan
plan = df_features_base.groupby("userId").agg(
    is_paid=("level", lambda x: int((x == "paid").any())),
    paid_ratio=("level", lambda x: (x == "paid").mean()),
)


# Windows
def window_count(days, page=None):
    cutoff = snapshot_ts - pd.Timedelta(days=days)
    subset = df_features_base[df_features_base["ts"] >= cutoff]
    if page:
        subset = subset[subset["page"] == page]
    return subset.groupby("userId").size()


windows = pd.DataFrame(index=activity.index)
windows["events_last_7d"] = window_count(7)
windows["events_last_30d"] = window_count(30)
windows["songs_last_30d"] = window_count(30, "NextSong")

# Merge
df_features = (
    activity.join(listening).join(engagement).join(friction).join(plan).join(windows)
).fillna(0)

# -----------------------------
# Churn definition
# -----------------------------
activity_after_snapshot = df[df["ts"] > snapshot_ts].groupby("userId")["ts"].count()

df_features["churn"] = (~df_features.index.isin(activity_after_snapshot.index)).astype(
    int
)

# -----------------------------
# Save
# -----------------------------
feature_path.parent.mkdir(parents=True, exist_ok=True)
df_features.reset_index(inplace=True)
df_features.to_csv(feature_path, index=False)

print(f"✅ Features dataset saved: {feature_path}")
print("Class distribution:\n", df_features["churn"].value_counts())
