import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

tree = ET.parse("config/users.xml")
root = tree.getroot()

users = []
for u in root.findall("user"):
    users.append({
        "name": u.get("name"),
        "group": u.get("group")
    })
users_df = pd.DataFrame(users)

rating_files = glob.glob("ratings/*.csv")
latest_files = {}

for f in rating_files:
    filename = os.path.basename(f)
    if "_" not in filename:
        continue
    user = filename.split("_")[0]  # usr01
    try:
        timestamp_str = filename.split("_ratings_")[1].replace(".csv", "")
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
    except Exception:
        timestamp = datetime.min  
    if user not in latest_files or timestamp > latest_files[user]["time"]:
        latest_files[user] = {"path": f, "time": timestamp}

print(f"{len(latest_files)} unique user rating files found.")


all_data = []
for user, info in latest_files.items():
    f = info["path"]

    match = users_df.loc[users_df["name"] == user, "group"]
    if match.empty:
        print(f"⚠️ Skipping {user}: not found in users.xml (probably removed or inactive)")
        continue

    group = match.values[0]
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue

    df["user"] = user
    df["group"] = group
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"📊 Loaded total {len(data)} rows from {len(all_data)} users.")

data[["rating", "naturalness"]] = data[["rating", "naturalness"]].apply(pd.to_numeric, errors="coerce")
group_means = (
    data.groupby(["group", "emotion", "wav_path"])
    [["rating", "naturalness"]]
    .mean()
    .reset_index()
)

os.makedirs("ratings_aggregated", exist_ok=True)
for g in sorted(group_means["group"].unique()):
    out = group_means[group_means["group"] == g]
    out.to_csv(f"ratings_aggregated/{g}_average.csv", index=False)
    print(f"💾 Saved {g}_average.csv")


data[["rating", "naturalness"]] = data[["rating", "naturalness"]].apply(pd.to_numeric, errors="coerce")
overall = (
    data.groupby(["emotion", "wav_path"])
    [["rating", "naturalness"]]
    .mean()
    .reset_index()
)
overall.to_csv("ratings_aggregated/ratings_average.csv", index=False)
print("✅ Saved ratings_average.csv")