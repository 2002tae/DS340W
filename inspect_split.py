import os, pickle
import pandas as pd

os.makedirs("splits", exist_ok=True)

# paper name -> assignment name
SPLITS = [("train", "Training"), ("valid", "Test"), ("test", "Validation (unseen)")]

summary, daily = [], []
for split, name in SPLITS:
    with open(f"data/opensource/csi300_dl_{split}.pkl", "rb") as f:
        dl = pickle.load(f)
    idx = dl.get_index().to_frame(index=False)
    print(split, idx.columns.tolist(), len(idx))

    date_col = "datetime"
    idx.to_csv(f"splits/{split}_index.csv.gz", index=False)

    counts = idx.groupby(date_col).size().rename("n_stocks").reset_index()
    counts["split_paper"] = split
    counts["split_assignment"] = name
    daily.append(counts)

    summary.append({
        "split_paper": split,
        "split_assignment": name,
        "samples": len(idx),
        "trading_days": idx[date_col].nunique(),
        "first_date": idx[date_col].min().date(),
        "last_date": idx[date_col].max().date(),
    })

summary = pd.DataFrame(summary)
total_days = summary["trading_days"].sum()
summary["pct_of_days"] = (summary["trading_days"] / total_days * 100).round(1)
summary.to_csv("splits/split_summary.csv", index=False)
pd.concat(daily).to_csv("splits/daily_counts.csv", index=False)
print(summary.to_string(index=False))

# timeline figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 3))
for c in daily:
    ax.plot(c["datetime"], c["n_stocks"], lw=0.8, label=c["split_assignment"].iloc[0])
ax.set_ylabel("stocks per day")
ax.set_title("CSI300 chronological split (MASTER, opensource data)")
ax.legend()
fig.tight_layout()
fig.savefig("splits/split_timeline.png", dpi=150)
