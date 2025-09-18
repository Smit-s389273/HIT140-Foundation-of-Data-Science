# HIT140 Foundations of Data Science
# Objective 1: Investigation A (Bats vs Rats)
# Do bats perceive rats as predators? Analysis in Python.

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# --------------------
# 1. Load datasets
# --------------------
# Define base path
base_path = r"C:\Users\amdar\OneDrive\Desktop"

# Load CSV files using full paths
d1 = pd.read_csv(os.path.join(base_path, "dataset1.csv"))
d2 = pd.read_csv(os.path.join(base_path, "dataset2.csv"))

# Parse datetimes for dataset1
for col in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
    if col in d1.columns:
        d1[col] = pd.to_datetime(d1[col], dayfirst=True, errors="coerce")

# Parse datetimes for dataset2
if "time" in d2.columns:
    d2["time"] = pd.to_datetime(d2["time"], dayfirst=True, errors="coerce")

# --------------------
# 2. Descriptive Stats
# --------------------
print("=== Summary Counts ===")
print(f"Total rows: {len(d1)}")
print(d1["risk"].value_counts(normalize=True))
print(d1["reward"].value_counts(normalize=True))

# Crosstab risk vs reward
ct = pd.crosstab(d1["risk"], d1["reward"])
print("\n=== Risk vs Reward Crosstab ===")
print(ct)

# Timing summaries
print("\n=== Timing Variables Summary ===")
print(d1[["seconds_after_rat_arrival", "bat_landing_to_food", "hours_after_sunset"]].describe())

# --------------------
# 3. Inferential Tests
# --------------------
# Chi-square: risk ~ reward
chi2, p_chi2, dof, exp = stats.chi2_contingency(ct)
print("\nChi-square test (risk vs reward):")
print(f"chi2={chi2:.3f}, p={p_chi2:.4g}, dof={dof}")

# Mann–Whitney U test for vigilance proxies
def mannwhitney(col):
    x = d1.loc[d1["risk"] == 1, col].dropna()
    y = d1.loc[d1["risk"] == 0, col].dropna()
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    print(f"\nMann–Whitney U for {col}:")
    print(f"U={U:.1f}, p={p:.4g}")
    print(f"Median (risk=1)={np.median(x):.2f}, Median (risk=0)={np.median(y):.2f}")

mannwhitney("seconds_after_rat_arrival")
mannwhitney("bat_landing_to_food")

# Logistic regression: risk ~ timing + reward + hours_after_sunset
formula = "risk ~ seconds_after_rat_arrival + hours_after_sunset + reward"
model_df = d1.dropna(subset=["risk", "seconds_after_rat_arrival", "hours_after_sunset", "reward"])
logit = smf.logit(formula=formula, data=model_df).fit()
print("\n=== Logistic Regression Results ===")
print(logit.summary())
print("\nOdds Ratios:")
print(np.exp(logit.params))

# --------------------
# 4. Dataset2 Context
# --------------------
print("\n=== Dataset2 Correlations ===")
for var in ["rat_arrival_number", "rat_minutes"]:
    rho, p = stats.spearmanr(d2[var], d2["bat_landing_number"], nan_policy="omit")
    print(f"Spearman({var}, bat_landing_number) = {rho:.3f}, p={p:.4g}")

# --------------------
# 5. Visualisations
# --------------------
# Save figures to Desktop
def save_fig(name):
    plt.savefig(os.path.join(base_path, name))
    plt.close()

# Risk vs Reward proportions
(ct.T / ct.T.sum()).T.plot(kind="bar", stacked=True)
plt.title("Proportion of Rewards within Risk Groups")
plt.xlabel("Risk (0=avoid, 1=take risk)")
plt.ylabel("Proportion")
save_fig("fig_risk_reward_proportions.png")

# Boxplot: seconds_after_rat_arrival by risk
d1.boxplot(column="seconds_after_rat_arrival", by="risk")
plt.title("Seconds After Rat Arrival by Risk")
plt.suptitle("")
save_fig("fig_seconds_after_rat_arrival_by_risk.png")

# Boxplot: bat_landing_to_food by risk
d1.boxplot(column="bat_landing_to_food", by="risk")
plt.title("Bat Landing to Food Time by Risk")
plt.suptitle("")
save_fig("fig_bat_landing_to_food_by_risk.png")

# Scatter: rat arrivals vs bat landings
plt.scatter(d2["rat_arrival_number"], d2["bat_landing_number"], alpha=0.5)
plt.title("Rat Arrivals vs Bat Landings (30-min windows)")
plt.xlabel("Rat Arrival Number")
plt.ylabel("Bat Landing Number")
save_fig("fig_scatter_rat_arrivals_vs_bat_landings.png")

print("\nFigures saved to Desktop:")
print(" - fig_risk_reward_proportions.png")
print(" - fig_seconds_after_rat_arrival_by_risk.png")
print(" - fig_bat_landing_to_food_by_risk.png")
print(" - fig_scatter_rat_arrivals_vs_bat_landings.png")
