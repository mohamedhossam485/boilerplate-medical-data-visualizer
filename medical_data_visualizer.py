import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1) Load data
df = pd.read_csv("medical_examination.csv")

# 2) Add overweight column (BMI > 25 → 1 else 0)
bmi = df["weight"] / (df["height"] / 100) ** 2
df["overweight"] = (bmi > 25).astype(int)

# 3) Normalize cholesterol & gluc (0 = good, 1 = bad)
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"]        = (df["gluc"] > 1).astype(int)


def draw_cat_plot():
    # 4) Melt to long format
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
    )

    # 5) Group and get totals
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 6) Draw catplot
    g = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar"
    )

    # 7) Figure handle
    fig = g.fig

    # Do not modify
    fig.savefig("catplot.png")
    return fig


def draw_heat_map():
    # 8) Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 9) Correlation matrix
    corr = df_heat.corr()

    # 10) Upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 11) Set up figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 12) Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        ax=ax
    )

    # Do not modify
    fig.savefig("heatmap.png")
    return fig

