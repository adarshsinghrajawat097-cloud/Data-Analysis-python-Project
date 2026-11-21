"""
Diwali Sales Analysis 
"""

import numpy as np
import pandas as pd

CSV_PATH = "Diwali Sales Data.csv"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="unicode_escape")

    section("Shape & Head")
    print(df.shape)
    print(df.head())
    df.info()

    # blank/unrelated columns
    df = df.drop(columns=["Status", "unnamed1"], errors="ignore")

    section("Null values (cleaning se pehle)")
    print(df.isnull().sum())

    # null rows
    df = df.dropna().copy()

    # data type
    df["Amount"] = df["Amount"].astype("int")

    # Marital_Status code: 1 = Married, 0 = Single (numpy se label)
    df["Marital_Label"] = np.where(df["Marital_Status"] == 1, "Married", "Single")

    section("Cleaned data")
    print("Shape:", df.shape)
    print("Amount dtype:", df["Amount"].dtype)
    print("Columns:", list(df.columns))
    print("Duplicate rows:", df.duplicated().sum())

    section("describe()")
    print(df[["Age", "Orders", "Amount"]].describe())

    # numpy summary
    amt = df["Amount"].to_numpy()
    print(
        f"\nNumPy -> mean: {np.mean(amt):.2f} | median: {np.median(amt):.2f} "
        f"| std: {np.std(amt):.2f} | 90th percentile: {np.percentile(amt, 90):.2f}"
    )
    return df


def count_table(df, col, by=None):
    """Kitne buyers (rows)."""
    if by:
        return pd.crosstab(df[col], df[by])
    return df[col].value_counts()


def sum_table(df, group_cols, value_col, top=None):
    """Total Amount / Orders group-wise, sabse zyada upar."""
    out = (
        df.groupby(group_cols, as_index=False)[value_col]
        .sum()
        .sort_values(by=value_col, ascending=False)
    )
    return out.head(top) if top else out


def run_eda(df: pd.DataFrame) -> None:
    section("Gender")
    print(count_table(df, "Gender"))
    print(sum_table(df, ["Gender"], "Amount").to_string(index=False))

    section("Age Group")
    print(count_table(df, "Age Group", by="Gender"))
    print(sum_table(df, ["Age Group"], "Amount").to_string(index=False))

    section("State (Top 10)")
    print(sum_table(df, ["State"], "Orders", top=10).to_string(index=False))
    print(sum_table(df, ["State"], "Amount", top=10).to_string(index=False))

    section("Marital Status")
    print(count_table(df, "Marital_Label"))
    print(sum_table(df, ["Marital_Label", "Gender"], "Amount").to_string(index=False))

    section("Occupation")
    print(count_table(df, "Occupation"))
    print(sum_table(df, ["Occupation"], "Amount").to_string(index=False))

    section("Product Category")
    print(count_table(df, "Product_Category"))
    print(sum_table(df, ["Product_Category"], "Amount", top=10).to_string(index=False))

    section("Top 10 Product_ID by Orders")
    print(sum_table(df, ["Product_ID"], "Orders", top=10).to_string(index=False))


def print_conclusion() -> None:
    section("Conclusion")
    print(
        "Single women aged 26–35 drive Diwali spend, "
        "Based in UP, Maharashtra and Karnataka, working in IT, Healthcare and Aviation,"
        "and buying mostly Food, Clothing and Electronics.."
    )


def main() -> None:
    df = load_and_clean(CSV_PATH)
    run_eda(df)
    print_conclusion()
     # clean file save
    df.to_csv("Diwali Sales Cleaned.csv", index=False, encoding="utf-8-sig")
    print("Clean file : Diwali Sales Cleaned.csv")


    



if __name__ == "__main__":
    main()
