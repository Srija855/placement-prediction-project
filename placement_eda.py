
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from load_data import load_data


# ============================================================
# CHART DIRECTORY
# ============================================================

CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)


def _chart_path(filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename: str):
    plt.tight_layout()
    plt.savefig(
        _chart_path(filename),
        bbox_inches="tight",
        dpi=120
    )
    plt.close("all")


# ============================================================
# EDA
# ============================================================

def run_eda() -> dict:

    data = load_data()

    charts = []


    # ========================================================
    # 1. MISSING VALUES
    # ========================================================

    missing = data.isnull().sum()

    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

    if not missing_df.empty:

        plt.figure(figsize=(10, 5))

        sns.barplot(
            x=missing_df.index,
            y=missing_df["missing_pct"]
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.ylabel("Missing %")
        plt.xlabel("Column")
        plt.title("Missing Values by Column")

        _save("missing_values.png")
        charts.append("missing_values.png")


    # ========================================================
    # 2. DUPLICATE ROWS
    # ========================================================

    duplicate_count = int(
        data.duplicated().sum()
    )


    # ========================================================
    # 3. TARGET VARIABLE DISTRIBUTION
    # ========================================================

    target_counts = (
        data["PlacementStatus"]
        .value_counts()
        .to_dict()
    )

    plt.figure(figsize=(7, 5))

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.xlabel(
        "Placement Status"
    )

    plt.ylabel("Count")

    plt.title(
        "Placement Status Distribution"
    )

    _save("target_distribution.png")
    charts.append("target_distribution.png")


    # ========================================================
    # 4 & 5. NUMERIC FEATURE DISTRIBUTIONS
    # ========================================================

    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    for col in hist_cols:

        if col in data.columns:

            plt.figure(figsize=(8, 5))

            sns.histplot(
                data[col],
                kde=True
            )

            plt.title(
                f"Distribution of {col}"
            )

            plt.xlabel(col)
            plt.ylabel("Frequency")

            fname = (
                f"hist_{col.lower()}.png"
            )

            _save(fname)
            charts.append(fname)


    # ========================================================
    # 6. CGPA DISTRIBUTION WITH MEAN
    # ========================================================

    if "CGPA" in data.columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data["CGPA"],
            kde=True
        )

        mean_cgpa = data["CGPA"].mean()

        plt.axvline(
            x=mean_cgpa,
            linestyle="--",
            label=f"Mean = {mean_cgpa:.2f}"
        )

        plt.legend()

        plt.title(
            "CGPA Distribution with Mean"
        )

        plt.xlabel("CGPA")
        plt.ylabel("Frequency")

        _save("cgpa_mean.png")
        charts.append("cgpa_mean.png")


    # ========================================================
    # 7. OUTLIER DETECTION - BOX PLOTS
    # ========================================================

    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Salary Package"
    ]

    for col in box_cols:

        if col in data.columns:

            plt.figure(figsize=(10, 4))

            sns.boxplot(
                x=data[col]
            )

            plt.title(
                f"Box Plot - {col}"
            )

            plt.xlabel(col)

            fname = (
                f"box_{col.lower().replace(' ', '_')}.png"
            )

            _save(fname)
            charts.append(fname)


    # ========================================================
    # 8. CORRELATION HEATMAP
    # ========================================================

    numeric_data = data.select_dtypes(
        include=[np.number]
    )

    if not numeric_data.empty:

        corr = numeric_data.corr()

        plt.figure(
            figsize=(14, 10)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(
            "Correlation Heatmap"
        )

        _save("correlation_heatmap.png")
        charts.append(
            "correlation_heatmap.png"
        )


    # ========================================================
    # 9. CGPA VS SALARY PACKAGE
    # ========================================================

    if (
        "CGPA" in data.columns
        and "Salary Package" in data.columns
    ):

        plt.figure(figsize=(8, 5))

        sns.regplot(
            data=data,
            x="CGPA",
            y="Salary Package"
        )

        plt.title(
            "CGPA vs Salary Package"
        )

        plt.xlabel("CGPA")
        plt.ylabel("Salary Package")

        _save("cgpa_vs_salary.png")
        charts.append(
            "cgpa_vs_salary.png"
        )


    # ========================================================
    # 10. APTITUDE VS CODING TEST SCORE
    # ========================================================

    if (
        "AptitudeTestScore" in data.columns
        and "CodingTestScore" in data.columns
    ):

        plt.figure(figsize=(8, 5))

        sns.regplot(
            data=data,
            x="AptitudeTestScore",
            y="CodingTestScore"
        )

        plt.title(
            "Aptitude Test Score vs Coding Test Score"
        )

        plt.xlabel(
            "Aptitude Test Score"
        )

        plt.ylabel(
            "Coding Test Score"
        )

        _save("aptitude_vs_coding.png")
        charts.append(
            "aptitude_vs_coding.png"
        )


    # ========================================================
    # 11. CATEGORICAL FEATURE COUNTS
    # ========================================================

    categorical_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    for col in categorical_cols:

        if col in data.columns:

            plt.figure(figsize=(9, 5))

            sns.countplot(
                data=data,
                x=col
            )

            plt.title(
                f"Distribution of {col}"
            )

            plt.xlabel(col)
            plt.ylabel("Count")

            plt.xticks(
                rotation=45,
                ha="right"
            )

            fname = (
                f"categorical_{col.lower()}.png"
            )

            _save(fname)
            charts.append(fname)


    # ========================================================
    # 12. GENDER VS PLACEMENT STATUS
    # ========================================================

    if (
        "Gender" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=data,
            x="Gender",
            hue="PlacementStatus"
        )

        plt.title(
            "Gender vs Placement Status"
        )

        plt.xlabel("Gender")
        plt.ylabel("Number of Students")

        plt.legend(
            title="Placement Status"
        )

        _save("gender_vs_placement.png")
        charts.append(
            "gender_vs_placement.png"
        )


    # ========================================================
    # 13. COLLEGE TIER VS PLACEMENT STATUS
    # ========================================================

    if (
        "CollegeTier" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=data,
            x="CollegeTier",
            hue="PlacementStatus"
        )

        plt.title(
            "College Tier vs Placement Status"
        )

        plt.xlabel("College Tier")
        plt.ylabel("Number of Students")

        plt.legend(
            title="Placement Status"
        )

        _save("college_tier_vs_placement.png")
        charts.append(
            "college_tier_vs_placement.png"
        )


    # ========================================================
    # 14. SGPA TREND ACROSS SEMESTERS
    # ========================================================

    sem_cols = [
        "SGPA_Sem1",
        "SGPA_Sem2",
        "SGPA_Sem3",
        "SGPA_Sem4",
        "SGPA_Sem5",
        "SGPA_Sem6",
        "SGPA_Sem7",
        "SGPA_Sem8"
    ]

    existing_sem_cols = [
        col for col in sem_cols
        if col in data.columns
    ]

    if existing_sem_cols:

        avg_sgpa = data[
            existing_sem_cols
        ].mean()

        semesters = [
            i + 1
            for i in range(
                len(existing_sem_cols)
            )
        ]

        plt.figure(
            figsize=(10, 5)
        )

        sns.lineplot(
            x=semesters,
            y=avg_sgpa.values,
            marker="o"
        )

        plt.title(
            "Average SGPA Trend Across Semesters"
        )

        plt.xlabel("Semester")
        plt.ylabel("Average SGPA")

        plt.xticks(semesters)

        plt.grid(True)

        _save("sgpa_trend.png")
        charts.append(
            "sgpa_trend.png"
        )


    # ========================================================
    # 15. SALARY DISTRIBUTION OF PLACED STUDENTS
    # ========================================================

    if (
        "PlacementStatus" in data.columns
        and "Salary Package" in data.columns
    ):

        # Handles both string and numeric placement values
        placed_data = data[
            data["PlacementStatus"]
            .astype(str)
            .str.lower()
            .eq("placed")
        ]

        # If PlacementStatus uses 1 for placed
        if placed_data.empty:

            placed_data = data[
                data["PlacementStatus"] == 1
            ]

        if not placed_data.empty:

            plt.figure(
                figsize=(8, 5)
            )

            sns.histplot(
                data=placed_data,
                x="Salary Package",
                kde=True
            )

            plt.title(
                "Salary Distribution of Placed Students"
            )

            plt.xlabel(
                "Salary Package"
            )

            plt.ylabel(
                "Number of Students"
            )

            _save(
                "salary_distribution.png"
            )

            charts.append(
                "salary_distribution.png"
            )


    # ========================================================
    # 16. PAIRPLOT - MULTIVARIATE
    # ========================================================

    pair_cols = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
        "PlacementStatus"
    ]

    existing_pair_cols = [
        col for col in pair_cols
        if col in data.columns
    ]

    if len(existing_pair_cols) >= 3:

        pair_data = data[
            existing_pair_cols
        ].copy()

        # Pairplot can become very slow for large datasets.
        # Use a sample for visualization.
        if len(pair_data) > 3000:
            pair_data = pair_data.sample(
                3000,
                random_state=42
            )

        sns.pairplot(
            data=pair_data,
            hue="PlacementStatus"
            if "PlacementStatus"
            in pair_data.columns
            else None,
            diag_kind="hist"
        )

        plt.suptitle(
            "Pairwise Relationships Between Student Performance Features",
            y=1.02
        )

        fname = "pairplot_performance.png"

        plt.savefig(
            _chart_path(fname),
            bbox_inches="tight",
            dpi=120
        )

        plt.close("all")

        charts.append(fname)


    # ========================================================
    # RETURN EDA RESULTS
    # ========================================================

    missing_dict = {
        col: int(cnt)
        for col, cnt in missing.items()
        if cnt > 0
    }

    return {

        "n_rows": int(len(data)),

        "n_cols": int(len(data.columns)),

        "duplicate_count": duplicate_count,

        "missing": missing_dict,

        "target_counts": {
            str(k): int(v)
            for k, v in target_counts.items()
        },

        "charts": charts
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_eda()

    print(results)

