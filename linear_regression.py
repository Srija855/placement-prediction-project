import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# DATASET PATH
# =========================================================

CSV_PATH = r"/Users/garimellasrija/PycharmProjects/PythonProject/placement_predict_50k Dataset (3) (1).csv"


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

TARGET = "Salary Package"


# =========================================================
# GRAPH FOLDER
# =========================================================

GRAPH_FOLDER = os.path.join(
    "static",
    "linear_regression"
)

os.makedirs(GRAPH_FOLDER, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(CSV_PATH)

    df.columns = df.columns.str.strip()

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data():

    df = load_data()

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[required_columns].copy()

    for col in required_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna()

    X = data[FEATURES]

    y = data[TARGET]

    return data, X, y


# =========================================================
# CREATE EQUATION
# =========================================================

def create_equation(model):

    equation = f"Salary = {model.intercept_:.4f}"

    for feature, coefficient in zip(
        FEATURES,
        model.coef_
    ):

        sign = "+" if coefficient >= 0 else "-"

        equation += (
            f" {sign} "
            f"{abs(coefficient):.4f} × {feature}"
        )

    return equation


# =========================================================
# CREATE GRAPHS
# =========================================================

def create_regression_graphs(
    data,
    y_test,
    predictions,
    prefix=""
):

    # =====================================================
    # CGPA VS SALARY
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.scatter(
        data["CGPA"],
        data[TARGET],
        alpha=0.4
    )

    plt.xlabel("CGPA")
    plt.ylabel("Salary Package")

    if prefix:
        plt.title(
            f"CGPA vs Salary Package - {prefix}"
        )
    else:
        plt.title(
            "CGPA vs Salary Package"
        )

    plt.tight_layout()

    cgpa_filename = (
        f"cgpa_salary_{prefix.lower().replace(' ', '_')}.png"
        if prefix
        else "cgpa_salary.png"
    )

    cgpa_path = os.path.join(
        GRAPH_FOLDER,
        cgpa_filename
    )

    plt.savefig(cgpa_path)

    plt.close()


    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.5
    )

    min_value = min(
        y_test.min(),
        predictions.min()
    )

    max_value = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")

    if prefix:
        plt.title(
            f"Actual vs Predicted Salary - {prefix}"
        )
    else:
        plt.title(
            "Actual vs Predicted Salary"
        )

    plt.tight_layout()

    actual_filename = (
        f"actual_vs_predicted_{prefix.lower().replace(' ', '_')}.png"
        if prefix
        else "actual_vs_predicted.png"
    )

    actual_path = os.path.join(
        GRAPH_FOLDER,
        actual_filename
    )

    plt.savefig(actual_path)

    plt.close()


    # =====================================================
    # RESIDUAL ANALYSIS
    # =====================================================

    residuals = y_test - predictions

    plt.figure(figsize=(8, 5))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0
    )

    plt.xlabel("Predicted Salary")
    plt.ylabel("Residual")

    if prefix:
        plt.title(
            f"Residual Analysis - {prefix}"
        )
    else:
        plt.title(
            "Residual Analysis"
        )

    plt.tight_layout()

    residual_filename = (
        f"residual_analysis_{prefix.lower().replace(' ', '_')}.png"
        if prefix
        else "residual_analysis.png"
    )

    residual_path = os.path.join(
        GRAPH_FOLDER,
        residual_filename
    )

    plt.savefig(residual_path)

    plt.close()


    return {
        "cgpa_graph":
            f"linear_regression/{cgpa_filename}",

        "actual_predicted_graph":
            f"linear_regression/{actual_filename}",

        "residual_graph":
            f"linear_regression/{residual_filename}"
    }


# =========================================================
# WITHOUT REGULARIZATION
# =========================================================

def run_linear_regression():

    data, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )


    # =====================================================
    # GRAPHS
    # =====================================================

    graphs = create_regression_graphs(
        data,
        y_test,
        predictions
    )


    # =====================================================
    # CORRELATION
    # =====================================================

    correlation = data.corr(
        numeric_only=True
    )[TARGET]

    correlation_data = []

    for feature in FEATURES:

        correlation_data.append({

            "feature": feature,

            "correlation":
                round(
                    correlation[feature],
                    4
                )
        })


    # =====================================================
    # EQUATION
    # =====================================================

    equation = create_equation(
        model
    )


    return {

        "total_rows":
            len(data),

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "features":
            FEATURES,

        "correlation":
            correlation_data,

        "equation":
            equation,

        "mae":
            round(mae, 4),

        "mse":
            round(mse, 4),

        "rmse":
            round(rmse, 4),

        "r2":
            round(r2, 4),

        "cgpa_graph":
            graphs["cgpa_graph"],

        "actual_predicted_graph":
            graphs["actual_predicted_graph"],

        "residual_graph":
            graphs["residual_graph"]
    }


# =========================================================
# WITH REGULARIZATION
# =========================================================

def run_linear_regularization(selected_model="ridge"):

    data, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # =====================================================
    # SELECT MODEL
    # =====================================================

    if selected_model == "ridge":

        model_name = "Ridge Regression"
        model = Ridge(alpha=1.0)

    elif selected_model == "lasso":

        model_name = "Lasso Regression"
        model = Lasso(alpha=0.01)

    elif selected_model == "elastic":

        model_name = "Elastic Net"
        model = ElasticNet(
            alpha=0.01,
            l1_ratio=0.5
        )

    else:

        model_name = "Ridge Regression"
        model = Ridge(alpha=1.0)


    # =====================================================
    # TRAIN
    # =====================================================

    model.fit(
        X_train,
        y_train
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    predictions = model.predict(
        X_test
    )


    # =====================================================
    # METRICS
    # =====================================================

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )


    # =====================================================
    # GRAPHS
    # =====================================================

    graphs = create_regression_graphs(
        data,
        y_test,
        predictions,
        selected_model
    )


    # =====================================================
    # EQUATION
    # =====================================================

    equation = create_equation(
        model
    )


    # =====================================================
    # RETURN
    # =====================================================

    return {

        "total_rows":
            len(data),

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "features":
            FEATURES,

        "model":
            model_name,

        "mae":
            round(mae, 4),

        "mse":
            round(mse, 4),

        "rmse":
            round(rmse, 4),

        "r2":
            round(r2, 4),

        "equation":
            equation,

        "cgpa_graph":
            graphs["cgpa_graph"],

        "actual_predicted_graph":
            graphs["actual_predicted_graph"],

        "residual_graph":
            graphs["residual_graph"]
    }

# =========================================================
# SALARY PREDICTION WITHOUT REGULARIZATION
# =========================================================

def predict_salary(
    cgpa,
    aptitude,
    coding,
    interview
):

    data, X, y = prepare_data()

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    input_data = pd.DataFrame([{

        "CGPA":
            float(cgpa),

        "AptitudeTestScore":
            float(aptitude),

        "CodingTestScore":
            float(coding),

        "MockInterviewScore":
            float(interview)

    }])

    prediction = model.predict(
        input_data
    )[0]

    return round(
        float(prediction),
        2
    )


# =========================================================
# SALARY PREDICTION WITH REGULARIZATION
# =========================================================

def predict_regularized_salary(
    cgpa,
    aptitude,
    coding,
    interview,
    model_type="ridge"
):

    data, X, y = prepare_data()


    if model_type == "ridge":

        model = Ridge(
            alpha=1.0
        )

    elif model_type == "lasso":

        model = Lasso(
            alpha=0.01
        )

    else:

        model = ElasticNet(
            alpha=0.01,
            l1_ratio=0.5
        )


    model.fit(
        X,
        y
    )


    input_data = pd.DataFrame([{

        "CGPA":
            float(cgpa),

        "AptitudeTestScore":
            float(aptitude),

        "CodingTestScore":
            float(coding),

        "MockInterviewScore":
            float(interview)

    }])


    prediction = model.predict(
        input_data
    )[0]


    return round(
        float(prediction),
        2
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    result = run_linear_regression()

    print(
        "\n===== LINEAR REGRESSION WITHOUT REGULARIZATION ====="
    )

    print(
        "Total rows:",
        result["total_rows"]
    )

    print(
        "Training rows:",
        result["train_rows"]
    )

    print(
        "Testing rows:",
        result["test_rows"]
    )

    print(
        "\nEquation:"
    )

    print(
        result["equation"]
    )

    print(
        "\nPerformance:"
    )

    print(
        "MAE :",
        result["mae"]
    )

    print(
        "MSE :",
        result["mse"]
    )

    print(
        "RMSE:",
        result["rmse"]
    )

    print(
        "R2  :",
        result["r2"]
    )


    print(
        "\n===== LINEAR REGRESSION WITH REGULARIZATION ====="
    )

    regularized = run_linear_regularization()


    for model in regularized["models"]:

        print(
            model["model"],
            "| MAE:",
            model["mae"],
            "| MSE:",
            model["mse"],
            "| RMSE:",
            model["rmse"],
            "| R2:",
            model["r2"]
        )


    print(
        "\nBest Model:",
        regularized["best_model"]
    )

    print(
        "Best R2:",
        regularized["best_r2"]
    )

    print(
        "\nBest Model Equation:"
    )

    print(
        regularized["equation"]
    )