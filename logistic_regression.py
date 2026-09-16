import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


CSV_PATH = r"/Users/garimellasrija/PycharmProjects/PythonProject/placement_predict_50k Dataset (3) (1).csv"


FEATURES = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

TARGET = "PlacementStatus"


GRAPH_FOLDER = os.path.join(
    "static",
    "logistic_regression"
)

os.makedirs(
    GRAPH_FOLDER,
    exist_ok=True
)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    df = pd.read_csv(CSV_PATH)

    df.columns = df.columns.str.strip()

    return df


# ==========================================================
# PREPARE DATA
# ==========================================================

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

    # Convert feature columns to numbers
    for col in FEATURES:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Convert target to binary
    def convert_target(value):

        if pd.isna(value):
            return np.nan

        # Numeric target
        if isinstance(value, (int, float, np.integer, np.floating)):

            return 1 if float(value) == 1 else 0

        value = str(value).strip().lower()

        if value in [
            "placed",
            "1",
            "yes",
            "true"
        ]:
            return 1

        if value in [
            "not placed",
            "notplaced",
            "0",
            "no",
            "false"
        ]:
            return 0

        return np.nan

    data[TARGET] = data[TARGET].apply(
        convert_target
    )

    data = data.dropna()

    data[TARGET] = data[TARGET].astype(int)

    X = data[FEATURES]

    y = data[TARGET]

    return data, X, y


# ==========================================================
# SIGMOID FUNCTION
# ==========================================================

def sigmoid(z):

    z = np.clip(
        z,
        -500,
        500
    )

    return 1 / (1 + np.exp(-z))


# ==========================================================
# CREATE CONFUSION MATRIX
# ==========================================================

def create_confusion_matrix(
    y_test,
    predictions,
    filename="confusion_matrix.png"
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Placed",
            "Placed"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display.plot(
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    path = os.path.join(
        GRAPH_FOLDER,
        filename
    )

    plt.savefig(path)

    plt.close()

    return (
        "logistic_regression/"
        + filename
    )


# ==========================================================
# WITHOUT REGULARIZATION
# ==========================================================

def run_logistic_regression():

    data, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Very large C means almost no regularization
    model = LogisticRegression(
        C=1e10,
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    confusion_graph = create_confusion_matrix(
        y_test,
        predictions,
        "confusion_matrix.png"
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

        "model":
            "Logistic Regression",

        "accuracy":
            round(
                accuracy,
                4
            ),

        "threshold":
            0.5,

        "confusion_matrix":
            cm.tolist(),

        "confusion_graph":
            confusion_graph
    }


# ==========================================================
# REGULARIZED LOGISTIC REGRESSION
# ==========================================================

def run_logistic_regularization(
    selected_model="ridge"
):

    data, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # ------------------------------------------------------
    # RIDGE / L2
    # ------------------------------------------------------

    if selected_model == "ridge":

        model_name = "Ridge Logistic Regression (L2)"

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000
        )


    # ------------------------------------------------------
    # LASSO / L1
    # ------------------------------------------------------

    elif selected_model == "lasso":

        model_name = "Lasso Logistic Regression (L1)"

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=1000
        )


    # ------------------------------------------------------
    # ELASTIC NET
    # ------------------------------------------------------

    elif selected_model == "elastic":

        model_name = "Elastic Net Logistic Regression"

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=2000
        )


    else:

        selected_model = "ridge"

        model_name = "Ridge Logistic Regression (L2)"

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000
        )


    # Train model
    model.fit(
        X_train,
        y_train
    )


    # Probability
    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # 0.5 threshold
    predictions = (
        probabilities >= 0.5
    ).astype(int)


    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )


    # Different filename for each model
    confusion_filename = (
        f"confusion_matrix_{selected_model}.png"
    )

    confusion_graph = create_confusion_matrix(
        y_test,
        predictions,
        confusion_filename
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

        "model":
            model_name,

        "accuracy":
            round(
                accuracy,
                4
            ),

        "threshold":
            0.5,

        "confusion_matrix":
            cm.tolist(),

        "confusion_graph":
            confusion_graph
    }


# ==========================================================
# PREDICT WITHOUT REGULARIZATION
# ==========================================================

def predict_placement(
    cgpa,
    aptitude,
    coding,
    interview
):

    data, X, y = prepare_data()

    model = LogisticRegression(
        C=1e10,
        max_iter=1000
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


    probability = model.predict_proba(
        input_data
    )[0][1]


    prediction = (
        1
        if probability >= 0.5
        else 0
    )


    result = (
        "Placed"
        if prediction == 1
        else "Not Placed"
    )


    return {

        "result":
            result,

        "probability":
            round(
                float(probability) * 100,
                2
            ),

        "prediction":
            prediction
    }


# ==========================================================
# PREDICT WITH REGULARIZATION
# ==========================================================

def predict_regularized_placement(
    cgpa,
    aptitude,
    coding,
    interview,
    model_type="ridge"
):

    data, X, y = prepare_data()


    if model_type == "ridge":

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000
        )


    elif model_type == "lasso":

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            max_iter=1000
        )


    elif model_type == "elastic":

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=2000
        )


    else:

        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000
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


    probability = model.predict_proba(
        input_data
    )[0][1]


    prediction = (
        1
        if probability >= 0.5
        else 0
    )


    result = (
        "Placed"
        if prediction == 1
        else "Not Placed"
    )


    return {

        "result":
            result,

        "probability":
            round(
                float(probability) * 100,
                2
            ),

        "prediction":
            prediction
    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    result = run_logistic_regression()

    print("\n===== LOGISTIC REGRESSION =====")

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
        "Accuracy:",
        result["accuracy"]
    )

    print(
        "Threshold:",
        result["threshold"]
    )


    print(
        "\n===== REGULARIZATION ====="
    )

    for model_type in [
        "ridge",
        "lasso",
        "elastic"
    ]:

        result = run_logistic_regularization(
            model_type
        )

        print(
            result["model"],
            "| Accuracy:",
            result["accuracy"]
        )