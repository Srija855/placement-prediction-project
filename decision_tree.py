import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score


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
    "decision_tree"
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
# CONVERT TARGET
# =========================================================

def convert_target(value):

    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
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


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data():

    df = load_data()

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = df[required_columns].copy()

    for column in FEATURES:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        data[column] = data[column].fillna(
            data[column].median()
        )

    data[TARGET] = data[TARGET].apply(
        convert_target
    )

    data = data.dropna(
        subset=[TARGET]
    )

    data[TARGET] = data[TARGET].astype(int)

    X = data[FEATURES]

    y = data[TARGET]

    return data, X, y


# =========================================================
# CREATE MODEL
# =========================================================

def get_model(model_type):

    if model_type == "decision_tree":

        return (
            "Decision Tree",

            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                random_state=42
            )
        )


    elif model_type == "random_forest":

        return (
            "Random Forest",

            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        )


    elif model_type == "adaboost":

        return (
            "AdaBoost",

            AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=42
            )
        )


    elif model_type == "gradient_boosting":

        return (
            "Gradient Boosting",

            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        )


    elif model_type == "xgboost":

        try:

            from xgboost import XGBClassifier

        except ImportError:

            raise ImportError(
                "XGBoost is not installed. "
                "Run: pip install xgboost"
            )

        return (
            "XGBoost",

            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss"
            )
        )


    elif model_type == "lightgbm":

        try:

            from lightgbm import LGBMClassifier

        except ImportError:

            raise ImportError(
                "LightGBM is not installed. "
                "Run: pip install lightgbm"
            )

        return (
            "LightGBM",

            LGBMClassifier(
                boosting_type="gbdt",
                n_estimators=100,
                learning_rate=0.1,
                max_depth=-1,
                num_leaves=31,
                max_bin=255,
                random_state=42,
                verbosity=-1,
                importance_type="gain"
            )
        )


    return get_model("decision_tree")


# =========================================================
# GINI
# =========================================================

def calculate_gini(y):

    probabilities = pd.Series(y).value_counts(
        normalize=True
    )

    return 1 - sum(
        probability ** 2
        for probability in probabilities
    )


# =========================================================
# ENTROPY
# =========================================================

def calculate_entropy(y):

    probabilities = pd.Series(y).value_counts(
        normalize=True
    )

    return -sum(
        probability * np.log2(probability)
        for probability in probabilities
        if probability > 0
    )


# =========================================================
# DECISION TREE INFORMATION GAIN
# =========================================================

def calculate_information_gain(
    model,
    X_train,
    y_train
):

    tree = model.tree_

    root = 0

    feature_index = tree.feature[root]

    threshold = tree.threshold[root]

    if feature_index == -2:

        return 0.0

    feature_values = X_train.iloc[
        :,
        feature_index
    ]

    left_mask = (
        feature_values <= threshold
    )

    right_mask = ~left_mask

    y_left = y_train[left_mask]

    y_right = y_train[right_mask]

    parent_entropy = calculate_entropy(
        y_train
    )

    left_entropy = calculate_entropy(
        y_left
    )

    right_entropy = calculate_entropy(
        y_right
    )

    total = len(y_train)

    weighted_entropy = (

        (len(y_left) / total)
        * left_entropy

        +

        (len(y_right) / total)
        * right_entropy

    )

    information_gain = (
        parent_entropy
        - weighted_entropy
    )

    return information_gain


# =========================================================
# FEATURE IMPORTANCE GRAPH
# =========================================================

def create_feature_importance(
    model,
    model_type
):

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return None

    importance = model.feature_importances_

    importance_df = pd.DataFrame({

        "feature": FEATURES,

        "importance": importance

    })

    importance_df = importance_df.sort_values(
        "importance"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    filename = (
        f"feature_importance_{model_type}.png"
    )

    path = os.path.join(
        GRAPH_FOLDER,
        filename
    )

    plt.savefig(path)

    plt.close()

    return (
        "decision_tree/"
        + filename
    )


# =========================================================
# GET ALGORITHM TOPICS
# =========================================================

def get_topics(
    model,
    model_type,
    X_train,
    y_train
):

    topics = {}


    # =====================================================
    # DECISION TREE
    # =====================================================

    if model_type == "decision_tree":

        gini = calculate_gini(y_train)

        entropy = calculate_entropy(y_train)

        information_gain = (
            calculate_information_gain(
                model,
                X_train,
                y_train
            )
        )

        root_feature_index = (
            model.tree_.feature[0]
        )

        if root_feature_index >= 0:

            root_feature = FEATURES[
                root_feature_index
            ]

            threshold = model.tree_.threshold[0]

            splitting = (
                f"{root_feature} <= "
                f"{threshold:.4f}"
            )

        else:

            splitting = "No split"


        topics["Gini Impurity"] = (
            f"{gini:.4f}"
        )

        topics["Entropy"] = (
            f"{entropy:.4f}"
        )

        topics["Information Gain"] = (
            f"{information_gain:.4f}"
        )

        topics["Tree Depth"] = (
            model.get_depth()
        )

        topics["Splitting Criteria"] = (
            f"Gini | {splitting}"
        )


    # =====================================================
    # RANDOM FOREST
    # =====================================================

    elif model_type == "random_forest":

        topics["Bagging"] = (
            "Applied"
        )

        topics["Bootstrap Sampling"] = (
            "Enabled"
            if model.bootstrap
            else "Disabled"
        )

        topics["Multiple Decision Trees"] = (
            f"{len(model.estimators_)} trees"
        )

        topics["Random Feature Selection"] = (
            f"max_features = "
            f"{model.max_features}"
        )

        topics["Number of Trees"] = (
            model.n_estimators
        )


    # =====================================================
    # ADABOOST
    # =====================================================

    elif model_type == "adaboost":

        weak_learner = type(
            model.estimators_[0]
        ).__name__

        learner_depth = (
            model.estimators_[0]
            .get_depth()
        )

        topics["Weak Learners"] = (
            f"{weak_learner}, "
            f"Depth = {learner_depth}"
        )

        topics["Boosting"] = (
            "Sequential boosting applied"
        )

        # Actual learner/sample weighting information
        if hasattr(
            model,
            "estimator_weights_"
        ):

            weights = (
                model.estimator_weights_
            )

            topics["Sample Weights"] = (
                f"Initial = "
                f"{1 / len(X_train):.6f}, "
                f"Final learner weight = "
                f"{weights[-1]:.4f}"
            )

        else:

            topics["Sample Weights"] = (
                "Updated during boosting"
            )

        topics["Number of Estimators"] = (
            model.n_estimators
        )

        topics["Learning Rate"] = (
            model.learning_rate
        )


    # =====================================================
    # GRADIENT BOOSTING
    # =====================================================

    elif model_type == "gradient_boosting":

        topics["Sequential Trees"] = (
            f"{model.n_estimators} trees"
        )

        # Calculate actual residual values
        staged_probabilities = (
            list(
                model.staged_predict_proba(
                    X_train
                )
            )
        )

        if staged_probabilities:

            first_probability = (
                staged_probabilities[0][:, 1]
            )

            final_probability = (
                staged_probabilities[-1][:, 1]
            )

            first_residual = (
                y_train.values
                - first_probability
            )

            final_residual = (
                y_train.values
                - final_probability
            )

            topics["Residuals"] = (
                f"Initial MAE = "
                f"{np.mean(np.abs(first_residual)):.4f}, "
                f"Final MAE = "
                f"{np.mean(np.abs(final_residual)):.4f}"
            )

        else:

            topics["Residuals"] = (
                "Calculated sequentially"
            )

        topics["Gradient Descent"] = (
            "Applied"
        )

        topics["Learning Rate"] = (
            model.learning_rate
        )

        topics["Number of Estimators"] = (
            model.n_estimators
        )

        topics["Tree Depth"] = (
            model.max_depth
        )


    # =====================================================
    # XGBOOST
    # =====================================================

    elif model_type == "xgboost":

        topics["Gradient Boosting"] = (
            "Applied"
        )

        topics["Regularization"] = (
            f"L1 = {model.reg_alpha}, "
            f"L2 = {model.reg_lambda}"
        )

        topics["Learning Rate"] = (
            model.learning_rate
        )

        topics["Number of Estimators"] = (
            model.n_estimators
        )

        topics["Maximum Depth"] = (
            model.max_depth
        )

        topics["Subsampling"] = (
            f"{model.subsample * 100:.0f}%"
        )


    # =====================================================
    # LIGHTGBM
    # =====================================================

    elif model_type == "lightgbm":

        topics["Gradient Boosting"] = (
            f"Boosting type = "
            f"{model.boosting_type}"
        )

        topics["Leaf-Wise Growth"] = (
            f"Applied | "
            f"num_leaves = {model.num_leaves}"
        )

        topics["Histogram-Based Learning"] = (
            f"Applied | "
            f"max_bin = {model.max_bin}"
        )

        topics["Learning Rate"] = (
            model.learning_rate
        )

        topics["Number of Estimators"] = (
            model.n_estimators
        )

        topics["Maximum Depth / Leaves"] = (
            f"Depth = {model.max_depth}, "
            f"Leaves = {model.num_leaves}"
        )


    return topics


# =========================================================
# RUN ALGORITHM
# =========================================================

def run_decision_tree(
    model_type="decision_tree"
):

    data, X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    model_name, model = get_model(
        model_type
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Test prediction
    predictions = model.predict(
        X_test
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Feature importance
    feature_graph = create_feature_importance(
        model,
        model_type
    )

    feature_importance = {}

    if hasattr(
        model,
        "feature_importances_"
    ):

        for feature, importance in zip(
            FEATURES,
            model.feature_importances_
        ):

            feature_importance[feature] = round(
                float(importance),
                4
            )

    # Algorithm-specific topics
    topics = get_topics(
        model,
        model_type,
        X_train,
        y_train
    )

    return {

        "model": model_name,

        "total_rows": len(data),

        "train_rows": len(X_train),

        "test_rows": len(X_test),

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "topics": topics,

        "feature_importance": feature_importance,

        "feature_importance_graph": feature_graph
    }


# =========================================================
# PREDICTION
# =========================================================

def predict_decision_tree(
    cgpa,
    aptitude,
    coding,
    interview,
    model_type="decision_tree"
):

    data, X, y = prepare_data()

    model_name, model = get_model(
        model_type
    )

    # Train on complete dataset
    model.fit(
        X,
        y
    )

    input_data = pd.DataFrame([{

        "CGPA": float(cgpa),

        "AptitudeTestScore": float(
            aptitude
        ),

        "CodingTestScore": float(
            coding
        ),

        "MockInterviewScore": float(
            interview
        )

    }])

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    if prediction == 1:

        result = "Placed"

    else:

        result = "Not Placed"

    return {

        "model": model_name,

        "result": result,

        "probability": round(
            float(probability) * 100,
            2
        ),

        "prediction": int(prediction)

    }