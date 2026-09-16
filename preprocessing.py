import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# DATASET PATH
# =========================================================

CSV_PATH = r"/Volumes/D/class project placement/placement_predict_50k Dataset (3) (1).csv"


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():
    return pd.read_csv(CSV_PATH)


# =========================================================
# FIND TARGET COLUMN
# =========================================================

def find_target_column(df):

    possible_targets = [
        "Placement_Status",
        "placement_status",
        "Placement",
        "placement",
        "Placed",
        "placed",
        "PlacementStatus",
        "status"
    ]

    for column in possible_targets:
        if column in df.columns:
            return column

    # fallback: commonly used last column
    return df.columns[-1]


# =========================================================
# IMPORTANT FUNCTION
# USED BY linear_regression.py
# =========================================================

def prepare_features(df=None, target_column=None):

    # -----------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------

    if isinstance(df, str):

        if df.lower().endswith(".csv"):
            df = pd.read_csv(df)
        else:
            target_column = df
            df = load_dataset()

    elif df is None:

        df = load_dataset()

    elif isinstance(df, pd.DataFrame):

        df = df.copy()

    else:

        raise TypeError(
            "Invalid input to prepare_features()"
        )

    # -----------------------------------------------------
    # 2. TARGET COLUMN
    # -----------------------------------------------------

    if target_column is None:
        target_column = find_target_column(df)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            f"not found in dataset."
        )

    # -----------------------------------------------------
    # 3. X AND y
    # -----------------------------------------------------

    X = df.drop(
        columns=[target_column]
    ).copy()

    y = df[target_column].copy()

    # -----------------------------------------------------
    # 4. REMOVE ID COLUMNS
    # -----------------------------------------------------

    id_columns = []

    for column in X.columns:

        name = column.lower()

        if (
            name == "id"
            or name.endswith("_id")
            or name == "member_id"
        ):
            id_columns.append(column)

    if id_columns:
        X = X.drop(
            columns=id_columns
        )

    # -----------------------------------------------------
    # 5. TRAIN / TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # IMPORTANT:
    # Keep targets as pandas Series
    y_train = pd.Series(
        y_train,
        index=X_train.index,
        name=target_column
    )

    y_test = pd.Series(
        y_test,
        index=X_test.index,
        name=target_column
    )

    # -----------------------------------------------------
    # 6. COLUMN TYPES
    # -----------------------------------------------------

    numerical_columns = (
        X_train
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "category", "bool"]
        )
        .columns
        .tolist()
    )

    # -----------------------------------------------------
    # 7. NUMERICAL PIPELINE
    # -----------------------------------------------------

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="mean")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # -----------------------------------------------------
    # 8. CATEGORICAL PIPELINE
    # -----------------------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # -----------------------------------------------------
    # 9. PREPROCESSOR
    # -----------------------------------------------------

    transformers = []

    if numerical_columns:

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # -----------------------------------------------------
    # 10. FIT ON TRAINING DATA ONLY
    # -----------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # -----------------------------------------------------
    # 11. TRANSFORM TEST DATA
    # -----------------------------------------------------

    X_test_processed = preprocessor.transform(
        X_test
    )

    # -----------------------------------------------------
    # 12. KEEP PROCESSED FEATURES AS DATAFRAMES
    # -----------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        feature_names = [
            f"Feature_{i}"
            for i in range(
                X_train_processed.shape[1]
            )
        ]

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    # -----------------------------------------------------
    # 13. TARGET NUMERIC CONVERSION
    # -----------------------------------------------------

    if (
        y_train.dtype == "object"
        or str(y_train.dtype) == "category"
    ):

        categories = (
            y_train.dropna()
            .unique()
        )

        mapping = {
            value: i
            for i, value in enumerate(categories)
        }

        y_train = y_train.map(mapping)
        y_test = y_test.map(mapping)

    y_train = pd.to_numeric(
        y_train,
        errors="coerce"
    )

    y_test = pd.to_numeric(
        y_test,
        errors="coerce"
    )

    # -----------------------------------------------------
    # 14. REMOVE INVALID TARGET ROWS
    # -----------------------------------------------------

    train_valid = y_train.notna()
    test_valid = y_test.notna()

    X_train_processed = X_train_processed.loc[
        train_valid
    ]

    y_train = y_train.loc[
        train_valid
    ]

    X_test_processed = X_test_processed.loc[
        test_valid
    ]

    y_test = y_test.loc[
        test_valid
    ]

    # -----------------------------------------------------
    # 15. COMPLETE PROCESSED DATA
    # -----------------------------------------------------

    X_processed = pd.concat(
        [
            X_train_processed,
            X_test_processed
        ],
        axis=0
    )

    y_processed = pd.concat(
        [
            y_train,
            y_test
        ],
        axis=0
    )

    # -----------------------------------------------------
    # 16. RETURN 8 VALUES
    # -----------------------------------------------------

    return (
        X_processed,
        y_processed,
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        list(feature_names)
    )


# =========================================================
# 1. DATA QUALITY
# =========================================================

def data_quality(df):

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df)
    ) * 100

    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count.values,
        "Missing Percentage": missing_percentage.values
    })

    missing_table = missing_table[
        missing_table["Missing Count"] > 0
    ]

    # Mean imputation demonstration
    df_imputed = df.copy()

    numerical_columns = df_imputed.select_dtypes(
        include=np.number
    ).columns

    for column in numerical_columns:

        if df_imputed[column].isnull().sum() > 0:

            df_imputed[column] = (
                df_imputed[column]
                .fillna(df_imputed[column].mean())
            )

    return {
        "missing_table":
            missing_table.to_dict("records"),

        "total_missing":
            int(missing_count.sum()),

        "remaining_missing":
            int(df_imputed.isnull().sum().sum())
    }


# =========================================================
# 2. OUTLIER HANDLING
# =========================================================

def outlier_analysis(df):

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numerical_columns:

        values = df[column].dropna()

        if len(values) == 0:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)

        outlier_count = (
            (values < lower_fence) |
            (values > upper_fence)
        ).sum()

        results.append({
            "Column": column,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower Fence": round(
                lower_fence, 3
            ),
            "Upper Fence": round(
                upper_fence, 3
            ),
            "Outliers": int(outlier_count)
        })

    return results


# =========================================================
# 3. DATA LEAKAGE PREVENTION
# =========================================================

def leakage_prevention(df):

    target_column = find_target_column(df)

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    except ValueError:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )

    return {
        "target": target_column,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "status":
            "Preprocessing should be fitted only on training data."
    }


# =========================================================
# 4. CATEGORICAL ENCODING
# =========================================================

def encoding_analysis(df):

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    results = []

    for column in categorical_columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
        )

        results.append({
            "Column": column,
            "Unique Values": len(values),
            "Sample Categories":
                ", ".join(values[:5])
        })

    return results


# ---------------------------------------------------------
# ONE-HOT ENCODING
# ---------------------------------------------------------

def one_hot_example(df):

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_columns) == 0:
        return []

    column = categorical_columns[0]

    data = df[[column]].fillna("Missing")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded = encoder.fit_transform(data)

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            [column]
        )
    )

    return encoded_df.head(5).to_dict(
        "records"
    )


# ---------------------------------------------------------
# ORDINAL ENCODING
# ---------------------------------------------------------

def ordinal_example(df):

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if len(categorical_columns) == 0:
        return []

    column = categorical_columns[0]

    data = df[[column]].fillna("Missing")

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    encoded = encoder.fit_transform(data)

    results = []

    for i in range(
        min(5, len(df))
    ):

        results.append({
            "Original":
                str(df[column].iloc[i]),

            "Encoded":
                float(encoded[i][0])
        })

    return results


# ---------------------------------------------------------
# TARGET ENCODING
# ---------------------------------------------------------

def target_encoding_example(df):

    target_column = find_target_column(df)

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    if (
        target_column is None
        or len(categorical_columns) == 0
    ):
        return []

    column = categorical_columns[0]

    target = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    temporary = pd.DataFrame({
        "category": df[column],
        "target": target
    })

    mapping = (
        temporary
        .groupby("category")["target"]
        .mean()
    )

    results = []

    for category in mapping.index[:10]:

        results.append({
            "Category": str(category),
            "Target Mean":
                round(
                    float(mapping[category]),
                    4
                )
        })

    return results


# =========================================================
# 5. FEATURE SCALING
# =========================================================

def scaling_analysis(df):

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numerical_columns:
        return []

    selected_columns = numerical_columns[:5]

    sample = (
        df[selected_columns]
        .copy()
    )

    sample = sample.fillna(
        sample.mean()
    )

    sample = sample.head(5)

    minmax_scaler = MinMaxScaler()

    standard_scaler = StandardScaler()

    minmax_values = (
        minmax_scaler
        .fit_transform(sample)
    )

    standard_values = (
        standard_scaler
        .fit_transform(sample)
    )

    results = []

    for i in range(len(sample)):

        row = {
            "Row": i + 1
        }

        for j, column in enumerate(
            selected_columns
        ):

            row[
                column + " MinMax"
            ] = round(
                float(minmax_values[i][j]),
                3
            )

            row[
                column + " Standardized"
            ] = round(
                float(standard_values[i][j]),
                3
            )

        results.append(row)

    return results


# =========================================================
# 6. MODEL READINESS
# =========================================================

def model_readiness(df):

    target_column = find_target_column(df)

    if target_column is None:

        return {
            "status":
                "Target column not detected."
        }

    X = df.select_dtypes(
        include=np.number
    ).copy()

    if target_column in X.columns:

        X = X.drop(
            columns=[target_column]
        )

    if X.empty:

        return {
            "status":
                "No numerical features available."
        }

    X = X.fillna(
        X.mean()
    )

    y = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    if y.nunique() < 2:

        return {
            "status":
                "Target does not contain enough classes."
        }

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    except ValueError:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    train_prediction = model.predict(
        X_train
    )

    test_prediction = model.predict(
        X_test
    )

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    difference = (
        train_accuracy -
        test_accuracy
    )

    if difference > 0.15:

        interpretation = (
            "Possible Overfitting"
        )

    elif (
        train_accuracy < 0.70
        and test_accuracy < 0.70
    ):

        interpretation = (
            "Possible Underfitting"
        )

    else:

        interpretation = (
            "Reasonable Generalization"
        )

    return {
        "train_accuracy":
            round(train_accuracy, 4),

        "test_accuracy":
            round(test_accuracy, 4),

        "difference":
            round(difference, 4),

        "interpretation":
            interpretation,

        "status": ""
    }


# =========================================================
# COMPLETE PREPROCESSING
# =========================================================

def run_preprocessing():

    df = load_dataset()

    return {
        "data_quality":
            data_quality(df),

        "outliers":
            outlier_analysis(df),

        "leakage":
            leakage_prevention(df),

        "encoding":
            encoding_analysis(df),

        "one_hot":
            one_hot_example(df),

        "ordinal":
            ordinal_example(df),

        "target_encoding":
            target_encoding_example(df),

        "scaling":
            scaling_analysis(df),

        "model_readiness":
            model_readiness(df)
    }