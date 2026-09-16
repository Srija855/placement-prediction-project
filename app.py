from flask import Flask, render_template, request

from load_data import get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing

from linear_regression import (
    run_linear_regression,
    run_linear_regularization,
    predict_salary,
    predict_regularized_salary
)
from logistic_regression import (
    run_logistic_regression,
    run_logistic_regularization,
    predict_placement,
    predict_regularized_placement
)
from decision_tree import (
    run_decision_tree,
    predict_decision_tree
)


app = Flask(__name__)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# =========================================================
# DATA LOADING
# =========================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        summary = get_data_summary()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda():

    error = None
    results = None

    try:

        results = run_eda()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=results,
        error=error
    )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    error = None
    results = None

    try:
        results = run_preprocessing()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    if error:
        return render_template(
            "preprocessing.html",
            active="preprocessing",
            results=None,
            error=error
        )

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=results,
        error=None
    )

@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():

    results = None
    regularized = None
    prediction = None

    regression_type = "without"

    if request.method == "POST":

        regression_type = request.form.get(
            "regression_type",
            "without"
        )

        # =============================================
        # WITHOUT REGULARIZATION
        # =============================================

        if regression_type == "without":

            results = run_linear_regression()


        # =============================================
        # WITH REGULARIZATION
        # =============================================

        elif regression_type == "with":

            regularization_type = request.form.get(
                "regularization_type",
                "ridge"
            )

            regularized = run_linear_regularization(
                regularization_type
            )


        # =============================================
        # SALARY PREDICTION
        # =============================================

        if (
            request.form.get("cgpa")
            and
            request.form.get("aptitude")
            and
            request.form.get("coding")
            and
            request.form.get("interview")
        ):

            cgpa = request.form.get("cgpa")

            aptitude = request.form.get("aptitude")

            coding = request.form.get("coding")

            interview = request.form.get("interview")


            if regression_type == "with":

                regularization_type = request.form.get(
                    "regularization_type",
                    "ridge"
                )

                prediction = predict_regularized_salary(

                    cgpa,
                    aptitude,
                    coding,
                    interview,

                    regularization_type

                )

            else:

                prediction = predict_salary(

                    cgpa,
                    aptitude,
                    coding,
                    interview

                )


    return render_template(

        "linear_regression.html",

        results=results,

        regularized=regularized,

        prediction=prediction,

        regression_type=regression_type

    )

@app.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression():

    results = None

    regularized = None

    prediction = None

    regression_type = "without"


    if request.method == "POST":

        regression_type = request.form.get(
            "regression_type",
            "without"
        )


        # ==========================================
        # WITHOUT REGULARIZATION
        # ==========================================

        if regression_type == "without":

            results = run_logistic_regression()


        # ==========================================
        # WITH REGULARIZATION
        # ==========================================

        elif regression_type == "with":

            regularization_type = request.form.get(
                "regularization_type",
                "ridge"
            )

            regularized = run_logistic_regularization(
                regularization_type
            )


        # ==========================================
        # PREDICTION FORM
        # ==========================================

        if (
            request.form.get("cgpa")
            and
            request.form.get("aptitude")
            and
            request.form.get("coding")
            and
            request.form.get("interview")
        ):

            cgpa = request.form.get(
                "cgpa"
            )

            aptitude = request.form.get(
                "aptitude"
            )

            coding = request.form.get(
                "coding"
            )

            interview = request.form.get(
                "interview"
            )


            # --------------------------------------
            # REGULARIZED PREDICTION
            # --------------------------------------

            if regression_type == "with":

                regularization_type = request.form.get(
                    "regularization_type",
                    "ridge"
                )

                prediction = predict_regularized_placement(
                    cgpa,
                    aptitude,
                    coding,
                    interview,
                    regularization_type
                )


            # --------------------------------------
            # NORMAL PREDICTION
            # --------------------------------------

            else:

                prediction = predict_placement(
                    cgpa,
                    aptitude,
                    coding,
                    interview
                )


    return render_template(
        "logistic_regression.html",

        results=results,

        regularized=regularized,

        prediction=prediction,

        regression_type=regression_type
    )
# =========================================================
# DECISION TREES
# =========================================================

@app.route(
    "/decision-tree",
    methods=["GET", "POST"]
)
def decision_tree():

    results = None

    prediction = None

    error = None

    model_type = "decision_tree"


    if request.method == "POST":

        model_type = request.form.get(
            "model_type",
            "decision_tree"
        )


        # Run selected algorithm

        try:

            results = run_decision_tree(
                model_type
            )

        except Exception as e:

            error = f"Model error: {e}"


        # Prediction

        if (
            request.form.get("cgpa")
            and request.form.get("aptitude")
            and request.form.get("coding")
            and request.form.get("interview")
        ):

            try:

                prediction = predict_decision_tree(

                    request.form.get("cgpa"),

                    request.form.get("aptitude"),

                    request.form.get("coding"),

                    request.form.get("interview"),

                    model_type

                )

            except Exception as e:

                error = f"Prediction error: {e}"


    return render_template(

        "decision_tree.html",

        results=results,

        prediction=prediction,

        model_type=model_type,

        active="decision-tree",

        error=error

    )
# =========================================================
# REGULARIZATION
# =========================================================

@app.route("/regularization")
def regularization():

    error = None
    results = None

    try:

        # IMPORTANT:
        # Function name must match regularization.py

        results = run_logistic_regularization()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "regularization.html",
        active="regularization",
        results=results,
        error=error
    )


# =========================================================
# REGULARIZED PLACEMENT PREDICTION
# =========================================================

@app.route(
    "/predict-regularized-placement",
    methods=["POST"]
)
def predict_regularized_placement_route():

    try:

        cgpa = request.form["cgpa"]

        aptitude = request.form["aptitude"]

        coding = request.form["coding"]

        interview = request.form["interview"]

        prediction = predict_regularized_placement(
            cgpa,
            aptitude,
            coding,
            interview
        )

        results = run_logistic_regularization()

        return render_template(
            "regularization.html",
            active="regularization",
            results=results,
            prediction=prediction
        )

    except Exception as e:

        try:

            results = run_logistic_regularization()

        except Exception:

            results = None

        return render_template(
            "regularization.html",
            active="regularization",
            results=results,
            error=f"Prediction error: {e}"
        )




# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )