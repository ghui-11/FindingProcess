"""
Model management: training, loading, and saving for field classifiers.
Supports Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
Now supports separate binary classifiers for case and activity columns only.
"""
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from .feature_extraction import extract_column_features_and_labels_from_dir

MODEL_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix'))

def train_case_act_classifiers(
    train_dir: str,
    nrows: int = 1000,
    max_word_threshold: int = 6,
    method: str = "logistic",
    params: dict = None,
    verbose: bool = True,
    save_model: bool = True,
    model_folder: str = None,
):
    X, y = extract_column_features_and_labels_from_dir(
        train_dir=train_dir, nrows=nrows,
        max_word_threshold=max_word_threshold, verbose=verbose
    )
    feature_names = ["n_unique_ratio", "max_freq", "entropy", "sim", "std_length"]
    params = params or {}
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    y_case = (y == "case").astype(int)
    y_act = (y == "activity").astype(int)
    if method == "logistic":
        case_clf = LogisticRegression(max_iter=1000, **params)
        act_clf = LogisticRegression(max_iter=1000, **params)
        case_model_name = "case_classifier_logistic.joblib"
        act_model_name = "act_classifier_logistic.joblib"
    elif method == "tree":
        case_clf = DecisionTreeClassifier(**params)
        act_clf = DecisionTreeClassifier(**params)
        case_model_name = "case_classifier_tree.joblib"
        act_model_name = "act_classifier_tree.joblib"
    elif method == "rf":
        case_clf = RandomForestClassifier(random_state=42, **params)
        act_clf = RandomForestClassifier(random_state=42, **params)
        case_model_name = "case_classifier_rf.joblib"
        act_model_name = "act_classifier_rf.joblib"
    elif method == "gb":
        case_clf = GradientBoostingClassifier(random_state=42, **params)
        act_clf = GradientBoostingClassifier(random_state=42, **params)
        case_model_name = "case_classifier_gb.joblib"
        act_model_name = "act_classifier_gb.joblib"
    else:
        raise ValueError("method must be 'logistic', 'tree', 'rf', or 'gb'")
    case_clf.fit(X_scale, y_case)
    act_clf.fit(X_scale, y_act)
    if verbose:
        print(f"Trained {method} case classifier on {X.shape[0]} samples.")
        print(f"Trained {method} activity classifier on {X.shape[0]} samples.")
    if save_model:
        if model_folder is None:
            model_folder = MODEL_FOLDER
        os.makedirs(model_folder, exist_ok=True)
        joblib.dump(case_clf, os.path.join(model_folder, case_model_name))
        joblib.dump(act_clf, os.path.join(model_folder, act_model_name))
        joblib.dump(scaler, os.path.join(model_folder, "case_act_scaler.joblib"))
        if verbose:
            print(f"Saved case model to {os.path.join(model_folder, case_model_name)}")
            print(f"Saved activity model to {os.path.join(model_folder, act_model_name)}")
            print(f"Saved scaler to {os.path.join(model_folder, 'case_act_scaler.joblib')}")
    return case_clf, act_clf, scaler, feature_names

def load_case_act_classifiers(model_folder: str = None, method: str = "logistic"):
    if model_folder is None:
        model_folder = MODEL_FOLDER
    if method == "logistic":
        case_model_name = "case_classifier_logistic.joblib"
        act_model_name = "act_classifier_logistic.joblib"
    elif method == "tree":
        case_model_name = "case_classifier_tree.joblib"
        act_model_name = "act_classifier_tree.joblib"
    elif method == "rf":
        case_model_name = "case_classifier_rf.joblib"
        act_model_name = "act_classifier_rf.joblib"
    elif method == "gb":
        case_model_name = "case_classifier_gb.joblib"
        act_model_name = "act_classifier_gb.joblib"
    else:
        raise ValueError("method must be 'logistic', 'tree', 'rf', or 'gb'")
    case_clf = joblib.load(os.path.join(model_folder, case_model_name))
    act_clf = joblib.load(os.path.join(model_folder, act_model_name))
    scaler = joblib.load(os.path.join(model_folder, "case_act_scaler.joblib"))
    feature_names = ["n_unique_ratio", "max_freq", "entropy", "sim", "std_length"]
    return case_clf, act_clf, scaler, feature_names