import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import os


def preprocess_data(filepath):

    data = pd.read_csv(filepath)

    #clean up
    data.columns = data.columns.str.strip()

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip().str.lower()

    le_color = LabelEncoder()
    le_type = LabelEncoder()
    le_spectral = LabelEncoder()

    data["Star color_encoded"] = le_color.fit_transform(data["Star color"])
    data["Star type_encoded"] = le_type.fit_transform(data["Star type"])
    data["Spectral Class_encoded"] = le_spectral.fit_transform(data["Spectral Class"])

    y = data["Spectral Class"]
    x = data.drop(columns=["Star color", "Star type", "Spectral Class"])


    return x, y, le_color, le_type, le_spectral


def train_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name=None, log=True):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    results_row = {
        "model": model_name,
        "accuracy": accuracy,
        "Macro f1": report["macro avg"]["f1-score"],
        "Weighted f1": report["weighted avg"]["f1-score"],
    }

    log_file = "../Models/model_results.csv"
    if log:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        try:
            existing = pd.read_csv(log_file)
            data = pd.concat([existing, pd.DataFrame([results_row])], ignore_index=True)
        except FileNotFoundError:
            data = pd.DataFrame([results_row])
        data.to_csv(log_file, index=False)

    return accuracy

filepath = "../Data/raw/star_dataset.csv"

X, Y, le_color, le_type, le_spectral = preprocess_data(filepath)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
print(Y_test.value_counts())

#LogisticRegression
logr = linear_model.LogisticRegression()
train_evaluate(logr, X_train_scaled, Y_train, X_test_scaled, Y_test, "Logistic Regression")


#DecisionTreeClassifier
tree = DecisionTreeClassifier()
train_evaluate(tree, X_train_scaled, Y_train, X_test_scaled, Y_test, "Decision Tree")

#RandomForestClassificer
forest = RandomForestClassifier()
train_evaluate(forest, X_train_scaled, Y_train, X_test_scaled, Y_test, "Random Forest")

#KNeighborsClassifier
knn = KNeighborsClassifier()
train_evaluate(knn, X_train_scaled, Y_train, X_test_scaled, Y_test, "K Neighbors")


#Small dataset giving overfitting and not full results.
#Not doing hyperparameter testing as its already close to 100% and moving on
