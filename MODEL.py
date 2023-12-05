from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def create_fit_predict_model(data, max_depth, min_samples_leaf):
    """
    Creates and fits a Decision Tree Classifier to make predictions on the specified dataset.

    """
    # Splitting the dataset into features and target variable
    X, y = data.drop(columns='churn'), data['churn']

    # Creating and fitting the model
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    tree.fit(X, y)

    # Making predictions
    y_pred = tree.predict(X)

    return tree, X, y, y_pred

def create_csv(tree_model, X_test, y_pred_test, test_index):
    """
    Creates a CSV DataFrame from the predictions of a decision tree model.

    Parameters:
    - tree_model: The trained decision tree model.
    - X_test: The test dataset features.
    - y_pred_test: The predicted churn labels for the test dataset.
    - test_index: The index of the test dataset, typically customer IDs.

    Returns:
    - csv: A DataFrame with customer ID, predicted probabilities of churn, and churn predictions.
    """
    # Creating a DataFrame from predicted probabilities
    csv = pd.DataFrame(tree_model.predict_proba(X_test))

    # Adding customer_id and churn predictions
    csv['customer_id'] = test_index
    csv['churn_predict'] = y_pred_test

    # Dropping the first column (probability of class 0)
    csv.drop(0, axis=1, inplace=True)

    # Reordering columns to make customer_id the first column
    column_order = ['customer_id'] + [col for col in csv.columns if col != 'customer_id']
    csv = csv[column_order]

    return csv