from sklearn.neighbors import KNeighborsClassifier
def run_knn_and_evaluate(X_train, y_train, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors model on the training data, makes predictions,
    and evaluates the model's performance.

    Parameters:
    - X_train (DataFrame): The features of the training data.
    - y_train (Series): The target of the training data.
    - n_neighbors (int, optional): The number of neighbors to use for kNN. Default is 5.

    Returns:
    - accuracy (float): The accuracy of the model on the training data.
    - predictions (array): Predictions made by the model on the training data.
    """
    # Initialize the KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model to the training data
    knn.fit(X_train, y_train)

    # Make predictions
    predicted = knn.predict(X_train)

    # Compute accuracy
    accuracy = knn.score(X_train, y_train)

    print(f'KNN Model with {n_neighbors} Neighbors')
    print(f'Accuracy on Training Data: {accuracy:.2f}')
    print(f'First 10 Predictions: {predicted[:10]}')

    return accuracy, predicted

# Example usage
# accuracy, predictions = run_knn_and_evaluate(X_train, y_train, n_neighbors=3)

from sklearn.metrics import classification_report, confusion_matrix

def print_classification_metrics(y_true, y_pred):
    """
    Prints classification metrics for a given set of true and predicted values.

    Parameters:
    y_true (array-like): True labels of the data.
    y_pred (array-like): Predicted labels from the model.

    Returns:
    None. Prints the classification metrics.
    """
    # Calculating confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculating metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    true_positive_rate = tp / (tp + fn)
    false_positive_rate = fp / (fp + tn)
    true_negative_rate = tn / (tn + fp)
    false_negative_rate = fn / (fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  # recall is the same as true_positive_rate
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Printing the metrics
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"True Positive Rate: {true_positive_rate:.2f}")
    print(f"False Positive Rate: {false_positive_rate:.2f}")
    print(f"True Negative Rate: {true_negative_rate:.2f}")
    print(f"False Negative Rate: {false_negative_rate:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

    # Printing classification report for additional metrics and support
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# Example usage
# print_classification_metrics(y_test, y_pred)