import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
from statistics import mode
from TextClassifier import TextClassifier
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  
 
class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.p_y = None  # Prior probability P(Y=1)
        self.p_x_given_y1 = None  # P(X_i=1|Y=1)
        self.p_x_given_y0 = None  # P(X_i=1|Y=0)

    def fit(self, X, y):
        y = y.flatten()
        n_samples, n_features = X.shape
        self.p_y = (np.sum(y) + self.alpha) / (n_samples + 2*self.alpha)
        y1_indices = (y == 1)
        y0_indices = (y == 0)
        count_y1 = np.sum(y1_indices)
        count_y0 = np.sum(y0_indices)
        X_y1 = X[y1_indices]
        self.p_x_given_y1 = (X_y1.sum(axis=0) + self.alpha) / (count_y1 + 2*self.alpha)
        X_y0 = X[y0_indices]
        self.p_x_given_y0 = (X_y0.sum(axis=0) + self.alpha) / (count_y0 + 2*self.alpha)

    def predict(self, X):
        n_samples, n_features = X.shape
        log_p_y1 = np.log(self.p_y)
        log_p_y0 = np.log(1.0 - self.p_y)
        log_p_x1_y1 = np.log(self.p_x_given_y1)
        log_p_x0_y1 = np.log(1.0 - self.p_x_given_y1)
        log_p_x1_y0 = np.log(self.p_x_given_y0)
        log_p_x0_y0 = np.log(1.0 - self.p_x_given_y0)

        X_float = X.astype(float)
        log_p_x1_y1_expanded = np.tile(log_p_x1_y1, (n_samples, 1))
        log_p_x0_y1_expanded = np.tile(log_p_x0_y1, (n_samples, 1))
        y1_likelihood = (X_float * log_p_x1_y1_expanded + (1 - X_float) * log_p_x0_y1_expanded).sum(axis=1)
        log_post_y1 = y1_likelihood + log_p_y1

        log_p_x1_y0_expanded = np.tile(log_p_x1_y0, (n_samples, 1))
        log_p_x0_y0_expanded = np.tile(log_p_x0_y0, (n_samples, 1))
        y0_likelihood = (X_float * log_p_x1_y0_expanded + (1 - X_float) * log_p_x0_y0_expanded).sum(axis=1)
        log_post_y0 = y0_likelihood + log_p_y0

        predictions = (log_post_y1 > log_post_y0).astype(int)
        return predictions

if __name__ == "__main__":
    print("Initializing TextClassifier")
    text_classifier = TextClassifier()
    X_train, y_train, X_dev, y_dev, X_test, y_test = text_classifier.get_data()

    print("Training custom NaiveBayes on full training set")
    nb_clf = NaiveBayes(alpha=1.0)
    nb_clf.fit(X_train, y_train)

    y_dev_pred = nb_clf.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\n Dev Accuracy: {dev_accuracy:.4f}")

    y_test_pred = nb_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\n Classification Report:")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    df_report = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
    df_report = df_report.drop(["macro avg", "weighted avg"], errors="ignore")
    print(df_report)

    print("\nRunning learning curve experiment")
    # Assuming data is sufficiently randomized.
    n_train_total = X_train.shape[0]
    # Create 10 steps between a minimum number (e.g., 10 examples) and the full training set
    training_sizes = np.linspace(10, n_train_total, 10, dtype=int)

    # Lists to hold metrics for class 1
    train_precision, train_recall, train_f1 = [], [], []
    dev_precision, dev_recall, dev_f1 = [], [], []

    # Choose the positive class (1) for evaluation (change if needed)
    target_class = 1

    for size in training_sizes:
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        model = NaiveBayes(alpha=1.0)
        model.fit(X_train_subset, y_train_subset)

        # Predictions on training subset
        y_train_pred = model.predict(X_train_subset)
        # Predictions on dev set (full)
        y_dev_pred = model.predict(X_dev)

        # Compute metrics for class 1 on training data
        p_train = precision_score(y_train_subset, y_train_pred, pos_label=target_class, zero_division=0)
        r_train = recall_score(y_train_subset, y_train_pred, pos_label=target_class, zero_division=0)
        f1_train = f1_score(y_train_subset, y_train_pred, pos_label=target_class, zero_division=0)

        # Compute metrics for class 1 on dev data
        p_dev = precision_score(y_dev, y_dev_pred, pos_label=target_class, zero_division=0)
        r_dev = recall_score(y_dev, y_dev_pred, pos_label=target_class, zero_division=0)
        f1_dev = f1_score(y_dev, y_dev_pred, pos_label=target_class, zero_division=0)

        train_precision.append(p_train)
        train_recall.append(r_train)
        train_f1.append(f1_train)
        dev_precision.append(p_dev)
        dev_recall.append(r_dev)
        dev_f1.append(f1_dev)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("learning curves Naive Bayes (First Category)", fontsize=14)

    axs[0].plot(training_sizes, train_precision, marker="o", label="Train")
    axs[0].plot(training_sizes, dev_precision, marker="o", label="Dev")
    axs[0].set_title("Precision (class 1)")
    axs[0].set_xlabel("Training examples")
    axs[0].set_ylabel("Precision")
    axs[0].legend()

    axs[1].plot(training_sizes, train_recall, marker="o", label="Train")
    axs[1].plot(training_sizes, dev_recall, marker="o", label="Dev")
    axs[1].set_title("Recall (class 1)")
    axs[1].set_xlabel("Training examples")
    axs[1].set_ylabel("Recall")
    axs[1].legend()

    axs[2].plot(training_sizes, train_f1, marker="o", label="Train")
    axs[2].plot(training_sizes, dev_f1, marker="o", label="Dev")
    axs[2].set_title("F1 Score (class 1)")
    axs[2].set_xlabel("Training examples")
    axs[2].set_ylabel("F1 Score")
    axs[2].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig("learning_curves.png")
    plt.show()

    prec_class, rec_class, f1_class, _ = precision_recall_fscore_support(y_test, y_test_pred, labels=[0, 1])

    metrics_table = pd.DataFrame({
        "Precision": prec_class,
        "Recall": rec_class,
        "F1 Score": f1_class
    }, index=["Class 0", "Class 1"])

    prec_micro = precision_score(y_test, y_test_pred, average="micro")
    rec_micro = recall_score(y_test, y_test_pred, average="micro")
    f1_micro = f1_score(y_test, y_test_pred, average="micro")

    prec_macro = precision_score(y_test, y_test_pred, average="macro")
    rec_macro = recall_score(y_test, y_test_pred, average="macro")
    f1_macro = f1_score(y_test, y_test_pred, average="macro")

    metrics_table.loc["Micro Average"] = [prec_micro, rec_micro, f1_micro]
    metrics_table.loc["Macro Average"] = [prec_macro, rec_macro, f1_macro]

    print("\nDetailed Test Metrics:")
    print(metrics_table)
    metrics_table.to_csv("test_metrics.csv")
