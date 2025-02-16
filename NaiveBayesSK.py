import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from TextClassifier import TextClassifier
import os, sys
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Κρύβει όλα τα INFO & WARNING logs, αφήνει μόνο τα ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Απενεργοποιεί τα oneDNN μηνύματα
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  


if __name__ == "__main__":
    print("Initializing TextClassifier...")
    text_classifier = TextClassifier()
    X_train, y_train, X_dev, y_dev, X_test, y_test = text_classifier.get_data()

    print("Training scikit-learn BernoulliNB...")
    nb_sklearn = BernoulliNB(alpha=1.0)
    nb_sklearn.fit(X_train, y_train)

    y_dev_pred = nb_sklearn.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\n Dev Accuracy: {dev_accuracy:.4f}")

    y_test_pred = nb_sklearn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\n Classification Report:")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    class_labels = [str(label) for label in np.unique(y_test)]  
    filtered_report = {k: report[k] for k in class_labels}  

    df_report = pd.DataFrame(filtered_report).T.drop(columns=["support"], errors="ignore")
    print(df_report)  

    train_sizes = np.linspace(0.1, 1.0, 10)
    train_precisions, train_recalls, train_f1s = [], [], []
    dev_precisions, dev_recalls, dev_f1s = [], [], []

    pos_label = 1

    for frac in train_sizes:
        idx = int(frac * len(X_train))
        X_train_subset, y_train_subset = X_train[:idx], y_train[:idx]
        nb_temp = BernoulliNB(alpha=1.0)
        nb_temp.fit(X_train_subset, y_train_subset)

        y_train_pred_temp = nb_temp.predict(X_train_subset)
        train_precisions.append(precision_score(y_train_subset, y_train_pred_temp, pos_label=pos_label))
        train_recalls.append(recall_score(y_train_subset, y_train_pred_temp, pos_label=pos_label))
        train_f1s.append(f1_score(y_train_subset, y_train_pred_temp, pos_label=pos_label))

        y_dev_pred_temp = nb_temp.predict(X_dev)
        dev_precisions.append(precision_score(y_dev, y_dev_pred_temp, pos_label=pos_label))
        dev_recalls.append(recall_score(y_dev, y_dev_pred_temp, pos_label=pos_label))
        dev_f1s.append(f1_score(y_dev, y_dev_pred_temp, pos_label=pos_label))

    plt.figure(figsize=(10, 6))
    num_train_examples = train_sizes * len(X_train)
    plt.plot(num_train_examples, train_precisions, label='Train Precision')
    plt.plot(num_train_examples, dev_precisions, label='Dev Precision', marker='o')
    plt.plot(num_train_examples, train_recalls, label='Train Recall')
    plt.plot(num_train_examples, dev_recalls, label='Dev Recall', marker='o')
    plt.plot(num_train_examples, train_f1s, label='Train F1')
    plt.plot(num_train_examples, dev_f1s, label='Dev F1', marker='o')
    plt.xlabel('Number of training examples')
    plt.ylabel('Rate')
    plt.title('learning curves for Category 1')
    plt.legend()
    plt.grid(True)
    plt.show()

    y_test_pred = nb_sklearn.predict(X_test)


    unique_classes = np.unique(y_test)
    results = {}

    for cls in unique_classes:
        prec = precision_score(y_test, y_test_pred, labels=[cls], average='micro')
        rec = recall_score(y_test, y_test_pred, labels=[cls], average='micro')
        f1 = f1_score(y_test, y_test_pred, labels=[cls], average='micro')
        results[f'Class {cls}'] = {'Precision': prec, 'Recall': rec, 'F1': f1}

    results['Micro Average'] = {
        'Precision': precision_score(y_test, y_test_pred, average='micro'),
        'Recall': recall_score(y_test, y_test_pred, average='micro'),
        'F1': f1_score(y_test, y_test_pred, average='micro')
    }
    results['Macro Average'] = {
        'Precision': precision_score(y_test, y_test_pred, average='macro'),
        'Recall': recall_score(y_test, y_test_pred, average='macro'),
        'F1': f1_score(y_test, y_test_pred, average='macro')
    }

    df_results = pd.DataFrame(results).T
    print("\n Evaluation Results on Test Data:")
    print(df_results)