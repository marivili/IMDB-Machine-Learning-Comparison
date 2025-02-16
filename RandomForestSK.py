import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from TextClassifier import TextClassifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
tf.get_logger().setLevel('ERROR')  
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Initializing TextClassifier")
    text_classifier = TextClassifier()
    X_train, y_train, X_dev, y_dev, X_test, y_test = text_classifier.get_data()
    
    print("Training Scikit-learn RandomForestClassifier...")
    
    rf = RandomForestClassifier(n_estimators=10, max_depth=4, min_samples_split=10, n_jobs=-1, random_state=42)
    
    rf.fit(X_train, y_train)
    
    y_dev_pred = rf.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\n Dev Accuracy: {dev_accuracy:.4f}")
    
    y_test_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    print("\n Classification Report:")
    report = classification_report(y_test, y_test_pred, output_dict=True)

    class_labels = [str(label) for label in np.unique(y_test)]  
    filtered_report = {k: report[k] for k in class_labels}  

    df_report = pd.DataFrame(filtered_report).T.drop(columns=["support"], errors="ignore")
    print(df_report)

    train_sizes = np.linspace(10, X_train.shape[0], 10, dtype=int)
    train_precisions, train_recalls, train_f1s = [], [], []
    dev_precisions, dev_recalls, dev_f1s = [], [], []

    perm = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    for size in train_sizes:
        X_subset = X_train_shuffled[:size]
        y_subset = y_train_shuffled[:size]
        model = RandomForestClassifier(n_estimators=10, max_depth=4, min_samples_split=10, n_jobs=-1, random_state=42)
        model.fit(X_subset, y_subset)
        
        y_subset_pred = model.predict(X_subset)
        y_dev_pred = model.predict(X_dev)
        
        train_precisions.append(precision_score(y_subset, y_subset_pred, pos_label=1, zero_division=0))
        train_recalls.append(recall_score(y_subset, y_subset_pred, pos_label=1, zero_division=0))
        train_f1s.append(f1_score(y_subset, y_subset_pred, pos_label=1, zero_division=0))
        
        dev_precisions.append(precision_score(y_dev, y_dev_pred, pos_label=1, zero_division=0))
        dev_recalls.append(recall_score(y_dev, y_dev_pred, pos_label=1, zero_division=0))
        dev_f1s.append(f1_score(y_dev, y_dev_pred, pos_label=1, zero_division=0))

    plt.figure(figsize=(12,8))

    plt.subplot(3,1,1)
    plt.plot(train_sizes, train_precisions, 'b-', label='Training Precision')
    plt.plot(train_sizes, dev_precisions, 'r-', label='Dev Precision')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(train_sizes, train_recalls, 'b-', label='Training Recall')
    plt.plot(train_sizes, dev_recalls, 'r-', label='Dev Recall')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(train_sizes, train_f1s, 'b-', label='Training F1')
    plt.plot(train_sizes, dev_f1s, 'r-', label='Dev F1')
    plt.xlabel('Number of training examples')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

    full_model = RandomForestClassifier(n_estimators=10, max_depth=4, min_samples_split=10, n_jobs=-1, random_state=42)
    full_model.fit(X_train, y_train)
    y_test_pred = full_model.predict(X_test)

    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    final_results = {k: test_report[k] for k in class_labels} 
    
    final_results["micro avg"] = {
        "precision": precision_score(y_test, y_test_pred, average="micro"),
        "recall": recall_score(y_test, y_test_pred, average="micro"),
        "f1-score": f1_score(y_test, y_test_pred, average="micro"),
    }
    final_results["macro avg"] = {
        "precision": precision_score(y_test, y_test_pred, average="macro"),
        "recall": recall_score(y_test, y_test_pred, average="macro"),
        "f1-score": f1_score(y_test, y_test_pred, average="macro"),
    }
    df_test_report = pd.DataFrame(final_results).T.drop(columns=["support"], errors="ignore")
    print("\n Evaluation Results on Test Data:")
    print(df_test_report)