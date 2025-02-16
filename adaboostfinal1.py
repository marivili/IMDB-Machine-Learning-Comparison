import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        self.training_loss = []
        self.dev_loss = []
        
        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            y_pred = stump.predict(X)
            err = np.sum(w * (y_pred != y)) / np.sum(w)
            
            if err > 0.5:
                continue
            
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)
            
            self.models.append(stump)
            self.alphas.append(alpha)
            
            train_loss = np.mean(y_pred != y)
            self.training_loss.append(train_loss)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.models):
            final_pred += alpha * stump.predict(X)
        return np.sign(final_pred)

if __name__ == "__main__":
    from TextClassifier import TextClassifier
    
    print("Initializing TextClassifier")
    text_classifier = TextClassifier()
    X_train, y_train, X_dev, y_dev, X_test, y_test = text_classifier.get_data()
    
    print("Training AdaBoost on full training set")
    adaboost = AdaBoost(n_estimators=50)
    adaboost.fit(X_train, y_train)
    
    y_dev_pred = adaboost.predict(X_dev)
    dev_loss = np.mean(y_dev_pred != y_dev)
    adaboost.dev_loss.append(dev_loss)
    
    # Plot training and development loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(adaboost.training_loss) + 1), adaboost.training_loss, label="Training Loss")
    plt.plot(range(1, len(adaboost.dev_loss) + 1), adaboost.dev_loss, label="Development Loss")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Loss")
    plt.title("Training and Development Loss over AdaBoost Iterations")
    plt.legend()
    plt.savefig("loss_curves_adaboost.png")
    plt.show()
    
    y_test_pred = adaboost.predict(X_test)
    print("\n Classification Report:")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    df_report = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
    print(df_report)
    
    # Compute precision, recall, F1 for each class
    precisions, recalls, f1s, _ = precision_recall_fscore_support(y_test, y_test_pred)
    metrics_dict = {
        "Category": ["0", "1"],
        "Precision": precisions,
        "Recall": recalls,
        "F1": f1s
    }
    df_class = pd.DataFrame(metrics_dict)
    print("\nMetrics per class:")
    print(df_class)
    
    # Compute micro and macro averages
    precision_micro = precision_score(y_test, y_test_pred, average='micro')
    recall_micro = recall_score(y_test, y_test_pred, average='micro')
    f1_micro = f1_score(y_test, y_test_pred, average='micro')
    
    precision_macro = precision_score(y_test, y_test_pred, average='macro')
    recall_macro = recall_score(y_test, y_test_pred, average='macro')
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    
    metrics_avg = {
        "Measurement": ["Micro Average", "Macro Average"],
        "Precision": [precision_micro, precision_macro],
        "Recall": [recall_micro, recall_macro],
        "F1": [f1_micro, f1_macro]
    }
    df_avg = pd.DataFrame(metrics_avg)
    print("\n Micro and Macro Averages on Test Data:")
    print(df_avg)
