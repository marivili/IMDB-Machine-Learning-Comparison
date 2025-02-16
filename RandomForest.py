from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample
from statistics import mode
import math, re, nltk, os, sys
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support
from nltk.corpus import movie_reviews
from TextClassifier import TextClassifier
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  

class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category

class ID3:
    def __init__(self, features, max_depth=None, min_samples_split=10):
        self.tree = None
        self.features = features  # expected to be a numpy array of candidate feature indices
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self, X, y):
        # Start with the most common class in the entire training set
        common_class = mode(y.flatten())
        self.tree = self.create_tree(X, y, features=self.features, category=common_class, depth=0)
        return self.tree
    
    def create_tree(self, X, y, features, category, depth):
        # Base conditions
        if len(X) == 0 or len(y) < self.min_samples_split:
            return Node(is_leaf=True, category=category)
        if np.all(y.flatten() == 0):
            return Node(is_leaf=True, category=0)
        if np.all(y.flatten() == 1):
            return Node(is_leaf=True, category=1)
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Node(is_leaf=True, category=mode(y.flatten()))
        
        # Compute information gains for each candidate feature.
        # Note: We pass the entire column at once instead of using a list comprehension.
        igs = [self.calculate_ig(y.flatten(), X[:, feat]) for feat in features]
        
        # Get the actual best feature (column index) rather than its position in the 'features' array.
        best_feature = features[np.argmax(igs)]
        root = Node(checking_feature=best_feature)
        
        # Partition the data using the best feature.
        # (Assuming binary features: 1 for yes and 0 for no.)
        left_mask = (X[:, best_feature] == 1)
        right_mask = (X[:, best_feature] == 0)
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Remove the chosen feature from the candidate list.
        new_features = np.setdiff1d(features, [best_feature])
        
        # Recursively build the subtrees.
        # Convention: left_child corresponds to feature value 1.
        root.left_child = self.create_tree(X_left, y_left, new_features, mode(y.flatten()), depth + 1)
        root.right_child = self.create_tree(X_right, y_right, new_features, mode(y.flatten()), depth + 1)
        
        return root
    
    def calculate_ig(self, classes_vector, feature_vector):
        total_samples = len(classes_vector)
        unique_classes, class_counts = np.unique(classes_vector, return_counts=True)
        # Calculate the overall entropy
        HC = -np.sum((class_counts / total_samples) * np.log2((class_counts + 1e-10) / total_samples))
    
        # Calculate the weighted entropy after splitting on the feature
        feature_values, feature_counts = np.unique(feature_vector, return_counts=True)
        HC_feature = 0
        for value, count in zip(feature_values, feature_counts):
            subset_classes = classes_vector[feature_vector == value]
            subset_unique, subset_counts = np.unique(subset_classes, return_counts=True)
            entropy_subset = -np.sum((subset_counts / count) * np.log2((subset_counts + 1e-10) / count))
            HC_feature += (count / total_samples) * entropy_subset
    
        return HC - HC_feature
       
    def predict(self, X):
        predicted_classes = []
        for sample in X:
            node = self.tree  
            # Traverse the tree until a leaf node is reached.
            while not node.is_leaf:
                # Follow the branch corresponding to the sample’s value for the feature.
                if sample[node.checking_feature] == 1:
                    node = node.left_child
                else:
                    node = node.right_child
            predicted_classes.append(node.category)
        return np.array(predicted_classes)



class RandomForestID3:
    def __init__(self, n_trees=5, max_depth=4, min_samples_split=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
    
    def fit(self, X, y):
        for i in range(self.n_trees):
            print(f"Training tree {i+1}/{self.n_trees}...")
            # Create a bootstrap sample (60% of the original data) with replacement.
            X_sample, y_sample = resample(X, y, replace=True, n_samples=int(len(X) * 0.6))
            tree = ID3(features=np.arange(X.shape[1]), max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        print("RandomForestID3 training completed!")
    
    def predict(self, X):
        # Get predictions from each tree.
        predictions = np.array([tree.predict(X) for tree in self.trees]).T
        # Take the mode (majority vote) of the predictions.
        final_predictions = [mode(pred) for pred in predictions]
        return np.array(final_predictions)


if __name__ == "__main__":
    print("Initializing TextClassifier")
    text_classifier = TextClassifier()
    X_train, y_train, X_dev, y_dev, X_test, y_test = text_classifier.get_data() 

    print("Training RandomForestID3 model...")
    rf_id3 = RandomForestID3(n_trees=10, max_depth=4, min_samples_split=10) 
    rf_id3.fit(X_train, y_train)

    y_dev_pred = rf_id3.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)
    print(f"\n Dev Accuracy: {dev_accuracy:.4f}")

    y_test_pred = rf_id3.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    print("\n Classification Report:")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    df_report = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
    print(df_report)
    
    # Compute precision, recall, F1 for each class (ignoring support)
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