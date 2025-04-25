# Text Classification with Custom ML & Deep Learning Models

This project implements several custom machine learning algorithms for binary text classification (e.g. positive/negative sentiment), evaluated on the IMDB dataset. It also includes comparisons with scikit-learn implementations and a deep learning model using a stacked bidirectional RNN with LSTM/GRU in PyTorch.


## 📚 Contents

### Part A: Custom ML Implementations
We implemented the following algorithms **from scratch**, without using built-in ML functions:
- ✅ Naive Bayes (Bernoulli or Multinomial)
- ✅ Random Forest with ID3 (max depth as a hyperparameter)
- ✅ AdaBoost with decision stumps (depth-1 decision trees)
- ✅ Logistic Regression with stochastic gradient ascent and regularization

**Features:**
- Texts are represented as binary vectors (0/1) indicating the presence of words.
- Vocabulary is built by removing the `n` most common and `k` rarest words.
- From the remaining words, we select the `m` most informative ones (based on information gain).
- The models are trained and evaluated using a portion of the **IMDB Large Movie Review Dataset**.


### Part B: Comparison with Standard Implementations
We compare our custom models with:
- scikit-learn’s versions of the same algorithms (Naive Bayes, Random Forest, AdaBoost, Logistic Regression)
- Additional models like `MLPClassifier` from scikit-learn (where applicable)

We use the **same feature representation** and **hyperparameters** (where possible) for a fair comparison.


### Part C: Deep Learning Model (PyTorch)
We implement a deep learning model:
- A **Stacked Bidirectional RNN** with LSTM or GRU units
- Uses **Global Max Pooling** after the RNN layers
- Trained with **Adam optimizer**
- Uses **pre-trained word embeddings**
- Training is stopped based on dev set performance


## 📈 Evaluation & Outputs

For all models, we include:
- Learning curves (training and dev sets)
- Precision, Recall, and F1 scores for both classes
- Micro- and Macro-averaged results on the test set

Example metrics:
- Precision / Recall / F1 per class
- Learning curves by number of training examples
- Loss curves (Part C)


## ⚙️ Hyperparameters (examples)
| Parameter            | Value (Example)    |
|----------------------|--------------------|
| n (most common)      | 100                |
| k (rarest words)     | 50                 |
| m (top informative)  | 1500               |
| λ (regularization)   | 0.01               |
| Random Forest Trees  | 100 (max depth 5)  |
| AdaBoost Iterations  | 50                 |
| RNN Layers           | 2 (Bi-LSTM)        |
| Optimizer            | Adam               |

Hyperparameters were selected through dev set tuning and common defaults from the literature.


## 📦 Technologies Used
- Python 3.x
- NumPy, Pandas, Matplotlib
- Scikit-learn (Part B)
- PyTorch (Part C)
- NLTK or spaCy for tokenization/preprocessing (optional)


## 💡 Notes
- No ML libraries were used for model training in Part A
- Ready-made tools were used **only** for preprocessing and visualization
- All algorithms can be applied to other binary text classification tasks


## 📂 Dataset
IMDB Large Movie Review Dataset  
📎 https://ai.stanford.edu/~amaas/data/sentiment/

---

## 👩‍💻 Contributors
Project developed as part of a university assignment on machine learning and natural language processing.

