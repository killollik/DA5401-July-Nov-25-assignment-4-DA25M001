

# GMM-Based Synthetic Sampling for Imbalanced Data
### Advanced Fraud Detection for Course DA5401

This repository contains the Jupyter Notebook for Assignment 4 of the DA5401 course (`DA5401_July_Nov_25_assignment_4_DA25M001.ipynb`). The project demonstrates an advanced technique for handling a severely imbalanced dataset in the context of credit card fraud detection. It uses a **Gaussian Mixture Model (GMM)** to model the minority (fraud) class, generate high-quality synthetic data, and train a robust classifier that significantly outperforms a baseline model.

---

## 📝 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Methodology & Workflow](#-methodology--workflow)
  - [Part A: Baseline Model](#part-a-baseline-model--data-analysis)
  - [Part B: GMM for Synthetic Sampling](#part-b-gaussian-mixture-model-gmm-for-synthetic-sampling)
  - [Part C: Multi-Stage Optimization](#part-c-multi-stage-optimization-for-superior-performance)
- [Key Findings & Results](#-key-findings--results)
- [Final Conclusion & Recommendation](#-final-conclusion--recommendation)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🎯 Problem Statement

Credit card fraud detection is a classic example of an imbalanced classification problem. In the provided dataset, fraudulent transactions represent only **0.173%** of the data, with an imbalance ratio of approximately **578:1**.

Standard machine learning models trained on such data tend to be biased towards the majority class ('Normal') and perform poorly at identifying the minority class ('Fraud'). This results in a high number of **False Negatives** (missed frauds), which poses a significant financial and security risk. The primary goal of this project is to develop a model with a superior **F1-score** by improving the detection of fraudulent transactions (recall) without unacceptably compromising the reliability of its predictions (precision).

## 📊 Dataset

The project utilizes the **Credit Card Fraud Detection** dataset, sourced from Kaggle.
- **Source:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** The dataset contains 30 numerical features. Features `V1` through `V28` are the result of a PCA transformation. The only features that have not been transformed are `Time` and `Amount`.
- **Target:** The `Class` column, where `1` indicates a fraudulent transaction and `0` indicates a normal one.

## ⚙️ Methodology & Workflow

The project is structured in three main parts, following a progression from a simple baseline to a fully optimized model.

### Part A: Baseline Model & Data Analysis
1.  **Data Loading & Analysis:** The dataset is loaded and analyzed to confirm the severe class imbalance.
2.  **Visualization:** Pie charts, bar plots, and KDE plots are used to visualize the class distribution and feature differences between normal and fraudulent transactions.
3.  **Baseline Model Training:** A standard `LogisticRegression` classifier is trained on the original, imbalanced, and scaled data. This model establishes the performance benchmark.
    - **Result:** The baseline model achieved a reasonable precision of **0.83** but a poor recall of **0.64**, resulting in an F1-score of **0.724**. It failed to identify 36% of all fraudulent transactions.

### Part B: Gaussian Mixture Model (GMM) for Synthetic Sampling
1.  **Theoretical Foundation:** A GMM was chosen over simpler methods like SMOTE because of its ability to probabilistically model complex, multi-modal distributions. This is ideal for capturing potentially different types of fraud patterns within the minority class.
2.  **Optimal Component Selection:** The fraudulent transactions from the training set were isolated. AIC and BIC scores were calculated for GMMs with a varying number of components, and `k=3` was identified as the optimal number. This suggests the fraud data has three distinct underlying patterns.
3.  **Visualization of Synthetic Data:** PCA was used to visualize the original fraud data points against a cloud of GMM-generated synthetic data, confirming that the GMM successfully learned and replicated the underlying distribution.

### Part C: Multi-Stage Optimization for Superior Performance
A naive oversampling to a 1:1 ratio initially led to poor results (high recall but extremely low precision). Therefore, a strategic, multi-stage optimization process was implemented:

1.  **Finding the Best Sampling Ratio:** Instead of a 1:1 balance, multiple minority-to-majority ratios were tested (from 5% to 95%). A **5% ratio** (1 fraud sample for every 20 normal samples) was found to yield the highest F1-score, providing a much better balance.
2.  **Hyperparameter Tuning:** Using the 5% resampled dataset, `GridSearchCV` was employed to find the optimal regularization parameter `C` for the `LogisticRegression` model, further improving its performance.
3.  **Decision Threshold Optimization:** The final and most impactful step. The model's default 0.5 probability threshold was adjusted by analyzing the precision-recall curve. A new threshold was found that maximized the F1-score, creating the best possible trade-off between identifying fraud and avoiding false alarms.

## 📈 Key Findings & Results

The multi-stage optimization process successfully transformed the GMM-based approach from a liability into a high-performing solution. The final optimized model demonstrated significant improvements across all key metrics compared to the baseline.

| Metric              | Baseline Model | Optimized GMM Model | Improvement (%) |
| ------------------- | :------------: | :-----------------: | :-------------: |
| **Precision (Fraud)** |     0.829      |       **0.840**       |     +1.3%       |
| **Recall (Fraud)**    |     0.643      |       **0.806**       |     +25.4%      |
| **F1-Score (Fraud)**  |     0.724      |       **0.823**       |     +13.6%      |

<p align="center">
  <img src="https://i.imgur.com/48mYn7r.png" alt="Comparison Bar Chart" width="700">
</p>

- The optimized model increased **recall by over 25%**, catching a much higher proportion of fraudulent transactions.
- It also managed to slightly **increase precision**, meaning its positive predictions are even more reliable than the baseline.
- This resulted in a **13.6% improvement in the overall F1-score**, indicating a more robust and effective model.

## ✅ Final Conclusion & Recommendation

**Recommendation: HIGHLY RECOMMENDED (with optimizations)**

GMM-based synthetic oversampling is a powerful and effective technique for this severely imbalanced dataset, but **only when implemented strategically**. Naive application (e.g., balancing to 1:1) is detrimental and leads to a model with unacceptably low precision.

The final optimized model, developed through strategic ratio selection and threshold tuning, is demonstrably superior to the baseline. It provides a much better balance between detecting fraud (high recall) and maintaining trust in its predictions (high precision). This approach should be strongly considered for classification problems where the minority class is complex and critically important to detect.

## 🚀 How to Run
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/killollik/DA5401_July_Nov_25_assignment_4_DA25M001.git
    cd DA5401_July_Nov_25_assignment_4_DA25M001
    ```
2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
    ```
4.  **Run the Jupyter Notebook**
    ```bash
    jupyter notebook DA5401_July_Nov_25_assignment_4_DA25M001.ipynb
    ```

## 👤 Author
- **killollik**
- **GitHub:** [https://github.com/killollik](https://github.com/killollik)
