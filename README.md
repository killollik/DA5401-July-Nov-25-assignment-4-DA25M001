# DA5401 A4: GMM-Based Synthetic Sampling for Imbalanced Data

**Course:** DA5401 - Advanced Data Analytics  
**Term:** July-Nov 2025  
**Student ID:** DA25M001

## 1. Project Overview

This project tackles the critical challenge of class imbalance in machine learning, specifically within the context of **credit card fraud detection**. The dataset used is heavily imbalanced, with fraudulent transactions representing a tiny fraction (0.173%) of the total data.

The core objective is to implement and evaluate a sophisticated, model-based oversampling technique—**Gaussian Mixture Models (GMM)**—to generate high-quality synthetic data for the minority (fraud) class. The effectiveness of this approach is measured by comparing the performance of a Logistic Regression classifier trained on the GMM-balanced data against a baseline model trained on the original, imbalanced data.

## 2. Problem Statement

A financial institution needs a robust fraud detection model. The primary challenge is creating a training set that enables a classifier to learn the subtle patterns of fraudulent transactions without overfitting or misclassifying the vast majority of normal transactions. This project implements a GMM-based data generation pipeline and provides a detailed analysis of its impact on model performance, focusing on metrics relevant to imbalanced classification problems like Precision, Recall, and F1-Score.

## 3. Repository Structure

```
.
├── _.ipynb      # Main Jupyter Notebook with all code, visualizations, and analysis.
├── creditcard.csv                    # The dataset file (if included in the repo).
└── README.md                         # This readme file.
```

## 4. Methodology

The project is structured into three main parts, as detailed in the Jupyter Notebook:

### Part A: Baseline Model and Data Analysis
1.  **Data Loading & Analysis**: The `creditcard.csv` dataset is loaded, and a thorough analysis of the severe class imbalance is conducted.
2.  **Data Splitting**: The data is split into stratified training (80%) and testing (20%) sets to ensure the original class distribution is preserved.
3.  **Baseline Model**: A standard **Logistic Regression** classifier is trained on the original, imbalanced training data to establish a performance benchmark.

### Part B: Gaussian Mixture Model (GMM) for Synthetic Sampling
1.  **Theoretical Foundation**: A markdown analysis explains the fundamental differences between GMM-based sampling and simpler methods like SMOTE, highlighting GMM's theoretical advantages in modeling complex, multi-modal distributions.
2.  **GMM Implementation**: A GMM is fitted exclusively to the minority (fraud) class data from the training set. The optimal number of components (`k`) is determined using the **Bayesian Information Criterion (BIC)**, which balances model fit and complexity.
3.  **Synthetic Data Generation**: The trained GMM is used to sample a sufficient number of new, synthetic fraud instances to create a perfectly balanced training dataset.
4.  **Visualization**: PCA is used to visually inspect and confirm the quality of the generated synthetic data compared to the original minority samples.

### Part C: Performance Evaluation and Conclusion
1.  **Enhanced Model Training**: A new Logistic Regression model is trained on the GMM-balanced training dataset.
2.  **Comparative Analysis**: This enhanced model is evaluated on the original, unseen, imbalanced test set. Its performance (Precision, Recall, F1-Score) is compared side-by-side with the baseline model.
3.  **Final Recommendation**: A detailed conclusion is drawn based on the empirical results, discussing the trade-offs and providing a clear recommendation on the practical use of this GMM-based approach.

## 5. Key Findings & Results

The analysis revealed a classic **precision-recall trade-off**, a common outcome of oversampling techniques.

| Metric (Fraud Class) | Baseline Model | GMM-Enhanced Model | Improvement |
| :------------------- | :------------: | :----------------: | :---------: |
| **Precision**        |     0.830      |       0.082        |  **-87.2%** |
| **Recall**           |     0.643      |       0.898        |  **+39.7%** |
| **F1-Score**         |     0.724      |       0.151        |  **-79.2%** |

-   **Success in Recall**: The GMM-enhanced model was highly successful at its primary goal: detecting more fraud. The **Recall rate skyrocketed by ~40%**, reducing missed frauds (False Negatives) from 35 to just 10.
-   **Failure in Precision**: This gain came at an extreme cost. The model became overly aggressive, causing **Precision to plummet by ~87%**. The number of false alarms (False Positives) exploded from 13 to 981.
-   **Overall Performance Decline**: The F1-Score, which balances this trade-off, dropped significantly, indicating that the GMM-enhanced model, in its current state, is less practical for production use than the baseline.

## 6. Final Recommendation

**Not Recommended for Direct Production Use.**

While GMM oversampling is a theoretically powerful method for teaching a model to recognize rare patterns, applying it naively leads to a model that is too aggressive and unreliable. The operational cost of investigating a **75-fold increase in false alarms** would be prohibitive.

However, the experiment is valuable as it demonstrates the technique's potential. The recommendation is to use GMM sampling not as a standalone solution but as part of a more nuanced strategy, including:
1.  **Decision Threshold Tuning**: Adjusting the classification threshold to improve precision.
2.  **Hybrid Sampling Methods**: Combining GMM oversampling with majority class undersampling.
3.  **Using More Complex Models**: Employing algorithms like LightGBM or XGBoost that may handle resampled data more effectively.

## 7. How to Run the Code

### Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`.

### Installation
You can install the necessary libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### Running the Notebook
1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/DA5401-July-Nov-25-assignment-4-DA25M001.git
    cd DA5401-July-Nov-25-assignment-3-DA25M001
    ```
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    ```
3.  Open the `DA5401_A4_GMM_Analysis.ipynb` file and run the cells sequentially. The notebook uses `kagglehub` to automatically download the dataset.

---
