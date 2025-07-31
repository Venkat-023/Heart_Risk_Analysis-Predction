🫀 Heart Disease Risk Prediction Using Machine Learning
This project presents an end-to-end pipeline for predicting heart disease risk using supervised machine learning. The workflow involves data analysis, preprocessing, dimensionality reduction, model training with hyperparameter tuning, evaluation with cross-validation, and performance visualization. The goal is to build interpretable and high-performing models for binary classification of heart disease presence.

📌 Objectives
Perform exploratory data analysis (EDA) to understand data distribution and relationships.

Clean and preprocess data, selecting relevant features.

Separate numerical and categorical features for targeted analysis.

Apply PCA for dimensionality reduction and visualization.

Train and tune multiple classification algorithms.

Validate results using cross-validation and confusion matrices.

Visualize decision boundaries and model comparisons.

🧪 Exploratory Data Analysis (EDA)
Initial Data Inspection

Reviewed data shape, types, and null values.

Analyzed correlations between each feature and the target variable.

Dropped features with negligible or zero correlation.

Feature Categorization

Numerical Features: Continuous variables such as age, cholesterol, etc.

Categorical Features: Discrete variables such as chest pain type, sex, etc.

Numerical Feature Analysis

Used hue-based plots (sns.scatterplot, sns.pairplot) to examine class separability.

Identified that two target classes are clearly separable, suggesting suitability of KNN.

Checked distributions using distplot and handled outliers appropriately.

Categorical Feature Analysis

Used count plots to assess class distributions within categorical features.

🔧 Feature Engineering
Removed low-impact features based on domain understanding and correlation heatmaps.

Handled categorical encoding where necessary.

Applied Principal Component Analysis (PCA) to reduce dimensionality for 2D visualization and meshgrid decision boundary plotting.

🤖 Machine Learning Models
Models Trained
Logistic Regression

K-Nearest Neighbors (KNN)

Random Forest Classifier

Hyperparameter Tuning
Used RandomizedSearchCV to find optimal hyperparameters for each model.

Evaluated models using standard accuracy and cross-validation.

Initial Model Results
Model	Accuracy (%)
Logistic Regression	79.60
Random Forest	98.53
K-Nearest Neighbors	100.00 (suspicious)

Due to the unrealistic perfect score from KNN, additional validation was performed.

KNN Cross-Validation
python
CV Accuracy Scores: [0.9805, 0.9707, 0.9707, 1.0, 0.9853]
Mean CV Accuracy: 98.15%
This confirmed KNN was not overfitting and indeed performed exceptionally well on this dataset.

📊 Evaluation & Visualization
Confusion Matrices: Plotted using heatmaps for each model to inspect class-level performance.

ROC Curves & AUC Scores: (Optional if included)

Decision Boundary Plots:

Applied PCA to reduce features to 2D.

Plotted meshgrid-based decision boundaries for KNN and Random Forest to visually compare classifier behavior.

📁 Project Structure
heart-disease-prediction/
├── data/                   # Dataset
├── notebooks/              # Jupyter notebooks (EDA, Modeling, Visualization)
├── models/                 # Serialized models (optional)
├── visuals/                # Plots and figures
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
📚 Technologies Used
Python 3

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Jupyter Notebook for iterative development and documentation

PCA for dimensionality reduction

RandomizedSearchCV and cross_val_score for tuning and validation

✅ Key Takeaways
KNN emerged as the best-performing model with strong class separation and excellent generalization.

Random Forest also performed well with high accuracy and interpretability.

PCA and visualization techniques provided valuable insights into model behavior and data structure.

Evaluation was robust through confusion matrices and cross-validation, ensuring confidence in the results.

📌 Future Improvements
Incorporate ensemble methods like Gradient Boosting or XGBoost.

Perform feature importance analysis for interpretability.

Deploy the best model using a web framework (e.g., Flask or Streamlit).

Address potential class imbalance (if dataset is updated).
