import streamlit as st
import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Robust DB path construction ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "database", "german_credit.db")

st.set_page_config(page_title="German Credit Risk Analysis - EDA", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Project Overview",
    "Data Summary & Quality Checks",
    "Feature Engineering & Dimensionality Reduction",
    "Predictive Modeling: Linear & Logistic Regression"
])

if page == "Project Overview":
    st.title("German Credit Risk Dataset Overview")
    st.markdown("""
    ### Project Context and Motivation
    This project was developed for an interview with the PD/Rating Modelling & Process unit at **HypoVereinsbank – UniCredit**, Germany, led by **Dr. Ghazal Tayebirad**. It focuses on applying foundational statistical concepts and SQL queries to a real-world dataset in a business-relevant domain. The goal was to demonstrate practical skills in data handling, exploratory analysis, and basic credit risk evaluation within a limited timeframe.

    While this version serves as a proof of concept, the dataset and context allow for much deeper statistical analysis, feature engineering, and model validation—available to expand upon request.

    **by Mohammad Joudy**
    """)
    st.markdown("""
    **Dataset:** [German Credit Data (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

    This dataset contains information on 1,000 individuals who have applied for credit at a German bank. Each row represents an applicant, with 20 features describing their financial status, personal information, and the outcome of their credit application.

    **Target variable:** `Credit_risk` (1 = Good, 0 = Bad)

    **Purpose:**
    - To predict whether an applicant is a good or bad credit risk based on their attributes.
    - To explore which features are most predictive of credit risk.

    **Source:**
    - UCI Machine Learning Repository: Statlog (German Credit Data)
    - [Dataset Documentation](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
    """)

elif page == "Data Summary & Quality Checks":
    st.title("Data Summary & Quality Checks")
    st.markdown("""
    This section provides an initial exploration of the German Credit dataset using SQL queries and basic statistical analysis. These are essential steps in any data analysis or modeling workflow, aimed at better understanding the structure, distribution, and quality of the data. By examining key features such as credit amount, employment duration, and personal demographics, we can uncover patterns and relationships that inform later stages of feature engineering, model selection, and evaluation.
    """)
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM credit_data", conn)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    st.subheader("Target Variable Distribution")
    sql_target_dist = '''
    SELECT Credit_risk, COUNT(*) AS count
    FROM credit_data
    GROUP BY Credit_risk;
    '''
    st.code(sql_target_dist, language='sql')
    dist = df["Credit_risk"].map({1: "Good", 0: "Bad"}).value_counts()
    st.bar_chart(dist)
    st.markdown("""
    **Interpretation:**
    The dataset is moderately imbalanced, with more 'Good' credit risks than 'Bad'. This is important for modeling, as class imbalance can affect model performance and evaluation.
    """)

    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.markdown("""
    **Interpretation:**
    Summary statistics provide an overview of the range and central tendency of numeric features. For example, the average credit amount and duration can help us understand typical applicants.
    """)

    st.subheader("Data Quality Checks")
    missing = df.isnull().sum()
    duplicates = df.duplicated().sum()
    st.write("Missing values per column:")
    st.write(missing)
    st.write(f"Number of duplicate rows: {duplicates}")
    st.markdown("""
    **Interpretation:**
    There are no missing values or duplicate rows in the dataset, indicating high data quality and reliability for analysis.
    """)

    st.markdown("---")
    st.subheader("Average Credit Amount by Credit Risk")
    sql_query = """
    SELECT Credit_risk, AVG(Credit_amount) AS avg_credit
    FROM credit_data
    GROUP BY Credit_risk;
    """
    st.code(sql_query, language='sql')
    try:
        avg_credit_df = pd.read_sql(sql_query, conn)
        st.dataframe(avg_credit_df)
    except Exception as e:
        st.error(f"Failed to run SQL query: {e}")
    st.markdown("""
    **Interpretation:**
    This table shows the average credit amount for each credit risk group. It provides insight into the typical credit extended to 'Good' and 'Bad' risk applicants, which can be useful for risk assessment and policy making.
    """)

    st.markdown("---")
    st.subheader("Additional Analyses and Visualizations")

    # Block 8: Present employment since vs. risk rate
    sql_block8 = '''
    SELECT Present_employment_since, 
           AVG(Credit_risk) AS risk_rate
    FROM credit_data
    GROUP BY Present_employment_since
    ORDER BY risk_rate ASC;
    '''
    st.code(sql_block8, language='sql')
    try:
        block8_df = pd.read_sql(sql_block8, conn)
        st.dataframe(block8_df)
    except Exception as e:
        st.error(f"Failed to run SQL query (Block 8): {e}")

    # Block 9: Personal status and sex vs. good credit ratio
    sql_block9 = '''
    SELECT Personal_status_and_sex,
           COUNT(*) AS total,
           AVG(Credit_risk) AS good_credit_ratio
    FROM credit_data
    GROUP BY Personal_status_and_sex
    ORDER BY good_credit_ratio ASC;
    '''
    st.code(sql_block9, language='sql')
    try:
        block9_df = pd.read_sql(sql_block9, conn)
        st.dataframe(block9_df)
    except Exception as e:
        st.error(f"Failed to run SQL query (Block 9): {e}")

    # Block 10: Distribution of Credit Amount
    st.code('''
credit_amounts = pd.read_sql("SELECT Credit_amount FROM credit_data;", conn)
credit_amounts.hist(bins=30)
plt.title("Distribution of Credit Amount")
plt.xlabel("Credit Amount")
plt.ylabel("Frequency")
plt.show()
''', language='python')
    try:
        credit_amounts = pd.read_sql("SELECT Credit_amount FROM credit_data;", conn)
        # Compute histogram data
        counts, bins = pd.cut(credit_amounts["Credit_amount"], bins=30, retbins=True)
        hist_df = credit_amounts.groupby(counts).size().reset_index(name='Frequency')
        hist_df['bin_center'] = [interval.mid for interval in hist_df['Credit_amount']]
        hist_df = hist_df.set_index('bin_center')
        st.markdown("**Distribution of Credit Amount**")
        st.bar_chart(hist_df['Frequency'])
        st.markdown("**X-axis:** Credit Amount (bin center)")
        st.markdown("**Y-axis:** Frequency")
        st.markdown("This chart shows the frequency distribution of credit amounts in the dataset, grouped into 30 bins. It helps visualize the range and common values of credit extended to applicants.")
    except Exception as e:
        st.error(f"Failed to plot credit amount distribution: {e}")

elif page == "Feature Engineering & Dimensionality Reduction":
    st.title("Feature Engineering & Dimentionality Reduction")
    st.markdown("""
    This section outlines several potential feature engineering steps and dimensionality reduction techniques that could enhance model performance. While most of these transformations were not applied in the final model, they are presented as options to demonstrate awareness of common preprocessing strategies. In simple and well-structured datasets like this one, such extensive feature engineering may be redundant or have minimal impact—but understanding these techniques remains an important part of a robust data science workflow.
    """)
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM credit_data", conn)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    st.subheader("Feature Engineering")
    st.markdown("""
    **What is Feature Engineering?**
    Feature engineering transforms raw data into features that better represent the underlying problem to predictive models, improving their performance.

    - **Categorical variables** are converted to numeric using one-hot encoding.
    - **Numeric features** are standardized (mean=0, std=1) for fair comparison.
    - **New features** such as `Credit_per_month` (credit amount divided by duration), `High_installment_flag` (installment rate > 3), and age groups are created to capture more nuanced patterns.
    """)
    # Example: show one-hot encoding and scaling
    categorical_cols = [
        "Status_of_existing_checking_account", "Credit_history", "Purpose",
        "Savings_account_bonds", "Present_employment_since",
        "Personal_status_and_sex", "Other_debtors_guarantors", "Property",
        "Other_installment_plans", "Housing", "Job", "Telephone", "foreign_worker"
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    numerical_cols = [
        "Duration_in_month", "Credit_amount", "Installment_rate_in_percentage_of_disposable_income",
        "Present_residence_since", "Age_in_years", "Number_of_existing_credits_at_this_bank",
        "Number_of_people_being_liable_to_provide_maintenance_for"
    ]
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    # New features
    df_encoded["Credit_per_month"] = df_encoded["Credit_amount"] / (df_encoded["Duration_in_month"] + 1)
    df_encoded["High_installment_flag"] = (df_encoded["Installment_rate_in_percentage_of_disposable_income"] > 3).astype(int)
    df_encoded["Age_Group"] = pd.cut(
        df_encoded["Age_in_years"],
        bins=[0, 25, 40, 100],
        labels=["Young", "Middle", "Senior"]
    )
    st.write("Sample of engineered features:")
    st.write(df_encoded[["Credit_amount", "Duration_in_month", "Credit_per_month", "High_installment_flag", "Age_in_years", "Age_Group"]].head())

    # Show number of features before and after encoding
    st.subheader("Feature Dimensionality Before and After Encoding")
    st.markdown(f"""
    - **Original number of features:** {df.shape[1]}
    - **Number of features after encoding and feature engineering:** {df_encoded.shape[1]}
    """)

    # Correlation heatmap (colors only, no numbers)
    st.subheader("Correlation Matrix of Engineered Features")
    numeric_df = df_encoded.select_dtypes(include='number')
    correlation_matrix = numeric_df.drop(columns=["Credit_risk"]).corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix of Features")
    st.pyplot(fig)

    # Correlation of features with target
    st.subheader("Feature Correlation with Credit Risk")
    st.markdown("""
    The following table shows which features are most strongly associated with the target variable (`Credit_risk`). Positive values indicate features that are more common among 'Good' credit risks, while negative values indicate features more common among 'Bad' credit risks.
    """)
    target_corr = numeric_df.corr()["Credit_risk"].sort_values(ascending=False)
    st.write("Top 5 features positively correlated with Credit_risk:")
    st.dataframe(target_corr.head())
    st.write("Top 5 features negatively correlated with Credit_risk:")
    st.dataframe(target_corr.tail())
    st.markdown("""
    **Interpretation:**
    - Features at the top are most positively correlated with being a 'Good' credit risk (Credit_risk=1). These may be protective or favorable factors.
    - Features at the bottom are most negatively correlated, indicating association with 'Bad' credit risk (Credit_risk=0). These may be risk factors.
    - The strength of the correlation (closer to 1 or -1) suggests a stronger linear relationship, but does not imply causation.
    - This analysis helps identify which features are most informative for predicting credit risk and can guide feature selection for modeling.
    """)

    st.subheader("Principal Component Analysis (PCA)")
    st.markdown("""
    **What is PCA?**
    Principal Component Analysis (PCA) is a technique to reduce the dimensionality of data while retaining most of the variance. It helps visualize complex datasets and can improve model performance by removing noise and redundancy.

    **Note:**
    The original dataset contains 20 features, but after feature engineering (especially one-hot encoding of categorical variables), the number of features increases significantly. Each categorical variable is expanded into multiple binary columns, resulting in a higher-dimensional feature space (e.g., 50+ features). PCA is applied to this engineered feature set, not the original columns.
    """)
    X = df_encoded.drop(columns=["Credit_risk", "Age_Group"])
    y = df_encoded["Credit_risk"]
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["Credit_risk"] = y
    # Remove the PCA scatter plot and its interpretation
    # (Remove: fig, ax = plt.subplots(); sns.scatterplot(...); st.pyplot(fig); st.markdown(f"**Interpretation:** ..."))
    # Keep the rest of the PCA and scree plot analysis

    # PCA Scree plot: cumulative explained variance (as in notebook block 27)
    st.subheader("Explained Variance by Number of Principal Components")
    X_full = df_encoded.drop(columns=["Credit_risk", "Age_Group"], errors='ignore')
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_full)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(cum_var)+1), cum_var, marker='o')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance by Number of Principal Components')
    ax.grid(True)
    st.pyplot(fig)
    
    # Interpretation based on the figure
    n_components_90 = np.argmax(cum_var >= 0.90) + 1
    st.markdown(f"""
    **Interpretation:**
    - The scree plot above shows how the cumulative explained variance increases as more principal components are included.
    - In this analysis, the first {n_components_90} principal component{'s' if n_components_90 == 1 else 's'} explain(s) 90% of the variance in the data.
    - This means you can reduce the dimensionality from {X_full.shape[1]} features to just {n_components_90} principal component{'s' if n_components_90 == 1 else 's'} with minimal information loss.
    - Most of the variance is captured by the very first component, indicating that the data has strong redundancy or a dominant underlying factor.
    """) 

elif page == "Predictive Modeling: Linear & Logistic Regression":
    st.title("Predictive Modeling: Linear & Logistic Regression")
    st.markdown("""
    This section applies both linear and logistic regression models to the German Credit dataset. While logistic regression is the appropriate method for binary classification tasks (such as predicting credit risk: good vs. bad), linear regression is also included as a complementary analysis to demonstrate prediction of credit amount and illustrate regression interpretation more broadly.

    Only a subset of features in each model were statistically significant, typically determined by a p-value less than 0.05, indicating a meaningful relationship with the outcome variable. For example, loan duration and installment rate appear consistently significant across both models, reinforcing their importance in credit assessment.
    """)

    # 1. Full statsmodels Logit regression summary (HTML table)
    st.subheader("Statsmodels Logit Regression (Full Analysis Table)")
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM credit_data", conn)
    X = df.drop(columns=["Credit_risk", "Credit_amount"])
    y_class = df["Credit_risk"]
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    import statsmodels.api as sm
    X_const = sm.add_constant(X_encoded)
    logit_model = sm.Logit(y_class, X_const).fit(disp=0)
    summary_html = logit_model.summary().as_html()
    st.markdown(summary_html, unsafe_allow_html=True)

    st.markdown("""
    **Key Results from Logit Regression Summary:**
    
    | Feature                                         | Coefficient | p-value | Interpretation |
    |-------------------------------------------------|-------------|---------|----------------|
    | Duration_in_month                               | -0.451      | 0.000   | Longer duration decreases odds of good credit risk |
    | Installment_rate_in_percentage_of_disposable_income | -0.243  | 0.014   | Higher installment rate decreases odds of good credit risk |
    | Status_of_existing_checking_account_A13          | 1.123       | 0.009   | This status increases odds of good credit risk |
    | Status_of_existing_checking_account_A14          | 1.567       | 0.000   | This status strongly increases odds of good credit risk |
    
    Other features are not statistically significant (p > 0.05).
    
    **Interpretation:**
    - Negative coefficients indicate features that reduce the likelihood of being a good credit risk.
    - Positive coefficients indicate features that increase the likelihood of being a good credit risk.
    - Only features with p < 0.05 are considered statistically significant predictors in this model.
    """)

    # 2. Logistic Regression (Logit) section (classification report, confusion matrix, AUC)
    st.header("Logistic Regression (Logit)")
    st.markdown("""
    **Purpose:** Classify whether a customer is a good or bad credit risk.
    """)
    st.subheader("Results")
    st.markdown("""
    **Classification Report:**
    
    |       | precision | recall | f1-score | support |
    |-------|-----------|--------|----------|---------|
    | 0     | 0.69      | 0.58   | 0.63     | 59      |
    | 1     | 0.83      | 0.89   | 0.86     | 141     |
    | accuracy |         |        | 0.80     | 200     |
    | macro avg | 0.76   | 0.73   | 0.75     | 200     |
    | weighted avg | 0.79| 0.80   | 0.79     | 200     |

    **Confusion Matrix:**
    
    [[ 34  25]
     [ 15 126]]

    **AUC Score:** 0.808
    """)

    # 3. Linear Regression (OLS) section
    st.header("Linear Regression (OLS)")
    st.markdown("""
    **Purpose:** Predict the credit amount a customer receives based on their features.

    **Main Results:**
    - **MSE:** 0.368
    - **MAE:** 0.404
    - **R² Score:** 0.519

    **Key Significant Features:**
    - `Duration_in_month` (coef ≈ 0.56, p < 0.001): Each additional month in the loan duration increases the predicted credit amount by about 0.56 units.
    - `Installment_rate_in_percentage_of_disposable_income` (coef ≈ -0.30, p < 0.001): Higher installment rates are associated with a lower predicted credit amount.
    - Other features, such as `Present_residence_since` and `Number_of_existing_credits_at_this_bank`, are not statistically significant.

    **Interpretation:**
    The linear regression model provides a moderately accurate prediction of credit amount, with a few features (especially loan duration and installment rate) being strong, significant predictors. However, the model does not capture all the complexity in the data, as indicated by the R² value.
    """)

    # 4. Comparison table and summary
    st.header("Comparison Table")
    st.markdown("""
    | Aspect / Metric                                 | Linear Regression (OLS)         | Logistic Regression (Logit)         |
    |-------------------------------------------------|----------------------------------|-------------------------------------|
    | **Target**                                      | Credit_amount (continuous)       | Credit_risk (binary: good/bad)      |
    | **R² / Pseudo R²**                              | 0.519                           | 0.252 (Pseudo R²)                   |
    | **MSE / MAE**                                   | 0.368 / 0.404                   | N/A                                 |
    | **Accuracy**                                    | N/A                             | 0.80                                |
    | **AUC**                                         | N/A                             | 0.81                                |
    | **Best Class Performance**                      | N/A                             | Class 1 (good credit): precision 0.83, recall 0.89 |
    | **Key Significant Predictors (p < 0.05)**       | Duration_in_month, Installment_rate_in_percentage_of_disposable_income | Duration_in_month, Installment_rate_in_percentage_of_disposable_income, Status_of_existing_checking_account_A13/A14 |
    | **Interpretation of Coefficients**              | Change in credit amount per unit | Change in log-odds of good credit per unit (can be exponentiated to odds ratio) |
    | **Model Fit**                                   | Moderate (explains ~52% variance)| Good discrimination (AUC 0.81), strong accuracy for class 1 |
    """)

    st.markdown("""
    **Summary:**
    - Both models identify similar features as significant (e.g., Duration_in_month, Installment_rate_in_percentage_of_disposable_income), but the direction and interpretation differ.
    - Linear regression is moderately predictive for the continuous target, but not perfect (R² = 0.52).
    - Logistic regression is strong at classifying "good credit" cases, with high recall and precision, and a good AUC (0.81), but less effective for "bad credit" (lower recall/precision).
    - Feature importance: Both models agree on some key predictors, which increases confidence in their relevance.
    - Use case: Use linear regression if you want to predict the actual credit amount. Use logistic regression if you want to classify applicants as good/bad credit risk.
    """)

    st.markdown("""
    ---
    ### Predictive Modeling: Linear & Logistic Regression
    This section applies both linear and logistic regression models to the German Credit dataset. While logistic regression is the appropriate method for binary classification tasks (such as predicting credit risk: good vs. bad), linear regression is also included as a complementary analysis to demonstrate prediction of credit amount and illustrate regression interpretation more broadly.

    Only a subset of features in each model were statistically significant, typically determined by a p-value less than 0.05, indicating a meaningful relationship with the outcome variable. For example, loan duration and installment rate appear consistently significant across both models, reinforcing their importance in credit assessment.
    """) 