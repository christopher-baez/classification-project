# 📊 Classification Project

## 📝 Project Description
This project aims to delve deeper into customer behavior to identify underlying factors that are not immediately apparent. Understanding key factors driving customer churn can help preemptively address issues and prevent both current and new customers from considering leaving the company.

## 🎯 Goals
The goal of this project is to identify key drivers of customer churn, pinpoint customers at risk of churn, and make actionable recommendations to reduce the monthly churn rate and enhance customer retention.

## 🗂 Data Dictionary
- **senior_citizen**: Whether the customer is a senior citizen.
- **partner**: If the customer has a partner on the line.
- **dependents**: Whether there are dependents on the line.
- **tenure**: Number of months the customer has been with the company.
- **phone_service**: If the customer has phone service.
- **multiple_lines**: If the customer has multiple lines.
- **streaming_tv**: Whether the customer pays for streaming TV.
- **streaming_movies**: Whether the customer pays for streaming movies.
- **monthly_charges**: The customer's monthly payment.
- **total_charges**: Total amount paid by the customer.
- **churn**: Whether the customer left the company.
- **has_add_ons**: Whether the customer has additional services.
- **has_internet**: Whether the customer pays for internet.
- **monthly_contract**: Whether the customer is on a monthly or yearly contract.
- **automatic_payment**: Whether the customer uses manual or automatic payment methods.
- **is_male**: Gender of the customer.

## 📊 Project Plan
- **Data Acquisition**: Acquired from a SQL database provided by Codeup, then transferred to a Python environment for analysis.
- **Data Preparation**: 
  - Converted 'total charges' to a float.
  - Combined 'online_security', 'online_backup', 'device_protection', 'tech_support' into 'add_ons'.
  - Removed 'payment_type_id', 'internet_service_id', 'contract_type_id', 'customer_id'.

## 🔍 Exploratory Data Analysis
Key relationships identified:
- **Senior Citizens**: Significant correlation with churn.
- **Monthly Contracts**: High churn rates among monthly contracts.
- **Automatic Payments**: Impact on churn likelihood.
- **Monthly Charges**: Direct association with customer churn.

## 🤖 Modeling
Used a Decision Tree Classifier:
- **Model Selection**: Chosen for interpretability and handling categorical data.
- **Parameter Tuning**: Experimented with tree depths for optimal performance.
- **Performance**: Achieved 81% accuracy, surpassing the 73% baseline.

## 🎯 Conclusions and Recommendations
Strategies to reduce churn:
- **Senior Citizens**: Improve communication and support.
- **Monthly Contracts**: Incentives for longer-term commitments.
- **Automatic Payments**: Promote automated payment methods with benefits.

## 🛠 Tools and Technologies Used
- **Languages and Environments**: SQL, Python, Jupyter Notebook.
- **Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, SciPy.stats.
- **Methodologies**: Data Cleaning, EDA, Statistical Testing, ML Modeling, Data Visualization, Function Creation.

## 🔄 Instructions for Reproducing the Project
- Environment setup.
- Data acquisition.
- Running the code.
- Generating the final report.

## 🚀 Future Work
With more time, I would:
- Explore additional variables.
- Experiment with different models.
- Investigate more solutions to address the causes of customer churn.

## 🙌 Acknowledgments
Special thanks to the Codeup staff.

