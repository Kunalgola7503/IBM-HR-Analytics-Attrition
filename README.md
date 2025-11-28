# IBM HR Analytics - Employee Attrition Prediction

## 📊 Project Overview

This project analyzes employee attrition patterns using the IBM HR Analytics dataset and builds predictive models to identify employees at risk of leaving. The analysis combines exploratory data analysis (EDA), machine learning classification models, and interactive visualizations to provide actionable insights for HR decision-making.

**Key Objectives:**
- Identify key factors contributing to employee attrition
- Build accurate classification models to predict attrition risk
- Provide data-driven recommendations for employee retention strategies
- Create interactive dashboards for stakeholder engagement

---

## 📁 Project Structure

```
IBM-HR-Analytics-Attrition/
├── 01_Attrition_EDA.ipynb              # Exploratory Data Analysis
├── 02_Attrition_Modeling.ipynb         # Classification Model Development
├── IBM_HR_Analytics.xlsx               # Original dataset
├── employee_attrition_eda_clean.csv    # Cleaned dataset after EDA
├── attrition_model.pkl                 # Trained classification model
├── attrition_scaler.pkl                # Feature scaler for predictions
└── README.md                           # Project documentation
```

---

## 📈 Dataset Information

**Source:** IBM HR Analytics Employee Attrition & Performance Dataset

**Dataset Overview:**
- **Total Records:** 1,470 employees
- **Features:** 35 attributes including demographics, job characteristics, and satisfaction metrics
- **Target Variable:** Attrition (Yes/No)
- **Attrition Rate:** ~16% of employees in the dataset

**Key Features:**
- Demographics: Age, Gender, Marital Status, Education
- Employment: Department, Job Role, Years at Company, Monthly Income
- Work-Life Balance: Overtime, Business Travel, Distance from Home
- Satisfaction Metrics: Job Satisfaction, Environment Satisfaction, Work-Life Balance
- Performance: Performance Rating, Years Since Last Promotion

---

## 🔍 Analysis Workflow

### 1. Exploratory Data Analysis (`01_Attrition_EDA.ipynb`)

**Key Activities:**
- Data cleaning and preprocessing
- Handling missing values and duplicates
- Univariate and bivariate analysis
- Feature correlation analysis
- Visualization of attrition patterns across different features
- Outlier detection and treatment

**Key Findings:**
- Employees working overtime have significantly higher attrition rates
- Younger employees and those in lower job levels show higher turnover
- Job satisfaction and work-life balance are strong predictors
- Frequent business travelers have increased attrition risk
- Distance from home impacts retention

### 2. Predictive Modeling (`02_Attrition_Modeling.ipynb`)

**Models Implemented:**
- Logistic Regression (Baseline)
- Random Forest Classifier
- Gradient Boosting Classifier (Best Performance)

**Model Performance:**
- **Accuracy:** ~87%
- **Precision:** High precision in identifying at-risk employees
- **Recall:** Effective detection of actual attrition cases
- **F1-Score:** Balanced performance metric

**Feature Engineering:**
- Encoding categorical variables
- Feature scaling and normalization
- Feature importance analysis
- Handling class imbalance

---

## 📊 Power BI Dashboard (Coming Soon)

**Interactive dashboard for HR analytics visualization**

### Planned Dashboard Features:

#### 1. **Executive Summary Page**
- Overall attrition rate KPI
- Total headcount and turnover trends
- Year-over-year attrition comparison
- Department-wise attrition overview

#### 2. **Employee Demographics Analysis**
- Attrition by age group, gender, and marital status
- Education level impact on retention
- Interactive filters for demographic segmentation

#### 3. **Job & Department Insights**
- Attrition rates by department and job role
- Salary analysis and compensation impact
- Job level and tenure relationship
- Overtime and work-life balance metrics

#### 4. **Satisfaction & Engagement Metrics**
- Job satisfaction scores vs. attrition
- Environment satisfaction trends
- Work-life balance ratings
- Relationship satisfaction analysis

#### 5. **Predictive Analytics Integration**
- ML model prediction visualization
- Risk scoring for current employees
- Feature importance dashboard
- What-if analysis for retention strategies

#### 6. **Actionable Recommendations**
- High-risk employee identification
- Targeted retention strategy suggestions
- Cost impact analysis of attrition
- Retention program effectiveness tracking

**Note:** The Power BI dashboard will be added to this repository soon, providing stakeholders with an interactive interface to explore insights and make data-driven HR decisions.

---

## 🛠️ Technologies Used

**Programming & Analysis:**
- Python 3.x
- Pandas, NumPy (Data manipulation)
- Matplotlib, Seaborn (Visualization)
- Scikit-learn (Machine Learning)

**Machine Learning:**
- Classification algorithms
- Model evaluation metrics
- Hyperparameter tuning
- Cross-validation

**Visualization:**
- Jupyter Notebook visualizations
- Power BI (Dashboard - upcoming)

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
jupyter notebook
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Kunalgola7503/IBM-HR-Analytics-Attrition.git
cd IBM-HR-Analytics-Attrition
```

2. **Install required packages:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

4. **Run the notebooks in sequence:**
   - Start with `01_Attrition_EDA.ipynb` for data exploration
   - Then run `02_Attrition_Modeling.ipynb` for model development

---

## 📊 How to Use the Trained Model

The repository includes pre-trained model files for immediate use:

```python
import pickle
import pandas as pd

# Load the trained model and scaler
with open('attrition_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('attrition_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare your employee data
# (Ensure features match the training data format)
employee_data = pd.DataFrame(...)  # Your employee data

# Scale features
employee_scaled = scaler.transform(employee_data)

# Make predictions
predictions = model.predict(employee_scaled)
probabilities = model.predict_proba(employee_scaled)

# predictions: 0 = No Attrition, 1 = Attrition
# probabilities: [prob_no_attrition, prob_attrition]
```

---

## 📌 Key Insights & Recommendations

### Critical Attrition Factors:
1. **Overtime Work:** Strong correlation with attrition
2. **Work-Life Balance:** Low ratings increase turnover risk
3. **Job Satisfaction:** Primary predictor of retention
4. **Career Development:** Stagnant growth leads to attrition
5. **Compensation:** Competitive pay is essential but not sufficient

### Recommended Actions:
- **Monitor high-risk profiles:** Young employees, overtime workers, low satisfaction scores
- **Improve work-life balance:** Reduce mandatory overtime, flexible work arrangements
- **Career development:** Clear promotion paths and skill development opportunities
- **Regular engagement:** Conduct satisfaction surveys and stay interviews
- **Competitive compensation:** Regular market analysis and salary benchmarking

---

## 📈 Results & Impact

**Model Performance:**
- Successfully predicts employee attrition with ~87% accuracy
- Identifies key risk factors for targeted interventions
- Enables proactive retention strategies

**Business Value:**
- Early identification of at-risk employees
- Data-driven HR decision making
- Reduced recruitment and training costs
- Improved employee retention rates
- Enhanced workplace culture insights

---

## 🔄 Future Enhancements

- [ ] Complete Power BI interactive dashboard
- [ ] Deploy model as web application
- [ ] Add real-time prediction API
- [ ] Implement advanced deep learning models
- [ ] Include text analysis of employee feedback
- [ ] Develop employee engagement scoring system
- [ ] Create automated reporting system

---

## 👤 Author

**Kunal Gola**
- GitHub: [@Kunalgola7503](https://github.com/Kunalgola7503)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/kunal-gola)

---

## 📄 License

This project is created for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- IBM for providing the HR Analytics dataset
- Open-source community for Python libraries
- Data science community for best practices and methodologies

---

## 📞 Contact & Feedback

For questions, suggestions, or collaboration opportunities, feel free to reach out through GitHub or LinkedIn!

**Last Updated:** November 2025
