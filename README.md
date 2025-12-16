# Bank Customer Churn Model 🏦

A machine learning project that predicts customer churn for banking institutions using Support Vector Machine (SVM) classification with hyperparameter tuning.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Objectives](#project-objectives)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Customer churn is a critical metric for banks and financial institutions. This project implements a machine learning model to predict whether a customer will leave the bank based on various features such as credit score, geography, gender, age, balance, and account activity.

## 📊 Dataset

The project uses the Bank Churn Modelling dataset containing **10,000 customer records** with the following features: 

- **CustomerId**: Unique identifier for each customer
- **Surname**: Customer's last name
- **CreditScore**: Credit score of the customer
- **Geography**: Customer's location (France, Germany, Spain)
- **Gender**: Male or Female
- **Age**: Customer's age
- **Tenure**: Number of years as a bank customer
- **Balance**: Account balance
- **Num Of Products**: Number of bank products used
- **Has Credit Card**: Whether customer has a credit card (0/1)
- **Is Active Member**: Whether customer is an active member (0/1)
- **Estimated Salary**: Customer's estimated salary
- **Churn**:  Target variable (0 = Retained, 1 = Churned)

**Data Source**: [YBI Foundation Dataset Repository](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Bank%20Churn%20Modelling. csv)

## 🎯 Project Objectives

1. **Data Encoding**: Convert categorical variables to numerical format
2. **Feature Scaling**: Normalize features for optimal model performance
3. **Handling Imbalanced Data**: 
   - Random Under Sampling
   - Random Over Sampling
4. **Support Vector Machine Classifier**:  Implement SVM for classification
5. **Grid Search for Hyperparameter Tuning**:  Optimize model parameters

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**:  Data manipulation and analysis
- **NumPy**: Numerical computing
- **Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **Google Colab**: Development environment

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/DishaKotian/Bank_Customer_Churn_Model. git
cd Bank_Customer_Churn_Model
```

2. Install required packages:
```bash
pip install pandas numpy seaborn scikit-learn
```

3. Open the Jupyter Notebook:
```bash
jupyter notebook Bank_Customer_Churn_Model.ipynb
```

Or run directly in [Google Colab](https://colab.research.google.com/)

## 🚀 Usage

1. Open the `Bank_Customer_Churn_Model.ipynb` notebook
2. Run all cells sequentially
3. The notebook will: 
   - Load and analyze the dataset
   - Perform data encoding
   - Apply feature scaling
   - Handle class imbalance
   - Train the SVM model
   - Evaluate model performance

## 📈 Methodology

### 1. Data Preprocessing
- Set `CustomerId` as index (no duplicates found)
- Encoded categorical variables: 
  - **Geography**: France=2, Germany=1, Spain=0
  - **Gender**: Male=0, Female=1
  - **Num Of Products**: 1=0, 2-4=1

### 2. Feature Engineering
- Removed non-predictive features (CustomerId, Surname)
- Applied feature scaling for numerical variables

### 3. Handling Imbalanced Data
- Implemented Random Under Sampling
- Implemented Random Over Sampling
- Compared model performance with both techniques

### 4. Model Training
- Support Vector Machine (SVM) Classifier
- Grid Search CV for hyperparameter optimization
- Cross-validation for robust evaluation

## 📊 Model Performance

The model is evaluated using:
- **Accuracy Score**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**

## 🤝 Contributing

Contributions are welcome!  Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Disha Kotian**
- GitHub: [@DishaKotian](https://github.com/DishaKotian)
- Email: kotiandishaj5335@gmail.com

## 🙏 Acknowledgments

- Dataset provided by [YBI Foundation](https://github.com/YBIFoundation/Dataset)
- Developed using Google Colab

---

⭐ If you found this project helpful, please consider giving it a star! 
```
