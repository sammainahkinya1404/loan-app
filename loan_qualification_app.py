import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and preprocess the dataset
@st.cache
def load_data():
    df = pd.read_csv('df1_loan.csv')
    df.drop(['Unnamed: 0', 'Loan_ID'], axis=1, inplace=True)

    label_encoder = LabelEncoder()
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    df['Total_Income'] = df['Total_Income'].str.replace('$', '').astype(float)
    df.fillna(df.mean(), inplace=True)

    return df

df = load_data()

# Train the model
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Streamlit app
st.title("Loan Qualification Checker")

# Input features
def user_input_features():
    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
    Education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
    Credit_History = st.selectbox("Credit History", ("Yes", "No"))
    Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

    Total_Income = ApplicantIncome + CoapplicantIncome
    data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'Total_Income': Total_Income,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode the input data to match the training data
input_df['Gender'] = LabelEncoder().fit_transform(input_df['Gender'])
input_df['Married'] = LabelEncoder().fit_transform(input_df['Married'])
input_df['Dependents'] = LabelEncoder().fit_transform(input_df['Dependents'])
input_df['Education'] = LabelEncoder().fit_transform(input_df['Education'])
input_df['Self_Employed'] = LabelEncoder().fit_transform(input_df['Self_Employed'])
input_df['Property_Area'] = LabelEncoder().fit_transform(input_df['Property_Area'])
input_df['Credit_History'] = input_df['Credit_History'].apply(lambda x: 1 if x == "Yes" else 0)

# Predict loan eligibility
prediction = model.predict(input_df)

st.subheader("Prediction")
st.write("You are eligible for a loan." if prediction[0] == 1 else "You are not eligible for a loan.")

# Display model performance (optional)
st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

