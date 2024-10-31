import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Headings
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# Prepare the data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to get user input data
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(user_report_data, index=[0])

# Get user data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Random Forest model training
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predictions = rf.predict(x_test)
rf_probs = rf.predict_proba(user_data)[:, 1]
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Logistic Regression model training
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predictions = lr.predict(x_test)
lr_probs = lr.predict_proba(user_data)[:, 1]
lr_accuracy = accuracy_score(y_test, lr_predictions)

# Model Selection
model_selection = st.sidebar.selectbox("Select Model for Prediction", ["Random Forest", "Logistic Regression"])

# Display prediction results based on selected model
if model_selection == "Random Forest":
    st.subheader('Random Forest Prediction')
    prediction = rf.predict(user_data)[0]
    probability = rf_probs[0]
    st.write(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    st.write(f"Probability of Diabetes: {probability:.2f}")
    st.write(f"Accuracy: {rf_accuracy:.2f}")

elif model_selection == "Logistic Regression":
    st.subheader('Logistic Regression Prediction')
    prediction = lr.predict(user_data)[0]
    probability = lr_probs[0]
    st.write(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    st.write(f"Probability of Diabetes: {probability:.2f}")
    st.write(f"Accuracy: {lr_accuracy:.2f}")

# Feature Importance
importances = rf.feature_importances_
features = x.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
st.subheader('Feature Importance')
st.bar_chart(importance_df.set_index('Feature'))

# Plotting the distribution of Glucose levels
st.subheader('Glucose Level Distribution')
plt.figure(figsize=(10, 5))
sns.histplot(df['Glucose'], kde=True, color='blue')
plt.title('Distribution of Glucose Levels')
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
st.pyplot(plt)

# Plotting a correlation heatmap
st.subheader('Feature Correlation Heatmap')
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
st.pyplot(plt)

# Plotting pairplot
st.subheader('Pairplot of Features')
sns.pairplot(df, hue='Outcome')
st.pyplot(plt)

# User feedback section
st.sidebar.subheader('User Feedback')
feedback = st.sidebar.text_area("Please provide your feedback or suggestions here:")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.write("Thank you for your feedback!")
