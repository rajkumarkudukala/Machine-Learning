import streamlit as st
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#page config
st.set_page_config(" Multiple Linear Regression",layout="centered")

#Load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")

#Title
st.markdown("""
<div class="card">
    <h1> Multiple Linear Regression </h1>
    <p>Predict  <b> Tip Amount </b> from <b> Total Bill </b> and <b> Party Size </b> using Multiple Linear Regression...</p>    
</div>
""",unsafe_allow_html=True)

#Load data 
@st.cache_data
def load_data():
    df = sns.load_dataset('tips')
    return df   
df = load_data()

#Data preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" Dataset Preview ")
st.dataframe(df[['total_bill','size','tip']].head())
st.markdown('</div>',unsafe_allow_html=True)


#Prepare data
x,y = df[['total_bill','size']], df['tip']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Train model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

#Visualization
import matplotlib.pyplot as plt
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip (with multiple linear regression)")
fig, ax = plt.subplots()
ax.scatter(df['total_bill'], df['tip'], color='blue', label='Actual Tips')
ax.plot(df['total_bill'], model.predict(scaler.transform(x)), color='red', label='Predicted Tips')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip Amount')
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)


# m & c
st.markdown('<div class="card">'
            f'<h3> Model Interception </h3>'
            f'<p> Coefficients: {model.coef_} </p>'
            f'<p> co-efficient (Group size):</p> {model.coef_[1]} </p>'
            f'<p> Intercept: {model.intercept_} </p>'
            '</div>',unsafe_allow_html=True)

#predictions
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" Predict Tip amount ")
bill = st.slider(" Total Bill Amount ($) ", float(df['total_bill'].min()), float(df['total_bill'].max()), float(df['total_bill'].mean()),30.0)
size = st.slider(" Group size ",int(df['size'].min()), int(df['size'].max()), 2)
input_data = scaler.transform([[bill, size]])
predicted_tip = model.predict(input_data)[0]
st.markdown(f'<div class="prediction-box"> For a total bill of <b>${bill}</b> with a group size of <b>{size}</b>, the predicted tip amount is <b>${predicted_tip:.2f}</b> </div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)