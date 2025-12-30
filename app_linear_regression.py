import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
import os
os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config("Linear Regression ", layout = "centered")
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")

#Title
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h1> Simple Linear Regression </h1>
    <p>Predict  <b> Tip Amount </b> from <b> Total Bill </b> using Simple Linear Regression...</p>
</div>
""",unsafe_allow_html=True)

@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df = load_data()
st.markdown('<div class = "card">', unsafe_allow_html = True)
st.subheader('Dataset Preview')
st.dataframe(df.head())
st.markdown('<div>', unsafe_allow_html=True)

#Data preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" Dataset Preview ")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

#Prepare data
x,y = df[['total_bill']], df['tip']
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
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" Total bill vs Tip Amount ")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(df['total_bill'], df['tip'], color='blue', label='Data points',alpha=0.5)
ax.plot(df['total_bill'], model.predict(scaler.transform(df[['total_bill']])), color='red', label='Regression line')
ax.set_xlabel('Total Bill ($)')
ax.set_ylabel('Tip Amount ($)')
ax.set_title('Total Bill vs Tip Amount')
ax.legend()
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#Performance
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" About Model Performance ")
c1,c2,c3,c4=st.columns(4)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("R²",f"{r2:.2f}")
c3.metric("MSE",f"{mse:.2f}")
c4.metric("RMSE",f"{rmse:.2f}")
st.markdown('</div>',unsafe_allow_html=True)