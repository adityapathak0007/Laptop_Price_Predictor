import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Add custom CSS for buttons
st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 8px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("💻 Laptop Price Predictor")
st.markdown("""
Welcome to the **Laptop Price Predictor**! This tool helps you estimate the price of a laptop based on various features.

Simply provide the details below, and click **Predict Price** to get an estimate.
""")

# Input fields with structured layout
with st.form("prediction_form"):
    st.header("Provide Laptop Specifications")

    # Brand selection
    company = st.selectbox('Select Laptop Brand', df['Company'].unique(), help="Choose the brand of the laptop.")

    # Type of laptop
    type = st.selectbox('Select Laptop Type', df['TypeName'].unique(), help="Choose the type or category of the laptop.")

    # RAM selection
    ram = st.selectbox('Select RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], help="Specify the amount of RAM.")

    # Weight input
    weight = st.number_input('Laptop Weight (in kg)', min_value=0.5, max_value=5.0, step=0.1, help="Enter the weight of the laptop.")

    # Touchscreen
    touchscreen = st.radio('Touchscreen Feature', ['No', 'Yes'], help="Does the laptop have a touchscreen?")

    # IPS display
    ips = st.radio('IPS Display', ['No', 'Yes'], help="Does the laptop have an IPS display?")

    # Screen size
    screen_size = st.number_input('Screen Size (in inches)', min_value=10.0, max_value=20.0, step=0.1, help="Enter the screen size in inches.")

    # Screen resolution
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'],
                               help="Choose the screen resolution.")

    # CPU brand
    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique(), help="Select the CPU brand.")

    # Hard disk capacity
    hdd = st.selectbox('HDD Capacity (in GB)', [0, 128, 256, 512, 1024, 2048], help="Specify the HDD storage capacity.")

    # SSD capacity
    ssd = st.selectbox('SSD Capacity (in GB)', [0, 8, 128, 256, 512, 1024], help="Specify the SSD storage capacity.")

    # GPU brand
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique(), help="Select the GPU brand.")

    # Operating System
    os = st.selectbox('Operating System', df['os'].unique(), help="Choose the operating system.")

    # Submit button
    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Convert user inputs into a query
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    try:
        # Calculate PPI (Pixels Per Inch)
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # Create query array
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        # Make prediction
        predicted_price = np.exp(pipe.predict(query)[0])  # Exponentiate if model uses log-transformed price

        # Display the result
        st.success(f"The estimated price of the laptop is: **₹{predicted_price:,.2f}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add footer
st.markdown("""
---
Developed using Streamlit.
""")
