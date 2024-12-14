# Laptop Price Prediction Application 💻

This repository contains a Streamlit-based Laptop Price Prediction App. The app allows users to predict the price of a used laptop based on various input features such as the brand, model, year of purchase, RAM, weight, touchscreen feature, IPS display, screen size, screen resolution, CPU brand, HDD capacity, SSD capacity, GPU brand, and operating system.

## 🛠️ Features
- **Predict the price of a used laptop** based on:
  - Laptop brand (company)
  - Laptop type
  - RAM (in GB)
  - Weight (in kg)
  - Touchscreen feature (Yes/No)
  - IPS display (Yes/No)
  - Screen size (in inches)
  - Screen resolution
  - CPU brand
  - HDD capacity (in GB)
  - SSD capacity (in GB)
  - GPU brand
  - Operating System
- **Responsive web interface** built using Streamlit.
- **Real-time price predictions** based on user input.

## 📊 How It Works
The app collects user inputs and utilizes a linear regression model to predict laptop prices. The model was trained on a dataset containing features such as laptop brand, model, year, RAM, weight, touchscreen feature, IPS display, screen size, screen resolution, CPU brand, HDD capacity, SSD capacity, GPU brand, and operating system.

## 📄 Data
The dataset (`Cleaned Laptop.csv`) contains key columns:
- `name`: The name of the laptop model.
- `company`: The laptop manufacturer or brand.
- `TypeName`: The type or category of the laptop.
- `Ram`: The amount of RAM in the laptop.
- `Weight`: The weight of the laptop.
- `Touchscreen`: Whether the laptop has a touchscreen.
- `IPS`: Whether the laptop has an IPS display.
- `ScreenSize`: The screen size in inches.
- `Resolution`: The screen resolution.
- `Cpu brand`: The CPU brand.
- `HDD`: The HDD capacity in GB.
- `SSD`: The SSD capacity in GB.
- `Gpu brand`: The GPU brand.
- `os`: The operating system.
- `Price`: The actual selling price of the laptop (used for model training).

## 🛠️ Technologies Used
- **Python**: The core programming language.
- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For building the machine learning model.
- **Pickle**: For saving and loading the pre-trained model.

## 📁 Files
- `app.py`: The main Python script for the Streamlit web app.
- `laptop_data.csv`: The dataset used for project.
- `df.pkl`: The dataset used for predictions.
- `pipe.pkl`: The saved model used to predict laptop prices.
- `requirements.txt`: The list of Python packages required.
- `README.md`: Documentation file (this file).

## 🎯 Usage
1. **Select the laptop brand** (company) from the dropdown.
2. **Choose the laptop type** (e.g., Gaming, Ultrabook) from the dropdown.
3. **Enter the amount of RAM** (in GB).
4. **Input the weight of the laptop** (in kg).
5. **Indicate if the laptop has a touchscreen feature** (Yes/No).
6. **Indicate if the laptop has an IPS display** (Yes/No).
7. **Specify the screen size** (in inches).
8. **Select the screen resolution**.
9. **Choose the CPU brand**.
10. **Specify the HDD capacity** (in GB).
11. **Specify the SSD capacity** (in GB).
12. **Choose the GPU brand**.
13. **Select the operating system**.
14. Click the "Predict Price" button to get the estimated price of the laptop.
15. The predicted laptop price will be displayed on the sidebar.

## 📂 Files
- `df.pkl`: The cleaned dataset used to predict laptop prices. Ensure it is placed in the root directory for the app to access.
- `pipe.pkl`: This is the pre-trained machine learning model that will be used for predicting laptop prices based on user input. The model has been serialized using Pickle. Make sure the file is located in the root directory so the app can load it when running predictions.

### If you want to train the model yourself, refer to the code provided to train and save the model using Scikit-learn.

## 📋 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/adityapathak0007/Laptop_Price_Predictor.git

## View the App

You can view the live Laptop Price Prediction App built using Streamlit by clicking on the link below:

[View Laptop Price Prediction App built using Streamlit](https://laptoppricepredictor-byocjkyctngsve5znjp5nt.streamlit.app/)

## Contact

For any questions, suggestions, or feedback, please feel free to reach out:

- **Aditya Pathak** 👤
- **Email:** [adityapathak034@gmail.com](mailto:adityapathak034@gmail.com) 📧
- **GitHub:** [adityapathak0007](https://github.com/adityapathak0007) 🐙
- **LinkedIn:** [adityapathak07](https://www.linkedin.com/in/adityapathak07) 🔗
