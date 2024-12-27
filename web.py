import streamlit as st
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.losses import MeanSquaredError

# Title of the web app
st.title('Dự đoán giá kim cương')

# Reset session state
def reset_inputs():
    st.session_state.clear()  # Xóa toàn bộ session_state
    st.rerun()  # Làm mới ứng dụng để reset widget

# Initialize session state if not already initialized
if 'carat' not in st.session_state:
    st.session_state['carat'] = 0.0
    st.session_state['cut'] = 'Fair'
    st.session_state['color'] = 'D'
    st.session_state['clarity'] = 'I1'
    st.session_state['depth'] = 0.0
    st.session_state['table'] = 0.0
    st.session_state['x'] = 0.0
    st.session_state['y'] = 0.0
    st.session_state['z'] = 0.0
    st.session_state['model_type'] = 'Linear Regression'

# Sidebar for inputs
st.sidebar.header('Nhập thông tin kim cương:')
carat = st.sidebar.number_input('Carat (trọng lượng)', min_value=0.0, step=0.01, format="%.2f", key='carat')
cut = st.sidebar.selectbox('Cut (độ cắt)', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], key='cut')
color = st.sidebar.selectbox('Color (màu)', ['D', 'E', 'F', 'G', 'H', 'I', 'J'], key='color')
clarity = st.sidebar.selectbox('Clarity (độ tinh khiết)', ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'], key='clarity')
depth = st.sidebar.number_input('Depth (%)', min_value=0.0, max_value=100.0, step=0.1, format="%.1f", key='depth')
table = st.sidebar.number_input('Table (%)', min_value=0.0, max_value=100.0, step=0.1, format="%.1f", key='table')
x = st.sidebar.number_input('X (chiều dài, mm)', min_value=0.0, step=0.01, format="%.2f", key='x')
y = st.sidebar.number_input('Y (chiều rộng, mm)', min_value=0.0, step=0.01, format="%.2f", key='y')
z = st.sidebar.number_input('Z (chiều cao, mm)', min_value=0.0, step=0.01, format="%.2f", key='z')

# Dropdown to select model
model_type = st.sidebar.selectbox(
    'Chọn mô hình dự báo',
    ['Linear Regression', 'Linear Regression từ Weka'],
    key='model_type'
)

# Load the appropriate model based on user selection
if model_type == 'Linear Regression':
    model_path = 'D:\BigDataAnalysis\model\linear_regression_python_model_final.pkl'
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
elif model_type == 'Linear Regression từ Weka':
    model_path = 'D:\BigDataAnalysis\model\linear_regression_weka_final.h5'
    model = load_model(model_path, custom_objects={'MeanSquaredError': MeanSquaredError})
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

# Tạo DataFrame từ đầu vào
input_data = {
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
}
df = pd.DataFrame(input_data)

# Khởi tạo LabelEncoder cho các cột phân loại
label_encoders = {
    'cut': LabelEncoder(),
    'color': LabelEncoder(),
    'clarity': LabelEncoder()
}

# Fit LabelEncoder và mã hóa các cột phân loại
df['cut'] = label_encoders['cut'].fit_transform(df['cut'])
df['color'] = label_encoders['color'].fit_transform(df['color'])
df['clarity'] = label_encoders['clarity'].fit_transform(df['clarity'])

# Áp dụng MinMaxScaler cho các cột liên tục
scaler_path = 'D:\BigDataAnalysis\scaler\MinMaxscaler.pkl'
scaler_loaded = joblib.load(scaler_path)
columns_to_scale = ['carat', 'depth', 'table', 'x', 'y', 'z']
df[columns_to_scale] = scaler_loaded.transform(df[columns_to_scale])

# Prepare the input features for prediction
input_features = df.values

# Predict button
if st.sidebar.button('Dự đoán giá'):
    try:
        if carat == 0:
            prediction = 0
        else:
            # Dự đoán
            if model_type == 'Linear Regression':
                prediction = model.predict(input_features)[0]
            elif model_type == 'Linear Regression từ Weka':
                prediction = model.predict(input_features)[0]
            # Chuyển đổi giá trị dự đoán thành số thực
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()
        st.write(f'### Giá dự đoán: ${prediction:,.2f}')
    except Exception as e:
        st.error(f'Đã xảy ra lỗi: {e}')

# Reset button
if st.sidebar.button('Reset'):
    reset_inputs()
    st.write("### Đã reset dữ liệu.")
