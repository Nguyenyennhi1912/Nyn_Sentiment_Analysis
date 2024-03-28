import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pyvi import ViTokenizer, ViPosTagger
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from datetime import datetime
from sklearn.model_selection import *
from xu_ly_tieng_viet import *
from select_id import *

# GUI
st.title("Data Science Project")
st.write("## Sentiment Analysis")
menu = ["Home", "Result Model", "Predict Comment", "Recommend based on IDRestaurant"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Github](https://github.com/Nguyenyennhi1912?tab=repositories)")

#elif choice == 'Result Model':



elif choice == 'Predict Comment':    
    st.subheader("Predict Comment")
    # Cho người dùng chọn nhập dữ liệu hoặc upload file
    type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu vào text area", "Nhập nhiều dòng dữ liệu trực tiếp", "Upload file"])
    # Nếu người dùng chọn nhập dữ liệu vào text area
    if type == "Nhập dữ liệu vào text area":
        st.subheader("Nhập dữ liệu vào text area")
        content = st.text_area("Nhập ý kiến:")
        comment = optimized_process_text(content)
        # Nếu người dùng nhập dữ liệu đưa content này vào thành 1 dòng trong DataFrame
        with open(r'C:\Users\Admin\Desktop\Project Python\GUI\pipeline_SVC_1.pkl', 'rb') as file:  
            model = pickle.load(file)
        comment_pred = model.predict([comment])
        # st.code("New predictions: " + str(comment_pred))
            
# Nếu người dùng chọn nhập nhiều dòng dữ liệu trực tiếp vào một table
    elif type == "Nhập nhiều dòng dữ liệu trực tiếp":
        st.subheader("Nhập nhiều dòng dữ liệu trực tiếp")        
        df = pd.DataFrame(columns=["Comment"])
        for i in range(3):
            df = df.append({"Comment": st.text_area(f"Nhập ý kiến {i+1}:")}, ignore_index=True)
        df.dropna(inplace=True)
        df['Cleaned'] = df['Comment'].apply(lambda x: optimized_process_text(x))        
        with open(r'C:\Users\Admin\Desktop\Project Python\GUI\pipeline_SVC_1.pkl', 'rb') as file:  
            model = pickle.load(file)
        df['Prediction'] = df['Cleaned'].apply(lambda x: model.predict([x]))  
        # st.dataframe(df[["Comment","Prediction"]])

# Nếu người dùng chọn upload file
    elif type == "Upload file":
        st.subheader("Upload file")
        # Upload file
        uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "txt"])
        if uploaded_file is not None:
            # Đọc file dữ liệu
            df = pd.read_csv(uploaded_file)
            st.write(df)
#######        
    # Từ df này, người dùng có thể thực hiện các xử lý dữ liệu khác nhau
    submitted_project1 = st.button("Submit")
    if submitted_project1:
        st.write("Hiển thị kết quả dự đoán cảm xúc...")
        if type == "Nhập dữ liệu vào text area":
            st.code("New predictions: " + str(comment_pred))
        elif type == "Nhập nhiều dòng dữ liệu trực tiếp":
            st.dataframe(df[["Comment","Prediction"]])
        # elif type == "Upload file":
            # st.write(df)

elif choice == 'Recommend based on IDRestaurant':
    st.subheader("Recommend based on IDRestaurant")
    data = pd.read_csv(r"C:\Users\Admin\Desktop\Project Python\Project1\Reviews_concat.csv")
    # Cho người dùng chọn nhập ID nhà hàng và xem thông tin nhà hàng đó
    type = st.selectbox("What's IDRestaurant?", list(data['IDRestaurant'].unique()))
    visual = st.radio("Do you want to visualize the data?", ("Yes", "No"))
 
    # Từ df này, người dùng có thể thực hiện các xử lý dữ liệu khác nhau
    submitted_project2 = st.button("Submit")
    if submitted_project2:
        if visual == "No": 
            # Nếu người dùng chọn nhập dữ liệu vào text area
            number = int(type)
            st.write("Restaurant information")
            st.code('Restaurant name: '+ data[data['IDRestaurant'] == number]['Restaurant'].iloc[0])
            st.code('Address: '+ data[data['IDRestaurant'] == number]['Address'].iloc[0])
            st.code('Operate time: '+ data[data['IDRestaurant'] == number]['Time'].iloc[0])
            st.code('Price range: '+ data[data['IDRestaurant'] == number]['Price'].iloc[0])
            st.code('Average Rating: '+ str(round(data['Rating'].mean(), 2)))
            st.code('Number of reviews: '+ str(data[data['IDRestaurant'] == number]['ID'].count()))
        if visual == "Yes":
            # Nếu người dùng chọn nhập dữ liệu vào text area
            number = int(type)
            st.write("Restaurant information")
            st.code('Restaurant name: '+ data[data['IDRestaurant'] == number]['Restaurant'].iloc[0])
            st.code('Address: '+ data[data['IDRestaurant'] == number]['Address'].iloc[0])
            st.code('Operate time: '+ data[data['IDRestaurant'] == number]['Time'].iloc[0])
            st.code('Price range: '+ data[data['IDRestaurant'] == number]['Price'].iloc[0])
            st.code('Average Rating: '+ str(round(data['Rating'].mean(), 2)))
            st.code('Number of reviews: '+ str(data[data['IDRestaurant'] == number]['ID'].count()))
            
            visual_data(type)

               
    
    
       
