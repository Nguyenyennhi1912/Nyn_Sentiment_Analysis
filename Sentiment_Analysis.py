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
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from datetime import datetime
from sklearn.model_selection import *
from xu_ly_tieng_viet import *
from select_id import *

# GUI
st.title("DATA SCIENCE PROJECT")
st.write("## Project 1: SENTIMENT ANALYSIS")
menu = ["ABOUT PROJECT", "PREDICT COMMENT", "RECOMMEND ON ID"]
choice = st.sidebar.selectbox('CONTENTS', menu)
if choice == 'ABOUT PROJECT':
    st.subheader("Business Objective", divider='rainbow')
    st.write(""" ###### Sentiment Analysis (Phân tích tình cảm/cảm xúc) là một trong những cách sử dụng ngôn ngữ tự nhiên để nhận diện và nghiên cứu trạng thái cảm xúc và thông tin chủ quan một cách có hệ thống. Sentiment Analysis là quá trình phân tích, đánh giá quan điểm (tích cực, trung tính, tiêu cực,...) của 1 đối tượng bằng việc sử dụng các thuật toán của Machine Learning """) 
    st.write(""" ###### Sentiment Analysis thông qua đánh giá và nhận xét của khách hàng khi tham gia trãi nghiệm dịch vụ có vai trò quan trọng trong việc quảng bá kinh doanh của doanh nghiệp. Phân tích được cảm xúc của khách hàng về sản phẩm/dịch vụ là tích cực hay tiêu cực giúp doanh nghiệp tổng quan được tình hình hoạt động, đề ra các chiến lược để quảng bá sản phẩm.""")
    st.write("###### => Mục tiêu: Sử dụng các thuật toán Machine Learning trong Python để thực hiện Sentiment Analysis")
    st.image(r"sentiment.jpg")

    st.subheader("EDA", divider='rainbow')
    
    # Tạo khung subplot 
    st.write("##### 1. Visualize Rating Distribution")
    data = pd.read_csv(r"Reviews_concat.csv") 
    fig1 = sns.histplot(data=data, x='Rating').set_title('Rating Distribution')
    st.pyplot(fig1.figure)
    st.write("###### Thang điểm Rating của về dịch vụ của các nhà hàng dựa trên thang đo 10. Các đánh giá của khách hàng chiếm phần lớn ở mức từ 7 điểm trở lên")
    st.write("###### => Đề xuất phân loại cảm xúc: Positive (Rating > 8), Neutral (Rating > 6), Negative")

    ## Preprocessing
    st.write("##### 2. Preprocessing")
    a = pd.read_csv(r'Sentiment_by_Rating.csv')
    st.dataframe(a[["Rating","Comment","Sentiment"]].head(10))
    fig = sns.countplot(data=a,
              x='Sentiment',
              palette="mako",
              order = a['Sentiment'].value_counts().index)

    st.pyplot(fig.figure)
    st.write("###### Nhận xét: Sự mất cân bằng dữ liệu là không đáng kể khi phân loại theo thang đó đề xuất.")
    st.write("###### Tuy nhiên, đánh giá của khách hàng theo Rating không khớp với lời nhận xét của họ. Bởi vì có thể khách hàng bị hiểu nhầm Rating đang được đo trên thang điểm 5. Vì vậy nếu dùng Rating để phân tích sẽ cho kết quả không hiệu quả.") 
    st.write("###### => Dùng nhận xét (Comment) của khách hàng để phân tích")

    st.write("##### 3. Build Model")
    reviews = pd.read_csv(r'Reviews_after_EDA_1.csv')
    y_test = pd.read_csv(r"y_test.csv")
    y_pred_SVC = pd.read_csv(r"y_pred_SVC.csv")
    y_pred_SVC = y_pred_SVC.to_numpy()
    
    st.code(confusion_matrix(y_test, y_pred_SVC))
    st.code(classification_report(y_test, y_pred_SVC))

elif choice == 'PREDICT COMMENT':    
    st.subheader("Predict Comment")
    # Cho người dùng chọn nhập dữ liệu hoặc upload file
    type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu vào text area", "Nhập nhiều dòng dữ liệu trực tiếp"])
    # Nếu người dùng chọn nhập dữ liệu vào text area
    if type == "Nhập dữ liệu vào text area":
        st.subheader("Nhập dữ liệu vào text area")
        content = st.text_area("Nhập ý kiến:")
        comment = optimized_process_text(content)
        # Nếu người dùng nhập dữ liệu đưa content này vào thành 1 dòng trong DataFrame
        with open(r'pipeline_SVC_1.pkl', 'rb') as file:  
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
        with open(r'pipeline_SVC_1.pkl', 'rb') as file:  
            model = pickle.load(file)
        df['Prediction'] = df['Cleaned'].apply(lambda x: model.predict([x]))  
        # st.dataframe(df[["Comment","Prediction"]])

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

elif choice == 'RECOMMEND ON ID':
    st.subheader("Recommend on IDRestaurant")
    data = pd.read_csv(r"Reviews_concat.csv")
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

               
    
    
       
