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
menu = ["Business Objective", "Result Model", "Predict Comment", "Recommend based on IDRestaurant"]
choice = st.sidebar.selectbox('### Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
             #### Sentiment Analysis through reviews and comments on products and services is one of the common natural language processing tasks in business. Natural language processing and machine learning can help classify customer emotions accurately.""") 
    st.write("#### => Problem/ Requirement: Use Machine Learning algorithms in Python for Sentiment analysis.")
    st.image(r"GUI\sentiment.jpg")

if choice == 'Result Model':
    st.subheader("Result Model")
    reviews = pd.read_csv(r'C:\Users\Admin\Desktop\Project Python\GUI\Cung_cap_HV_ShopeeFood\2_Reviews.csv')
    # Tạo khung subplot 
    st.write("##### 1. Visualize Rating Distribution")
    fig = sns.histplot(data=reviews, x='Rating').set_title('Rating Distribution')
    st.pyplot(fig.figure)
    st.write("Đánh giá của khách hàng về nhà hàng chiếm phần lớn ở mức từ 7 điểm trở lên")
    st.write("=> Đề xuấT phân loại khách hàng: Positive (Rating > 8), Neutral (Rating > 6), Negative")

    ## Preprocessing
    st.write("##### 2. Preprocessing")
    a = pd.read_csv(r'Project1\Sentiment_by_Rating.csv')
    st.dataframe(a[["Rating","Comment","Sentiment"]].head(10))
    fig = sns.countplot(data=a,
              x='Sentiment',
              palette="mako",
              order = a['Sentiment'].value_counts().index)

    st.pyplot(fig.figure)
    st.write("Nhận xét: Sự mất cân bằng dữ liệu là không đáng kể khi phân loại theo thang đó đề xuất.")
    st.write("Tuy nhiên, đánh giá của khách hàng theo Rating không khớp với lời nhận xét của họ. Bởi vì có thể khách hàng bị hiểu nhầm Rating đang được đo trên thang điểm 5. Vì vậy nếu dùng Rating để phân tích sẽ cho kết quả không hiệu quả.") 
    st.write("=> Dùng nhận xét (Comment) của khách hàng để phân tích")

    st.write("##### 3. Build Model")
    reviews = pd.read_csv(r'GUI\Reviews_after_EDA_1.csv')
    y_test = pd.read_csv(r"GUI\y_test.csv")
    y_pred_SVC = pd.read_csv(r"GUI\y_pred_SVC.csv")
    y_pred_SVC = y_pred_SVC.to_numpy()
    
    st.code(confusion_matrix(y_test, y_pred_SVC))
    st.code(classification_report(y_test, y_pred_SVC))

elif choice == 'Predict Comment':    
    st.subheader("Predict Comment")
    # Cho người dùng chọn nhập dữ liệu hoặc upload file
    type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu vào text area", "Nhập nhiều dòng dữ liệu trực tiếp"])
    # Nếu người dùng chọn nhập dữ liệu vào text area
    if type == "Nhập dữ liệu vào text area":
        st.subheader("Nhập dữ liệu vào text area")
        content = st.text_area("Nhập ý kiến:")
        comment = optimized_process_text(content)
        # Nếu người dùng nhập dữ liệu đưa content này vào thành 1 dòng trong DataFrame
        with open(r'GUI\pipeline_SVC_1.pkl', 'rb') as file:  
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
        with open(r'GUI\pipeline_SVC_1.pkl', 'rb') as file:  
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

elif choice == 'Recommend based on IDRestaurant':
    st.subheader("Recommend based on IDRestaurant")
    data = pd.read_csv(r"Project1\Reviews_concat.csv")
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

               
    
    
       