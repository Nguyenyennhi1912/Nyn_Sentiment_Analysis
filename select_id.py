import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
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
import pickle

data = pd.read_csv(r"C:\Users\Admin\Desktop\Project Python\Project1\Reviews_concat.csv")

def enter_your_idrestaurant(id):
    # Nếu người dùng chọn nhập dữ liệu vào text area
    number = int(id)
    st.write("Restaurant information")
    st.code('Restaurant name: '+ data[data['IDRestaurant'] == number]['Restaurant'].iloc[0])
    st.code('Address: '+ data[data['IDRestaurant'] == number]['Address'].iloc[0])
    st.code('Operate time: '+ data[data['IDRestaurant'] == number]['Time'].iloc[0])
    st.code('Price range: '+ data[data['IDRestaurant'] == number]['Price'].iloc[0])
    st.code('Average Rating: '+ str(round(data['Rating'].mean(), 2)))
    st.code('Number of reviews: '+ str(data[data['IDRestaurant'] == number]['ID'].count()))


def visual_data(id):
    def find_words(document, list_of_words):
        document_lower = document.lower()
        word_count = 0
        word_list = []
        for word in list_of_words:
            if word in document_lower:
                word_count += document_lower.count(word)
                word_list.append(word)
        return word_list
    positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn", "ngon",
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh","sạch",
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng","không bị","lạ miệng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "đậm đà",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo","đa dạng",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm","quá tuyệt","đúng vị",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận", "khen",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền","nhiệt tình",
    "nghiện","nhanh","ngon nhất","quá ngon","điểm cộng","niềm nở","ok"
    ]
    negative_words = [
    "kém", "tệ", "đau", "xấu", "dở", "ức","tức",
    "buồn", "rối", "thô", "lâu", "chán",
    "tối", "chán", "ít", "mờ", "mỏng", "mắc",
    "lỏng lẻo", "khó", "cùi", "yếu","hôi","sống",
    "kém chất lượng", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp","không ngon",
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp","nhạt nhẽo",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập","không thể chấp nhận",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng","không thoải mái",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp","tanh","khủng khiếp","thất vọng",
    "trộm cướp","cau có","điểm trừ","đợi lâu","không đặc sắc","không đặc biệt"
    ]
    
    
    data['positive_words'] = data[data['IDRestaurant'] == id]['Cleaned'].dropna().apply(lambda x: find_words(x, positive_words))
    data['negative_words'] = data[data['IDRestaurant'] == id]['Cleaned'].dropna().apply(lambda x: find_words(x, negative_words))
    
    def listToString(s):
 
        # initialize an empty string
        str1 = " "
 
        # traverse in the string
        for ele in s:
            str1 = str1 +" "+ ele
 
        # return string
        return str1


    def get_top_words(word_list, n=30):
        all_words = [word for lst in word_list if isinstance(lst, list) for word in lst]
        word_counter = Counter(all_words)
        return [word for word, _ in word_counter.most_common(n)]

    top_positive_words = get_top_words(data['positive_words'])
    top_negative_words = get_top_words(data['negative_words'])
                                         
    #print('Top negative words:', top_negative_words)
    #print('Top positive words:', top_positive_words)



    # Plot Word cloud
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    # Negative Wordcloud
    text_negative = listToString(top_negative_words)
    wordcloud_negative = WordCloud(background_color='white', width=400, height=400, max_words=50).generate(text_negative)
    axes[0].imshow(wordcloud_negative, interpolation='bilinear')
    axes[0].set_title('Negative Wordcloud')
    axes[0].axis('off')
    
    # Positive Wordcloud
    text_positive = listToString(top_positive_words)
    wordcloud_positive = WordCloud(background_color='white', width=400, height=400, max_words=50).generate(text_positive)
    axes[1].imshow(wordcloud_positive, interpolation='bilinear')
    axes[1].set_title('Positive Wordcloud')
    axes[1].axis('off')
    plt.show()

    return st.pyplot(fig)
