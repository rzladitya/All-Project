import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import pickle
import joblib
import re
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- SET PAGES ---
st.set_page_config(page_title="APP | StockMorrow", page_icon=":label:", layout="wide")

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 1

def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Projects"],  # required
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Projects"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

# -- Upload Lottiefiles --
def load_lottie(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

lottie_lokal = load_lottie("src/stock.json")

if selected == "Home":
    def main(): 
        st.markdown('''
        <div style="background-color: #1DCAA3; padding: 8px; margin-bottom: 1rem;border: 1px solid; border-radius: 9px;">
        <h1 style="color: #232142; text-align:center;"> StockMorrow</h1>
        </div>
        ''', unsafe_allow_html=True)
    
        col1, col2 = st.columns(2)
        with col1:
            st_lottie(lottie_lokal, speed=1, reverse=False, loop=True, quality='low', height=310, width=310, key=None)
        with col2:
            st.write("")
            st.write("")
            st.subheader(":globe_with_meridians: StockMorrow")
            st.write("")
            st.write("""
            StockMorrow is a website that predicts market movement by analyzing daily headline news all over the world. 
            We’re dedicated to create not only the best but the easiest platform for stock traders  to use. 
            We in StockMorrow want to provide additional insight to help you decide on which stock you want to invest. 
            By using  Machine Learning  method  to analyze and calculate the correlation between world headline news and stock price movement.
            """)
        st.markdown('---')
        lottie_online = load_lottie("src/market.json")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('')
            st.subheader("What StockMorrow has to offer :question::question:")
            st.markdown('')
            st.write("StockMorrow offers 5 stocks worth investing tomorrow based on headlines news.")
            st.write('')
            st.write("")
            st.write(":one: Market Movement Prediction Based on headline news")
            st.write(":two: Accurate Movement Prediction")
            st.write(":three: Stock Suggestion")
        with col2:
            st_lottie(lottie_online, speed=1, reverse=False, loop=True, quality='low', height=310, width=310, key=None)
    
        st.markdown("---")

    if __name__ == '__main__':
        main()

# Load model and vectorizer
with open("nb_model.pkl", "rb") as model_file:
    nb_model = pickle.load(model_file)
# Load Vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Load model stock predict
with open("preprocessor.pkl", "rb") as file:
    stock_model = pickle.load(file)

if selected == "Projects":
    # Set Title
    c = st.container()
    c.title(":bar_chart: StockMorrow")
    st.subheader('Predict tomorrow’s movement of 5 worth investing Stock from S&P 500 using Headline News Sentiment Analysis.')

    # Set Text box
    txt = st.text_area('Input Headline News', '  ')

    # Create Fuction
    def stock_prediction(sample_news):
        sample_news = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_news)
        sample_news = sample_news.lower()
        sample_news_words = sample_news.split()
        sample_news_words = [word for word in sample_news_words if not word in set(stopwords.words('english'))]
        ps = PorterStemmer()
        final_news = [ps.stem(word) for word in sample_news_words]
        final_news = ' '.join(final_news)

        temp = vectorizer.transform([final_news]).toarray()
        return nb_model.predict(temp)

    if st.button("Analysis"):
        with st.spinner(text='Analyzing...'):
            time.sleep(5)
        st.markdown(f'News: {txt}')
        if stock_prediction(txt):
            st.error('Prediction: The S&P 500 Index will remain the same or will go down.')
        else:
            st.success('Prediction: The S&P 500 Index will go up!')

        html_temp = """
        <div style="background-color: #46D9B8  ;padding:6px; border: 1px solid; border-radius: 9px; margin-bottom:8px;">
        <h4 style="color:black;text-align:center;">Stocks Recommendation</h4>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        stock = ['CPRT', 'MCD', 'MNST', 'NVDA', 'WST']
        res = stock_model.predict([txt])
        data = pd.DataFrame(data=res, columns= stock)
        datas = data[data>=0]
        datas2 = data[data<=0]
        datas3 = datas2.dropna(axis=1)
        datas1 = datas.dropna(axis=1)
        col1, col2 = st.columns(2)
        with col1:
            st.write('Stock price will go up!')
            st.table(datas1)
        with col2:
            st.write('Stock price will go down!')
            st.table(datas3)

        st.write('''
        Note : The stock price changes by percentage.
        ''')