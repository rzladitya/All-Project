import streamlit as st
import requests

# --- SET PAGES ---
st.set_page_config(page_title="Web Application", page_icon=":football:", layout="wide")

st.title(":football: Aplikasi Pengecekan Draft NFL")
left_column, right_column = st.columns(2)
with left_column:
    age = st.number_input("Age")
    height = st.number_input("Height")
    verticaljump = st.number_input("Vertical Jump")
with right_column:
    school = st.selectbox("School", ['Alabama', 'LSU', 'Ohio St', 'USC', 'Florida', 'Oklahoma', 'Georgia', 'Clemson', 'Miami', 'Illinois', 'California', 'Virginia Tech', 'Boston'])
    weight = st.number_input("Weight")
    benchpressreps = st.number_input("Maximum bench press repetitions achieved")

# inference
data = {'age':age,
        'school': school,
        'height':height,
        'weight':weight,
        'verticaljump':verticaljump,
        'benchpressreps':benchpressreps}

URL = "https://backend-ml2-deployment.herokuapp.com/predict"

# komunikasi
r = requests.post(URL, json=data)
res = r.json()
if res['code'] == 200:
    st.write("")
    if st.button("Predict"):
        st.text("Apakah pemain akan direkrut selama NFL Draft berlangsung?")
        st.success(res['data']['result']['target_names'])
else:
    st.write("Mohon maaf terjadi kesalahan")
    st.write(f"keterangan : {res['summary']}")





