import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import lightgbm as lgb

st.title('アヤメの花の判別')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    input = df.values
    output = model.predict(input)
    prediction_prob_df = pd.DataFrame(output, columns=['setosa', 'versicolor', 'virginica'])

    st.subheader('予測結果')
    st.write('各種類である確率')
    st.write(prediction_prob_df)

    fig, ax = plt.subplots()
    ax.pie(x=prediction_prob_df.iloc[0].values,
            labels=['setosa', 'versicolor', 'virginica'],
            colors=['r', 'g', 'b'],
            autopct='%1.1f%%',
            labeldistance=1.2)
    st.pyplot(fig)
    