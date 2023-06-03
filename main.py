import streamlit as st
import pandas as pd
import pickle
import numpy as np

# emotions
labeled_emotions = { 0:'anger',  1:'fear', 2:'joy', 3:'love', 4:'sadness',5:'surprise'}

# load model using pickle
model = pickle.load(open('emotion_predictor_model.pkl','rb'))

# title
st.title('Emotions Recognizer')
# till here

# message and its predicted label
msg = st.text_input(label = 'Message:', key = 'text')

if msg:
    prediction = model.predict([msg])
    emotion = labeled_emotions.get(prediction[0])
    st.write("""
             ### You're feeling: """+ emotion.title())
st.markdown('-' * 80)
# till here

# read and concat all the messages
df1 = pd.read_csv('train.txt',sep=';', header=None)
df2 = pd.read_csv('test.txt',sep=';', header=None)
df3 = pd.read_csv('val.txt',sep=';', header=None)

df = pd.concat([df1,df2,df3])
df = df.rename(columns={0:'message',1:'emotion'})
# till here

# what people have to say
random = np.random.randint(0, df.shape[0])

said = df.iloc[random, 0].title()
felt = df.iloc[random, 1].title()

st.write("""
         #### Someone said : 
         ##### {}
         #### We think they felt : 
         ##### {}""".format(said, felt))
st.markdown('-' * 80)
# till here

# Bar chart for overall emotions count
st.markdown("Emotions Felt Chart:  ")
st.bar_chart(df.emotion.value_counts(normalize=True), use_container_width=True, )
st.markdown('-' * 80)
# till here




