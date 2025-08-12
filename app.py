import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## load the lstm model
model=load_model('next_word_lstm.keras')

## load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_next_word(model, tokenizer, text, maxSeqLen):
    tokenList=tokenizer.texts_to_sequences([text])[0]
    if len(tokenList)>= maxSeqLen:
        tokenList=tokenList[-maxSeqLen+1:]
    tokenList=pad_sequences([tokenList],maxlen=maxSeqLen-1,padding='pre')
    predicted=model.predict(tokenList,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)[0]
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

## streamlist app
st.title("Next word prediction with LSTM and early stopping")
input_text=st.text_input("Enter the sequence of words","Fran. For this releefe much thankes:")
if st.button("Predict next word"):
    maxSeqLen=model.input_shape[1]+1
    next_word=predict_next_word(model, tokenizer, input_text, maxSeqLen)
    st.write(f"Predicted next word: {next_word}")