# app.py
import streamlit as st
import torch
from models import tokenize, pad_sequence, SentimentClassifier, vocab

@st.cache_resource
def load_model():
    model = SentimentClassifier()
    model.load_state_dict(torch.load("sentiment_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("Social Media Sentiment Analysis 🧠")
text = st.text_area("Enter text:", "This movie was amazing!")

if st.button("Analyze"):
    token_ids = tokenize(text)
    padded_ids = pad_sequence(token_ids)
    input_tensor = torch.tensor(padded_ids).unsqueeze(0)
    
    # Create mask (ignore padding)
    mask = (input_tensor != vocab["<pad>"]).float()
    
    with torch.no_grad():
        logits = model(input_tensor, mask)
    pred = torch.argmax(logits).item()
    
    st.write("## Result")
    st.success("Positive 😊" if pred == 1 else "Negative 😞")