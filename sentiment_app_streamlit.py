import streamlit as st
from tensorflow.keras.models import load_model
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# import numpy as np
from sentence_transformers import SentenceTransformer
import os
from nltk.corpus import WordNetCorpusReader, wordnet


wn = WordNetLemmatizer()
stopwords_En = stopwords.words('english')
model_ber = SentenceTransformer('all-MiniLM-L6-v2')
stopwords_En.remove('no')
stopwords_En.remove('not')

@st.cache_resource
def load_sentiment_model():
    model_path = 'Amazon_review_sent_analysis model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path)
    return model


# def clean_text(text):
#     text = "".join([word.lower() for word in text if word not in string.punctuation])
#     tokens = word_tokenize(text)
#     text = " ".join([wn.lemmatize(word) for word in tokens if word not in stopwords_En])   
#     return text




def preprocess(df):
    # Clean the text
    df['cleaned_Text'] = df['text'].apply(lambda x: clean_text(x)) 
    print(df['cleaned_Text'])
    print("Done Cleaning")
    
    text_embedding = model_ber.encode(df['cleaned_Text'])
    print("Done Embedding")
    df['embedding'] = list(text_embedding)
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x).astype(float) if isinstance(x, list) else x)
    print("Done Embedding listing ")
    return df
    

def predict_review_label(df):

    cleaned_review = clean_text(df)
    review_embedding = model_ber.encode([cleaned_review])
    model=load_sentiment_model()
    prediction_prob = model.predict(review_embedding)
    # predicted_label = (prediction_prob > 0.5).astype(int)
    if prediction_prob[0] >0.65:
        sentiment_label = "Positive "
    elif prediction_prob[0] >0.4:    
        sentiment_label = "Negative "  
    
    else:
        sentiment_label = "Natural "     
        
    return sentiment_label,  float(prediction_prob[0][0])


# functions for cleaning text 
def remove_punct(text):
    return "".join([char for char in text if char not in string.punctuation])

# Define the function to tokenize text
def tokenize(text):
    return word_tokenize(text)

# Define the function to remove stopwords
def remove_stopwords(tokenized_list):
    return [word for word in tokenized_list if word not in stopwords_En]


def clean_text(text):
    
    text = remove_punct(text.lower())

    tokens = tokenize(text)
    
    tokens = remove_stopwords(tokens)
    
    text = " ".join([wn.lemmatize(word) for word in tokens])
    
    return text

def main():
    st.image("image_app.png")
    # st.image("img.png")


    st.title("Review Sentiment Analysis App")
    
    review = st.text_input("Enter a review:", "")
    
    # st.write(f"Current working directory: {os.getcwd()}")
    
    if st.button("Analyze Review"):
        if review:                    
            label, prob = predict_review_label(review)          
            # st.write( label ,prob)
            # st.subheader( "This Review is ")
            if label=='Positive ':
                st.subheader("This Review is Positive 😊")

            elif label=='Negative ':
                st.subheader("This Review is Negative 😔")    
            else:
                st.subheader("This Review is Natural 🙂")       
               

            # st.write( label )
            # st.subheader( "with Probability equal ")
            st.write(f"with Probability equal to  {prob:.3f}")
            
        else:
            st.write("Please enter a review for analysis.")
   

if __name__=='__main__':
    main()

