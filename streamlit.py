#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pickle
import re
from time import time
from turtle import width
from click import File
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import docx
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from prettytable import PrettyTable
from PIL import Image
import numpy as np


# In[21]:


image = Image.open('cactus_logo.png')

################ all libraries imported

st.image(image, width = 200)
st.write(""" 
# Welcome to Document Classifier """)

#### subheader
st.subheader("Upload a file")
docx_file = st.file_uploader("Upload Document", type=["docx","pdf"],accept_multiple_files = True)


######## text cleaning done ###########


# In[31]:


def clean_text(text):
    '''
    for removing special characters, lowering the case and removing 
    punctuations
    
    params:
        text: str - text string of the document to be cleaned 
    returns:
        text_list : list of cleaned text that has length one. 
                    tfidf_vectoriser accepts only list of strings
    '''
    pattern = r'[^A-Za-z ]+'
    clean_text = [re.sub(pattern,'', str(x.lower())) for x in text.split(" ")]
    text_list = " ".join(clean_text)
    #if you are sending this to the model,

    return text_list


# In[23]:


# In[24]:


def convert_multilabel(y_pred):
    
    '''
    It takes multilable prediction and converts it into a single Label
    '''
    
    labels = {('non-rp', 'non-rtrc', 'abstract'): 'Abstract', 
    ('non-rp', 'non-rtrc', 'non-abstract'): 'Other', 
    ('non-rp', 'rtrc', 'non-abstract'): 'Response to reviewer comments', 
    ('rp', 'non-rtrc', 'non-abstract'): 'Research paper_Journal article'}
    labels_ = ["Abstract", "Other", "Response to reviewer comments", "Research paper_Journal article"]

    y_pred_simple = []
    
    for i in np.array(y_pred):
        y_pred_simple.append(labels[tuple(i)])
    
    return y_pred_simple


# In[28]:


### importing the pickled models ###########
base_path = './'
model_path = pickle.load(open(base_path + "estimator_v6_rfc_bucket.sav",'rb'))
vectoriser_path = pickle.load(open(base_path + "tfidf_v6_bucket.sav",'rb'))

x = PrettyTable()
x.field_names = ["File","MVP","MVP+Others"]

def prediction_others(clean_text,vectorizer,model):
  pred_start_time = time.time()
  
  print("****** here ******")
  vector = vectorizer.transform([clean_text])
  print("****** crosses ******")
  pred = model.predict(vector)
  pred_prob = model.predict_proba(vector)
  pred_end_time = time.time()
  total_pred_time = round(pred_end_time-pred_start_time,2)
  return pred,pred_prob,total_pred_time
  
def simplify_probability(pred_proba):
	'''
	model.predict_proba will give a 3x1x2 array of probabilities. Each of the these three 2-D vectors are 
	probabilties for [non-rp,rp],[non-rtrc, rtrc],[abstract, non-abstract] labels. 
	For eg
	[
	  "array([[0.92, 0.08]])",
	  "array([[0.95, 0.05]])",
	  "array([[0.5, 0.5]])"
	]
	means probability of document being a non-rp is 0.92 or it being a research paper is 0.08
	Similary, probabiity for it being abstract is 0.5
	and probability for it being rtrc document is 0.05
	'''
	simple_prob = np.prod(np.max(np.array(pred_prob).squeeze(1), axis = 1))
	return simple_prob
# In[29]:


if st.button("Predict Me !"):

    for file in docx_file:
        #split = file.name.split(".")
        
        #pdf_path = path + split[0]+".pdf"
        st.write("""\n\n\n\n\n\n""")
        st.write("""**File Name:** """,file.name)
        #displayPDF(pdf_path)
        t = time.time()
        d = docx.Document(file)
        doc_clean_text = ''
        all_para = d.paragraphs
        clean_time_start = time.time()
        for i in all_para:
            a = clean_text(i.text.strip())
            doc_clean_text+=a+" "
        clean_time_end = time.time()
        print(doc_clean_text)

        pred, pred_prob, pred_time = prediction_others(doc_clean_text,vectoriser_path,model_path)
        #st.write(["Probability of it being a Research Paper:", pred_prob])
        y_pred = convert_multilabel(pred)
        x = PrettyTable()
        x.field_names = ["Metrics","MVP + Others"]
        x.add_row(["Probability(RP)",pred_prob[0][0][1]])
        x.add_row(["Probability(RTRC)",pred_prob[1][0][1]])
        x.add_row(["Probability(Abstract)",pred_prob[2][0][0]])
        x.add_row(["Class Prediction",y_pred[0]])
        simple_prob = simplify_probability(pred_prob)
        x.add_row(["Overall Probability (RP x RTRC x Abstract)",simple_prob])
        x.add_row(["Time(secs)",pred_time])
        st.write("""\n\n""")
        st.write("""\t\t\t""",x)
        st.write("""\n""")

