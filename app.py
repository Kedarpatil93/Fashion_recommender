import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input,ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3
from numpy.linalg import norm 
from sklearn.neighbors import NearestNeighbors



feature_list = np.array(pickle.load(open('all_embeddings_efficient_max.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

#model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = EfficientNetB3(weights='imagenet', include_top=False, pooling='max')


#st.title('Fashion Recommender System')
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE; font-size: 38px;'>🛍️ Fashion Recommender System</h1>", unsafe_allow_html=True)
st.markdown(" Upload an image of a clothing item to get similar recommendations:")
#file upload ==> save 

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0    

#uploaded_file = st.file_uploader("Upload an Image")
uploaded_file = st.file_uploader("")

def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)    
    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)
    neighbors.fit(feature_list)
    distances,indices = neighbors.kneighbors([features])
    return indices


# Display Image 
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
    # display the file 
        st.markdown("##### Uploaded Image:")
        display_image = Image.open(uploaded_file)
        st.image(Image.open(uploaded_file), width=150)
        #st.image(display_image)
    #extract features 
        features = feature_extraction(os.path.join('uploads',uploaded_file.name),model)
        #st.text(features)
        #recommendation
        indices = recommend(features,feature_list)
        #st.text(indices)
        # show recommendations

        st.markdown("##### 🔍 Recommended Similar Items:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            img = Image.open(filenames[indices[0][i]])
            col.image(img.resize((125, 125)))
    else:
        st.header("Error occured in file upload")




