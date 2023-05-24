import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import cv2 as cv
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	pixels = plt.imread(filename)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
 
# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	faces = [extract_face(f) for f in filenames]
	samples = asarray(faces, 'float32')
	samples = preprocess_input(samples, version=2)
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	return model.predict(samples)
 
# determine if the two faces match
# get the cosine distance between them
def match_embeddings(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding) 
	if score <= thresh:
		return True
	else:
		return False

st.write("""
# Face Verification App
This app takes in two images and tries to determine if both images are the same person.
""")

st.write("""## Choose image input method""")
options = ['Camera', 'Upload', 'Both']

option = st.radio('Method', options, index=1)

st.sidebar.header('User Input')

# initial value for img
img = []

# Collect image from the user
uploaded_file_1 = st.sidebar.file_uploader("Upload your first image in .jpg format", type=["jpg"], key='firstFile')

if option == 'Upload':
    uploaded_file_2 = st.sidebar.file_uploader("Upload your second image in .jpg format", type=["jpg"], key='secondFile')

    st.write("""## Prediction""")
    
    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        img.append(uploaded_file_1)
        img.append(uploaded_file_2)

        try:
            prediction = get_embeddings(img)
            if match_embeddings(prediction[0], prediction[1]):
                st.write('This photo is of the same person')
            else:
                st.write('This photo is not of the same person')

        except IndexError:
             st.write('Couldn\'t detect face in one of the images. Are the faces in the images clear?')

elif option == 'Camera':
    image_1 = st.camera_input('Capture first Image', key='FirstCamera', 
                            help="""This is a basic camera that takes a photo. 
                                    Don\'t forget to allow access in order for the app to be able to use the devices camera.""")
    image_2 = st.camera_input('Capture second Image', key='SecondCamera', 
                            help="""This is a basic camera that takes a photo.. 
                                    Don\'t forget to allow access in order for the app to be able to use the devices camera.""")
    
    st.write("""## Prediction""")

    if image_1 is not None and image_2 is not None:
        img.append(image_1)
        img.append(image_2)

        try:
            prediction = get_embeddings(img)
            if match_embeddings(prediction[0], prediction[1]):
                st.write('This photo is of the same person')
            else:
                st.write('This photo is not of the same person')

        except IndexError:
            st.write('Couldn\'t detect face in one of the images. Are the faces in the images clear?')
            
elif option == 'Both':
    image_3 = st.camera_input('Capture Image', key='ThirdCamera', 
                            help="""This is a basic camera that takes a photo.. 
                                    Don\'t forget to allow access in order for the app to be able to use the devices camera.""")
    
    st.write("""## Prediction""")
    
    if uploaded_file_1 is not None and image_3 is not None:
        img.append(uploaded_file_1)
        img.append(image_3)
        
        try:
            prediction = get_embeddings(img)
            if match_embeddings(prediction[0], prediction[1]):
                st.write('This photo is of the same person')
            else:
                st.write('This photo is not of the same person')

        except IndexError:
            st.write('Couldn\'t detect face in one of the images. Are the faces in the images clear?')

else:
    img = cv.imread("face.jpg")
    st.image(img, channels='BGR', caption='Image')
