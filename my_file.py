#Import libraries
import streamlit as st
import numpy as np
import cv2
from  PIL import Image, ImageEnhance 

#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.2])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
    


#Add a header and expander in side bar
st.sidebar.markdown('<p class="font">Turn your photo to amazing anime shots appplication</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
        Use this simple app to convert your favorite photo to a drawing anime cartoon.  \n  \nThis app was created by Khang Nguyen as a final project for FPT Machine Learning course. Hope you enjoy!
     """) 

#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg']) 

#Add 'before' and 'after' columns
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
    st.image(image, width= 500)  
    st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True) 

# # Add conditional statements to take the user input values
#     filter = st.sidebar.radio('Covert your photo to:', ['Original','Gray Image','Black and White', 'Pencil Sketch', 'Blur Effect'])
#     if filter == 'Gray Image':
#             converted_img = np.array(image.convert('RGB'))
#             gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
#             st.image(gray_scale, width=300)
#     elif filter == 'Black and White':
#             converted_img = np.array(image.convert('RGB'))
#             gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
#             slider = st.sidebar.slider('Adjust the intensity', 1, 255, 127, step=1)
#             (thresh, blackAndWhiteImage) = cv2.threshold(gray_scale, slider, 255, cv2.THRESH_BINARY)
#             st.image(blackAndWhiteImage, width=300)
#     elif filter == 'Pencil Sketch':
#             converted_img = np.array(image.convert('RGB')) 
#             gray_scale = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
#             inv_gray = 255 - gray_scale
#             slider = st.sidebar.slider('Adjust the intensity', 25, 255, 125, step=2)
#             blur_image = cv2.GaussianBlur(inv_gray, (slider,slider), 0, 0)
#             sketch = cv2.divide(gray_scale, 255 - blur_image, scale=256)
#             st.image(sketch, width=300) 
#     elif filter == 'Blur Effect':
#             converted_img = np.array(image.convert('RGB'))
#             slider = st.sidebar.slider('Adjust the intensity', 5, 81, 33, step=2)
#             converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2BGR)
#             blur_image = cv2.GaussianBlur(converted_img, (slider,slider), 0, 0)
#             st.image(blur_image, channels='BGR', width=300) 
#     else: 
#             st.image(image, width=300) 

    #Turn the photo into anime pic
    converted_img = np.array(image.convert('RGB'))
    data = np.float32(converted_img).reshape((-1, 3))
    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Applying cv2.kmeans function
    k_values = st.sidebar.slider('Adjust the K-values in K-means', 4, 15, 9, step=1)
    _, label, center = cv2.kmeans(data, k_values, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    # Reshape the output data to the size of input image
    result = center[label.flatten()]
    result = result.reshape(converted_img.shape)
    # Convert the input image to gray scale
    gray = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
    # Perform adaptive threshold
    if k_values is not None:
      slider = st.sidebar.slider('Adjust the edges intensity', 5, 15, 9, step=2)
      edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, slider, 8)
      # Smooth the result
      blurred = cv2.medianBlur(result, 3)
      # Combine the result and edges to get final cartoon effect
      cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
      # Show the results
      st.image(cartoon)

#Add a feedback section in the sidebar
st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.subheader('Please help us improve!')
with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
    text=st.text_input(label='Please leave your feedback here')
    submitted = st.form_submit_button('Submit')
    if submitted:
      st.write('Thanks for your feedback!')
      st.markdown('Your Rating:')
      st.markdown(rating)
      st.markdown('Your Feedback:')
      st.markdown(text) 
