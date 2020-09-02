import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

from face_detection import select_face
from face_swap import face_swap
from PIL import Image


import cv2
import argparse
# sys.path.append('tl_gan')
# sys.path.append('pg_gan')
# import feature_axis
# import tfutil
# import tfutil_cpu

# This should not be hashed by Streamlit when using st.cache.
TL_GAN_HASH_FUNCS = {
    tf.Session : id
}

def main():

    #Upload images
    uploaded_file = st.file_uploader("Choose a picture", type=['jpg', 'png'])
    # if uploaded_file is not None:
    #     st.image(uploaded_file, width=200)
    second_uploaded_file = st.file_uploader("Choose another picture", type=['jpg', 'png'])
    # if second_uploaded_file is not None:
    #     st.image(second_uploaded_file, width=200)
    try:
        image1 = Image.open(uploaded_file)
        image2 = Image.open(second_uploaded_file)

        image1_arr = np.array(image1)
        image2_arr = np.array(image2)

        print(image1_arr.shape)
        print(image2_arr.shape)


        show_file = st.empty()
        show_file1 = st.empty()
        show_file2 = st.empty()
        show_file3 = st.empty()
        show_file4 = st.empty()
        show_file5 = st.empty()

        if not uploaded_file:
            show_file.info('Please upload a file')
            return
        show_file.title('Input Images')
        show_file1.image(uploaded_file, width=100)
        show_file2.title('+')
        show_file3.image(second_uploaded_file, width=100)
        show_file4.title('=')
        # show_file5.image(image1, width=300)

        # Read images to opencv
        # src_img = cv2.imencode('jpg', image1)
        # dst_img = cv2.imencode('jpg',image2)
    except:
        show_file = st.empty()
        show_file.info('Please upload a file')

    src_img = cv2.imencode('jpg', image1_arr)
    dst_img = cv2.imencode('jpg', image2_arr)

if __name__ == "__main__":
    main()
