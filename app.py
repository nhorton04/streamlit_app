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
    try:
        #Upload images
        uploaded_file = st.file_uploader("Choose a picture", type=['jpg', 'png'])
        if uploaded_file is not None:
            st.image(uploaded_file, width=200)
        second_uploaded_file = st.file_uploader("Choose another picture", type=['jpg', 'png'])
        if second_uploaded_file is not None:
            st.image(second_uploaded_file, width=200)

        # Read images to opencv
        # src_img = cv2.imread(uploaded_file)
        # dst_img = cv2.imread(second_uploaded_file)
        image1 = Image.open(uploaded_file)
        image2 = Image.open(second_uploaded_file)

        image1_arr = np.array(image1)
        image2_arr = np.array(image2)

        # Select src face
        src_points, src_shape, src_face = select_face(image1_arr)
        src_points1, src_shape1, src_face1 = select_face(image2_arr)
        # Select dst face
        dst_points, dst_shape, dst_face = select_face(image2_arr)
        dst_points1, dst_shape1, dst_face1 = select_face(image1_arr)

        if src_points is None or dst_points is None:
            print('No Face Detected')
            exit(-1)
        output = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, image2_arr)

        output2 = face_swap(src_face1, dst_face1, src_points1, dst_points1, dst_shape1, image1_arr)


    except:
        st.title('No Images Uploaded Yet')

    if st.button('Swap Faces ⬇️'):
        st.image(output)

    if st.button('Swap Faces ⬆️'):
        st.image(output2)

# @st.cache(allow_output_mutation=True, hash_funcs=TL_GAN_HASH_FUNCS)
# def load_model():
#     """
#     Create the tensorflow session.
#     """
#     # Open a new TensorFlow session.
#     config = tf.ConfigProto(allow_soft_placement=True)
#     session = tf.Session(config=config)
#
#     # Must have a default TensorFlow session established in order to initialize the GAN.
#     with session.as_default():
#         # Read in either the GPU or the CPU version of the GAN
#         with open(MODEL_FILE_GPU if USE_GPU else MODEL_FILE_CPU, 'rb') as f:
#             G = pickle.load(f)
#     return session, G

if __name__ == "__main__":
    main()
