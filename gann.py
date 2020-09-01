import numpy as np
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

from face_detection import select_face
from face_swap import face_swap
from PIL import Image
from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from torchvision.utils import save_image
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
        show_file5.image(image1, width=300)
        # Read images to opencv
        # src_img = cv2.imencode('jpg', image1)
        # dst_img = cv2.imencode('jpg',image2)
    except:
        show_file = st.empty()
        show_file.info('Please upload a file')

    os.makedirs("images/outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = style_transform()

    # Define model and load model checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load('/home/nick/Downloads/faceswap_app/models/starry_night_10000.pth'))
    transformer.eval()

    # Prepare input
    image_tensor = Variable(transform(Image.open(uploaded_file))).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()

    # Save image
    fn = args.image_path.split("/")[-1]
    save_image(stylized_image, f"images/outputs/stylized-{fn}")
if __name__ == "__main__":
    main()
