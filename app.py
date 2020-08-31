import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

# sys.path.append('tl_gan')
# sys.path.append('pg_gan')
# import feature_axis
# import tfutil
# import tfutil_cpu


uploaded_file = st.file_uploader("Choose a picture", type=['jpg', 'png'])
if uploaded_file is not None:
    st.image(uploaded_file)
