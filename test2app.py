# import
# import matplotlib.pyplot as plt
# # load the images into memory
# (trainX, trainy), (testX, testy) = load_data()
# # summarize the shape of the dataset
# # plot images from the training dataset
# for i in range(100):
# 	# define subplot
# 	plt.subplot(10, 10, 1 + i)
# 	# turn off axis
# 	plt.axis('off')
# 	# plot raw pixel data
# 	plt.imshow(trainX[i], cmap='gray_r')
# plt.show()

import streamlit as st
import time

options = ("male", "female")

a = st.empty()

value = a.radio("gender", options, 0)

st.write(value)

time.sleep(2)

value = a.radio("gender", options, 1)

st.write(value)
