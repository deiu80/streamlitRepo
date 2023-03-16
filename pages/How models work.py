import  streamlit as st

st.title('Deep learning in Computer Vision')



st.write("Deep learning has revolutionized computer vision, enabling computers to understand and interpret visual information with remarkable accuracy."
         "\nDeep learning algorithms rely on neural networks, which are designed to learn and recognize patterns in data. With computer vision, deep learning can be used to detect and classify objects in images and video, recognize faces, track motion, and even generate realistic images and videos."
        )

st.write("\nDeep learning has made it possible to build more intelligent and sophisticated computer vision systems, with applications ranging from self-driving cars to medical diagnosis to security and surveillance.")
st.write(" As a result, deep learning has become an essential tool for researchers and practitioners in computer vision,"
         " and it is likely to continue driving advances in this field for years to come.")

st.subheader("Convolutional Neural Networks")
st.image('Plots/architecture-cnn-en.jpeg')

st.write("Play the below animation. It is showing how CNNs layers work when given an image (of a digit in this case). ")
st.write('With each layer, convolution operations are performed on reduced dimension of the image')
video_file = open('Plots/cnn_video.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)
st.subheader('Support Vector Machines')
