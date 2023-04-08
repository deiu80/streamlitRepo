import streamlit as st
import cv2
from helpers import image_resizer


@st.cache_data
def load_images():
    example = cv2.imread('Plots/example_image.jpg')
    mono = cv2.imread('Plots/monochrome_image.jpg')
    equalized = cv2.imread('Plots/equalized_image.jpg')
    denoised = cv2.imread('Plots/denoised_image.jpg')
    filtered = cv2.imread('Plots/filtered_img.jpg')
    svm_image = cv2.imread("Plots/svm_multiclass.png")
    hog_image = cv2.imread("Plots/HOG_scikit-image_AngelaMerkel.jpeg")
    hog_image = cv2.cvtColor(hog_image, cv2.COLOR_BGR2RGB)
    conf_matrix = cv2.cvtColor(cv2.imread('Plots/65acc_only_faces.png'), cv2.COLOR_BGR2RGB)

    return example, mono, equalized, denoised, filtered, svm_image, hog_image, conf_matrix


example, mono, equalized, denoised, filtered, svm_image, hog_image, conf_matrix = load_images()

st.title('Deep learning in Computer Vision')

st.write(
    "Deep learning has revolutionized computer vision, enabling computers to understand and interpret visual information with remarkable accuracy."
    "\nDeep learning algorithms rely on neural networks, which are designed to learn and recognize patterns in data. With computer vision, deep learning can be used to detect and classify objects in images and video, recognize faces, track motion, and even generate realistic images and videos."
)

st.write(
    "\nDeep learning has made it possible to build more intelligent and sophisticated computer vision systems, with applications ranging from self-driving cars to medical diagnosis to security and surveillance.")
st.write(
    " As a result, deep learning has become an essential tool for researchers and practitioners in computer vision,"
    " and it is likely to continue driving advances in this field for years to come.")

st.subheader("Convolutional Neural Networks")
st.image('Plots/architecture-cnn-en.jpeg')

st.write("Play the below animation. It is showing how CNNs layers work when given an image (of a digit in this case). ")
st.write('With each layer, convolution operations are performed on reduced dimension of the image')
video_file = open('Plots/cnn_video.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

st.subheader('Support Vector Machines')
st.markdown(
    "SVMs are considered to be a traditional machine learning technique, as they have been in use since the 1990s."
    " They are based on the concept of finding a hyperplane that separates data points into different classes. "
    "The hyperplane is chosen in such a way that it maximizes the margin between the two classes, i.e., the distance between the hyperplane and the closest data points of each class.")
st.image(svm_image)
st.markdown(
    "In computer vision tasks, features such as Histogram of Oriented Gradients (HOG) are commonly used to extract important information from images, which are then fed into the SVM algorithm for training and classification."
    " HOG is particularly useful in SVM-based object detection tasks because it can capture the shape and edges of objects in an image, which allows for robust classification even in the presence of partial occlusion or changes in lighting conditions.")
st.image(hog_image)

st.write(
    "For training the SVM i have used a custom built dataset called Only_facesFER."
    "This is a subset of the original FER2013 dataset, containing around 17 000 images of size 150px. These have been extracted using  face detection algorithm HaarCascades from OpenCV , and processed using the following pipeline:",

    "\n- rgb image is converted to monochrome"
    "\n- enhance the contrast using histogram equalization from OpenCV"
    "\n- remoive the noise and blur the image"
    "\n- sharpen the edges"
    "\n\nCheck the below example image its transformations, following the above steps of processing:")

col1, col2, col3, col4, col5 = st.columns(5)

col1.image(example)
col1.text('1 grayscaled')
col2.image(mono)
col3.image(equalized)
col3.text("2 equalized")
col4.image(denoised)
col4.text("3 denoised")
col5.image(filtered)
col5.text("4 edge-sharpened")

st.subheader('Evalutation for the used model')
col1, col2 = st.columns(2)
col1.image('Plots/svm_class.png')
col2.write('Confusion matrix')
col2.image('Plots/65acc_only_faces.png')
st.write('Finally, we pass the image to the model and we receive an output looking like this')
st.code( {
  "angry": 3.6081,
  "fear": 0.78539,
  "happy": 88.64721,
  "neutral": 3.69275,
  "sadness": 0.70454,
  "surprise": 2.56201
})
st.write("containing values representing the predicted probabilities of different emotions")