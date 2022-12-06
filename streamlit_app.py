import streamlit as st
import pandas as pd
from keras.models import load_model
import tensorflow as tf
import csv


def downsample_rows(happy_limit, neutral_limit, filename):
    input = open(filename, 'r')
    result_filename = filename.replace(".csv", "_downsampl.csv")
    output = open(result_filename, 'w')

    writer = csv.writer(output)
    counter = 1
    neutralcounter = 1
    for row in csv.reader(input):
        if row[1] == 'happy' and counter <= happy_limit:
            counter += 1
            output.write(row[0])
            output.write(",")
            output.write(row[1])
            output.write("\n")
        elif row[1] == 'neutral' and neutralcounter <= neutral_limit:
            neutralcounter += 1
            output.write(row[0])
            output.write(",")
            output.write(row[1])
            output.write("\n")
        elif row[1] != 'happy' and row[1] != 'neutral':
            output.write(row[0])
            output.write(",")
            output.write(row[1])
            output.write("\n")

    input.close()
    output.close()


def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")


def page2():
    st.markdown("# Real Time️")
    st.sidebar.markdown("# Real time feed")


data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

img_width = 48
img_height = 48
batch_size = 64

downsample_rows(2800, 2800, "./full_test_dataset.csv")
test_df = pd.read_csv("./full_test_dataset_downsampl.csv")
test_df = test_df.loc[test_df["class_names"] != "disgust"]
test_gen = data_gen.flow_from_dataframe(
    test_df,
    batch_size=batch_size,
    x_col='img_path',
    y_col='class_names',
    target_size=(img_width, img_height),
    shuffle=False,
    color_mode="grayscale",
    class_mode="categorical"
)

loaded_model = load_model('./best_model_optimised_cnn.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# preds = loaded_model.predict(test_gen,verbose=1)
# pred_argmaxed = preds.argmax(axis=1)Q
# from sklearn.metrics import confusion_matrix
# conf_matrix_log = confusion_matrix(test_gen.labels, pred_argmaxed)
#
# from sklearn.metrics import ConfusionMatrixDisplay
#
# labels = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
#
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_log,
#                               display_labels=labels)
#
# disp.plot(cmap='PuRd', xticks_rotation=35)
# st.title("CE301 - Capstone Project")
#
#
#
#
# from sklearn.metrics import classification_report
#
# print(classification_report(test_gen.labels, pred_argmaxed, target_names=labels))
