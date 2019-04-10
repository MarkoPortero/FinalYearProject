import csv
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import tensorflow as tf
import numpy as np
import cv2


RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

TEST_IMAGES_DIR = os.getcwd() + "/test_images"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

filename = "C:/Users/BASEDMARK/Documents/photo1.jpg"


def UploadAction():
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    path = filename

    # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = Image.open(path)
    img = img.resize((800, 500), Image.ANTIALIAS)
    imgtemp = ImageTk.PhotoImage(img)
    image.configure(image=imgtemp)
    image.img = imgtemp
    classifications = []

    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    # end for

    print("classifications = " + str(classifications))

    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph, note that we don't need to be concerned with the return value
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    with tf.Session() as sess:
        # show the file name on std out
        print(filename)

        # get the file name and full path of the current image file
        imageFileWithPath = os.path.join(filename)
        # attempt to open the image with OpenCV
        openCVImage = cv2.imread(imageFileWithPath)

        # if we were not able to successfully open the image, continue with the next iteration of the for loop
        if openCVImage is None:
            print("unable to open " + filename + " as an OpenCV image")
            # end if

        # get the final tensor from the graph
        finalTensor = sess.graph.get_tensor_by_name('final_result:0')

        # convert the OpenCV image (numpy array) to a TensorFlow image
        tfImage = np.array(openCVImage)[:, :, 0:3]

        # run the network to get the predictions
        # DecodeJpeg is the name of the tensor that is being fed the images
        predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

        # sort predictions from most confidence to least confidence
        sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print("---------------------------------------")

        # keep track of if we're going through the next for loop for the first time so we can show more info about
        # the first prediction, which is the most likely prediction (they were sorted descending above)
        onMostLikelyPrediction = True
        # for each prediction . . .
        for prediction in sortedPredictions:
            strClassification = classifications[prediction]

            # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
            if strClassification.endswith("s"):
                strClassification = strClassification[:-1]
                # end if

            # get confidence, then get confidence rounded to 2 places after the decimal
            confidence = predictions[0][prediction]

            # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
            if onMostLikelyPrediction:
                # get the score as a %
                scoreAsAPercent = confidence * 100.0
                # show the result to std out
                # labelPrediction = Label(main_window, text="the object appears to be a " + strClassification + ", " + "{0:.2f}".format(
                #   scoreAsAPercent) + "% confidence")

                labelPrediction.config(text="the object appears to be a " + strClassification + ", " + "{0:.2f}".format(
                    scoreAsAPercent) + "% confidence")
                # labelPrediction.pack()
                print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(
                    scoreAsAPercent) + "% confidence")

                # mark that we've show the most likely prediction at this point so the additional information in
                # this if statement does not show again for this image
                onMostLikelyPrediction = False
                # end if

            # for any prediction, show the confidence as a ratio to five decimal places
            labelPredictionPercentage.config(text=strClassification + " (" + "{0:.5f}".format(confidence) + ")")
            # labelPredictionPercentage = Label(main_window, text=strClassification + " (" + "{0:.5f}".format(confidence) + ")")
            # labelPredictionPercentage.pack()

            print(strClassification + " (" + "{0:.5f}".format(confidence) + ")")
            # end for
        # end for
    # end with

    # write the graph to file so we can view with TensorBoard

    return


def DirectoryOptions():
    file_classifications = {}
    selectedDirectory = filedialog.askdirectory()
    # get a list of classifications from the labels file
    classifications = []
    # for each line in the label file . . .
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    # end for

    # show the classifications to prove out that we were able to read the label file successfully
    print("classifications = " + str(classifications))

    # load the graph from file
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph, note that we don't need to be concerned with the return value
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    with tf.Session() as sess:
        fileName_List = []
        Classification_List = []
        fileHeader = "File Name"
        classificationHeader = "Classification"
        fileName_List.append(fileHeader)
        Classification_List.append(classificationHeader)
        # for each file in the test images directory . . .
        for fileName in os.listdir(selectedDirectory):
            # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            # end if

            # show the file name on std out
            print(fileName)

            # get the file name and full path of the current image file
            imageFileWithPath = os.path.join(selectedDirectory, fileName)
            # attempt to open the image with OpenCV
            openCVImage = cv2.imread(imageFileWithPath)

            # if we were not able to successfully open the image, continue with the next iteration of the for loop
            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue
            # end if

            # get the final tensor from the graph
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # convert the OpenCV image (numpy array) to a TensorFlow image
            tfImage = np.array(openCVImage)[:, :, 0:3]

            # run the network to get the predictions
            # DecodeJpeg is the name of the tensor that is being fed the images
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # sort predictions from most confidence to least confidence
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            print("---------------------------------------")

            # keep track of if we're going through the next for loop for the first time so we can show more info about
            # the first prediction, which is the most likely prediction (they were sorted descending above)
            onMostLikelyPrediction = True

            fileName_List.append(fileName)
            Classification_List.append(classification)
            # for each prediction . . .
            for prediction in sortedPredictions:

                strClassification = classifications[prediction]
                # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]
                # end if

                # get confidence, then get confidence rounded to 2 places after the decimal
                confidence = predictions[0][prediction]

                # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
                if onMostLikelyPrediction:
                    # get the score as a %
                    scoreAsAPercent = confidence * 100.0
                    # show the result to std out
                    print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(
                        scoreAsAPercent) + "% confidence")
                    # mark that we've show the most likely prediction at this point so the additional information in
                    # this if statement does not show again for this image
                    onMostLikelyPrediction = False
                # end if
                # for any prediction, show the confidence as a ratio to five decimal places
                print(strClassification + " (" + "{0:.5f}".format(confidence) + ")")
            # end for
        # end for

        with open('Classifications.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(zip(fileName_List, Classification_List))
        messagebox.showinfo("Successful", "Your classifications have been written to file.")


# GUI AREA
main_window = Tk()
main_window.geometry("800x800")

label_1 = Label(main_window, text="Choose a file:")
label_1.pack()

btn_choose = Button(main_window, text="Choose a File", command=UploadAction).pack()
btn_choose = Button(main_window, text="Choose a directory", command=DirectoryOptions).pack()
# Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img = ImageTk.PhotoImage(Image.open(filename))

# The Label widget is a standard Tkinter widget used to display a text or image on the screen.
image = Label(main_window, image=img)
image.pack()

labelPrediction = Label(main_window, text="")
labelPrediction.pack()

labelPredictionPercentage = Label(main_window, text="")
labelPredictionPercentage.pack()

main_window.mainloop()
