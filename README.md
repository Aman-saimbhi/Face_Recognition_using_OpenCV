# Face_Recognition_using_OpenCV
A face recognition system which can be used as 1:1 face verification as well as 1:k face recognition problem.
Data directory is not added as the dataset which I used is personal.

You can make a directory named data/ and add the folders of the images of different persons. Given the images in the folders,
128 Dimensional feature vectors namely encodings of the image is calculated and is appended in the list of encodings and the 
name of the person is also stored in another list. Once all the images are done, pickle is used to store the encodings and 
the names so that they can be used elsewhere. This process can be done by running the face.py.

Face_predict.py : This file performs the predictions and draw a bounding box around the face as well as display the name
on the openCV frame. This script uses the encodings calculated previously. The encodings of the face in the frame are 
calculated and then the cosine or euclidean distance is calculated between this face and the stored encodings, if the 
distance is less than a threshold value then we say the faces are similar and hence the name is displayed.

cleaning.py : This file only has some pre-processing steps for the images in the directory.
