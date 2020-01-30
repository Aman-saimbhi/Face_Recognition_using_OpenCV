import cv2
import os
import dlib
import face_recognition
import pickle

#ImageNames = os.listdir("/Users/aman/Face_recognition/data/Aman")
DirNames = os.listdir("/Users/aman/Face_recognition/data/")
#basedir = "/Users/aman/Face_recognition/data/Aman/"
basedir = "/Users/aman/Face_recognition/data/"
Knownencodings = list()
Knownnames = list()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for dirname in DirNames:
    if dirname == '.DS_Store':
        continue
    imagenames = os.listdir(basedir + dirname)
    print(imagenames)
    imagepaths = []
    for imagename in imagenames:
        imagepaths.append(os.path.join(basedir, dirname, imagename))
    #print(images)
    for imagepath in imagepaths:
        #name = image.split('_')[0]
        name = dirname       # the name of the person will be same as directory
        #print(imagepath)
        print(imagepath)
        bgr = cv2.imread(imagepath)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(bgr, 1.3, 5)
        encodings = face_recognition.face_encodings(rgb, faces)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            Knownencodings.append(encoding)
            Knownnames.append(name)

    # dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": Knownencodings, "names": Knownnames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
