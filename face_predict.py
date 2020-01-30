import cv2
import os
import dlib
import face_recognition
import pickle


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
while True:
    check, frame = video.read()
    #print(check)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgb = cv2.resize(rgb, (int(rgb.shape[1]/2), int(rgb.shape[0]/2)))
    faces = face_cascade.detectMultiScale(rgb, 1.3, 5)
    #print(faces)
    encodings =face_recognition.face_encodings(rgb, faces)
    names = []
    #print(encodings)
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown"
    if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
    names.append(name)



    for ((sx, sy, sw, sh), name) in zip(faces, names):
        cv2.rectangle(frame, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 0), 2)
        y = sy - 15 if sy - 15 > 15 else sy + 15
        cv2.putText(frame, name, (sx, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

video.release()
cv2.destroyALLWindows()
