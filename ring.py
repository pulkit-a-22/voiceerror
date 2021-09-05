import json
import getpass
from pathlib import Path
from pprint import pprint
import cv2
from ring_doorbell import Ring, Auth
from oauthlib.oauth2 import MissingTokenError
import time
import os
from os import system
import imutils
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
import mtcnn

# saving password and username for easier login
cache_file = Path("test_token.cache")

# loading prereqs for facial recognition + detection
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read(), encoding="latin1")
le = pickle.loads(open("output/le.pickle", "rb").read(), encoding="latin1")

def token_updated(token):
    cache_file.write_text(json.dumps(token))


def otp_callback():
    auth_code = input("2FA code: ")
    return auth_code

def say_text(text):
    system("say{}".format(text))


def main():

        #logging in
        if cache_file.is_file():
            auth = Auth("RingProject/1.234e", json.loads(cache_file.read_text()), token_updated)
        else:
            username = input("Username: ")
            password = getpass.getpass("Password: ")
            auth = Auth("RingProject/1.234e", None, token_updated)
            try:
                auth.fetch_token(username, password)
            except MissingTokenError:
                auth.fetch_token(username, password, otp_callback())
    
        ring = Ring(auth)
        ring.update_data()
        
        devices = ring.devices()

        id = -1
        current_id = None
        
        while True:
            try:
                ring.update_data()
            except:
                continue

            doorbell = devices['authorized_doorbots'][0]
            for event in doorbell.history(limit=20, kind='ding'):
                current_id = event['id']
                break

            if current_id != id:
                id = current_id
                handle = handle_video(ring)
                while handle == 0:
                    handle = handle_video(ring)


face_detector = mtcnn.MTCNN()
conf_t = 0.50

def detect_faces(cv2_img):
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)
    faces = []
    coords = []
    for res in results:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        confidence = res['confidence']
        if confidence < conf_t:
            continue
        faces.append(cv2_img[y1:y2, x1:x2])
        coords.append([x1, y1, x2, y2])
    return faces, coords


def get_specific_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(50):
        if (i%10 != 0):
            continue 
        success, image = vidcap.read()
        if success:
            frames.append(image)
    return frames

def handle_video(ring):
    devices = ring.devices()
    doorbell = devices['authorized_doorbots'][0]
    try:
        doorbell.recording_download(
            doorbell.history(limit=100, kind='ding')[0]['id'],
            filename='new_motion.mp4',
            override=True)
    except:
        print('unable to dowbload video')
        return 0
    
    frames = get_specific_frames('new_motion.mp4')

    count = [0] * len(le.classes_)
    for frame in frames:
        faces, coords = detect_faces(frame)
        if faces is None:
            continue
        else:
            for i in range(len(faces)):
                face = faces[i]
                coord = coords[i]           
           
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                
                text = None
                if (proba < 0.5):
                    name = "unknown"
                    text = "Person at Door is Unknown"
                    say_text(text)
                else:
                    name = le.classes_[j]
                    count[j] = count[j] + 1
                    if (count[j] == 1):
                        text = le.classes_[j] + "is here"
                        say_text(text)

                # draw the bounding box of the face along with the associated probability
                if (name == "unknown"):
                    text = "{}".format(name)
                else:
                    text = "{}: {:.2f}%".format(name, proba * 100)

                y = coord[1] - 10 if coord[1] - 10 > 10 else coord[1] + 10
                cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
                cv2.putText(frame, text, (coord[0], y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)#pauses for 2 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()
