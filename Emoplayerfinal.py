import cv2               ## Image handling
import numpy as np
from keras.models import model_from_json  ##loading JSON model 
#import play_music_pygame
from pygame import mixer
from collections import Counter
import os
import random
import time

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
predictions=[]


def calc_freq(L):
    most_common, num_most_common = Counter(L).most_common(1)[0]  # 4, 6 times
    return most_common

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json) # Loading emotion model using import function

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5") # trained features
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()     # Reading the frames 
    frame = cv2.resize(frame, (850, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # Initializing to find face
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it, loop through ecah rectangle using its coordinates
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        print(maxindex)
        predictions.append(maxindex)
        if len(predictions)== 15:
            #os.system('cls')
            print("\nDone")
            print(emotion_dict)
            print("predictions = ",predictions)
            predicted=int(calc_freq(predictions))
            print("predicted freq= ",predicted)
            print()
            print("I think you are ", emotion_dict[predicted],"\n")
        
            mixer.init()
            if emotion_dict[predicted] == 'Fearful':
##                random.choice(os.listdir("E:\\EMOPLAYER_folder_music\\Emoplayer\\Emoplayer\\sad\\*.mp3"))
##                import os
##                import random 
                path="C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\Fearfull"
                files=os.listdir(path)
                d=random.choice(files)
                os.startfile("C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\Fearfull\\" + d)
##                mixer.music.load(d)
            if emotion_dict[predicted] == 'Happy':
                path="C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\Happy"
                files=os.listdir(path)
                d=random.choice(files)
                os.startfile("C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\Happy\\" + d)
##                mixer.music.load("song.mp3")
            if emotion_dict[predicted] == 'Sad':
                path="C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\sad"
                files=os.listdir(path)
                d=random.choice(files)
                os.startfile("C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\sad\\" + d)
##                mixer.music.load("song.mp3")
            if emotion_dict[predicted] == 'Surprised':
                path="C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\surprised"
                files=os.listdir(path)
                d=random.choice(files)
                os.startfile("C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\surprised\\" + d)
##                mixer.music.load("song.mp3")
            
##                mixer.music.load("song.mp3")
            if emotion_dict[predicted] == 'Neutral':
                path="C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\neutral"
                files=os.listdir(path)
                d=random.choice(files)
                os.startfile("C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\neutral\\" + d)
##                mixer.music.load("song.mp3")
            if emotion_dict[predicted] == 'Disgusted':
                path="C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\sad"
                files=os.listdir(path)
                d=random.choice(files)
                os.startfile("C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\sad\\" + d)
            if emotion_dict[predicted] == 'Angry':
                path="C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\Angry"
                files=os.listdir(path)
                d=random.choice(files)
                os.startfile("C:\\Users\\Jyothsna\\Desktop\\Emoplayer\\Angry\\" + d)
##                mixer.music.load("song.mp3")    
##            mixer.music.set_volume(0.7)
##            mixer.music.play()
            # infinite loop
            while True:
                  
                print("Press 'p' to pause, 'r' to resume")
                print("Press 'e' to exit the program")
                query = input("  ")
                  
                if query == 'p':
              
                    # Pausing the music
                    mixer.music.pause()     
                elif query == 'r':
              
                    # Resuming the music
                    mixer.music.unpause()
                elif query == 'e':
              
                    # Stop the mixer
                    mixer.music.stop()
                    break
            
            predictions.clear()
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) ## anti-aliased line

    cv2.imshow('Emotion Detection', frame)  # displays image in window
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
