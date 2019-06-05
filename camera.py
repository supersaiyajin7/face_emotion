# -*- coding: utf-8 -*-
import cv2
from model import FacialExpressionModel
import numpy as np
import matplotlib.pyplot as plt
import plotterfunc as pf
import face_recognition
import os

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

list_dict = dict((el,[]) for el in ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"])

known_face_encodings = []
known_face_names = []

path = "C:\\Users\\shringar.kashyap\\Documents\\products industry AI\\cnn funda\\face_recog\\"
files_path = os.listdir(path)
for i in range(len(files_path)):
    #files[i] = files[i][0:-4]
    selectedImage = face_recognition.load_image_file(path+files_path[i])
    face_encode = face_recognition.face_encodings(selectedImage)[0]
    known_face_encodings.append(face_encode)
    known_face_names.append(files_path[i][0:-4])



def __get_data__():
    """
    __get_data__: Gets data from the VideoCapture object and classifies them
    to a face or no face. 
    
    returns: tuple (faces in image, frame read, grayscale frame)
    """
    _, fr = rgb.read()
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    return faces, fr, gray

def face_recognizer():
     _, rgb_frame = rgb.read()
     ##rgb_frame = fr[:, :, ::-1]
     #gray_frame = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
     gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
     face_locations = face_recognition.face_locations(rgb_frame)
     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
     matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
     name_face = "Unknown"
     if True in matches:
            first_match_index = matches.index(True)
            name_face = known_face_names[first_match_index]
     else:
            newName = input("Please the name of this person:")
            cv2.imwrite(path+newName+".jpg",rgb_frame)
            #selectedNewImage = face_recognition.face_encodings(frame)
            #face_encode_new = face_recognition.face_encodings(selectedNewImage)[0]            
            print("Adding this face into the database!!!")
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(newName)
     
     return gray_frame,face_locations,rgb_frame,name_face


def start_app(cnn):
    #skip_frame = 10
    #data = []
    #flag = False
    ix = 0
    while True:
        ix += 1
        
        #faces, fr, gray_fr = __get_data__()
        try:
            gray_fr,faces,fr,name_face = face_recognizer()
        except:
            gray_fr,faces,fr,name_face = face_recognizer()
        print ("face:",faces)
        for (x, y, w, h) in faces:
            try:
                fc = gray_fr[y:y+h, x:x+w]
                
                roi = cv2.resize(fc, (48, 48))
                pred,list1 = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
    
                cv2.putText(fr, name_face+":"+pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
                
                j = 0
                for v in list_dict.values():
                    if j != len(list1): 
                        v.extend([list1[j]])
                        j = j + 1
                    else:
                        j = 0
            except:
                 pass   
                    
               
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
    cv2.destroyAllWindows()
    rgb.release()
    for k,v in list_dict.items():
        plt.plot(v[::int(len(v)/10)],label=k)
    plt.legend(loc='upper left')
    #print (list_dict)
    pf.plot_3d(list_dict)
    print("The plot is ready!")

if __name__ == '__main__':
    model = FacialExpressionModel("C:\\Users\\shringar.kashyap\\Documents\\products industry AI\\cnn funda\\face_model.json", "C:\\Users\\shringar.kashyap\\Documents\\products industry AI\\cnn funda\\face_model.h5")
    start_app(model)
