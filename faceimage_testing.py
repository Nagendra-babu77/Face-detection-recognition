import cv2
import numpy as np
import pickle
import os
import datetime
labels={}
reg=cv2.face.LBPHFaceRecognizer_create()
reg1=cv2.face.LBPHFaceRecognizer_create()
reg.read('trainer.yml')
reg1.read('etrainer.yml')
with open("labels.pickle","rb")as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
eye_cascade=cv2.CascadeClassifier('C:/Users/BABU/Desktop/xml/haarcascade_eye.xml')
face_cascade=cv2.CascadeClassifier('C:/Users/BABU/Desktop/xml/haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while 1:
    ret,frame=cap.read()
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.imshow('frame',frame)
    if cv2.waitKey(30)&0xFF==ord('p'):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        r,t1=cv2.threshold(gray,120,255,cv2.THRESH_TRUNC)
        faces=face_cascade.detectMultiScale(t1,1.3,5)
        for(x,y,w,h) in faces:
                roi_gray=t1[y:y+h,x:x+w]
                id1,con1=reg.predict(roi_gray)
                #print(con1)
                name=labels[id1]
                cv2.rectangle(t1,(x,y),(x+w,y+h),(255,255,255),2)
                eyes=eye_cascade.detectMultiScale(t1,1.3,5)
                for (ex,ey,ew,eh) in eyes:
                    eroi_gray=t1[ey:ey+eh,ex:ex+ew]
                    id2,con2=reg1.predict(eroi_gray)
                    cv2.rectangle(t1,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
                    #print(con2)
                    if id1==id2:
                        cv2.putText(t1,name,(x-10,y-10),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)
                        cv2.imshow('frame2',t1)
    if cv2.waitKey(30) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

