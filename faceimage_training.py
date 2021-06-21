import cv2
import numpy as np
import pickle
import os
from PIL import Image
eye_cascade=cv2.CascadeClassifier("C:/Users/BABU/Desktop/xml/haarcascade_eye.xml")
face_cascade=cv2.CascadeClassifier('C:/Users/BABU/Desktop/xml/haarcascade_frontalface_default.xml')
name=input("Enter your name\n")
if not os.path.exists('C:/Users/BABU/Desktop/image/face/'+name):
    if not os.path.exists('C:/Users/BABU/Desktop/image/eye/'+name):
        os.makedirs('C:/Users/BABU/Desktop/image/eye/'+name)
        os.makedirs('C:/Users/BABU/Desktop/image/face/'+name)
cap=cv2.VideoCapture(0)
sam=0
sam1=0
while 1:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    r,t1=cv2.threshold(gray,100,255,cv2.THRESH_TRUNC)
    faces=face_cascade.detectMultiScale(t1,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(t1,(x,y),(x+w,y+h),(0,0,0),2)
        roi_gray=t1[y:y+h,x:x+w]
        sam=sam+1
        cv2.imwrite("C:/Users/BABU/Desktop/image/face/"+name+"/"+str(sam)+".jpg",roi_gray)
        eyes=eye_cascade.detectMultiScale(t1,1.3,5)
        for(ex,ey,ew,eh) in eyes:
            if(x<ex and y<ey and x+w>ex+ew and y+h>ey+eh):
                cv2.rectangle(t1,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
                eroi_gray=t1[ey:ey+eh,ex:ex+ew]
                sam1=sam1+1
                cv2.imwrite("C:/Users/BABU/Desktop/image/eye/"+name+"/"+str(sam1)+".jpg",eroi_gray)
    cv2.imshow('frame',t1)
    if cv2.waitKey(30) &  0xFF==ord('n'):
        break
cap.release()
cv2.destroyAllWindows()
###
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,"image/face")
#eye_cascade=cv2.CascadeClassifier('C:/Users/BABU/Desktop/xml/haarcascade_eye.xml')
#face_cascade=cv2.CascadeClassifier('C:/Users/BABU/Desktop/xml/haarcascade_frontalface_default.xml')
reg=cv2.face.LBPHFaceRecognizer_create()
x_train=[]
y_label=[]
ex_train=[]
ey_label=[]
c_id=0
l_id={} 
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg")or file.endswith("png"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","_").lower()
            if not label in l_id:
                l_id[label]=c_id
                c_id+=1
            id_=l_id[label]
            pil_image=Image.open(path)
            image_array=np.array(pil_image,"uint8")
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5,minSize=(30,30))
            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_label.append(id_)
            eyes=eye_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5,minSize=(30,30))
            for (x,y,w,h) in eyes:
                    eroi=image_array[y:y+h,x:x+w]
                    ex_train.append(eroi)
                    ey_label.append(id_)
with open("labels.pickle","wb")as f:
    pickle.dump(l_id,f)
reg.train(x_train,np.array(y_label))
reg.save("trainer.yml")
reg.train(ex_train,np.array(ey_label))
reg.save("etrainer.yml")
print('success')


quit()
