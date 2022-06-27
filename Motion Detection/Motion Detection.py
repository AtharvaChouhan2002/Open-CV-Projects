import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cap=cv.VideoCapture(0)



while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    diff=cv.absdiff(frame1,frame2)

    gray=cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
    blur=cv.GaussianBlur(gray,(5,5),0)
    _,th1=cv.threshold(blur,20,255,cv.THRESH_BINARY)
    dilated=cv.dilate(th1,None,iterations=3)

    contours,_=cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        (x,y,w,h)=cv.boundingRect(c)

        if cv.contourArea(c)<1500:
            continue

        cv.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)


    cv.imshow("window",frame1)

    frame1=frame2
    ret,fram2=cap.read()




    if cv.waitKey(30)==27:
        break


cv.destroyAllWindows()
cap.release()













