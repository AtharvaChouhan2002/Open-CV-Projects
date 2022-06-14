import cv2 as cv
import time
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
import pytesseract as pt
pt.pytesseract.tesseract_cmd=r"C:\Users\athar\AppData\Local\tesseract.exe"




img=cv.imread("4 .jpg")


gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

bfilter=cv.bilateralFilter(gray,11,17,17)
edge=cv.Canny(bfilter,30,200)

keypoints=cv.findContours(edge.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
contours=imutils.grab_contours(keypoints)
contours=sorted(contours,key=cv.contourArea,reverse=True)[:10]


location=None

for c in contours:
    approx=cv.approxPolyDP(c,10,True)
    if len(approx)==4:
        location=approx
        break


mask=np.zeros(gray.shape,np.int8)
new_img=cv.drawContours(mask,[location],0,255,-1)
new_img=cv.bitwise_and(img,img,mask=mask)


cv.imshow("real",img)



text=pt.image_to_string(new_img)
print(text)
time.sleep(1)
cv.imshow("window",new_img)

plt.show()
cv.waitKey()
cv.destroyAllWindows()


