import cv2
import numpy as np
import time

gray = [0,0]


cap = cv2.VideoCapture('demo1.avi')
count = 0

while(1):
    # get a frame
    try:
        ret, frame = cap.read()

        if count == 0:
            gray[0] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif count == 1:
            gray[1] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            del gray[0]
            tem = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray.append(tem)
        count += 1
    
        blur0 = cv2.GaussianBlur(gray[0],(5,5),0)
        blur1 = cv2.GaussianBlur(gray[1],(5,5),0)
        
        d = cv2.absdiff(blur0, blur1)
        ret, th = cv2.threshold( d, 10, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, None, iterations=2)
        #cv2.imshow('dilated',dilated)

        img,contours,hierarchy = \
          cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

       
        markColor=(0,255,0)
        '''
        fin = cv2.drawContours(frame,contours,-1,markColor,2)
        '''
        if contours == []:
            pass
        else:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
            fin= cv2.rectangle(frame,(x,y),(x+w,y+h), markColor,2)
            cv2.imshow('rec',fin)
   
    # show a frame
        cv2.imshow("capture", fin)
    
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        



    except cv2.error:
        break
        #print('error')
cv2.destroyAllWindows()

