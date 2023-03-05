import numpy as np
import cv2
import time
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from cv2 import imread,resize
import serial


ard = serial.Serial(port='/dev/ttyACM0')
cap = cv2.VideoCapture(0) 	#640,480

w = 960
h = 480
blink=0
flag1=0
sums=0

i=101

model = keras.models.load_model('eye.h5')
out = 0

while(cap.isOpened()):
	ret, frame1 = cap.read()
	if ret==True:		
		frame = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
		faces=cv2.CascadeClassifier('haarcascade_eye.xml')
		detected = faces.detectMultiScale(frame, 1.3, 5)
			
		pupilFrame = frame
		pupilO = frame

		try:
			(x,y,w,h) = detected[0]
			img=frame[y-5:(y+h+5), x-5:(x+w+5)]
			eqhist=cv2.equalizeHist(img)
			cv2.imshow('frame',eqhist)
			p=cv2.waitKey(1)

			img = eqhist/255 #Normalizing pixels into 0-1
			img = resize(img, (60,60)) #This can be resized to any shape, choosing 224 randomly. Larger sizes can result in computation delayprint(model.predict(img))

			color_img = np.expand_dims(img, axis=-1)
			color_img = np.repeat(color_img, 3, axis=-1)
			
			#print(color_img.shape)
			test = []
			test.append(color_img)
			test = np.array(test)
			out = np.argmax(model.predict(test))
			
			if(out == 0):
			        print('right')
		        	ard.write(str.encode('r'))

			elif(out == 1):
			        print('left')
		        	ard.write(str.encode('l'))

			else:
			        print('center')
		        	ard.write(str.encode('u'))

			if p == ord('s'):
				cv2.imwrite('train/'+str(i)+'.jpg',eqhist)
				i+=1
			elif p == ord('q'): #cv2.waitKey(1) & 0xFF == ord('q'):
				break

			cv2.waitKey(1)

		except:
			pass

cap.release()
cv2.destroyAllWindows()
