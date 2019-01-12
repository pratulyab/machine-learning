''' Face Detection using Haar Cascade '''
import cv2
import numpy as np

# Working with live feed through webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, 1.3, 5)

	for face in faces:
		x,y,w,h = face
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
		cv2.putText(frame, "Some Text", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)

		only_face = frame[y:y+h, x:x+w]
		cv2.imshow('gray face', cv2.cvtColor(only_face, cv2.COLOR_RGB2GRAY))
		eyes = eye_cascade.detectMultiScale(only_face, 1.3, 5)
		for eye in eyes:
			print(len(eyes))
			ex,ey,ew,eh = eye
			cv2.rectangle(only_face, (ex,ey), (ex+ew, ey+eh), (255,0,0), 1)

	cv2.imshow("Feed", frame)
	if not ret:
		print("Couldn't access camera. Trying again")
		continue

	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break
	
#	print(key)
#	print(bin(key))
#	print(type(key))
#	if key & 0xff:
#		break
	

cap.release()
cv2.destroyAllWindows()
