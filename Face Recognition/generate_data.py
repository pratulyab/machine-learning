import numpy as np
import cv2
import time

name = input("Enter Name: ")
num = int(input("#mugshots: "))

mugshots = []

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print("Start Smiling..")

while True and num:
	time.sleep(3)
	ret, frame = cap.read()
	cv2.imshow("Feed", frame)

	if not ret:
		continue

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[:1] # Consider only the main face while generating data
	
	for face in faces:
		x,y,w,h = face
		only_face = frame[y:y+h, x:x+w]
		only_face = cv2.resize(only_face, (100,100))
		mugshots.append(only_face)
		num -= 1
		print("Clicked")

#	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#	cv2.imshow("Gray", gray)

	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break
	

mugshots = np.asarray(mugshots)
print(mugshots.shape)
mugshots = mugshots.reshape((mugshots.shape[0], -1))
print(mugshots.shape)
print(mugshots[0].shape)
np.save("./dataset/"+name+".npy", mugshots)
cap.release()
cv2.destroyAllWindows()
