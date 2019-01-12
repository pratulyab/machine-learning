import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

mugshots = []
names = []

files = [each for each in os.listdir('./dataset') if each.endswith('.npy')]

for filename in files:
	mugshots.append(np.load('./dataset/'+filename))
	names.append(filename[:-4])

face_data = np.concatenate(mugshots, axis=0)
print(face_data.shape)
names = np.repeat(names, [x.shape[0] for x in mugshots])
print(names.shape)
print(names)
labels = names.reshape((-1, 1))
print(labels.shape)
print(labels)


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(face_data, names)

while True:
	ret, frame = cap.read()
	
	faces = face_cascade.detectMultiScale(frame, 1.3, 5)

	for face in faces:
		x,y,w,h = face
		only_face = frame[y:y+h, x:x+h]
		only_face = cv2.resize(only_face, (100,100))
		only_face = only_face.reshape((1,-1))
		pred = knn.predict(only_face)
		print(pred)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(frame, pred[0], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
	cv2.imshow("Feed", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
