import cv2
import face_recognition
import numpy as np
import geocoder
from twilio.rest import Client

def smsMe(name,lat,long):
    print(1)
    print(f"{name}{lat}{long}")
    # client = Client(account_sid, auth_token)
    # message = client.messages.create(
    #    from_='+18087847010',
    #     body='This person {name} was detected at lat{lat} and long{long}',
    #    to='+917756946031'
    # )
    # print(message.sid)



def findMe():
    location = geocoder.ip('me')
    print(2)
    if location and location.latlng:
        latitude = location.latlng[0]
        longitude = location.latlng[1]
        print(f"Location found: Latitude: {latitude}, Longitude: {longitude}")
    else:
        print("Location could not be determined.")


video_capture = cv2.VideoCapture(0)
try:
    obama_image = face_recognition.load_image_file("Devesh/pic1.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    print(3)
except IndexError:
    print("Error: Face not found in provided images.")
    video_capture.release()
    print(4)
    cv2.destroyAllWindows()
    exit()



known_face_encodings = [obama_face_encoding]
known_face_names = ["Barack Obama"]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
print(5)
while True:
    ret, frame = video_capture.read()

    if not ret:
        print(6)
        print("Error: Could not read frame from camera.")
        break

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            print(7)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                print(8)

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print(9)

                face_names.append(name)

                if name != 'Unknown':
                    findMe()
                    print(10)  

    process_this_frame = not process_this_frame

    if face_locations and face_names:
        print(11)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            print(12)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    cv2.imshow('Video', frame)
    print(13)
    print(f"Faces recognized: {face_names}")
    print(14)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(15)
video_capture.release()
print(16)
cv2.destroyAllWindows()
