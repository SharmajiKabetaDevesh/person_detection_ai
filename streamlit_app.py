import cv2
import face_recognition
import numpy as np
import geocoder

import streamlit as st
from PIL import Image
import tempfile




def smsMe(name, lat, long):
    print(f'This person {name} was detected at lat {lat} and long {long}')
    st.write(f'This person {name} was detected at lat {lat} and long {long}')
    print(1)
  


def findMe():
    print(2)
    location = geocoder.ip('me')
    if location and location.latlng:
        latitude = location.latlng[0]
        longitude = location.latlng[1]
        print(f"Location found: Latitude: {latitude}, Longitude: {longitude}")
        return latitude, longitude
    else:
        print("Location could not be determined.")
        return None, None


st.title("Real-Time Face Recognition with SMS Notification")
uploaded_image = st.file_uploader("Upload an image of the person to track", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_image.read())
    print(3)
    image = face_recognition.load_image_file(tfile.name)
    face_encodings = face_recognition.face_encodings(image)
    
    if face_encodings:
        known_face_encoding = face_encodings[0]
        print(4)
        known_face_name = "Tracked Person" 
        st.success("Face encoding created for the uploaded image.")
    else:
        print(5)
        st.error("No face found in the uploaded image.")


video_capture = cv2.VideoCapture(0)
print(6)

if st.button("Start Face Recognition", key="start_button"):
    known_face_encodings = [known_face_encoding]
    known_face_names = ["Tracked Person"]
    print(7)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    stframe = st.empty()  
    i=0 
    while True:
        i+=1
        ret, frame = video_capture.read()
        print(8)

        if not ret:
            st.error("Error: Could not read frame from camera.")
            print(9)
            break

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            print(10)

            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                print(11)
 
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    print(12)

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                    if name != 'Unknown':
                        lat, long = findMe()
                        print(13)
                        if lat and long:
                            print(14)
                            smsMe(name, lat, long)

        process_this_frame = not process_this_frame


        if face_locations and face_names:
            print(15)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                print(16)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        stframe.image(frame, channels="BGR")
        print(17)
        if st.button("Stop Face Recognition", key=f"stop_button{i}"):
            break

print(18)
video_capture.release()
print(19)
cv2.destroyAllWindows()
