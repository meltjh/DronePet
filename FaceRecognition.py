# Changed by us
# Original code: http://thecodacus.com/ modified by Nazmi Asri

import os
import cv2 
from Actions import Action
import matplotlib.pyplot as plt
import face_recognition



class FaceRecognition:
    
    def __init__(self, drone_controller):
        # Create Local Binary Patterns Histograms for face recognization
#        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Load the trained mode
#        self.recognizer.read('face_recognition_data/trainer/trainer.yml')
        # Load prebuilt model for Frontal Face
#        cascadePath = "face_recognition_data/haarcascade_frontalface_alt.xml"
        # Create classifier from prebuilt model
#        self.faceCascade = cv2.CascadeClassifier(cascadePath);
        
        # Set the font style
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
#        self.scaleFactor = 1.3
#        self.minNeighbors = 3
        # high res
#        self.minSize = (88,88)
#        self.maxSize = (320, 320)
        # low res
        # self.minSize = (50,50)
        # self.maxSize = (200, 200)
        
        # Detection parameters
#        self.confidenceThreshold =  0.25
#        self.train_count = -1
        
        self.drone_controller = drone_controller
        
        face_folder = "faces/"
        richard_image = face_recognition.load_image_file(face_folder + "richard.jpg")
        richard_face_encoding = face_recognition.face_encodings(richard_image)[0]
        
        melissa_image = face_recognition.load_image_file(face_folder + "melissa.jpg")
        melissa_face_encoding = face_recognition.face_encodings(melissa_image)[0]
        
        self.known_face_encodings = [richard_face_encoding, melissa_face_encoding]
        self.known_face_ids = [0, 1]
    
      
    def id_to_name(self, face_id):
        if face_id == 0:
            return 'Richard'
        elif face_id == 1:
            return 'Melissa'
        else:
            return 'Unknown'
        
#    # Stores the cropped faces for training.
#    # Modified, but originally from: http://thecodacus.com/
#    def store_training_faces(self, image, face_id, display_faces = False, image_drawn = None):
#        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#        
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize, maxSize=self.maxSize)
#
#        # Loops for each faces
#        for (x,y,w,h) in faces:
#            if display_faces:
#                image_drawn = self.draw_box(image_drawn, (y,y+h,x,x+w), "{}x{}".format(w,h), box_color = (0, 192, 0))
#            self.store_training_face(image_gray[y:y+h,x:x+w], face_id, isGray=True)
#    
#    def store_training_face(self, image, face_id, isGray = False):
#        if image is None:
#            print('No face was saved')
#            return
#        
#        if not isGray:
#            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#        
#
#        path = "face_recognition_data/dataset/"
#        dir = os.path.dirname(path)
#        if not os.path.exists(dir):
#            os.makedirs(dir)
#    
#        # Add to the set. So create a new name as long as the previous one exists
#        while True:
#            self.train_count += 1
#            
#            # Save the captured image into the datasets folder
#            f_name = "{}User.{}.{}.jpg".format(path, str(face_id), str(self.train_count))
#
#            if not os.path.isfile(f_name):
#                cv2.imwrite(f_name, image)
#                break
    
    # Returns all confident faces, sorted high to low confidence
    # returns [(face_id, (x,y,w,h)), ...]
    def recognize_faces(self, image):
        small_frame = cv2.resize(image, (0, 0), fx=1/2, fy=1/2)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_ids = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            id = -1

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                id = self.known_face_ids[first_match_index]

                face_ids.append(id)
    
        # Display the results
        coordinates = []
        for (top, right, bottom, left), good_id in zip(face_locations, face_ids):            
            
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2            
            
            coordinates.append((good_id, (top, bottom, left, right)))
        
        return coordinates
            
#            
#    
#            # Draw a box around the face
#            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#    
#            # Draw a label with a name below the face
#            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#            font = cv2.FONT_HERSHEY_DUPLEX
#            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            
        
#        # Convert the captured frame into grayscale
#        image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#        
#        # Get all face from the video frame
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize, maxSize=self.maxSize)
#    
#        faces_confident = []
#        # For each face in faces
#        for (x,y,w,h) in faces:
#            # Recognize the face belongs to which ID
#            face_id, confidence = self.recognizer.predict(image_gray[y:y+h,x:x+w])
#    
#            # Weet niet waarom ze dat hebben gedaan
#            confidence = round(100 - confidence, 2)
#            
#            if confidence >= self.confidenceThreshold:
#                faces_confident.append((face_id, (y,y+h,x,x+w), confidence))
#            
#        # Sort faces so that the most confident are in front of the list.
#        faces_confident.sort(key=lambda x: x[2], reverse = True)
#            
#        return faces_confident
            

    # Returns the image with rectangle and title
    def draw_box(self, image, corners, title = 'Unknown', box_color = (0, 255, 0)):
        (top, bottom, left, right) = corners
        
        cv2.rectangle(image, (left, top), (right, bottom), box_color, 4)            
        cv2.putText(image, title, (left, top - 20), self.font, 1, (255, 255, 255), 3)
        
        return image
    
    # Returns the patch corners
    def get_body_patch(self, image, face):
        (face_top, face_bottom, face_left, face_right) = face
        
        face_w = face_right - face_left
        face_h = face_bottom - face_top
        
        [img_height, img_width, _] = image.shape
        
        # TODO: goede margins bedenken
        margin_width_ratio = 5
        margin_top_ratio = 2.3
        margin_bottom_ratio = 6
        
        margin_width = int(round(margin_width_ratio * face_w))
        margin_top = int(round(margin_top_ratio) * face_h)
        margin_bottom = int(round(margin_bottom_ratio) * face_h)
        
        # TODO margins boven hoofd en lengte ook ipv einde plaatje?
        left = max(0, face_left - margin_width)
        right = min(img_width, face_right + margin_width)
        
        top = max(0, face_top - margin_top)
        bottom = min(img_height, face_bottom + margin_bottom)
        
        print(bottom - top, right-left)
        
        return top, bottom, left, right

    def get_ratio_distance(self, x, center_ratio, margin_ratio):
        if x < center_ratio - margin_ratio or x > center_ratio + margin_ratio:
            return x - center_ratio
        return 0
    
    def get_ratio_degrees(self, x, center_ratio, margin_ratio, max_degree):
        distance_ratio = self.get_ratio_distance(x, center_ratio, margin_ratio)
        return distance_ratio * max_degree
    
    def perform_action(self, face_corners, image_w, image_h, image_drawn):
        
        face_top, face_bottom, face_left, face_right = face_corners
        center_x = (face_right-face_left)/2 + face_left
        center_y = (face_bottom-face_top)/2 + face_top

#        image_drawn = self.draw_box(image_drawn, (int(center_y-2), int(center_y+2), int(center_x-2), int(center_x+2)), '', box_color = (0, 192, 0))

#        print('face_left', face_left)        
#        print("center_y", center_y)
        
        # Somehow /2...
        ratio_x = center_x/(image_w*2)
        ratio_y = center_y/(image_h/2)
 
#        print("ratio_x",ratio_x)

        x_center_ratio = 0.5
        y_center_ratio = 0.42
        x_margin_ratio = 0.025
        y_margin_ratio = 0.025
        
        x_max_degree = 50.0
        y_max_degree = 30.0
        
        x_degrees = int(self.get_ratio_degrees(ratio_x, x_center_ratio, x_margin_ratio, x_max_degree))
        y_degrees = -int(self.get_ratio_degrees(ratio_y, y_center_ratio, y_margin_ratio, y_max_degree))
        
#        print("x_degrees", x_degrees)

        duration = 0.25
        
        if self.drone_controller.bebop.IsOnlineBebop:
            self.drone_controller.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=y_degrees/duration, duration=duration)
    
            if x_degrees != 0:
                max_rotation_speed = 35
                
                speed = (x_degrees/duration)/max_rotation_speed
#                print("Speed: {} / {}".format(speed, max_rotation_speed))
                self.drone_controller.bebop.fly_direct(roll=0, pitch=0, yaw=int(speed*100), vertical_movement=0, duration=duration)
        
    # If the original image exists, obtain faces. Show all faces in red, and show the correct one in green.
    # If the correct face is found, a patch around the body is drawn in black, and the coordinates are returned
    # as well as the drawn image.
    def main(self, image_original, image_drawn, specific_face_id):
        if image_original is None:
            print('No image given for face_recognition')
            return (-1, -1, -1, -1), (-1, -1, -1, -1)
        
        
        # Recogize faces
        faces = self.recognize_faces(image_original)
        if len(faces) <= 0:
            print('No faces are found.')
            return (-1, -1, -1, -1), (-1, -1, -1, -1)
        
        # Draw the faces and select the first face that has the given id.
        correct_face_corners = None
        for face_id, face_corners in faces:
            name = "{}".format(self.id_to_name(face_id))
            if face_id == specific_face_id and correct_face_corners is None:
                # Set the found face as the correct face. This is only the first instance.
                correct_face_corners = face_corners
                # Draw the correct face green
                image_drawn = self.draw_box(image_drawn, face_corners, name, box_color = (0, 192, 0))
            else:
                # Draw the incorrect faces red
                image_drawn = self.draw_box(image_drawn, face_corners, name, box_color = (192, 0, 0))
        
        if correct_face_corners is None:
            print('The correct face is not found.')
            return (-1, -1, -1, -1), (-1, -1, -1, -1)
        
        image_w, image_h, _ = image_original.shape
        self.perform_action(correct_face_corners, image_w, image_h, image_drawn)
        
        # Determine the body patch corners
        estimate_body_corners = self.get_body_patch(image_original, correct_face_corners)
        image_drawn = self.draw_box(image_drawn, estimate_body_corners, "Posture estimation focus", box_color = (0, 0, 0))
        
        return estimate_body_corners, correct_face_corners