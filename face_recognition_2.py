import glob
import cv2
import face_recognition
import time

class FaceRecognition:
    
    def __init__(self, drone_controller):
        # Set the font style
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        self.drone_controller = drone_controller
        
        # Obtaining the encodings given the images.
        self.known_face_encodings = self.get_face_encodings("faces/")
        self.known_face_ids = list(range(len(self.known_face_encodings)))
        
        self.scaleFactor = 1.3
        self.minNeighbors = 2
        self.minSize = (66, 66)
        self.maxSize = (320, 320)
        cascadePath = "face_recognition_data/haarcascade_frontalface_alt.xml"
        # Create classifier from prebuilt model
        self.faceCascade = cv2.CascadeClassifier(cascadePath);
    
    def get_face_encodings(self, face_folder):
        files = glob.glob("{}/*.jpg".format(face_folder))
        
        face_encodings = []
        for file in files:
            person_image = face_recognition.load_image_file(file)
            person_face_encoding = face_recognition.face_encodings(person_image)[0]
            face_encodings.append(person_face_encoding)
        
        return face_encodings
        
    def id_to_name(self, face_id):
        return "Person {}".format(face_id)
    
    def get_face_locations(self, image):        
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize, maxSize=self.maxSize)
        
#        # Loops for each faces
        face_locations = []
        for (x,y,w,h) in faces:
            face_locations_temp = face_recognition.face_locations(image[y:y+h, x:x+w])
            if len(face_locations_temp) == 0:
                print("Could not find a face in the location")
            elif len(face_locations_temp) > 1:
                print("Found multiple faces in location")
            else:
                t_y, t_x, t_yy, t_xx = face_locations_temp[0]
                face_locations.append((y+t_y, x+t_x, y+t_yy, x+t_xx))
 
        return face_locations

    # Returns all faces with patch information.
    def recognize_faces(self, image):
        shrink_ratio = 1
        small_frame = cv2.resize(image, (0, 0), fx=1/shrink_ratio, fy=1/shrink_ratio)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = self.get_face_locations(rgb_small_frame)
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
            top *= shrink_ratio
            right *= shrink_ratio
            bottom *= shrink_ratio
            left *= shrink_ratio      
            
            coordinates.append((good_id, (int(top), int(bottom), int(left), int(right))))
        
        return coordinates
            

    # Returns the image with rectangle and title
    def draw_box(self, image, corners, title = 'Unknown', box_color = (0, 255, 0)):
        (top, bottom, left, right) = corners
        
        cv2.rectangle(image, (int(left),int(top)), (int(right), int(bottom)), box_color, 4)            
        cv2.putText(image, title, (int(left), int(top) - 20), self.font, 1, (255, 255, 255), 3)
        
        return image
    
    # Returns the patch corners given the face patch
    def get_body_patch(self, image, face):
        (face_top, face_bottom, face_left, face_right) = face
        
        face_w = face_right - face_left
        face_h = face_bottom - face_top
        
        [img_height, img_width, _] = image.shape
        
        margin_width_ratio = 5
        margin_top_ratio = 2.5
        margin_bottom_ratio = 5
        
        margin_width = int(round(margin_width_ratio * face_w))
        margin_top = int(round(margin_top_ratio) * face_h)
        margin_bottom = int(round(margin_bottom_ratio) * face_h)
        
        # Ensure not out of bounds
        left = max(0, face_left - margin_width)
        right = min(img_width, face_right + margin_width)
        
        top = max(0, face_top - margin_top)
        bottom = min(img_height, face_bottom + margin_bottom)

        return top, bottom, left, right

    def get_ratio_distance(self, x, center_ratio, margin_ratio):
        if x < center_ratio - margin_ratio or x > center_ratio + margin_ratio:
            return x - center_ratio
        return 0
    
    def get_ratio_degrees(self, x, center_ratio, margin_ratio, max_degree):
        distance_ratio = self.get_ratio_distance(x, center_ratio, margin_ratio)
        return distance_ratio * max_degree
    
    # Given the position of the face, the drone will tilt the camera and rotate itself.
    def perform_action(self, face_corners, image_w, image_h, image_drawn):
        
        face_top, face_bottom, face_left, face_right = face_corners
        center_x = (face_right-face_left)/2 + face_left
        center_y = (face_bottom-face_top)/2 + face_top

        ratio_x = center_x/(image_w*2)
        ratio_y = center_y/(image_h/2)

        # Set the center to be in the middle, but sligtly to the top.
        # Also ignore if it is within the acceptable margin.
        x_center_ratio = 0.5
        y_center_ratio = 0.42
        x_margin_ratio = 0.025
        y_margin_ratio = 0.025
        
        x_max_degree = 50.0
        y_max_degree = 30.0
        
        x_degrees = int(self.get_ratio_degrees(ratio_x, x_center_ratio, x_margin_ratio, x_max_degree))
        y_degrees = -int(self.get_ratio_degrees(ratio_y, y_center_ratio, y_margin_ratio, y_max_degree))
        
        # How long should it last. The degrees are per second. So the velocities are divided by the duration.
        duration = 0.25
        
        if self.drone_controller.bebop.IsOnlineBebop:
            self.drone_controller.bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=y_degrees/duration, duration=duration)
    
            if x_degrees != 0:
                max_rotation_speed = 35
                
                speed = (x_degrees/duration)/max_rotation_speed
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