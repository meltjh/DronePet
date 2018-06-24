# Changed by us
# Original code: http://thecodacus.com/ modified by Nazmi Asri

import os
import cv2
import timeit

class FaceRecognition:
    
    def __init__(self):
        # Create Local Binary Patterns Histograms for face recognization
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Load the trained mode
        self.recognizer.read('face_recognition_data/trainer/trainer.yml')
        # Load prebuilt model for Frontal Face
        cascadePath = "face_recognition_data/haarcascade_frontalface_default.xml"
        # Create classifier from prebuilt model
        self.faceCascade = cv2.CascadeClassifier(cascadePath);
        
        # Set the font style
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Detection parameters
        self.confidenceThreshold =  -1000#0.4
        self.train_count = -1
    
      
    def id_to_name(self, face_id):
        if face_id == 0:
            return 'Richard'
        elif face_id == 1:
            return 'Melissa'
        else:
            return 'Unknown'
        
    # Stores the cropped faces for training.
    # Modified, but originally from: http://thecodacus.com/
    def store_training_faces(self, image, face_id, display_faces = False, image_drawn = None):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect frames of different sizes, list of faces rectangles
#        faces = self.faceCascade.detectMultiScale(image_gray, 1.3, 5) # TODO: welke waardes etc
        
        # high resoltution
        sf = 1.30
        mn = 2
        mis = (88, 88)
        mas = (320, 320)
        
        # low resolution
#        sf = 1.30
#        mn = 2
#        mis = (50, 50)
#        mas = (200, 200)
        
#        start = timeit.default_timer()
        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=sf, minNeighbors=mn, minSize=mis, maxSize=mas)
#        stop = timeit.default_timer()
#        print("For 10 times:")
#        print("sf:{}, mn:{}, mis:{}, mas:{}".format(sf, mn, mis, mas))
#        print("Amount of faces: {}",format(len(faces)))
#        print("Took: {} seconds".format(stop - start))
        
        # Loops for each faces
        for (x,y,w,h) in faces:
            if display_faces:
                image_drawn = self.draw_box(image_drawn, (y,y+h,x,x+w), "{}x{}".format(w,h), box_color = (0, 192, 0))
            self.store_training_face(image_gray[y:y+h,x:x+w], face_id, isGray=True)
    
    def store_training_face(self, image, face_id, isGray = False):
        if image is None:
            print('No face was saved')
            return
        
        if not isGray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        

        path = "face_recognition_data/dataset/"
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    
        # Add to the set. So create a new name as long as the previous one exists
        while True:
            self.train_count += 1
            
            # Save the captured image into the datasets folder
            f_name = "{}User.{}.{}.jpg".format(path, str(face_id), str(self.train_count))

            if not os.path.isfile(f_name):
                cv2.imwrite(f_name, image)
                break
    
    # Returns all confident faces, sorted high to low confidence
    # returns [(face_id, (x,y,w,h)), ...]
    def recognize_faces(self, image):
        # Convert the captured frame into grayscale
        image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
        # Get all face from the video frame
        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5) # TODO: waardes vinden
    
        faces_confident = []
        # For each face in faces
        for (x,y,w,h) in faces:
            # Recognize the face belongs to which ID
            face_id, confidence = self.recognizer.predict(image_gray[y:y+h,x:x+w])
    
            # Weet niet waarom ze dat hebben gedaan
            confidence = round(100 - confidence, 2)
            
            if confidence >= self.confidenceThreshold:
                faces_confident.append((face_id, (y,y+h,x,x+w), confidence))
            
        # Sort faces so that the most confident are in front of the list.
        faces_confident.sort(key=lambda x: x[2], reverse = True)
            
        return faces_confident
            

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
        margin_width_ratio = 4
        margin_top_ratio = 2.2
        margin_bottom_ratio = 4
        
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
        for face_id, face_corners, confidence in faces:
            name = "{} ({})".format(self.id_to_name(face_id), round(confidence, 2))
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
        
        # Determine the body patch corners
        estimate_body_corners = self.get_body_patch(image_original, correct_face_corners)
        
        return estimate_body_corners, correct_face_corners