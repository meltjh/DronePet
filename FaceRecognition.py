# Changed by us
# Original code: http://thecodacus.com/ modified by Nazmi Asri


import cv2
import os

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
        
        self.CONFIDENCE_THRESHOLD = -1000#0.25
    
        
    # Stores the cropped faces for training.
    # Modified, but originally from: http://thecodacus.com/
    def store_training_faces(self, image, face_id):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Initialize sample face image
        count = -1
    
        path = "face_recognition_data/dataset/"
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
    
        # Detect frames of different sizes, list of faces rectangles
        faces = self.faceCascade.detectMultiScale(image_gray, 1.3, 5)
    
        # Loops for each faces
        for (x,y,w,h) in faces:
    
    
            # Add to the set. So create a new name as long as the previous one exists
            while True:
                count += 1
                
                # Save the captured image into the datasets folder
                f_name = "{}User.{}.{}.jpg".format(path, str(face_id), str(count))
    
                if not os.path.isfile(f_name):
                    cv2.imwrite(f_name, image_gray[y:y+h,x:x+w])
                    break
    
    # Returns all confident faces
    # returns [(face_id, (x,y,w,h)), ...]
    def recognize_faces(self, image):
        # Convert the captured frame into grayscale
        image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
        # Get all face from the video frame
        faces = self.faceCascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)
    
        faces_confident = []
        # For each face in faces
        for (x,y,w,h) in faces:
            # Recognize the face belongs to which ID
            face_id, confidence = self.recognizer.predict(image_gray[y:y+h,x:x+w])
    
            # Weet niet waarom ze dat hebben gedaan
            confidence = round(100 - confidence, 2)
            
            if confidence >= self.CONFIDENCE_THRESHOLD:
                faces_confident.append((face_id, (x,y,w,h)), confidence)
            
        # Sort faces so that the most confident are in front of the list.
        faces_confident.sort(key=lambda x: x[2], reverse = True)
            
        return faces_confident
            
    # Returns the image with rectangle and name
    def draw_face(self, image, face, face_id, box_color = (0, 255, 0)):
        (face_x, face_y, face_w, face_h) = face
        
        cv2.rectangle(image, (face_x - 20, face_y - 20), (face_x + face_w + 20, face_y + face_h + 20), box_color, 4)
        
        if face_id == 0:
            name = 'Richard'
        if face_id == 1:
            name = 'Melissa'
        else:
            name = 'Unknown'
            
        cv2.putText(image, str(name), (face_x, face_y - 40), self.font, 1, (255, 255, 255), 3)
        
        return image
    
    # Returns the patch corners
    def body_patch(self, image, face):
        (face_x, face_y, face_w, face_h) = face
        
        # Patch pakken
        [img_height, img_width, _] = image.shape
        
        # TODO: goede margins bedenken
        margin_width_ratio = 3
        margin_top_ratio = 1
        margin_bottom_ratio = 5
        
        margin_width = int(round(margin_width_ratio * face_w))
        margin_top = int(round(margin_top_ratio) * face_h)
        margin_bottom = int(round(margin_bottom_ratio) * face_h)
        
        # TODO margins boven hoofd en lengte ook ipv einde plaatje?
        left = max(0, face_x - margin_width)
        right = min(img_width, face_x + face_w + margin_width)
        
        top = max(0, face_y - margin_top)
        bottom = min(img_height, face_y + face_h + margin_bottom)
        
        return top, bottom, left, right