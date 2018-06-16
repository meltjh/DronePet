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
                faces_confident.append((face_id, (x,y,w,h), confidence))
            
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
    
    
    # Given the eyes, nose and if possible the ears, estimate the face patch
    # which can then be fed to face recognition to determine the id of the skeleton.
    # Returns the (top, bottom, left, right) pixel values of the patch.
    # (-1, -1, -1, -1) if no patch found.
    def face_patch_given_skeleton(self, image, skeleton):
        # If no nose and eyes found, continue.
        if not (0 in skeleton.body_parts and
                14 in skeleton.body_parts and
                15 in skeleton.body_parts):
            
            print('No eyes and nose found')
            return (-1, -1, -1, -1)
        
        
        image_h = image.shape[0]
        image_w = image.shape[1]
        
        nose = skeleton.body_parts[0]
        eye_r = skeleton.body_parts[15] # l and r switched to relative to camera
        eye_l = skeleton.body_parts[14] # l and r switched to relative to camera
        
        cv2.circle(image, (int(round(image_w*eye_r.x)), int(round(image_h*eye_r.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
        cv2.circle(image, (int(round(image_w*eye_l.x)), int(round(image_h*eye_l.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
        cv2.circle(image, (int(round(image_w*nose.x)), int(round(image_h*nose.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)


        best_right = eye_r.x + (eye_r.x - nose.x)
        if 17 in skeleton.body_parts:
            ear_r = skeleton.body_parts[17] # l and r switched to relative to camera
            
            cv2.circle(image, (int(round(image_w*ear_r.x)), int(round(image_h*ear_r.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
            
            
            best_right = ear_r.x
        
        best_left = eye_l.x - (nose.x - eye_l.x)
        if 16 in skeleton.body_parts:
            ear_l = skeleton.body_parts[16] # l and r switched to relative to camera
            
            cv2.circle(image, (int(round(image_w*ear_l.x)), int(round(image_h*ear_l.y))), 3, (0,0,255), thickness=3, lineType=8, shift=0)
                 
            best_left = ear_l.x
        
        face_width = best_right - best_left
        
        # Crop the head, and add a little bit the difference between eyes and nose.
        # When this increases, the face is in front of the camera, so the head seems biggest.
        # When the head is seen from above or below, the head is smaller, and so is the difference between the eyes and the nose.
        avg_eyes = (eye_r.y + eye_l.y) / 2
        nose_eyes_difference = nose.y - avg_eyes
        face_height = face_width*2 + nose_eyes_difference
        
        left = best_left
        top = nose.y - (face_height / 2) * 1.2
        right = best_right
        bottom = nose.y + (face_height / 2)
        
        
        
        
        # Round and normalize to the actual image pixels
        left_inner = int(round(left*image_w))
        top_inner = int(round(top*image_h))
        right_inner = int(round(right*image_w))
        bottom_inner = int(round(bottom*image_h))
        
        cv2.rectangle(image, (left_inner, top_inner), (right_inner, bottom_inner), (255, 255, 255), 4)


        width = right-left
        height = bottom - top

        FACE_MARGIN = 0.5
        # Add some margin:
        left = left - FACE_MARGIN * width
        top = top - FACE_MARGIN * height 
        right = right + FACE_MARGIN * width
        bottom = bottom + FACE_MARGIN * height
        

        # Round and normalize to the actual image pixels
        left = int(round(left*image_w))
        top = int(round(top*image_h))
        right = int(round(right*image_w))
        bottom = int(round(bottom*image_h))
        
        left = max(0, left)
        top = max(0, top)
        right = min(image_w, right)
        bottom = min(image_h, bottom)
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 4)
        
        return top, bottom, left, right
        