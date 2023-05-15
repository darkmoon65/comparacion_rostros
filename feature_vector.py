
import cv2
import face_recognition

class feature: 

    def get_feature_vector(self, pathImg):
        image = cv2.imread(pathImg)
        
        vector = face_recognition.face_encodings(image, known_face_locations=None, num_jitters=1, model="small")
        return vector
