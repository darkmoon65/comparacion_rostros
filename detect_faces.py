
import cv2
import numpy as np
import time

class detect: 

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        #configuramos el backend para usar cuda
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detectAndSaveFaces(self, pathImg):
        
        # Set the GPU device for OpenCV
        cv2.cuda.setDevice(0)

        # Load the input image
        img = cv2.imread("./imagenes_prueba/"+str(pathImg))
        resized = cv2.resize(img, (300, 300))

        # Create a blob from the input image and preprocess it
        blob = cv2.dnn.blobFromImage(resized, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Seteamos el input para la red nueronal y clasificamos la imagen 
        self.net.setInput(blob)

        start_time = time.time()
        detections = self.net.forward()
        print("--- %s segundos ---" % (time.time() - start_time))

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
            
                #Extraemos el array de la imagen recortada (rostro)
                face_img_result = img[startY-50:endY+50, startX-50:endX+50]
                cv2.imwrite("./rostros_detectados/"+str(pathImg), face_img_result)
                
        return False
