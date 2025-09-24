from utilities.DeepLearningFoundationOperations import LogEmitter, Downloader, DownloadLogPopup
import os
from os.path import isfile, join
import time
import json
try:
    os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax", "torch"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    import keras
    from keras.preprocessing.image import load_img, save_img, img_to_array
    from keras.preprocessing import image
    from keras import callbacks
    from keras.callbacks import Callback
    from keras.utils import to_categorical
    from keras.models import Sequential , Model, load_model
    from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
    from keras.applications.imagenet_utils import preprocess_input
except:
    print("Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare2!")
try:
    import numpy as np
except:
    print("You Should Install numpy Library")
try:
    from PIL import Image
except:
    print("You Should Install PIL Library")
try:
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
except:
    print("You Should Install OpenCV-Python and cv2_enumerate_cameras Libraries")
try:
    import matplotlib.pyplot as plt
except:
    print("You Should Install matplotlib Library!")
try:
    from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt
    from PyQt6.QtWidgets import QMessageBox,QTextEdit, QWidget, QVBoxLayout, QPushButton, QLabel, QDialog, QTextEdit,QScrollArea
except:
    print("You Should Install PyQt6 Library!")

class FaceRecognitionOperation(QObject):
    def __init__(self,ImagesAndColorsHandler,DLOperationsHandler,parent=None):
        super().__init__()
        # Internal Variable to Access Data inside All Functions in the Class 
        self.DLOperationsHandler = DLOperationsHandler
        self.DLOperationsHandler.LoadModelDetails()
        self.models = self.DLOperationsHandler.models 
        self.ImagesAndColorsHandler = ImagesAndColorsHandler
        self.DownloadLogPopup = None
        self._is_running = False
        self.downloadResult = None
        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.Append_Log)       
        self.log_emitter.progressbar_signal.connect(self.Update_Progress)  
        self.log_emitter.finished_signal.connect(self.On_Finished)
        self.input_shape = (224,224, 3)
        self.model = None
        self.face_detector = None
        self.vgg_face_descriptor = None
        self.epsilon = 0.40
        self.faces_collection = dict()
                         
    # There are functions here for Creating a Simple CNN Model for Face Recognition
    # Find Comments and Explanation for each function related to ML and CV
    # UI functions do not have Comments because this is not a QT Training but they are Clear to Understand by its names and contents

    # Updating Logs After Download Finished
    def On_Finished(self, success, info ,modelType,filepath, imagePath,operationType):
        if not success:
            log = "Download Failed.\n" + str(info)
            if not "Download Cancelled" in str(info):
                log += "\nCheck Internet Connectivity!"
            
            self.DownloadLogPopup.Append_Log(log)
            return
                                
        else:
            self.DownloadLogPopup.Append_Log(str(info)+"\nDownload Complete.")   
            self.CreateLoadVGGFaceModel(filepath)              

    # Updating ProgressBar
    def Update_Progress(self, percent):
        if self._is_running:
           self.DownloadLogPopup.Update_Progress(percent)

    # Updating Logs
    def Append_Log(self,message):
        if self._is_running:
            self.DownloadLogPopup.Append_Log(message)

    # Create VGGFace Model | Load its Pre-Trained Weights
    def CreateLoadVGGFaceModel(self,filepath):
        # Create Sequential Model
        model = Sequential()
        # To display the summary of the model so far, include the current output shape
        # Start model by passing an Input object to the model, so it knows its input shape which is 28 x 28 x 1
        model.add(keras.Input(shape = self.input_shape ))        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Convolution2D(4096, (7, 7), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation('softmax'))

        self.model = model
        if os.path.exists(filepath):
            # From tensorflow.keras.models import model_from_json
            self.model.load_weights(filepath)
            self.vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
            QMessageBox.information(None,"Success","VGGFace Model Created, its Weights Loaded.")
        else:
            QMessageBox.warning(None,"Weights not Found","Failed Loading weights.")

    # Chheck if VGGFace Weights not exist then Download it
    def CheckVGGFaceModel(self):
        # Open Log Popup
        self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)   
        self.DownloadLogPopup.show()

        self._is_running = True  

        modelType   = "VGGFace"
        # Get Model Info
        if len(self.models) > 0 and self.models.get(modelType): 
            self.log_emitter.log_signal.emit("Checking for existing model file...")
                            
            url =  self.models[modelType]["url"] 
            filename = self.models[modelType]["name"] 
            fileSize = self.models[modelType]["size"] 
            expected_hash = self.models[modelType]["md5hash"] 

            expected_size = fileSize
            folder = os.path.normpath(join("resources","models"))
            filepath = os.path.join(folder, filename)
                
            # Only Download if File is Missing or File Size is Greater than Approximate Expected Size - Tolerance 
            # Hash Validation Is not Active to Accept Mirror Image of Models, Expected Size Validation is not Active for same Reason.
            if not os.path.exists(filepath) or not self.DLOperationsHandler.FileSize_Approximate_Validation("md5",modelType,filepath, expected_size,expected_hash,self.log_emitter,True):  
                self.log_emitter.log_signal.emit(filename + "\nModel file not found or  Size is not Valid! \nDownloading from internet...\n"+
                                                    "Make Sure your System Connected to the Internet\nFile is Approximately "+expected_size+"\n"+
                                                    "It takes a while Depending on the Speed of your System and Internet!\nDownload Url: \n" + url )
                if os.path.exists(filepath):
                    os.remove(filepath) 

                self.downloader = Downloader(url, filepath, modelType,"imagePath",self.log_emitter, expected_size,"operationType",self._is_running)
                self.DownloadLogPopup.Set_Downloader(self.downloader)
                self.downloader.Start()   
                
            else:
                self.log_emitter.log_signal.emit(filename + "\nModel file found locally.\n Hash and Size are not Validated in Config!\nLoading from cache...")                                  
                self.CreateLoadVGGFaceModel(filepath) 
        
        else:
            self.log_emitter.log_signal.emit("Error: 'models.json' not found or Not Contains Details for this Operation ( "+modelType+" ).\nPlease ensure the file exists in the root and contains Details for this Operation ( "+modelType+" ).")

    # Compare and Verify Similarity of 2 Faces
    def VerifySimilarity(self,face1,face2):
        if self.model is not None and self.vgg_face_descriptor is not None:
            plt.close()
            cv2.destroyAllWindows()
            img1_representation = self.vgg_face_descriptor.predict(self.preprocess_image('./resources/images/faces/%s' % (face1)))[0,:]
            img2_representation = self.vgg_face_descriptor.predict(self.preprocess_image('./resources/images/faces/%s' % (face2)))[0,:]
            
            cosine_similarity = self.findCosineSimilarity(img1_representation, img2_representation)
            
            f = plt.figure("Two Images for Comparing Similarity")
            f.add_subplot(1,2, 1)
            plt.imshow(image.load_img('./resources/images/faces/%s' % (face1)))
            plt.xticks([]); plt.yticks([])
            f.add_subplot(1,2, 2)
            plt.imshow(image.load_img('./resources/images/faces/%s' % (face2)))
            plt.xticks([]); plt.yticks([])
            plt.show(block=True)

            check = ""
            if(cosine_similarity < self.epsilon):
                check = "They are same person"
                QMessageBox.information(None,"Similarity Comparison Result:", "Cosine similarity: "+ str(cosine_similarity) + "\n" + check)
            else:
                check = "They are not same person!"
                QMessageBox.critical(None,"Similarity Comparison Result:", "Cosine similarity: "+ str(cosine_similarity) + "\n" + check)
        else:
            QMessageBox.warning(None,"Model not Found", "First, Create, Load the Model.")

    # Face Recognition on Camera
    def FaceRecognitionOnCamera(self):
        if os.path.exists("resources/haarcascades/haarcascade_frontalface_default.xml"):
            face_detector = cv2.CascadeClassifier('resources/haarcascades/haarcascade_frontalface_default.xml')
            self.face_detector = face_detector
            if self.model is not None and self.vgg_face_descriptor is not None:
                self.RetrieveFaceDataset()
                # Open Webcam
                self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.camera)
                while(True):
                    ret, img = self.ImagesAndColorsHandler.videoCapturer.read()
                    if not ret: break
                    if img is not None:
                        faces = self.face_detector.detectMultiScale(img, 1.3, 5)
                        if len(faces) > 0:
                            for (x,y,w,h) in faces:
                                    # Draw rectangle on Image
                                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                                    # Crop detected face
                                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
                                    # Resize detected face to 224x224
                                    detected_face = cv2.resize(detected_face, (224, 224))

                                    img_pixels = image.img_to_array(detected_face)
                                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                                    img_pixels /= 255

                                    captured_representation = self.vgg_face_descriptor.predict(img_pixels)[0,:]

                                    found = 0
                                    for i in self.faces_collection:
                                        person_name = i
                                        representation = self.faces_collection[i]

                                        similarity = self.findCosineSimilarity(representation, captured_representation)
                                        if(similarity < 0.35):
                                            cv2.putText(img, person_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                            found = 1
                                            break

                                    # Point Text to Face 
                                    cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255, 0, 0),1)
                                    cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255, 0, 0),1)
                                    # If image is not in face database
                                    if(found == 0): 
                                        cv2.putText(img, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        cv2.imshow('img',img)

                        if cv2.waitKey(1) in range(0,255):
                            break
                        
                self.ImagesAndColorsHandler.videoCapturer.release()
                cv2.destroyAllWindows()

            else:
                QMessageBox.warning(None,"Model not Found", "First, Create, Load the Model.")

        else:
            QMessageBox.warning(None,"Haarcascade not found","haarcascade_frontalface_default.xml File not found in: resources/haarcascades Path")
                
    # Face Recognition on Video
    def FaceRecognitionOnVideo(self,video):
        if os.path.exists("resources/haarcascades/haarcascade_frontalface_default.xml"):
            face_detector = cv2.CascadeClassifier('resources/haarcascades/haarcascade_frontalface_default.xml')
            self.face_detector = face_detector
            if self.model is not None and self.vgg_face_descriptor is not None:
                self.RetrieveFaceDataset()
                # Open Webcam
                self.ImagesAndColorsHandler.video = cv2.VideoCapture(video)
                frame_count = 0
                frame_rate = 5
                prev = 0
                while(True):
                        time_elapsed = time.time() - prev
                        ret, img = self.ImagesAndColorsHandler.video.read()
                        if not ret: break
                        if img is not None:
                            img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0,0,0])
                            # Do process for Frame Rate number in a Second to Spped up process
                            if time_elapsed >= 1./frame_rate:
                                prev = time.time()
                                faces = face_detector.detectMultiScale(img, 1.3, 5)
                                frame_count+=1
                                if len(faces) > 0:
                                    for (x,y,w,h) in faces:
                                        # Draw rectangle on Image
                                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
                                        # Crop detected face
                                        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                                        # Resize detected face to 224x224
                                        detected_face = cv2.resize(detected_face, (224, 224)) 

                                        img_pixels = image.img_to_array(detected_face)
                                        img_pixels = np.expand_dims(img_pixels, axis = 0)
                                        img_pixels /= 255

                                        captured_representation = self.vgg_face_descriptor.predict(img_pixels)[0,:]

                                        found = 0
                                        for i in self.faces_collection:
                                            person_name = i
                                            representation = self.faces_collection[i]

                                            similarity = self.findCosineSimilarity(representation, captured_representation)
                                            if(similarity < self.epsilon):
                                                cv2.putText(img, person_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                                found = 1
                                                break

                                        # Point Text to Face
                                        cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255, 0, 0),1)
                                        cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255, 0, 0),1)
                                        # If image is not in face database
                                        if(found == 0): 
                                            cv2.putText(img, 'UnKnown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                            break

                            cv2.imshow("Face",img)

                            if cv2.waitKey(1) in range(0,255): 
                                break
                           
                # Close openCV video capture
                self.ImagesAndColorsHandler.video.release()
                cv2.destroyAllWindows()

    # Retrieve Face Dataset
    def RetrieveFaceDataset(self):
        if len(self.faces_collection) == 0:
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)   
            self.DownloadLogPopup.show()
            self.Append_Log("Wait for loading Face Dataset.")
            for file in os.listdir("resources/images/faces"):
                self.Append_Log("Face: >> " + file + " << Added to the Dataset.")
                person_face, extension = file.split(".")
                self.faces_collection[person_face] = self.vgg_face_descriptor.predict(self.preprocess_image("./resources/images/faces/" + file))[0,:]

    # Pre Process on Image
    def preprocess_image(self,image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    # Find Similarity based on Cosine
    def findCosineSimilarity(self,source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
