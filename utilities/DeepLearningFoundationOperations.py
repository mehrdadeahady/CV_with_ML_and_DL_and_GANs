# Import Essential Libraries
from utilities.ScrollableMessageBox import show_scrollable_message
import os
from os.path import isfile, join
import sys
import time
import threading
import urllib.request
import urllib.error
import hashlib
import json
import traceback
import random
import requests
from urllib3.exceptions import IncompleteRead
import tarfile
import gzip
import shutil
try:
    os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax", "torch"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    from keras.applications import VGG16
    from keras.applications import VGG19
    from keras.applications import ResNet50
    from keras.applications import InceptionV3
    from keras.applications import Xception
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg19 import VGG19
    from keras.applications.resnet50 import ResNet50
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.xception import Xception
    from keras.applications import imagenet_utils
    from keras.applications.imagenet_utils import preprocess_input
    from keras.preprocessing.image import img_to_array, load_img
    from keras.models import load_model
    from keras.utils import get_file
    import urllib.request   
except:
    print("Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare!")
try:
    import numpy as np
except:
    print("You Should Install numpy Library")
try:
    import cv2
except:
    print("You Should Install OpenCV-Python and cv2_enumerate_cameras Libraries")
try: 
    from  PyQt6.QtGui import  QTextCursor   
    from PyQt6.QtCore import QObject, pyqtSignal,QTimer, Qt, QThread,QUrl
    from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
    from PyQt6.QtWidgets import QScrollArea,QProgressBar,QMessageBox, QFileDialog, QApplication, QDialog, QVBoxLayout, QTextEdit, QPushButton,QWidget,QLabel
except:
    print("You Should Install PyQt6 Library!")
        
class DeepLearningFoundationOperations(QObject):
    def __init__(self,ImagesAndColorsHandler,CreateSimpleCNNHandler,parent=None):
        super().__init__()
        # Internal Variable to Access Images, Videos and Cameras inside All Functions in the Class 
        self.models = {} 
        self.accuracy = 50    
        self.ImagesAndColorsHandler = ImagesAndColorsHandler
        self.CreateSimpleCNNHandler = CreateSimpleCNNHandler
        self.DownloadLogPopup = None
        self._is_running = False
        self.downloadResult = None
        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.Append_Log)       
        self.log_emitter.progressbar_signal.connect(self.Update_Progress)  
        self.log_emitter.finished_signal.connect(self.On_Finished)
        self.LoadModelDetails()
        
    # Consider|Attention: 
    # Process Functions Contains Computer Vision Functions with Comments and Explanations
    # Rest of Functions are Pre-Processor and Helpers
    
################################################### Processor Functions:
    
    # Processing the Operation on Pre-Trained Model
    def ProcessImage(self,model,imagePath,newSize,processMode):
        # Show Model Architecture in A new Popup
        show_scrollable_message("Model Summary:", self.CreateSimpleCNNHandler.ModelSummaryCapture(model))
        self.log_emitter.log_signal.emit("Do not Close the Log Window.\nWait for Prediction Result:\n")
        # ***If Selected Image not Closed Bring it to the Top***
        # imageName = self.ImagesAndColorsHandler.imageName or self.ImagesAndColorsHandler.tempImageName
        # if imageName is not None:
        #     print(imageName, cv2.getWindowProperty(imageName, cv2.WND_PROP_VISIBLE))
        #     if cv2.getWindowProperty(imageName, cv2.WND_PROP_VISIBLE) >= 1:
        #         cv2.setWindowProperty(imageName, cv2.WND_PROP_TOPMOST, 1)   
               
        # Loading the Image to Predict
        img = load_img(imagePath)          
        # Resize the Image to X * Y Square Shape Required for Specific Pre-Trained Model
        img = img.resize(newSize)
        # Convert the Image to Array
        img_array = img_to_array(img)
        # Convert the Image into a 4 Dimensional Tensor
        # Convert from (Height, Width, Channels), (Batchsize, Height, Width, Channels)
        img_array = np.expand_dims(img_array, axis=0)
        '''                  
        The keras.applications.imagenet_utils.preprocess_input function is a utility designed to preprocess image data before it is fed into 
        Keras models that have been pre-trained on the ImageNet dataset. These models, such as VGG16, ResNet50, MobileNet, etc., 
        expect input images to be preprocessed in a specific way that matches the preprocessing applied during their original training.
        Functionality:
        This function takes a tensor or NumPy array representing a batch of images as input and applies transformations based on the specified mode. 
        The available modes are: 
            caffe:
            This mode converts images from RGB to BGR and then zero-centers each color channel with respect to the ImageNet dataset's channel means, 
            without scaling the pixel values.
            tf:
            This mode scales pixel values to be between -1 and 1, on a sample-wise basis.
            torch:
            This mode scales pixel values between 0 and 1 and then normalizes each channel using the ImageNet dataset's channel means and standard deviations.
        '''
        # Preprocess the Input Image Array
        img_array = imagenet_utils.preprocess_input(img_array, mode=processMode)
        '''                    
        The model.predict() method in machine learning is used to generate predictions from a trained model on new, unseen input data.
        Functionality:
            Input: It takes input data (often in the form of NumPy arrays or tf.data.Dataset objects in frameworks like TensorFlow/Keras) 
            that the model has not previously encountered during training.
        Processing: 
            The input data is passed through the layers of the trained model.
        Output: 
            It returns the model's output, which represents the predictions for the given input. The format of the output depends on the type of model: 
            Classification Models: 
                    For classification tasks, model.predict() often returns probabilities for each class (e.g., a vector of probabilities 
                    for a multi-class classification, or a single probability for binary classification if a sigmoid activation is used in the output layer)
            Regression Models: 
                    For regression tasks, it returns the predicted numerical values.
        Usage:
            The primary purpose of model.predict() is to apply a trained model to new data to obtain its predictions, enabling tasks such as forecasting, 
            classification of new instances, or generating recommendations.
        '''
        # Predict Using Predict() method (New Method)
        prediction = model.predict(img_array)
         # Loading the Image to Predict
        '''                    
        The keras.imagenet_utils.decode_predictions function is a utility within Keras (now primarily integrated with TensorFlow as tf.keras) 
        designed to interpret the raw predictions generated by models trained on the ImageNet dataset.
        Purpose:
        This function translates the numerical output (typically a 1000-dimensional vector of probabilities) from a pre-trained ImageNet model into 
        human-readable class labels and their corresponding confidence scores.
        Usage:
        The function takes two main arguments:

            preds:
            A NumPy array representing a batch of predictions from an ImageNet-trained model. This array should have a shape of (samples, 1000), 
            where samples is the number of images in the batch and 1000 corresponds to the 1000 ImageNet classes.
            top:
            An optional integer specifying how many top-ranking predictions (classes) to return for each sample. The default value is typically 5. 

        Output:
        It returns a list of lists. Each inner list corresponds to a sample in the input batch and contains a list of tuples. Each tuple represents 
        a top prediction and consists of:

            class_name: The ImageNet class identifier (e.g., 'n02129165').
            class_description: A human-readable description of the class (e.g., 'lion').
            score: The confidence score (probability) assigned to that class by the model.
        '''
        # Decode the Prediction
        actual_prediction = imagenet_utils.decode_predictions(prediction)
        # Display the Result of Prediction in a Window on Top of Image
        if self.DownloadLogPopup:
           self.log_emitter.log_signal.emit("***********************\nDetection Result\n\nPredicted Object: \t" + str(actual_prediction[0][0][1]).title() + "\nStated Accuracy: \t" + str(actual_prediction[0][0][2]*100) +"\n***********************")
        # Display the Result of Prediction in a Log Window
        msgBox = QMessageBox(parent=None)
        msgBox.setWindowTitle("Detection Result")
        msgBox.setText("Predicted Object: \t" + str(actual_prediction[0][0][1]).title() + "\nStated Accuracy: \t" + str(actual_prediction[0][0][2]*100))
        msgBox.setWindowFlags(msgBox.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        msgBox.exec()
       
    # Processing the Operation on Mobilenet SSD Pre-Trained Model   
    def ProcessMobilenetSSD(self,img_to_detect,mobilenetssd,class_labels):
        # Get width, height of Image 
        img_height , img_width = img_to_detect.shape[0:2]

        # Resize to Match Input Size
        resized_img_to_detect = cv2.resize(img_to_detect,(300,300))

        # Convert to Blob to Pass into Model
        # Recommended Scale Factor is 0.007843, width,height of blob is 300,300, mean of 255 is 127.5
        img_blob = cv2.dnn.blobFromImage(resized_img_to_detect,0.007843,(300,300),127.5)

        # Pass Blob into Model
        mobilenetssd.setInput(img_blob)

        '''
        The mobilenetssd.forward() function is typically used in object detection pipelines involving the MobileNet-SSD model. Here's a breakdown of what it does and how it's used:
        🧠 What forward() Does
        - It performs inference: This method runs a forward pass through the MobileNet-SSD neural network.
        - It returns detection results: These are usually bounding boxes, class labels, and confidence scores for objects detected in the input image.
        
        🛠️ Typical Usage in Python (OpenCV + Caffe)
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        📦 Output Format
        The detections variable is usually a 4D array with shape [1, 1, N, 7], where:
        - N is the number of detected objects
        - Each detection has 7 values: [image_id, label, confidence, x_min, y_min, x_max, y_max]
        
        🔍 Example Interpretation
        You can loop through the detections like this:
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                # Draw box or label

        If you're using a different framework like PyTorch or TensorFlow, the method and output might vary slightly. Want help adapting this to your specific setup?
        '''
        obj_detections = mobilenetssd.forward()

        # Returned obj_detections[0, 0, index, 1]:
        # 1 => will have the Prediction Class Index
        # 2 => will have Confidence
        # 3 to 7 => will have the Bounding Box Co-Ordinates
        no_of_detections = obj_detections.shape[2]

        # loop over the Detections
        for index in np.arange(0, no_of_detections):

            prediction_confidence = obj_detections[0, 0, index, 2]

            # Take only Predictions with Confidence more than 
            if prediction_confidence > self.accuracy:

                # Get the Predicted Label
                predicted_class_index = int(obj_detections[0, 0, index, 1])
                predicted_class_label = class_labels[predicted_class_index]     

                # Obtain the Bounding Box Co-Oridnates for Actual Image from Resized Image Size
                bounding_box = obj_detections[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
                (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")

                # Create Prediction Label
                predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)

                # Display the Result of Prediction in Log Window if not Closed
                if self.DownloadLogPopup:    
                   self.log_emitter.log_signal.emit("predicted object {}: {} \t Stated Accuracy: {}".format(index +1 ,class_labels[predicted_class_index], prediction_confidence * 100) )           
                
                # Draw Rectangle Around Detected Object in the Image
                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0,255,0), 2)
               
                # Put the Result of Prediction as Text on Detected Object in the Image
                cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Display the Image
        cv2.imshow("Detection Output", img_to_detect)
                
     # Processing the Operation on Mobilenet SSD Pre-Trained Model   
    
    # Processing the Operation on All YOLO Pre-Trained Model
    def ProcessYOLO(self,img_to_detect,model,class_labels,class_colors,yolo_output_layer):
        # Get width, height of Image 
        img_height , img_width = img_to_detect.shape[0:2]

        # Convert to Blob to Pass into Model
        # Recommended by yolo Authors: 
        # Scale Factor is 0.003922=1/255, 
        # width,height of blob is 320,320
        # Accepted sizes are 320×320, 416×416, 608×608. Bigger size means Higher Accuracy but Smaller Speed
        img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)
        
        # Pass Blob into Model
        model.setInput(img_blob)
        '''
        The MODEL.forward method refers to the forward pass of the TinyYOLO neural network model—typically implemented in frameworks like PyTorch or TensorFlow. 
        This method defines how input data (usually an image) flows through the network to produce predictions.
        🧠 What happens in forward?
        - The input image is passed through a series of convolutional layers, activation functions, and pooling layers.
        - Intermediate feature maps are generated and refined.
        - The final output is a set of bounding boxes, class probabilities, and confidence scores for detected objects.
        📦 In PyTorch, for example:
        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            ...
            return detections
        🔍 Output format
        - A tensor containing object detection results
        - Each detection includes coordinates (x, y, width, height), objectness score, and class probabilities
        '''
        # Obtain the Detection Layers by Forwarding Model through till the Output Layer
        obj_detection_layers = model.forward(yolo_output_layer)

        # Loop over each of the layer outputs
        for indexA, object_detection_layer in enumerate(obj_detection_layers):
            # Loop over the detections
            for indexB,object_detection in enumerate(object_detection_layer):
                
                # obj_detections[1 to 4] => will have the two center points, box width and box height
                # obj_detections[5] => will have scores for all objects within bounding box
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]
            
                # Take only Predictions with Confidence Above
                if prediction_confidence > self.accuracy:

                    # Get the predicted label
                    predicted_class_label = class_labels[predicted_class_id]

                    # Obtain the bounding box co-oridnates for actual image from resized image size
                    bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                    (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                    start_x_pt = int(box_center_x_pt - (box_width / 2))
                    start_y_pt = int(box_center_y_pt - (box_height / 2))
                    end_x_pt = start_x_pt + box_width
                    end_y_pt = start_y_pt + box_height
                    
                    # Get a random mask color from the numpy array of colors
                    box_color = class_colors[predicted_class_id]
                    
                    # Convert the color numpy array as a list and apply to text and box
                    box_color = [int(c) for c in box_color]

                    # Create Prediction Label
                    predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                             
                    # Display the Result of Prediction in Log Window if not Closed
                    if self.DownloadLogPopup:    
                       self.log_emitter.log_signal.emit("predicted object {}: {} \t Stated Accuracy: {}".format( (indexA,indexB) ,predicted_class_label, prediction_confidence * 100) )           
                    
                    # Draw Rectangle Around Detected Object in the Image
                    cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
                
                    # Put the Result of Prediction as Text on Detected Object in the Image
                    cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # Display the Image
        cv2.imshow("Detection Output", img_to_detect)     
    
    # Processing the Operation on Optimizing YOLO Pre-Trained Model
    def ProcessOptimizedYOLO(self,img_to_detect,model,class_labels,class_colors,yolo_output_layer):
        # Get width, height of Image 
        img_height , img_width = img_to_detect.shape[0:2]

        # Convert to Blob to Pass into Model
        # Recommended by yolo Authors: 
        # Scale Factor is 0.003922=1/255, 
        # width,height of blob is 320,320
        # Accepted sizes are 320×320, 416×416, 608×608. Bigger size means Higher Accuracy but Smaller Speed
        img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)
        
        # Pass Blob into Model
        model.setInput(img_blob)
        '''
        The MODEL.forward method refers to the forward pass of the TinyYOLO neural network model—typically implemented in frameworks like PyTorch or TensorFlow. 
        This method defines how input data (usually an image) flows through the network to produce predictions.
        🧠 What happens in forward?
        - The input image is passed through a series of convolutional layers, activation functions, and pooling layers.
        - Intermediate feature maps are generated and refined.
        - The final output is a set of bounding boxes, class probabilities, and confidence scores for detected objects.
        📦 In PyTorch, for example:
        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            ...
            return detections
        🔍 Output format
        - A tensor containing object detection results
        - Each detection includes coordinates (x, y, width, height), objectness score, and class probabilities
        '''
        # Obtain the Detection Layers by Forwarding Model through till the Output Layer
        obj_detection_layers = model.forward(yolo_output_layer)

        ############## NMS Change 1 ###############
        # Initialization for non-max Suppression (NMS)
        # Declare List for [class id], [box center, width & height[], [confidences]
        class_ids_list = []
        boxes_list = []
        confidences_list = []
        ############## NMS Change 1 END ###########

        # Loop over each of the Layer Outputs
        for indexA, object_detection_layer in enumerate(obj_detection_layers):
            # Loop over the Detections
            for indexB,object_detection in enumerate(object_detection_layer):
                
                # obj_detections[1 to 4] => will have the two center points, box width and box height
                # obj_detections[5] => will have scores for all objects within bounding box
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]
            
                # Take only Predictions with Confidence Above 
                if prediction_confidence > self.accuracy:

                    # Get the Predicted Label
                    predicted_class_label = class_labels[predicted_class_id]

                    # Obtain the bounding box co-oridnates for actual image from resized image size
                    bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                    (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                    start_x_pt = int(box_center_x_pt - (box_width / 2))
                    start_y_pt = int(box_center_y_pt - (box_height / 2))
                    
                    ############## NMS Change 2 ###############
                    # Save class id, start x, y, width & height, confidences in a list for nms processing
                    # Make sure to pass confidence as float and width and height as integers
                    class_ids_list.append(predicted_class_id)
                    confidences_list.append(float(prediction_confidence))
                    boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])
                    ############## NMS Change 2 END ###########

        ############## NMS Change 3 ###############
        # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
        # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
        max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

        # Loop through the final set of detections remaining after NMS and draw bounding box and write text
        for max_valueid in max_value_ids:
            max_class_id = max_valueid#[0]
            box = boxes_list[max_class_id]
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]
            
            #get the predicted class id and label
            predicted_class_id = class_ids_list[max_class_id]
            predicted_class_label = class_labels[predicted_class_id]
            prediction_confidence = confidences_list[max_class_id]
        ############## NMS Change 3 END ###########        
            
            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height
            
            # Get a random mask color from the numpy array of colors
            box_color = class_colors[predicted_class_id]
            
            # Convert the color numpy array as a list and apply to text and box
            box_color = [int(c) for c in box_color]

            # Create Prediction Label
            predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                    
            # Display the Result of Prediction in Log Window if not Closed
            if self.DownloadLogPopup:    
                self.log_emitter.log_signal.emit("predicted object {}: {} \t Stated Accuracy: {}".format( (indexA,indexB) ,predicted_class_label, prediction_confidence * 100) )           
            
            # Draw Rectangle Around Detected Object in the Image
            cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        
            # Put the Result of Prediction as Text on Detected Object in the Image
            cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Display the Image
        cv2.imshow("Detection Output", img_to_detect) 

    # Processing the Operation on Mobilenet SSD Pre-Trained Model 
    def ProcessMaskRCNN(self,img_to_detect,maskrcnn,class_labels):

        # Resize Image to Smaller to Speed up Detection but Decrease Accuracy
        # img_to_detect = cv2.resize(img_to_detect, (640, 480))  # or even (320, 240)

        # Get width, height of Image 
        img_height , img_width = img_to_detect.shape[0:2]

        # Convert to Blob to Pass into Model
        # Swap BGR to RGB without Cropping
        img_blob = cv2.dnn.blobFromImage(img_to_detect,swapRB=True,crop=False)
        
        # Pass Blob into Model
        maskrcnn.setInput(img_blob)
        
        '''
        maskrcnn.forward refers to the forward pass function of a Mask R-CNN model, used to compute predictions from input data.
        🔍 What It Does
        The forward method:
        - Takes an input image (or batch of images)
        - Passes it through the Mask R-CNN network
        - Returns outputs such as:
        - Region Proposal Network (RPN) scores
        - Bounding box predictions
        - Class scores
        - Segmentation masks
        🧪 Example (PyTorch)
        In PyTorch, if you're using torchvision.models.detection.maskrcnn_resnet50_fpn, the forward pass looks like:
        outputs = model(images)
        Internally, this calls forward() and returns a list of dictionaries with keys like 'boxes', 'labels', and 'masks'.
        🧠 MATLAB Variant
        In MATLAB, maskrcnn.forward(detector, dlX) returns multiple outputs:
        [dlRPNScores, dlRPNReg, dlProposals, dlBoxScores, dlBoxReg, dlMasks] = forward(detector, dlX)
        This is part of the Computer Vision Toolbox for Mask R-CNN.
        '''
        (obj_detections_boxes,obj_detections_masks)  = maskrcnn.forward(["detection_out_final","detection_masks"])

        # Returned obj_detections[0, 0, index, 1]:
        # 1 => will have the Prediction Class Index
        # 2 => will have Confidence
        # 3 to 7 => will have the Bounding Box Co-Ordinates
        no_of_detections = obj_detections_boxes.shape[2]

         # loop over the Detections
        for index in np.arange(0, no_of_detections):

            prediction_confidence = obj_detections_boxes[0, 0, index, 2]

            # Take only Predictions with Confidence more than
            if prediction_confidence > self.accuracy:

                # Get the Predicted Label
                predicted_class_index = int(obj_detections_boxes[0, 0, index, 1])
                predicted_class_label = class_labels[predicted_class_index]     

                # Obtain the Bounding Box Co-Oridnates for Actual Image from Resized Image Size
                bounding_box = obj_detections_boxes[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
                (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")

                # Create Prediction Label
                predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)

                # Display the Result of Prediction in Log Window if not Closed
                if self.DownloadLogPopup:    
                   self.log_emitter.log_signal.emit("predicted object {}: {} \t Stated Accuracy: {}".format(index +1 ,class_labels[predicted_class_index], prediction_confidence * 100) )           
                
                # Draw Rectangle Around Detected Object in the Image
                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0,255,0), 2)
               
                # Put the Result of Prediction as Text on Detected Object in the Image
                cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # Display the Image
        cv2.imshow("Detection Output", img_to_detect)

################################################### Pre-Processor Functions:

    # Loading Mobilenet SSD Pre-Trained Model
    def PreProcessMobilenetSSD(self,imagePath,filepath,MobileNetSSD_Prototext_Path,operationType):
        '''
        The MobileNetSSD Caffe model is a lightweight deep learning model designed for real-time object detection. 
        It combines the efficient MobileNet architecture with the SSD (Single Shot MultiBox Detector) framework,
          making it ideal for deployment on devices with limited computational power like Raspberry Pi or mobile platforms.
        🔍 Key Features
        - MobileNet: Uses depthwise separable convolutions to reduce computation.
        - SSD: Detects objects in a single forward pass, enabling fast inference.
        - Caffe Format: Compatible with the Caffe deep learning framework, often used in embedded systems.
        📦 Common Files
        - MobileNetSSD_deploy.caffemodel: Pre-trained weights.
        - MobileNetSSD_deploy.prototxt: Network architecture definition.
        🛠️ Use Cases
        - Real-time object detection on edge devices.
        - Integration with OpenVINO for Intel hardware acceleration.
        - Robotics, surveillance, and smart cameras.
        '''
        MobileNetSSD_CaffeModel = filepath
        '''
        📄 MobileNetSSD_Prototext
        The MobileNetSSD deploy.prototxt file—sometimes casually referred to as "MobileNetSSD_Prototext" is the architectural script that defines 
        how the MobileNetSSD model operates within the Caffe deep learning framework. It outlines the neural network's structure, layer by layer, 
        and is essential for pairing with the trained weights (.caffemodel) during inference.
        🔧 Key Components
        - Input Layer: Accepts images, typically sized 300x300x3 (RGB).
        - Depthwise Separable Convolutions: Core of MobileNet's efficiency, reducing computation.
        - BatchNorm & ReLU: Stabilizes and activates the network after each convolution.
        - SSD Detection Layers: Multi-scale feature maps for detecting objects of various sizes.
        - PriorBox Layers: Defines anchor boxes for bounding box predictions.
        - Softmax & DetectionOutput: Final layers that classify objects and output bounding boxes.

        📦 Common Usage
        - Used with MobileNetSSD_deploy.caffemodel for object detection tasks.
        - Compatible with OpenCV's dnn module for fast inference.
        - Can be customized for different input sizes or object classes.

        🛠️ Example Applications
        - Real-time detection on mobile and embedded platforms.
        - Edge AI deployments using OpenVINO or TensorRT.
        - Robotics, smart cameras, and IoT vision systems.
        '''
        # Cleaning MobileNetSSD_Prototext File
        MobileNetSSD_Prototext = MobileNetSSD_Prototext_Path
        '''
        cv2.dnn.readNetFromCaffe is a function in OpenCV's Deep Neural Network (DNN) module that allows you to load a pre-trained Caffe model for inference.
        🧠 What It Does
        This function reads:
        - A .prototxt file: Defines the model architecture (layers, connections, etc.)
        - A .caffemodel file: Contains the trained weights
        Together, these files represent a complete neural network that can be used to make predictions on new data.
        🧪 Syntax
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        📦 Example Use Case
        Suppose you want to run object detection using a MobileNet SSD model trained in Caffe:
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet.caffemodel")

        Once loaded, you can pass input images to the network using net.forward() after preprocessing them with cv2.dnn.blobFromImage.
        🔍 Why Use It?
        - It's lightweight and fast for inference
        - Great for deploying models without needing full deep learning frameworks
        - Compatible with CPU and GPU backends in OpenCV
        '''
        # Loading Pre-Trained Model from Prototext and Caffemodel files
        mobilenetssd = cv2.dnn.readNetFromCaffe(MobileNetSSD_Prototext, MobileNetSSD_CaffeModel) 
        
        self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")

        # Set of 21 Class Labels in Alphabetical Order
        class_labels = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
        
        cv2.destroyAllWindows()

        if self.DownloadLogPopup:
           self.log_emitter.log_signal.emit("***********************\nDetection Results\n")

        img_to_detect = None

        match operationType:
            case "Images":
                # Load the Image to Detect
                img_to_detect = cv2.imread(imagePath)
                self.ProcessMobilenetSSD(img_to_detect,mobilenetssd,class_labels)
                
            case "Pre-Saved":
                # Get the Saved Video File as Stream
                self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.video)

                # Create a While Loop until Video Still Streaming
                while (self.ImagesAndColorsHandler.videoCapturer.isOpened):
                    # Wait for Pressing a Keyboard Key to Exit
                    if cv2.waitKey(1) in range(0,255):
                       break
                    # Get the Current Frame from Video Stream
                    ret,current_frame = self.ImagesAndColorsHandler.videoCapturer.read()
                    # Use Video Current Frame instead of Image
                    if current_frame is not None and len(current_frame.shape) > 1:
                        self.ProcessMobilenetSSD(current_frame,mobilenetssd,class_labels)             
                
                self.ImagesAndColorsHandler.videoCapturer.release()

            case "Realtime":
                # Get the Camera Video File as Stream
                self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.camera)

                # Create a While Loop until Camera Still is Open and Video is Streaming
                while True:
                    # Wait for Pressing a Keyboard Key to Exit
                    if cv2.waitKey(1) in range(0,255):
                       break
                    # Get the Current Frame from Camera Video Stream
                    ret,current_frame = self.ImagesAndColorsHandler.videoCapturer.read()
                    # Use Video Current Frame instead of Image
                    if current_frame is not None and len(current_frame.shape) > 1:
                        self.ProcessMobilenetSSD(current_frame,mobilenetssd,class_labels)

                self.ImagesAndColorsHandler.videoCapturer.release()
                          
        if self.DownloadLogPopup:            
            self.log_emitter.log_signal.emit("\n***********************")
            
    # Loading MaskRCNN Pre-Trained Model
    def PreProcessMaskRCNN(self,imagePath,filepath,MaskRCNN_Pbtxt_Path,operationType):
        '''
        📁 maskrcnn_buffermodel.pb — What It Is
        This file is likely a TensorFlow frozen model for Mask R-CNN, saved in Protocol Buffers (.pb) format. It contains:
        - The graph definition (i.e. the structure of the neural network)
        - The trained weights for inference
        - Everything needed to run the model on new images (but not to train it further)
        🧩 Why “BufferModel”?
        The name buffermodel suggests it might be:
        - A preprocessed or optimized version of the original model
        - Possibly used in streaming or buffered inference, where input data is processed in chunks
        - Or simply a naming convention used by the developers
        🛠️ Typical Use Case
        You'd use this .pb file to:
        - Load the model in TensorFlow (usually with tf.import_graph_def)
        - Run inference on images to detect and segment objects
        - Integrate it into an application for real-time or batch image analysis
        '''
        MaskRCNN_BufferModel = filepath
        '''
        MaskRCNN_BufferConfig.pbtxt is a text-based configuration file used with OpenCV's Deep Neural Network (DNN) module 
        to define the structure of a Mask R-CNN model. It's written in Protocol Buffers Text Format (pbtxt), 
        which is a human-readable version of the binary .pb model file.
        📄 What's Inside MaskRCNN.pbtxt?
        This file contains:
        - Layer definitions: Each node (layer) in the neural network, including its type (e.g., Conv2D, Relu, MaxPool)
        - Input/output specifications: Like image_tensor as the input placeholder
        - Operations and parameters: Such as strides, padding, dilation, and data formats
        It's essentially a blueprint that tells OpenCV how to interpret and run the frozen .pb model file.
        🧠 Why It's Needed
        When using OpenCV's cv2.dnn.readNetFromTensorflow, you typically need:
        net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "mask_rcnn.pbtxt")

        - The .pb file contains the trained weights and graph
        - The .pbtxt file describes the graph structure in a readable format
        🔍 Example Use Case
        You might use it for:
        - Object detection and instance segmentation in images or video
        - Running inference with OpenCV without needing TensorFlow directly
        '''
         # Cleaning MaskRCNN_BufferConfig File
        MaskRCNN_BufferConfig = MaskRCNN_Pbtxt_Path
        '''
        cv2.dnn.readNetFromTensorflow is an OpenCV function used to load a TensorFlow model for inference using the Deep Neural Network (DNN) module.
        🧠 What It Does
        It reads:
        - A .pb file: The frozen TensorFlow model containing the graph and weights.
        - An optional .pbtxt file: A text version of the graph structure, often required for complex models like Mask R-CNN.
        🧪 Syntax
        net = cv2.dnn.readNetFromTensorflow(modelPath, configPath)
        - modelPath: Path to the .pb file.
        - configPath: Path to the .pbtxt file (optional but often needed).
        📦 Example
        net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "mask_rcnn.pbtxt")
        This loads the model into OpenCV so you can run inference on images using net.forward() after preparing input with cv2.dnn.blobFromImage.
        '''
        # Loading Pre-Trained Model from BufferConfig and BufferModel files
        maskrcnn = cv2.dnn.readNetFromTensorflow(MaskRCNN_BufferModel,MaskRCNN_BufferConfig)

        self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")

        # Set of 90 Class Labels in Predefined Order
        class_labels = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
                        "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse",
                        "sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses",
                        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
                        "skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife",
                        "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
                        "cake","chair","sofa","pottedplant","bed","mirror","diningtable","window","desk","toilet","door","tv",
                        "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
                        "blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
        
        cv2.destroyAllWindows()

        if self.DownloadLogPopup:
           self.log_emitter.log_signal.emit("***********************\nDetection Results\n")

        img_to_detect = None

        match operationType:
            case  "BoundingBoxImages" | "ObjectMaskImages":
                # Load the Image to Detect
                img_to_detect = cv2.imread(imagePath)
                self.ProcessMaskRCNN(img_to_detect,maskrcnn,class_labels)
                
            case "Pre-Saved":
                # Get the Saved Video File as Stream
                self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.video)

                # Create a While Loop until Video Still Streaming
                while (self.ImagesAndColorsHandler.videoCapturer.isOpened):
                    # Wait for Pressing a Keyboard Key to Exit
                    if cv2.waitKey(1) in range(0,255):
                       break
                    # Get the Current Frame from Video Stream
                    ret,current_frame = self.ImagesAndColorsHandler.videoCapturer.read()
                    # Use Video Current Frame instead of Image
                    if current_frame is not None and len(current_frame.shape) > 1:
                        self.ProcessMaskRCNN(current_frame,maskrcnn,class_labels) 

                self.ImagesAndColorsHandler.videoCapturer.release()            

            case "Realtime":
                # Get the Camera Video File as Stream
                self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.camera)

                # Create a While Loop until Camera Still is Open and Video is Streaming
                while True:
                    # Wait for Pressing a Keyboard Key to Exit
                    if cv2.waitKey(1) in range(0,255):
                       break
                    # Get the Current Frame from Camera Video Stream
                    ret,current_frame = self.ImagesAndColorsHandler.videoCapturer.read()
                    # Use Video Current Frame instead of Image
                    if current_frame is not None and len(current_frame.shape) > 1:
                        self.ProcessMaskRCNN(current_frame,maskrcnn,class_labels)

                self.ImagesAndColorsHandler.videoCapturer.release()
                          
        if self.DownloadLogPopup:            
            self.log_emitter.log_signal.emit("\n***********************")

     # Loading MaskRCNN Pre-Trained Model

    # Loading All YOLO Pre-Trained Model
    def PreProcessAllYOLOModels(self,imagePath,filepath,CFG_Path,operationType,modelType=""):
        '''
        The .weights file contains the pre-trained weights for the YOLOv4-Tiny object detection model. 
        Think of it as the "learned knowledge" from training the model on a large dataset—typically the COCO dataset, 
        which includes 80 common object classes like people, cars, dogs, and more.
        🔍 What's inside?
        - Numerical values for each layer in the neural network (e.g., convolutional filters, biases)
        - These values were learned during training and are used to make predictions on new images
        ⚙️ How is it used?
        - Paired with the .cfg file (which defines the model architecture)
        - Loaded into frameworks like Darknet, OpenCV, or TensorFlow to perform real-time object detection
        - Enables fast inference on devices with limited computing power (e.g., Raspberry Pi, Jetson Nano)
        🧠 Why it matters Without the .weights file, the model would be like a brain with no memories—it knows how to process data, 
        but not what to look for. The weights give it the ability to recognize patterns and objects.
        If you're planning to train your own model, you can start with these weights and fine-tune them on your custom dataset—a process called transfer learning. 
        '''
        Weights = filepath
        '''
        The .cfg file is a configuration file used in the YOLO (You Only Look Once) object detection framework, 
        specifically for the YOLOv4-Tiny model. Here's what it does:
        🔧 Purpose
        It defines the architecture of the neural network—layer by layer—including:
        - Input dimensions (e.g. width, height, channels)
        - Convolutional layers and their parameters (filters, kernel size, stride)
        - Activation functions (like Leaky ReLU)
        - YOLO detection layers and anchor boxes
        ⚡ Why “Tiny”?
        YOLOv4-Tiny is a lightweight version of YOLOv4, optimized for speed and efficiency. It's ideal for:
        - Real-time detection on edge devices (like Raspberry Pi or mobile)
        - Scenarios where computational resources are limited
        📁 Usage
        This .cfg file is typically used with:
        - .weights (pre-trained weights)
        - A class names file (e.g. coco.names)
        - Frameworks like Darknet, OpenCV, or PyTorch implementations
        '''
         # Cleaning MaskRCNN_BufferConfig File
        Config = CFG_Path
        '''
       🧠 cv2.dnn.readNetFromDarknet() - Definition & Explanation
        🔍 What It Is:
        cv2.dnn.readNetFromDarknet() is a function in OpenCV's Deep Neural Network (DNN) module that loads a YOLO model trained using the Darknet framework.
          It reads the model architecture from a .cfg file and the learned weights from a .weights file.

        🧩 Parameters:
        - Config: Path to the .cfg file
        This file defines the structure of the neural network—how many layers, what types (convolutional, pooling, etc.), and how they're connected.
        - Weights: Path to the .weights file
        This file contains the trained parameters of the model—what the network has learned during training.

        ⚙️ What It Does:
        - Parses the .cfg file to build the network architecture.
        - Loads the .weights file to initialize the model with pretrained values.
        - Returns a cv2.dnn.Net object that can be used for inference, such as detecting objects in images or video frames.

        ✅ Why It's Useful:
        - Enables real-time object detection using YOLO directly in OpenCV.
        - No need for external frameworks like PyTorch or TensorFlow.
        - Works seamlessly with OpenCV's image and video processing tools.
       '''
        # Loading Pretrained Model from:
        # TinyYOLO_Weights or YOLO_Weights or OptimizedYOLO_Weights and 
        # TinyYOLO_Config files or YOLO_Config  or OptimizedYOLO_Config 
        model = cv2.dnn.readNetFromDarknet(Config,Weights)

        self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")

        # Set of 80 Class Labels in Pre-Defined Order
        class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                        "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                        "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                        "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                        "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                        "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                        "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                        "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]   
        
        # Declare List of Colors as an Array: Green, Blue, Red, cyan, yellow, purple      
        # Split by seperator ',' and for each split, Change Type to int
        # Convert that to a numpy Array to apply Color Mask to the Image numpy Array
        class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
        class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
        class_colors = np.array(class_colors)
        class_colors = np.tile(class_colors,(16,1))
        
        # Get all Layers from the yolo Network
        # Loop and find the last Layer (Output Layer) of the yolo Network 
        yolo_layers = model.getLayerNames()

        # It returns the layer numbers (not names) of the network’s output layers. 
        # These are the layers where model generates its predictions—bounding boxes, class scores, and confidence values.
        unconnected_out_layers = model.getUnconnectedOutLayers()

        yolo_output_layer = None

        # Check Old Version to Retrieve yolo_output_layer or New Version
        if isinstance(unconnected_out_layers[0], (list, tuple, np.ndarray)):
            yolo_output_layer = [yolo_layers[i[0] - 1] for i in unconnected_out_layers]
        else:
            yolo_output_layer = [yolo_layers[i - 1] for i in unconnected_out_layers]

        cv2.destroyAllWindows()

        if self.DownloadLogPopup:
           self.log_emitter.log_signal.emit("***********************\nDetection Results\n")

        img_to_detect = None

        match operationType:
            case  "Images":
                # Load the Image to Detect
                img_to_detect = cv2.imread(imagePath)

                # Process on Image
                if modelType == "OptimizedYOLO":
                     self.ProcessOptimizedYOLO(img_to_detect,model,class_labels,class_colors,yolo_output_layer)
                else:
                     self.ProcessYOLO(img_to_detect,model,class_labels,class_colors,yolo_output_layer)
                
            case "Pre-Saved":
                # Get the Saved Video File as Stream
                self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.video)

                # Create a While Loop until Video Still Streaming
                while (self.ImagesAndColorsHandler.videoCapturer.isOpened):                    
                    # Get the Current Frame from Video Stream
                    ret,current_frame = self.ImagesAndColorsHandler.videoCapturer.read()
                    # Use Video Current Frame instead of Image
                    if current_frame is not None and len(current_frame.shape) > 1:
                        # Process on Image
                        if modelType == "OptimizedYOLO":
                            self.ProcessOptimizedYOLO(current_frame,model,class_labels,class_colors,yolo_output_layer)
                        else:
                            self.ProcessYOLO(current_frame,model,class_labels,class_colors,yolo_output_layer)

                    # Wait for Pressing a Keyboard Key to Exit
                    if cv2.waitKey(1) in range(0,255):
                       break

                self.ImagesAndColorsHandler.videoCapturer.release()            

            case "Realtime":
                # Get the Camera Video File as Stream
                self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.camera)

                # Create a While Loop until Camera Still is Open and Video is Streaming
                while True:                 
                    # Get the Current Frame from Camera Video Stream
                    ret,current_frame = self.ImagesAndColorsHandler.videoCapturer.read()
                    # Use Video Current Frame instead of Image
                    if current_frame is not None and len(current_frame.shape) > 1:
                        # Process on Image
                        if modelType == "OptimizedYOLO":
                            self.ProcessOptimizedYOLO(current_frame,model,class_labels,class_colors,yolo_output_layer)
                        else:
                            self.ProcessYOLO(current_frame,model,class_labels,class_colors,yolo_output_layer)

                    # Wait for Pressing a Keyboard Key to Exit
                    if cv2.waitKey(1) in range(0,255):
                       break

                self.ImagesAndColorsHandler.videoCapturer.release()
                          
        if self.DownloadLogPopup:            
           self.log_emitter.log_signal.emit("\n***********************")

    # Loading Downloaded or Existing Pre-Trained Model
    def Loading_Model_Operation(self,modelType, filepath, imagePath, operationType):
            self.log_emitter.log_signal.emit("Loading model weights...")
            match modelType:
                case "VGGNet16":
                    # This is a command for Download/Load with Tracing only in Console Not UI
                    # model = VGG16(weights="imagenet") 
                    '''             
                    model.load_weights() is a function used in deep learning frameworks like Keras and TensorFlow to load pre-trained weights into a neural network model. 
                    This function is particularly useful in scenarios such as: 

                    Resuming training:
                    If training is interrupted or needs to be continued from a specific point, load_weights() allows the model to pick up where it left off without starting from scratch.
                    Transfer learning:
                    Pre-trained weights from a model trained on a large dataset can be loaded into a new model, which is then fine-tuned on a smaller, related dataset, leveraging the learned features.
                    Deployment for inference:
                    Once a model is trained, its weights can be saved and then loaded into a new model instance for making predictions without needing to re-train. 

                    Key considerations when using model.load_weights():

                    Architectural compatibility:
                    The model receiving the weights must have the exact same architecture (layer types, order, and configurations) as the model from which the weights were saved. TensorFlow and Keras match weights based on layer order and naming.
                    File format:
                    Weights are typically saved in formats like HDF5 (.h5) or TensorFlow's native format. The load_weights() function expects the path to the saved weights file.
                    Optimizer state:
                    When loading weights, the optimizer's state (e.g., learning rate, momentum) is typically reset. If resuming training and preserving the optimizer state is crucial, saving and loading the entire model (including the optimizer) might be necessary using model.save() and tf.keras.models.load_model().
                    by_name and skip_mismatch arguments:
                    These optional arguments can be used to control how weights are loaded, particularly when dealing with minor architectural differences or partial weight loading. by_name=True loads weights based on layer names, while skip_mismatch=True allows loading even if some layers don't have matching weights. 
                    '''
                    # Creating an empty VGGNet16 Model
                    model = VGG16(weights=None)
                    # Loding Pre-Trained Weights into the Model
                    model.load_weights(filepath)
                    self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")
                    # Resize the Image to 224x224 Square Shape
                    newSize = (224,224)
                    # Mode of Processing the Image in Keras
                    processMode = "caffe"
                    self.ProcessImage(model,imagePath,newSize,processMode)
                    
                case "VGGNet19":
                    # This is a command for Download/Load with Tracing only in Console Not UI
                    # model = VGG19(weights="imagenet") 

                    # Creating an empty VGGNet19 Model
                    model = VGG19(weights=None)
                    # Loding Pre-Trained Weights into the Model
                    model.load_weights(filepath)
                    self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")
                    # Resize the Image to 224x224 Square Shape
                    newSize = (224,224)
                    # Mode of Processing the Image in Keras
                    processMode = "caffe"
                    self.ProcessImage(model,imagePath,newSize,processMode)

                case "ResNet50":
                    # This is a command for Download/Load with Tracing only in Console Not UI
                    # model = ResNet50(weights="imagenet") 

                    # Creating an empty ResNet50 Model
                    model = ResNet50(weights=None)
                    # Loding Pre-Trained Weights into the Model
                    model.load_weights(filepath)
                    self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")
                    # Resize the Image to 224x224 Square Shape
                    newSize = (224,224)
                    # Mode of Processing the Image in Keras
                    processMode = "caffe"
                    self.ProcessImage(model,imagePath,newSize,processMode)

                case "Inception_v3":
                    # This is a command for Download/Load with Tracing only in Console Not UI
                    # model = InceptionV3(weights="imagenet") 

                    # Creating an empty Inception_v3 Model
                    model = InceptionV3(weights=None)
                    # Loding Pre-Trained Weights into the Model
                    model.load_weights(filepath)
                    self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")
                    # Resize the Image to 299x299 Square Shape
                    newSize = (299,299)
                    # Mode of Processing the Image in Keras
                    processMode = "tf"
                    self.ProcessImage(model,imagePath,newSize,processMode)
 
                case "Xception":
                    # This is a command for Download/Load with Tracing only in Console Not UI
                    # model = Xception(weights="imagenet") 

                    # Creating an empty Xception Model
                    model = Xception(weights=None)
                    # Loding Pre-Trained Weights into the Model
                    model.load_weights(filepath)
                    self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Model successfully.")
                    # Resize the Image to 299x299 Square Shape
                    newSize = (299,299)
                    # Mode of Processing the Image in Keras
                    processMode = "tf"
                    self.ProcessImage(model,imagePath,newSize,processMode)

                case "MobilenetSSD" | "MobileNetSSDPrototxt":      
                    # Check/Download required MobilenetSSD.prototxt File              
                    MobileNetSSD_Prototext_Path = os.path.splitext(filepath)[0] + ".prototxt"
                    if not os.path.exists(MobileNetSSD_Prototext_Path):
                       modelType = "MobileNetSSDPrototxt"
                       self.PreProcessImage(imagePath, modelType,operationType)
                    else:
                        filepath = os.path.splitext(filepath)[0] + ".caffemodel"
                        self.PreProcessMobilenetSSD(imagePath,filepath,MobileNetSSD_Prototext_Path,operationType) 

                case "MaskRCNN" | "MaskRCNNPbtxt": 
                    # Check/Download required MaskRCNN.pbtxt File              
                    MaskRCNN_Pbtxt_Path = os.path.splitext(filepath)[0] + ".pbtxt"
                    if not os.path.exists(MaskRCNN_Pbtxt_Path):
                       modelType = "MaskRCNNPbtxt"
                       self.PreProcessImage(imagePath, modelType, operationType)
                    else:
                        filepath = os.path.splitext(filepath)[0] + ".pb"
                        self.PreProcessMaskRCNN(imagePath,filepath,MaskRCNN_Pbtxt_Path,operationType) 
                
                case "TinyYOLO" | "TinyYOLOCFG" | "YOLO" | "YOLOCFG" | "OptimizedYOLO":
                    # Check/Download required .cfg Files  
                    Optimized = False           
                    CFG_Path = os.path.splitext(filepath)[0] + ".cfg"
                    if not os.path.exists(CFG_Path):
                        if modelType == "TinyYOLO":
                           modelType = "TinyYOLOCFG"
                        elif modelType == "YOLO":
                            modelType = "YOLOCFG"
                        elif modelType == "OptimizedYOLO":
                            Optimized = True
                            modelType = "YOLOCFG"

                        self.PreProcessImage(imagePath, modelType, operationType, Optimized)
                    else:
                        filepath = os.path.splitext(filepath)[0] + ".weights"
                        self.PreProcessAllYOLOModels(imagePath,filepath,CFG_Path,operationType,modelType) 

                case "VGGFace":
                    # VGGFace Operation located on another page
                    # Here we only download its Model Weights
                    pass

    # Check, Validation for Downloading Pre-Trained Model                
    def PreProcessImage(self, imagePath,modelType,operationType,Optimized = False):
        ConditionToCheck = None
        ContentMessage = None
        TitleMessage = None
        match operationType:
            case "Images" | "BoundingBoxImages" | "ObjectMaskImages":
                ConditionToCheck = self.ImagesAndColorsHandler.image is not None and self.ImagesAndColorsHandler.imageName is not None
                ContentMessage = "First, Select an Image!"
                TitleMessage = "No Image Selected" 
                
            case "Pre-Saved":
                ConditionToCheck = self.ImagesAndColorsHandler.video is not None and self.ImagesAndColorsHandler.Check_Camera_Availability(self.ImagesAndColorsHandler.video)
                ContentMessage = "First, Select a Video!" 
                TitleMessage = "No Video Selected"

            case "Realtime":
                ConditionToCheck = self.ImagesAndColorsHandler.camera is not None and self.ImagesAndColorsHandler.Check_Camera_Availability(self.ImagesAndColorsHandler.camera)
                ContentMessage = "First, Select a Camera!" 
                TitleMessage = "No Camera Selected"
            
            case _: 
                ConditionToCheck = self.ImagesAndColorsHandler.image is not None and self.ImagesAndColorsHandler.imageName is not None
                ContentMessage = "First, Select an Image!"
                TitleMessage = "No Image Selected" 

        if ConditionToCheck: 

            if self.DownloadLogPopup == None or not self.DownloadLogPopup:
                self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)   
                self.DownloadLogPopup.show()

            self._is_running = True  

            modelTypeTemp = modelType
            if str(modelType).startswith("Optimized"): 
                modelTypeTemp = str(modelType).replace("Optimized","")

            # Get Model Info
            if len(self.models) > 0 and self.models.get(modelTypeTemp): 
                self.log_emitter.log_signal.emit("Checking for existing model file...")
                                
                url =  self.models[modelTypeTemp]["url"] 
                filename = self.models[modelTypeTemp]["name"] 
                fileSize = self.models[modelTypeTemp]["size"] 
                expected_hash = self.models[modelTypeTemp]["md5hash"] 

                expected_size = fileSize
                folder = os.path.normpath(join("resources","models"))
                filepath = os.path.join(folder, filename)
                 
                # Only Download if File is Missing or File Size is Greater than Approximate Expected Size - Tolerance 
                # Hash Validation Is not Active to Accept Mirror Image of Models, Expected Size Validation is not Active for same Reason.
                if not os.path.exists(filepath) or not self.FileSize_Approximate_Validation("md5",modelType,filepath, expected_size,expected_hash,self.log_emitter,True):  
                    self.log_emitter.log_signal.emit(filename + "\nModel file not found or  Size is not Valid! \nDownloading from internet...\n"+
                                                     "Make Sure your System Connected to the Internet\nFile is Approximately "+expected_size+"\n"+
                                                        "It takes a while Depending on the Speed of your System and Internet!\nDownload Url: \n" + url )
                    if os.path.exists(filepath):
                       os.remove(filepath) 

                    self.downloader = Downloader(url, filepath, modelType,imagePath,self.log_emitter, expected_size,operationType,self._is_running,Optimized)
                    self.DownloadLogPopup.Set_Downloader(self.downloader)
                    self.downloader.Start()   
                    
                else:
                    self.log_emitter.log_signal.emit(filename + "\nModel file found locally.\n Hash and Size are not Validated in Config!\nLoading from cache...")                                  
                    self.Loading_Model_Operation(modelType, filepath,imagePath,operationType)
            
            else:
                self.log_emitter.log_signal.emit("Error: 'models.json' not found or Not Contains Details for this Operation ( "+modelType+" ).\nPlease ensure the file exists in the root and contains Details for this Operation ( "+modelType+" ).")

        else:
            QMessageBox.warning(None, TitleMessage,ContentMessage)

################################################### Helper Functions:

    # Selecting Desired Operation
    def SelectDeepLearningOperations(self,operation,imagePath,accuracy):
        self.DownloadLogPopup = None
        self.accuracy = accuracy
        operationString = operation.strip().split(" ")
        match operation.strip():
            case "Image Recognition using Pre-Trained VGGNet16 Model":
                modelType = operationString[4]
                self.PreProcessImage(imagePath, modelType,None)

            case "Image Recognition using Pre-Trained VGGNet19 Model":
                modelType = operationString[4]
                self.PreProcessImage(imagePath, modelType,None)
    
            case "Image Recognition using Pre-Trained ResNet50 Model":
                modelType = operationString[4]
                self.PreProcessImage(imagePath, modelType,None)

            case "Image Recognition using Pre-Trained Inception_v3 Model":
                modelType = operationString[4]
                self.PreProcessImage(imagePath, modelType,None)

            case "Image Recognition using Pre-Trained Xception Model":
                modelType = operationString[4]
                self.PreProcessImage(imagePath, modelType,None)

            case "Object Detection by Pre-Trained Mobilenet SSD Model on Images":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Mobilenet SSD Model on Pre-Saved Video":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Mobilenet SSD Model on Realtime Video":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Mask Implementation by Pre-Trained MaskRCNN Model on Images":
                modelType = operationString[5]
                operationType = operationString[0] + operationString[1] + operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Bounding Box Implementation by Pre-Trained MaskRCNN Model on Images":
                modelType = operationString[5]
                operationType = operationString[0] + operationString[1] + operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained MaskRCNN Model on Pre-Saved Video":
                modelType = operationString[4]
                operationType = operationString[7]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained MaskRCNN Model on Realtime Video":
                modelType = operationString[4]
                operationType = operationString[7]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Tiny YOLO Model on Images":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Tiny YOLO Model on Pre-Saved Video":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Tiny YOLO Model on Realtime Video":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)
        
            case "Object Detection by Pre-Trained YOLO Model on Images":
                modelType = operationString[4]
                operationType = operationString[7]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Optimized YOLO Model on Images":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained YOLO Model on Pre-Saved Video":
                modelType = operationString[4]
                operationType = operationString[7]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Optimized YOLO Model on Pre-Saved Video":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained YOLO Model on Realtime Video":
                modelType = operationString[4]
                operationType = operationString[7]
                self.PreProcessImage(imagePath, modelType, operationType)

            case "Object Detection by Pre-Trained Optimized YOLO Model on Realtime Video":
                modelType = operationString[4] + operationString[5]
                operationType = operationString[8]
                self.PreProcessImage(imagePath, modelType, operationType)

        self.ImagesAndColorsHandler.WaitKeyCloseWindows()       

    # Selecting Active Camera
    def SelectDeepLearningCamera(self,text):
        if text.strip() != "":
           self.ImagesAndColorsHandler.camera = int((text.split(",")[0]).split(":")[1].strip())
    
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
            self.Loading_Model_Operation(modelType, filepath, imagePath,operationType)              

    # Updating ProgressBar
    def Update_Progress(self, percent):
        if self._is_running:
           self.DownloadLogPopup.Update_Progress(percent)

    # Updating Logs
    def Append_Log(self,message):
        if self._is_running:
            self.DownloadLogPopup.Append_Log(message)

    # Validating Approximate Size of Files
    def FileSize_Approximate_Validation(self,type,modelType,filepath, expected_size,expected_hash,log_emitter,check):
        """Check if the file's SHA256 or MD5 hash matches the expected value."""
        if os.path.exists(filepath):
            fileSize = os.path.getsize(filepath) or os.stat(filepath).st_size
            try:
                value, unit = expected_size.split()
                value = float(value)
                unit = unit.upper()
                if unit == "KB":
                    expected_size = int(value * 1024)
                elif unit == "MB":
                    expected_size = int(value * 1024 * 1024)
                elif unit == "GB":
                    expected_size = int(value * 1024 * 1024 * 1024)
                else:
                    expected_size = int(value)
            except:
                expected_size = 0

            match type:
                case "sha256":
                    sha256 = hashlib.sha256()
                    try:
                        with open(filepath, "rb") as f:
                            for chunk in iter(lambda: f.read(8192), b""):
                                sha256.update(chunk)
                                actual_hash = sha256.hexdigest()
                                if check:
                                    log_emitter.log_signal.emit("Expected Hash: " + str(expected_hash))
                                    log_emitter.log_signal.emit("File Hash: " + str(actual_hash))
                                    log_emitter.log_signal.emit("Expected Size: " + str(expected_size) + "B")
                                    log_emitter.log_signal.emit("File Size: " + str(fileSize) + "B")
                        # return str(actual_hash).lower() == str(expected_hash).lower() and fileSize == expected_size
                    except Exception as e:
                        log_emitter.log_signal.emit("Error:", str(e))
                        # return False
                    
                case "md5":
                    md5 = hashlib.md5()
                    try:
                        with open(filepath, "rb") as f:
                            for chunk in iter(lambda: f.read(8192), b""):
                                md5.update(chunk)
                        actual_hash = md5.hexdigest()
                        if check:
                            log_emitter.log_signal.emit("Expected Hash: " + str(expected_hash))
                            log_emitter.log_signal.emit("File Hash: " + str(actual_hash))
                            log_emitter.log_signal.emit("Expected Size: " + str(expected_size) + "B")
                            log_emitter.log_signal.emit("File Size: " + str(fileSize) + "B")
                        # return str(actual_hash).lower() == str(expected_hash).lower() and fileSize == expected_size
                    except Exception as e:
                        log_emitter.log_signal.emit("Error:", str(e))
                        # return False
           
            tolerance = 10 * 1024 * 1024
            if modelType in ["MobileNetSSDPrototxt", "MaskRCNNPbtxt"]:
               tolerance = 10 * 1024
            if modelType in ["TinyYOLOCFG", "YOLOCFG"]:
               tolerance = 1 * 1024
               
            return fileSize > (expected_size - tolerance)
        
    # Loading Model Details from models.json file in the Root
    def LoadModelDetails(self):
        if len(self.models) <= 0:    
            try:
                with open('models.json', 'r') as f:
                    self.models = json.load(f)
            except FileNotFoundError:
                self.log_emitter.log_signal.emit("Error: 'models.json' not found.\nPlease ensure the file exists in the root.")
            except json.JSONDecodeError:
                self.log_emitter.log_signal.emit("Error: Could not decode JSON from 'models.json'.\nCheck the file's format.")
        
    # Cleaning Downloaded Text Prototext file
    def Clean_TXT_Prototext(self,path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and HTML tags
            if stripped and not stripped.startswith("<") and not stripped.endswith(">"):
                cleaned_lines.append(stripped)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines))

        self.log_emitter.log_signal.emit(f"Cleaned Text Prototext file in: {path}")
        return path
   
################################################### Helper Classes:
    
# Signal emitter for Thread-Safe logging
class LogEmitter(QObject):
    log_signal = pyqtSignal(str) # All Communications, Updates and Text Messages
    progressbar_signal = pyqtSignal(int) #  Percent: ProgressBar Update
    finished_signal = pyqtSignal(bool, str, str , str, str, str) # success ,message info [error, Cancle, success, fail] ,modelType , filepath, imagePath, operationType

# Dialog for showing logs During Model Download/Load and Cancle Download/Load Operation
class DownloadLogPopup(QDialog):
    def __init__(self, log_emitter):
        super().__init__()
        self.downloader = None
        self.log_emitter = log_emitter
        self.setWindowTitle("Model Download/Loading Log")
        self.setFixedSize(800, 400)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        # Create a scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: transparent;
            }
        """)
        # Create a container widget for the scroll area
        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Add QTextEdits to the layout
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("Cancel Download")
        self.layout.addWidget(self.cancel_button)
        self.cancel_button.clicked.connect(self.Cancel_Download)
         # Set the container as the scroll area's widget
        self.scroll_area.setWidget(container)
        # Final layout for the dialog
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)

    def Set_Downloader(self, downloader):
        self.downloader = downloader

    def Append_Log(self, message):       
        if str(message).startswith("Progress"):
            lines = self.log_output.toPlainText().splitlines()
            if  lines[-1].strip().startswith("Progress"):
                lines[-1] = message
                message = "\n".join(lines)
                self.log_output.setText(message)
                scrollBarMinimumValue = self.scroll_area.verticalScrollBar().minimum()
                self.scroll_area.verticalScrollBar().setValue(scrollBarMinimumValue)
                QApplication.processEvents() 
            else:
                self.log_output.append(message)
                scrollBarMinimumValue = self.scroll_area.verticalScrollBar().minimum()
                self.scroll_area.verticalScrollBar().setValue(scrollBarMinimumValue)
                QApplication.processEvents() 
        else:
            self.log_output.append(message)
            scrollBarMinimumValue = self.scroll_area.verticalScrollBar().minimum()
            self.scroll_area.verticalScrollBar().setValue(scrollBarMinimumValue)
            QApplication.processEvents()          

    def Update_Progress(self, percent):
        self.progress_bar.setValue(percent)
        QApplication.processEvents()

    def Cancel_Download(self):
        if self.downloader:
            self.downloader.Cancle()
            self.Append_Log("Download Cancelled by User!")

    def closeEvent(self, event):
        log = self.log_output.toPlainText()
        if self.downloader and not "Download Success" in log and not "Download Complete" in log and not "Download Cancelled" in log  and not "Download Failed" in log:
            response = QMessageBox.warning(None,"Warning!","If Download has not completed, it will be aborted!",QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if response == QMessageBox.StandardButton.Yes:
                self.Cancel_Download()
                event.accept()
            elif response == QMessageBox.StandardButton.No:
                event.ignore()

# Downloader for Required Models   
class Downloader(QObject):
    def __init__(self, url: str, filepath: str,modelType: str,imagePath: str ,log_emitter,expected_size: str,operationType: str,_is_running: bool,Optimized= False, parent=None):
        super().__init__(parent)
        self.url = QUrl(url)
        self.filepath = filepath
        self.log_emitter = log_emitter
        self.manager = QNetworkAccessManager(self)
        self.start_time = None
        self.reply = None
        self.cancelled = False
        self._is_running = _is_running
        self.Optimized = Optimized
        self.modelType = modelType
        self.imagePath = imagePath
        self.expected_size = expected_size
        self.operationType = operationType
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3
        self.user_agents = [
            # Windows
            b"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            
            # macOS
            b"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            
            # Linux
            b"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            
            # Android
            b"Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Mobile Safari/537.36",
            
            # iOS
            b"Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/134.0.0.0 Mobile/15E148 Safari/604.1",
            
            # Chrome on iPad
            b"Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/134.0.0.0 Mobile/15E148 Safari/604.1",
            
            # Chrome on Chromebook
            b"Mozilla/5.0 (X11; CrOS x86_64 14526.83.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        ]
        match modelType:
            case "MaskRCNN":
                  self.archive_model_filename = "frozen_inference_graph.pb" 
            case _:
                   self.archive_model_filename ="None"

    def Start(self):
        request = QNetworkRequest(self.url)
        random_user_agent = random.choice(self.user_agents)
        #
        #request.setRawHeader(b"User-Agent", b"MyDownloader/1.0")
        request.setRawHeader(b"User-Agent", random_user_agent)
        self.reply = self.manager.get(request)
        #
        self.reply.sslErrors.connect(lambda errors: self.reply.ignoreSslErrors())
        self.start_time = time.time()
        # Connect signals from QNetworkReply, not QNetworkAccessManager
        self.reply.downloadProgress.connect(self.On_Progress)
        self.reply.finished.connect(self.On_Finished)

    def Cancle(self):
        self.cancelled = True
        if self.reply:
            self.reply.abort()
        self._is_running = False

    def On_Progress(self, downloaded: int, total_size: int):
        if self.cancelled:
            self.log_emitter.finished_signal.emit(False, "Download Cancelled.","","","",self.operationType)
            return

        if self._is_running:
            elapsed = time.time() - self.start_time
            percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
            speed = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else float('inf')
            speed_text = f"{speed:.2f} MB/s" if speed > 1 else f"{speed * 1024:.2f} KB/s"
            time_str = time.strftime("%M:%S", time.gmtime(elapsed))
            bar = '=' * (percent // 5) + '-' * (20 - (percent // 5))
            progress_text = f"Progress: {downloaded} B from {total_size} B [{bar}] {time_str} {speed_text}"

            self.log_emitter.progressbar_signal.emit(percent)
            self.log_emitter.log_signal.emit(progress_text)
        else:
            return

    def On_Finished(self):
        if self.cancelled:
            self.log_emitter.finished_signal.emit(False, "Download Cancelled.","","","",self.operationType)
            return

        if self.reply.error() != QNetworkReply.NetworkError.NoError:
            self.log_emitter.finished_signal.emit(False, self.reply.errorString(),"","","",self.operationType)
            # Fallback to requests with streaming
            if os.path.exists(self.filepath):
               os.remove(self.filepath)
            self.log_emitter.progressbar_signal.emit(0)
            self.Fallback_Download()
            return


        data = self.reply.readAll().data()
        with open(self.filepath, 'wb') as f:
            f.write(data)

        if not self.ValidateSize():
            if os.path.exists(self.filepath):
               os.remove(self.filepath)
           # self.fallback_attempts += 1
            self.log_emitter.progressbar_signal.emit(0)
            # if self.fallback_attempts < self.max_fallback_attempts:
            #     self.log_emitter.log_signal.emit(f"Download Attempt {self.fallback_attempts + 1}/{self.max_fallback_attempts}")
            #     self.Fallback_Download()

            # else:
            #     self.log_emitter.finished_signal.emit(False, "Download failed after multiple fallback attempts.", "", "", "",self.operationType)
            self.Fallback_Download()
            return

        self.Handle_Archive_Files()
        if self.Optimized == True: self.modelType = "OptimizedYOLO"
        self.log_emitter.finished_signal.emit(True, "Download Success.", self.modelType, self.filepath, self.imagePath, self.operationType)
        self._is_running = False
        return True

    def Fallback_Download(self):
        headers = { "User-Agent": random.choice(self.user_agents).decode("utf-8") }
        while self.fallback_attempts < self.max_fallback_attempts:
            self.fallback_attempts += 1
            self.log_emitter.log_signal.emit(f"Fallback attempt {self.fallback_attempts}/{self.max_fallback_attempts}")
            try:
                with requests.get(self.url.toString(), headers=headers, stream=True, timeout=10) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("Content-Length", 0))
                    downloaded = 0
                    start_time = time.time()
                    with open(self.filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if self.cancelled:
                                self.log_emitter.finished_signal.emit(False, "Download Cancelled.", "", "", "",self.operationType)
                                return
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)

                                percent = int(downloaded * 100 / total_size) if total_size else 0
                                elapsed = time.time() - start_time
                                speed = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else float('inf')
                                speed_text = f"{speed:.2f} MB/s" if speed > 1 else f"{speed * 1024:.2f} KB/s"
                                time_str = time.strftime("%M:%S", time.gmtime(elapsed))
                                bar = '=' * (percent // 5) + '-' * (20 - (percent // 5))
                                progress_text = f"Progress: {downloaded} B from {total_size} B [{bar}] {time_str} {speed_text}"

                                self.log_emitter.progressbar_signal.emit(percent)
                                self.log_emitter.log_signal.emit(progress_text)

                if not self.ValidateSize():
                    os.remove(self.filepath)
                    time.sleep(2)
                    continue  # retry

                self.Handle_Archive_Files()
                if self.Optimized == True: self.modelType = "OptimizedYOLO"
                # Success
                self.log_emitter.finished_signal.emit(True, "Download Success (via fallback).", self.modelType, self.filepath, self.imagePath,self.operationType)
                self._is_running = False
                return True

            except Exception as e:
                self.log_emitter.log_signal.emit(f"Fallback error:\n {str(e)}. Retrying...")
                if os.path.exists(self.filepath):
                    os.remove(self.filepath)
                time.sleep(2)
                continue  # retry
        
        # Final failure after all retries
        self.log_emitter.finished_signal.emit(False, "Download failed after multiple fallback attempts.", "", "", "",self.operationType)
        self._is_running = False
        if os.path.exists(self.filepath):
               os.remove(self.filepath)
        return
    
    def Handle_Archive_Files(self):
        try:
            if self.Is_TarGz_File(self.filepath):
                # Step 1: Extract archive
                extract_dir = os.path.splitext(os.path.splitext(self.filepath)[0])[0]
                self.log_emitter.log_signal.emit(f"Extracting to: {extract_dir}")
                with tarfile.open(self.filepath, "r:gz") as tar:
                    tar.extractall(path=extract_dir)

                # Step 2: Locate frozen_inference_graph.pb
                found_pb = None
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        if file == self.archive_model_filename: # "frozen_inference_graph.pb":
                            found_pb = os.path.join(root, file)
                            break
                    if found_pb:
                        break

                if found_pb:
                    # Step 3: Move .pb file to final location
                    final_pb_path = os.path.join(os.path.dirname(self.filepath), os.path.basename(self.filepath)) # "MaskRCNN.pb")
                    shutil.move(found_pb, final_pb_path)
                    self.filepath = final_pb_path
                    self.log_emitter.log_signal.emit(f"Model file extracted and renamed to: {final_pb_path}")

                    # Step 4: Clean up extracted folder
                    shutil.rmtree(extract_dir)
                    self.log_emitter.log_signal.emit(f"Cleaned up extracted folder: {extract_dir}")
                else:
                    self.log_emitter.log_signal.emit("Extraction succeeded, but frozen_inference_graph.pb not found.")
            else:
                self.log_emitter.log_signal.emit("Downloaded file is not a valid tar.gz archive.")
        except Exception as e:
            self.log_emitter.log_signal.emit(f"Archive extraction failed: {str(e)}")

    def Is_TarGz_File(self,filepath):
        return tarfile.is_tarfile(filepath)

    def Is_Zip_File(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                signature = f.read(4)
                return signature == b'PK\x03\x04'
        except Exception as e:
            self.log_emitter.log_signal.emit(f"Error reading file: {e}")
            return False

    def ValidateSize(self):
        actual_size = os.path.getsize(self.filepath) or os.stat(self.filepath).st_size
        try:
            value, unit = self.expected_size.split()
            value = float(value)
            unit = unit.upper()
            if unit == "KB":
                expected_size = int(value * 1024)
            elif unit == "MB":
                expected_size = int(value * 1024 * 1024)
            else:
                expected_size = int(value)
        except:
            expected_size = 0

        tolerance = 10 * 1024 * 1024
        if self.modelType in ["MobileNetSSDPrototxt", "MaskRCNNPbtxt"]:
            tolerance = 10 * 1024
        if self.modelType in ["TinyYOLOCFG", "YOLOCFG"]:
               tolerance = 1 * 1024

        if actual_size < expected_size - tolerance:
            self.log_emitter.log_signal.emit(f"Downloaded file too small ({actual_size} < {expected_size - tolerance})")
            os.remove(self.filepath)
            return False
        return True

