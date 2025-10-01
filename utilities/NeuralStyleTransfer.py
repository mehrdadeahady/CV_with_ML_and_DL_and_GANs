import os
from os import listdir
from os.path import isfile, join
try:
    import numpy as np
except:
    print("You Should Install numpy Library")
try:
    import cv2
except:
    print("You Should Install OpenCV-Python and cv2_enumerate_cameras Libraries")
try:
    import matplotlib.pyplot as plt
except:
    print("You Should Install matplotlib Library!")
try:
    from PyQt6.QtCore import QObject
    from PyQt6.QtWidgets import QMessageBox, QFileDialog
except:
    print("You Should Install PyQt6 Library!")

class NeuralStyleTransfer(QObject):
    def __init__(self, parent=None):
        super().__init__()
        # Internal Variable to Access Data inside All Functions in the Class
        self.image = None
        self.style = None
        self.model = None
        self.ImageExtension = None
        self.ImageAfterStyleTransfer =  None
 
    # Consider|Attention:
    # Process Functions Contains Computer Vision Functions with Comments and Explanations
    # Rest of Functions are Pre-Processor and Helpers
    
    # Display Base Image
    def SelectShowImage(self, ImageName):
        if ImageName.strip() != "":
            self.ImageExtension = os.path.splitext(ImageName)[1]
            ImagePath = "resources/images/" + ImageName
            self.image = cv2.imread(ImagePath)
            cv2.imshow("Base Image",self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Display Style Image
    def SelectShowStyle(self, StyleName):
        if StyleName.strip() != "":
            if self.image is not None:
                StylePath = "resources/styles/" + StyleName
                self.style = cv2.imread(StylePath)
                cv2.imshow("Style",self.style)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                QMessageBox.warning(None, "No Image Selected","First, Select an Image.")

    # Sync Image Size with Style Size
    def SyncImageStyleSize(self, value):
        cv2.destroyAllWindows()
        height, width = (self.image.shape[0]), (self.image.shape[1])
        newHeight = int((value / width) * height)
        self.image = cv2.resize(self.image, (value, newHeight))
        height, width = (self.style.shape[0]), (self.style.shape[1])
        newHeight = int((value / width) * height)
        self.style = cv2.resize(self.style, (value, newHeight))
        cv2.imshow("Image",self.image)
        cv2.imshow("Style",self.style)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Transfer Style to the Image by its Pre-Trained Torch Model
    def TransferStyle(self,ModelName,RedValue,GreenValue,BlueValue):
        ModelPath = "resources/style_transfer_models/" + ModelName
        # loading Pre-Trained Torch Neural Style Transfer Model 
        neuralStyleModel = cv2.dnn.readNetFromTorch(ModelPath)
        '''  
        Syntax: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
        Parameters: 
            image - This is the image that we want to preprocess (for our model)
            scalefactor- scale factor basically multiplies(scales) our image channels. And remember that it scales it down by a factor of 1/n, 
            where n is the scalefactor you provided.
            size - this is the target size that we want our image to become. Most common CNNs use 224x224 or 229x229 pixels as their input image array, 
            but you can set it to meet your model requirements.
            mean- this is the mean subtracting values. You can input a single value or a 3-item tuple for each channel RGB, 
            it will basically subtract the mean value from all the channels accordingly, this is done to normalize our pixel values. 
            Note: mean argument utilizes the RGB sequence
            swapRB- OpenCV by default reads an image in BGR format, but as I mentioned above that the mean argument takes in the values in RGB sequence, 
            so to prevent any disruption this function, as the name implies swaps the red and blue channels. ( i.e why its default value is True)
        '''
        # Create Blob from the Image
        inpputBlob = cv2.dnn.blobFromImage(self.image, 1.0, mean=(RedValue, GreenValue, BlueValue), swapRB=False, crop=False)
        # Put Blob inside Neural Style Transfer Model
        neuralStyleModel.setInput(inpputBlob)
        # Get Output by Forwarding the Model Containig the Blob
        output = neuralStyleModel.forward()

        # Reshaping the output Tensor
        output = output.reshape(3, output.shape[2], output.shape[3])
        # Adding Back  the Mean Subtraction and Re-Ordering the Channels 
        output[0] += RedValue
        output[1] += GreenValue
        output[2] += BlueValue
        output /= 255
        output = output.transpose(1, 2, 0)
        
        cv2.imshow("Image after Transfer Style", output)
        # Storing Output in Memory by Converting it for storing on the Disk as RGB
        self.ImageAfterStyleTransfer = (output*255).astype(np.uint8)
        # Converting Image, Style and Transfered Image from BGR to RGB for Display in the Plot
        b,g,r = cv2.split(self.image)    
        self.image = cv2.merge([r,g,b])
        b,g,r = cv2.split(self.style)    
        self.style = cv2.merge([r,g,b])
        b,g,r = cv2.split(output)    
        output = cv2.merge([r,g,b])
       
        plt.close()     
        # Displaying Base Image, Style and Tranfered Style Image in the Same Plot for Comparison           
        f = plt.figure("Neural Style Transfer:")
        sub1 = f.add_subplot(1,3, 1)
        sub1.set_title("Image", color=(0,0,0))
        plt.imshow(self.image)
        plt.xticks([]); plt.yticks([])
        sub2 = f.add_subplot(1,3, 2)
        sub2.set_title("Style", color=(0,0,0))
        plt.imshow(self.style)
        plt.xticks([]); plt.yticks([])
        sub3 = f.add_subplot(1,3, 3)
        sub3.set_title("Image after Transfer Style", color=(0,0,0))
        plt.imshow(output)
        plt.xticks([]); plt.yticks([])
        plt.show(block=True)

        cv2.waitKey(0)
        cv2.destroyAllWindows

    # Saving the Image
    def SaveImage(self):
        file_path = None 
        if self.ImageAfterStyleTransfer is None:
           QMessageBox.warning(None, "No Transfer Style Image","First, Transfer a Style to an Image.")
        else:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save File", "ImageAfterStyleTransfer" + self.ImageExtension)
            if file_path:
                  # imwrite is Saving Image Function in OpenCV that takes 2 parameter
                  # cv2.imwrite(Parameter1 = Name of Image, Parameter2 = Path to Image ) 
                  cv2.imwrite(file_path,self.ImageAfterStyleTransfer)
       