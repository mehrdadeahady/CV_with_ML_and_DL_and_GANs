from utilities.CreateSimpleCNN import SignalEmitter, ConsoleCallback, TrainingLogPopupClass
import os
from os.path import isfile, join
import time
import json
try:
    os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax", "torch"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    # print(tf.config.list_physical_devices('GPU'))
    import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.callbacks import Callback
    from keras.datasets import mnist 
    from keras.utils import to_categorical
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Activation, BatchNormalization
    from keras import backend as K
    from keras.optimizers import SGD 
except:
    print("Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare!")
try:
    import numpy as np
except:
    print("You Should Install numpy Library")
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

class CreateHandGestureRecognitionCNN(QObject):
    def __init__(self,ImagesAndColorsHandler,CreateSimpleCNNHandler,parent=None):
        super().__init__()
        # Internal Variable to Access Data inside All Functions in the Class 
        self.ImagesAndColorsHandler = ImagesAndColorsHandler
        self.CreateSimpleCNNHandler = CreateSimpleCNNHandler
        self.GestureName = None
        self.train_generator = None
        self.validation_generator = None
        self.number_of_classes = None
        self.input_shape = (28,28,1)
        self.numberOfClasses = 0
        self.numberOfPixels = 0
        self.model = None
        self.modelSummary = ""
        self.modelHistory = None
        self.TrainedModel  = None
        self.training_thread = None
        self._is_running = False
        self.steps_per_epoch = None 
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.batch_signal.connect(self.Show_Batch)
        self.signal_emitter.epoch_signal.connect(self.Show_Epoch_Summary)
        self.signal_emitter.exit_signal.connect(self.Exit_Message)
        self.signal_emitter.modelTrainedSignal.connect(self.Set_Model_Trained_Values)
                 
    # There are functions here for Creating a Simple CNN Model for Hand Gesture Recognition for Master Degree Level
    # Find Comments and Explanation for each function related to ML and CV
    # UI functions do not have Comments because this is not a QT Training but they are Clear to Understand by its names and contents

     # Assigning Results of Training including: Trained Model and History Log
    def Set_Model_Trained_Values(self,modelHistoryObject,TrainedModelObject):
        self.modelHistory = modelHistoryObject
        self.TrainedModel = TrainedModelObject
        # Obtain accuracy score by evalute function
        score = self.TrainedModel.evaluate(self.validation_generator,steps=self.validation_steps,verbose=1)
        QMessageBox.information(None,"Training Finished Successfully", "Test Loss: " +str(score[0]) + "\nTest Accuracy: " + str(score[1]) + "\nReady for Saving the Training Model.")
       
    # Message When Canceling the Training of Model
    def Exit_Message(self,message):
        QMessageBox.critical(None, "Training Canceled", "Training Canceled by User!")

    # Displaying Realtime Progress of the Model Training
    def Show_Batch(self, message, epoch):
        if self._is_running:
           self.TrainingLogPopup.Show_Batch_Progress(message, epoch)

    # Displaying Result of the Model Training for Each Epoch
    def Show_Epoch_Summary(self, message, epoch):
        if self._is_running:
           self.TrainingLogPopup.Show_Epoch_Summary(message, epoch)

    # Recording Hand Gesture Samples to Create Dataset
    def RecordHandGesture(self, GestureKind, GestureName):
        # Assign Gesture Name
        self.GestureName = GestureName
        # Get the Camera Video File as Stream
        self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.camera)
        gestureDirectory = "./temp/handgesture/" + GestureKind + "/" + GestureName + "/"
        # Create Gesture Directories
        if not self.MakeDirectory(gestureDirectory):
             QMessageBox.critical(None,"IO Error","Couldn't Create Gesture Directory.")
             return
        else:
             pass
          
        i = 0
        image_count = 0        
        # Create a While Loop until Camera Still is Open and Video is Streaming
        while i < 3:                 
            # Get the Current Frame from Camera Video Stream
            ret,frame = self.ImagesAndColorsHandler.videoCapturer.read()
            # Flip Frame
            frame = cv2.flip(frame, 1)
            # Define Region of Interest
            roi = frame[100:400, 320:620]           
            # Convert roi to Gray
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Resize roi
            roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
            # Display roi
            cv2.imshow('Roi Sacled and Gray', roi)
            # Copy Fame for Gesture Region
            copy = frame.copy()
            # Define Gesture Region
            cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
            # Put Guidaing Text on the Copy of Frame
            if i == 0:
                image_count = 0
                cv2.putText(copy, "Press a Key to Record when ready", (10 , 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            if i == 1:
                image_count+=1
                match GestureKind:
                    case "train":
                                cv2.putText(copy, "Recording gesture - Train", (10 , 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    
                    case "test":
                                cv2.putText(copy, "Recording gesture - Test", (10 , 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

                cv2.putText(copy, str(image_count), (400 , 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
              
                cv2.imwrite(gestureDirectory + str(image_count) + ".jpg", roi)
           
            if i == 2:
                cv2.putText(copy, "Press a Key to Exit", (10 , 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            # Display Copy of Frame 
            cv2.imshow('frame', copy)    
            # Waiting for a key for Progress to next Step
            if cv2.waitKey(1) in range(0,255): 
                image_count = 0
                i += 1
        # Release Camera
        self.ImagesAndColorsHandler.videoCapturer.release()
        cv2.destroyAllWindows() 

    # Improving Dataset Efficiency
    def EnhanceDataset(self, GestureName):
        # Define number of Rows and Columns (Height and Width)
        img_rows, img_cols = 28, 28
        # Size of Batch
        batch_size = 32
        # Path to Hand Gesture Training and Testing folders
        train_data_dir = "./temp/handgesture/train" 
        validation_data_dir = "./temp/handgesture/test"
        # Number of Classes/Gestures Already Recorded
        num_train_classes = self.CountNumberOfDirectories(train_data_dir)
        num_test_classes = self.CountNumberOfDirectories(validation_data_dir)
        # Check if Each Training Classe has a Test Class
        if num_test_classes == num_train_classes:
            # Assign number of Classes/Gestures
            self.number_of_classes = num_train_classes
            # Count number of Samples taken for Train and Test of the Gesture
            trainSamplesCount = self.CountFilesInDirectory(train_data_dir + "/" + GestureName)
            testSamplesCount = self.CountFilesInDirectory(validation_data_dir + "/" + GestureName)
            # Check if Train Dataset already Generated or not
            if self.train_generator is None:            
                if trainSamplesCount > 300 and testSamplesCount > 100:           
                    # Use Data Augmentaiton to Cover Tolerances
                    train_data_generator = ImageDataGenerator(
                        rescale = 1./255,
                        rotation_range = 30,
                        width_shift_range = 0.3,
                        height_shift_range = 0.3,
                        horizontal_flip = True,
                        fill_mode = "nearest")
                    
                    validation_data_generator = ImageDataGenerator(rescale = 1./255)
                    # Create/Enhance Train Dataset
                    self.train_generator = train_data_generator.flow_from_directory(
                            train_data_dir,
                            target_size = (img_rows, img_cols),
                            batch_size = batch_size,
                            color_mode = "grayscale",
                            class_mode = "categorical",
                            shuffle=True  # improves learning
                            )
                    
                    # Storing class_indices in a json file
                    if not os.path.exists("temp"):
                        QMessageBox.critical(None, "Missing Directory", "Please create temp directory in the root.")
                        return
                    else:
                        with open("temp/class_indices.json", "w") as f:
                            json.dump(self.train_generator.class_indices, f)

                    # Create/Enhance Test/Validation Dataset
                    self.validation_generator = validation_data_generator.flow_from_directory(
                            validation_data_dir,
                            target_size = (img_rows, img_cols),
                            batch_size = batch_size,
                            color_mode = "grayscale",
                            class_mode = "categorical",
                            shuffle=False  # ensures consistent evaluation
                            )
                    
                    QMessageBox.warning(None,"Success","Train Generator and Validation Generator are Configured.")
                    
                else:
                    QMessageBox.warning(None,"In-Sufficient Samples","Recorded samples are not enough.\nTrain Samples should be greater than 300\n600 recommended.\nIt is: " + str(trainSamplesCount) +
                                        "\nTest Samples should be greater than 100\n200 recommended.\nIt is: " + str(testSamplesCount))
            
            else:
              QMessageBox.warning(None,"Already Configured","Train Generator and Validation Generator Already are Configured.")
                
        else:
              QMessageBox.critical(None,"Error","Train or Test samples does not exist for one of the Gestures.")
              
    # Create Directories for Storing Images
    def MakeDirectory(self,directory):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                return True
            except:
                print("IO Error. Couldn't Create Gesture Directory")
                return False
        else:
            return True
        
    # Create a Simple CNN Model
    def CreateModel(self):
        if self.train_generator == None or self.validation_generator == None:
           QMessageBox.warning(None,"Dataset not Exist","First Record Samples and Create/Enhance Dataset!")
        else:
            if self.train_generator is not None:
                self.number_of_classes = len(self.train_generator.class_indices)
            try:             
                # Create Model
                model = Sequential()
                # To display the summary of the model so far, include the current output shape
                # Start model by passing an Input object to the model, so it knows its input shape which is 28 x 28 x 1
                model.add(keras.Input(shape = self.input_shape ))           
                # First Convolution Layer, contains 64 Filters which Reduces layer size to 24 x 24 x 64
                model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
                # First MaxPooling with a kernel size of 2 x 2 reduces size to 12 x 12 x 64
                model.add(MaxPooling2D(pool_size=(2, 2)))
                # Second Convolution Layer, contains 64 Filters which Reduces layer size to 24 x 24 x 64
                model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
                # Second MaxPooling with a kernel size of 2 x 2 reduces size to 12 x 12 x 64
                model.add(MaxPooling2D(pool_size=(2, 2)))
                # Third Convolution Layer, contains 64 Filters which Reduces layer size to 24 x 24 x 64
                model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu'))
                # Third MaxPooling with a kernel size of 2 x 2 reduces size to 12 x 12 x 64
                model.add(MaxPooling2D(pool_size=(2, 2)))
                # Flatten layer reshapes the tensor to have a shape equal to the number of elements in tensor, before input into Dense Layer
                # In this CNN it goes from 12 * 12 * 64 to 9216 * 1
                model.add(Flatten())
                # First Dense: Connect this layer to a Fully Connected/Dense layer of size 1 * 128
                model.add(Dense(128, activation='relu'))
                # Dropout layer to reduce overfitting
                model.add(Dropout(0.5))
                # Second Dence: Final Fully Connected/Dense layer with an output for each class
                # softmax gives a probability distribution across all classes
                model.add(Dense(self.number_of_classes, activation='softmax'))
                # Compile the Model, this creates an object that stores the model we just created
                # Set Optimizer to use Root-mean squared propagation
                # Set Loss function to be categorical_crossentropy as it is suitable for multiple problems. 
                # It quantifies the difference between the actual class labels (0 or 1)
                # Finally, the Metrics (for Measuring Performance) to be accuracy
                model.compile(loss = 'categorical_crossentropy',
                            optimizer = 'rmsprop',
                            metrics = ['accuracy'])

                self.model = model
                # Capture Summary function to Display Model Layers and Parameters             
                self.modelSummary = self.CreateSimpleCNNHandler.ModelSummaryCapture(model)       
                QMessageBox.information(None,"Model Summary:",self.modelSummary)

            except:
                    QMessageBox.critical(None, "Instalation Error", "Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare!")
        
    # Show Model Summary
    def ModelSummaryFunction(self):
        if self.model == None:
            QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        elif self.modelSummary.strip() == "":
            QMessageBox.warning(None,"Model Summary not Exist","First Create a Model, Model Summary does not Exist!")
        else:
            cv2.destroyAllWindows()
            plt.close("all")
            QMessageBox.information(None,"Model Summary:",self.modelSummary)
    
    # Train the Model in another Thread
    def TrainModel(self,total_epochs):
        if self.model == None and self.train_generator == None:
             QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        else:
            self.total_epochs = total_epochs
            self.batch_size = 32           
            self._is_running = True
            self.number_of_train_samples= 0
            self.number_of_validation_samples = 0  
            train_data_dir = "./temp/handgesture/train" 
            test_data_dir = "./temp/handgesture/test"
            for root, dirs, files in os.walk(train_data_dir):
                    for file in files:
                        self.number_of_train_samples += 1  
            for root, dirs, files in os.walk(test_data_dir):
                    for file in files:
                        self.number_of_validation_samples += 1  

            self.validation_steps = self.number_of_train_samples // self.batch_size
            self.steps_per_epoch =  self.number_of_validation_samples // self.batch_size
            try:
                # Running the Training Model in a seperate Thread to ba able cancel long running process and Inversion of Control (IOC)
                self.training_thread = TrainingThread(self._is_running,self.signal_emitter, self.total_epochs, self.steps_per_epoch,self.validation_steps, self.batch_size,self.model,self.train_generator,self.validation_generator)
                self.TrainingLogPopup = TrainingLogPopupClass(self.total_epochs,self.training_thread)   
                self.TrainingLogPopup.show()   
                self.training_thread.start()
            except:
                   QMessageBox.critical(None, "Instalation Error", "Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare!")
    
    # Cancel Training
    def CancelTraining(self):
        if self.training_thread is not None and self._is_running == True:
            self._is_running = False
            self.training_thread.stop()
            self.TrainingLogPopup.close()
        else:
            QMessageBox.warning(None,"No Training Process","No Model Training Process Running now.")
          
     # Save Trained Model
   
    # Save Trained Model
    def SaveTrainedModel(self):
        if self.train_generator is None  or self.validation_generator is None:
            QMessageBox.warning(None,"No Data","First Create/Enhance Dataset!")
        elif self.model is None:
            QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        elif self.modelHistory is None or self.TrainedModel is None:
            QMessageBox.warning(None,"Model not Trained","First Train the Model!")  
        else:
            '''         
            The .h5 extension in Keras refers to the Hierarchical Data Format version 5 (HDF5) file format, 
            which was a common and widely used method for saving Keras models in older versions.
            This format allowed for saving the entire model, including:
                Model architecture: The structure of the neural network, including layers and their connections.
                Model weights: The learned parameters of the network.
                Optimizer state: Information about the optimizer used during training, allowing for continued training from the saved state. 
            While the .h5 format is considered a legacy format in newer Keras versions (which now recommend the .keras extension or TensorFlow SavedModel format),
            it remains compatible for loading models saved with older Keras versions. 
            You can still load .h5 models using tf.keras.models.load_model() in current TensorFlow/Keras environments.
            '''
            # Depricated Legacy saving by .h5 extension below:
            # self.model.save("SimpleCNN.h5", include_optimizer=True)
            # New way of Saving Keras Model by .keras extension:
            self.TrainedModel.save("resources/models/SimpleHandGestureCNN.keras", overwrite= True,include_optimizer=True)
            # Obtain accuracy score by evalute function
            score = self.TrainedModel.evaluate(self.validation_generator,steps=self.validation_steps,verbose=1)
            QMessageBox.information(None,"Training Model Saved Successfully","Saving Path = resources/models folder in the root." + 
                                    "\nTest Loss: " +str(score[0]) + "\nTest Accuracy: " + str(score[1]) + 
                                    "\nBelow you can Test this Model.") 

    # Test Trained Hand Gesture Model
    def TestHandGestureModel(self):
        self.ImagesAndColorsHandler.videoCapturer = cv2.VideoCapture(self.ImagesAndColorsHandler.camera)
        HandGestureModelPath = os.path.normpath("./resources/models/SimpleHandGestureCNN.keras")
        # Load Model as a Classifier
        classifier = load_model(HandGestureModelPath)
        while True:
            # Capture Frame from Camera
            ret, frame = self.ImagesAndColorsHandler.videoCapturer.read()
            # Flip Frame
            frame=cv2.flip(frame, 1)

            # Define region of interest
            roi = frame[100:400, 320:620]
            # Convert roi to Gray
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Resize roi
            roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)     
            # Display roi       
            cv2.imshow('roi scaled and gray', roi)

            # Copy Frame
            copy = frame.copy()
            # Define Gesture Region
            cv2.rectangle(copy, (320, 100), (620, 400), (255,0,0), 5)
            
            # Reshape roi
            roi = roi.reshape(1,28,28,1) 
            roi = roi/255

            # Detect Gesture
            prediction = classifier.predict(roi)
            predicted_class = np.argmax(prediction[0])
            
            # Load class_indices from json file
            if not os.path.exists("temp/class_indices.json"):
                QMessageBox.critical(None, "Missing Data", "Class indices not found. Please enhance dataset again.")
                return
            else:
                with open("temp/class_indices.json", "r") as f:
                    class_indices = json.load(f)

            # Retrieve Class Name
            label_map = dict((v,k) for k,v in class_indices.items())
            gesture_name = label_map[predicted_class]
            
            # Put Detected Gesture Name on the Copy of Frame
            cv2.putText(copy, gesture_name , (300 , 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # Display Result
            cv2.imshow('Testing Result: ', copy)    
            
            # Wait for a key to Exit
            if cv2.waitKey(1) in range(0,255): 
               break
        
        # Release Camera
        self.ImagesAndColorsHandler.videoCapturer.release()
        # Close all cv2 Windows
        cv2.destroyAllWindows() 

    # Count Files In a Directory
    def CountFilesInDirectory(self, directory):
        if os.path.exists(directory):
            count = 0 
            for file in os.scandir(directory):
                if file.is_file():
                    count += 1
            return count
        else:
             return 0
    
    # Count Directories In a Path
    def CountNumberOfDirectories(self, path):
        count = sum(os.path.isdir(i) for i in os.listdir(path))
        return count
    
# Training Thread
class TrainingThread(QThread):
    def __init__(self,_is_running, signal_emitter, total_epochs,steps_per_epoch,validation_steps, batch_size, model,train_generator,validation_generator):
        super().__init__()
        # Store Results for Creating Plots
        # In fit function specify datsets (x_train & y_train)
        # batch size (typically 16 to 128 depending on RAM of the System) 
        # number of epochs (usually 10 to 100) is number of train
        # validation datasets (x_test & y_test)
        # verbose = 1, sets our training to output performance metrics every epoch
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.signal_emitter = signal_emitter
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self.model = model
        self.ConsoleCallback = None
        self._is_running = _is_running
        self.modelHistory = None
        
    def run(self):
        if self._is_running == True:
           try:
                '''              
                In machine learning, model.fit() is a method used to train a machine learning model. 
                This method is commonly found in libraries like Scikit-learn, TensorFlow, and Keras.
                Purpose of model.fit():
                The primary purpose of model.fit() is to adjust the internal parameters of a machine learning model so that 
                it can accurately learn the patterns and relationships within the provided training data. 
                This process is often referred to as "model fitting."
                How it works:
                  Input Data:
                    You provide the model.fit() method with your training data, typically consisting of input features (X) and corresponding target labels (y).
                    Iterative Optimization:
                    The method then employs an optimization algorithm (e.g., gradient descent) to iteratively adjust the model's parameters 
                    (weights and biases in neural networks, coefficients in linear models, etc.).
                  Loss Minimization:
                    During each iteration (or "epoch"), the model makes predictions on the training data, and a "loss function" calculates 
                    the discrepancy between these predictions and the actual target labels. 
                    The optimization algorithm then updates the parameters in a direction that minimizes this loss.
                  Learning Patterns:
                    This iterative process allows the model to learn the underlying patterns, relationships, and features within the training data, 
                    enabling it to make accurate predictions on unseen data in the future.
                  Hyperparameters:
                    model.fit() also often takes various hyperparameters as arguments, such as the number of epochs 
                    (how many times the training data is passed through the model), batch size (the number of samples processed before updating parameters), 
                    and validation data (a separate dataset used to monitor performance during training).
                Outcome:
                After the model.fit() method completes, the model is "trained," meaning its internal parameters have been optimized to 
                represent the learned patterns from the training data. This trained model can then be used for making predictions on new, unseen data.
                Training history (optional but common):
                In many frameworks, model.fit() returns a History object or similar structure. This object contains a record of the training process, typically including:
                   Loss values: The value of the loss function at each epoch or iteration, indicating how well the model is performing on the training data.
                   Metric values: Values of other evaluation metrics (e.g., accuracy, precision, recall) calculated at each epoch, providing further insights into the model's performance on the training data and potentially a validation set.
                '''
                # Store Results for Creating Plots
                # In fit function specify datsets (x_train & y_train)
                # batch size (typically 16 to 128 depending on RAM of the System) 
                # number of epochs (usually 10 to 100) is number of train
                # validation datasets (x_test & y_test)
                # verbose = 1, sets our training to output performance metrics every epoch
                batch_size = self.batch_size
                epochs = self.total_epochs
                # Callback Function for Communication with Model during Training for: Canceling Training, Updating UI by Logs, Exporting Results of Training
                self.ConsoleCallback = ConsoleCallback(self._is_running,self.signal_emitter, self.steps_per_epoch, self.total_epochs)
                self.modelHistory = self.model.fit( self.train_generator,
                                                    steps_per_epoch = self.steps_per_epoch,
                                                    epochs = epochs,
                                                    validation_data = self.validation_generator,
                                                    verbose = 1,
                                                    validation_steps = self.validation_steps,
                                                    callbacks=[self.ConsoleCallback]
                                                    ) 
                
                if self.modelHistory is not None:
                   self.signal_emitter.modelTrainedSignal.emit(self.modelHistory,self.model)
           except:
                  self.signal_emitter.exit_signal.emit("exit")

    def stop(self):
        if self._is_running == True:
            self._is_running = False
            self.ConsoleCallback._is_running = False
            self.quit()
            self.exit()
