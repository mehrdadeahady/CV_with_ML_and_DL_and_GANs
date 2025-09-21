'''
CNN in DL refers to the role and application of Convolutional Neural Networks (CNNs) within the field of Deep Learning (DL).
Convolutional Neural Networks (CNNs) are a specialized type of neural network architecture, widely used in deep learning, 
particularly for tasks involving visual data. 
They are designed to automatically learn spatial hierarchies of features from input data, making them highly effective for image and video processing.
Key aspects of CNNs in Deep Learning:

    Feature Extraction:
    CNNs utilize convolutional layers with filters (or kernels) to detect patterns and features within data, such as edges, shapes, and textures in images. 
    This process of automatic feature extraction is a significant advantage over traditional methods that require manual feature engineering.
    Hierarchical Learning:
    CNNs learn features in a hierarchical manner. Early layers typically identify basic features, 
    while deeper layers combine these basic features to recognize more complex patterns, like objects or faces.
    Parameter Sharing:
    CNNs employ parameter sharing through the use of filters, which are applied across different regions of the input. 
    This greatly reduces the number of parameters compared to fully connected neural networks, making them more computationally efficient.
    Applications:
    CNNs are a cornerstone of many deep learning applications, including:
        Computer Vision: Image recognition, object detection, image segmentation, facial recognition.
        Natural Language Processing (NLP): Text classification, sentiment analysis.
        Medical Image Analysis: Disease detection and diagnosis from medical scans.

In essence, CNNs are a powerful and efficient architecture within the broader domain of deep learning, 
particularly adept at handling and extracting meaningful information from structured data like images.
'''
import os
from os.path import isfile, join
import time
try:
    os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax", "torch"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    # print(tf.config.list_physical_devices('GPU'))
    import keras
    from keras import callbacks
    from keras.callbacks import Callback
    from keras.datasets import mnist 
    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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

class CreateSimpleCNN(QObject):
    LoadMNISTRawDataOrPreparedData = pyqtSignal(int)
    def __init__(self,parent=None):
        super().__init__()
        # Internal Variable to Access Data inside All Functions in the Class 
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.input_shape = ()
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
   
    # There are functions here for Creating a Simple CNN Model for Bachelor's Degree Level
    # In Master Degree Level it is possible to create Customisable Models with desired Layers and Import desired dataset
    # Find Comments and Explanation for each function related to ML and CV
    # UI functions do not have Comments because this is not a QT Training but they are Clear to Understand by its names and contents

    # Assigning Results of Training including: Trained Model and History Log
    def Set_Model_Trained_Values(self,modelHistoryObject,TrainedModelObject):
        self.modelHistory = modelHistoryObject
        self.TrainedModel = TrainedModelObject
        # Obtain accuracy score by evalute function
        score = self.TrainedModel.evaluate(self.x_test, self.y_test, verbose=1)
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

    # Wait for Clicking a Key on Keyboard to Close All cv2 Windows
    def WaitKeyCloseWindows(self):
        # Wait until Clicking a Key on Keyboard
        cv2.waitKey(0)
        # Close All cv2 Windows
        cv2.destroyAllWindows()

    # Loading MNIST Dataset    
    def LoadMNIST(self):
        '''
        The MNIST (Modified National Institute of Standards and Technology database) dataset contains a training set of 60,000 images and
        a test set of 10,000 images of handwritten digits. 
        The handwritten digit images have been size-normalized and centered in a fixed size of 28*28 pixels.
        '''
        # loads the MNIST dataset
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            try:
                (x_train, y_train), (x_test, y_test)  = mnist.load_data()
                self.x_train = x_train
                self.y_train = y_train
                self.x_test = x_test
                self.y_test = y_test
                self.LoadMNISTRawDataOrPreparedData.emit(0)
            except:
                  QMessageBox.critical(None, "Instalation Error", "Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare!")

        else:
            QMessageBox.warning(None,"Dataset Exist","MNIST Dataset Already Loaded!")

    # Testing MNIST by OpenCV
    def TestMNIST(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        else:
            cv2.destroyAllWindows()
            plt.close("all")
            random_num = np.random.randint(0, len(self.x_train))
            image = self.x_train[random_num]
            image_scaled = cv2.resize(image, None, fx=5, fy=5, interpolation = cv2.INTER_NEAREST)
            cv2.imshow("Randon MNIST Sample", image_scaled)
            self.WaitKeyCloseWindows()

    # Testinig MNIST by Plots
    def PlotMNIST(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        else:
            cv2.destroyAllWindows()
            plt.close("all")
            plt.figure("Randon MNIST Samples:")
            for i in range(1,7):
                fig = plt.subplot(2,3,i)
                random_num = np.random.randint(0, len(self.x_train))
                plt.imshow(self.x_train[random_num])
                fig.set_title("Sample: " + str(i)) 

            plt.show()            
            
    # Preparing Data for Creating Model and Training it
    def PrepareData(self):
        if len(self.x_train.shape) < 4:
            # Store the number of Rows and Columns in a Sample
            img_rows, img_cols = self.x_train[0].shape[0:2]
            # Getting data in the right 'shape' needed for Keras
            # It need to add a 4th dimenion to the data by changing original image shape of (60000,28,28) to (60000,28,28,1)
            self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)

            # Store the Shape of a single image 
            self.input_shape = (img_rows, img_cols, 1)

            # change image type to float32 data type
            self.x_train = self.x_train.astype('float32') #uint8 originally
            self.x_test = self.x_test.astype('float32')

            # Normalize data by changing the range from (0 to 255) to (0 to 1)
            self.x_train /= 255.0
            self.x_test /= 255.0

            '''
            keras.utils.to_categorical is a utility function in Keras used for converting a vector of class labels (integers) into a binary class matrix, often referred to as one-hot encoding. This transformation is essential when working with classification problems, particularly when using loss functions like categorical_crossentropy.
            Purpose:
            The primary purpose of to_categorical is to prepare target labels for models that expect a one-hot encoded representation. For instance, if you have a classification problem with 3 classes, and a sample belongs to class 1, its label might be 1. After applying to_categorical, this 1 would be converted to [0, 1, 0], where the 1 is placed at the index corresponding to the class.
            Arguments:

                x:
                An array-like object containing the class values (integers from 0 to num_classes - 1) to be converted.
                num_classes:
                (Optional) The total number of classes. If not provided, it will be inferred from the maximum value in x plus one.

            Example:
            Python

            import numpy as np
            from tensorflow.keras.utils import to_categorical

            # Example class labels
            labels = np.array([0, 1, 2, 1, 0])

            # Convert to one-hot encoded format
            one_hot_labels = to_categorical(labels, num_classes=3)

            print(one_hot_labels)

            Output:
            Code

            [[1. 0. 0.]
            [0. 1. 0.]
            [0. 0. 1.]
            [0. 1. 0.]
            [1. 0. 0.]]

            When to Use:
            to_categorical is typically used when your target labels are represented as integer class indices, and your chosen loss function (e.g., categorical_crossentropy) and output layer activation (e.g., softmax) require a one-hot encoded representation of the target classes.
            '''      
            # One Hot Encode Outputs
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
            # Count the number columns in Hot Encoded Matrix 
            self.numberOfClasses = self.y_test.shape[1]
            self.numberOfPixels =  self.x_train.shape[1] * self.x_train.shape[2]

            self.LoadMNISTRawDataOrPreparedData.emit(1)

        else:
            QMessageBox.warning(None,"Data is Ready","Data Already Prepared!")

    # Show HotOneEncode Map
    def EncodeMap(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) == 4:
            cv2.destroyAllWindows()
            plt.close("all")
            path = os.path.normpath("icons/HotOneEncode.JPG")
            image = cv2.imread(path)
            cv2.imshow("Hot One Encode Map",image)
            self.WaitKeyCloseWindows()
        else:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")

     # Show HotOneEncode Map
   
    # Show Model Map
    def ModelMap(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) < 4:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")
        elif self.model == None:
            QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        else:
            cv2.destroyAllWindows()
            plt.close("all")
            path = os.path.normpath("icons/SimpleCNN.JPG")
            image = cv2.imread(path)
            cv2.imshow("Hot One Encode Map",image)
            self.WaitKeyCloseWindows()
    
    # Create a Simple CNN Model
    def CreateModel(self):
        if len(self.x_train) == 0 or len(self.y_train) == 0 or len(self.x_test) == 0  or len(self.y_test) == 0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) < 4:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")
        elif self.model == None:
             try:
                # Create Model
                model = Sequential()
                # To display the summary of the model so far, include the current output shape
                # Start model by passing an Input object to the model, so it knows its input shape which is 28 x 28 x 1
                model.add(keras.Input(shape=self.input_shape))
                # First Convolution Layer, contains:
                # 32 Filters with Kernel size 3x3 which Reduces layer size to 26 x 26 x 32
                # ReLU activation 
                model.add(Conv2D(32, kernel_size=(3, 3), activation='relu')) 
                # Second Convolution Layer, contains 64 Filters which Reduces layer size to 24 x 24 x 64
                model.add(Conv2D(64, (3, 3), activation='relu'))
                # MaxPooling with a kernel size of 2 x 2 reduces size to 12 x 12 x 64
                model.add(MaxPooling2D(pool_size=(2, 2)))
                # First Dropout with P setting (Dropout Rate) of 0.25 to reduce overfitting
                model.add(Dropout(0.25))
                # Flatten layer reshapes the tensor to have a shape equal to the number of elements in tensor, before input into Dense Layer
                # In this CNN it goes from 12 * 12 * 64 to 9216 * 1
                model.add(Flatten())
                # First Dense: Connect this layer to a Fully Connected/Dense layer of size 1 * 128
                model.add(Dense(128, activation='relu'))
                # Second Dropout layer to reduce overfitting
                model.add(Dropout(0.5))
                # Second Dence: Final Fully Connected/Dense layer with an output for each class (10)
                model.add(Dense(self.numberOfClasses, activation='softmax'))
                # Compile the Model, this creates an object that stores the model we just created
                # Set Optimizer to use Stochastic Gradient Descent (learning rate of 0.01)
                # Set Loss function to be categorical_crossentropy as it is suitable for multiclass problems
                # Finally, the Metrics (for Measuring Performance) to be accuracy
                model.compile(loss = 'categorical_crossentropy',
                            optimizer = SGD(0.01),
                            metrics = ['accuracy'])
                
                self.model = model
                # Capture Summary function to Display Model Layers and Parameters             
                self.modelSummary = self.ModelSummaryCapture(model)       
                QMessageBox.information(None,"Model Summary:",self.modelSummary)

             except:
                   QMessageBox.critical(None, "Instalation Error", "Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare!")
        else:
             QMessageBox.warning(None,"Model Exist","Model Already Exist!")  

    # Capture Model Summary function to Display Model Layers and Parameters
    def ModelSummaryCapture(self,model):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        ModelSummary = ""
        for i,e in enumerate(stringlist[0].split("\n")):
            if i != 1 and i != 3:
                if i == 2:
                    ModelSummary += "<pre>"+(str(e).replace("#","")).replace("┃"," ")+"</pre>"
                else:   
                    ModelSummary += "<pre>"+e+"</pre>"
                    
        return ModelSummary

    # Show Model Summary
    def ModelSummaryFunction(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) < 4:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")
        elif self.model == None:
            QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        elif self.modelSummary.strip() == "":
            QMessageBox.warning(None,"Model Summary not Exist","First Create a Model, Model Summary does not Exist!")
        else:
            cv2.destroyAllWindows()
            plt.close("all")
            QMessageBox.information(None,"Model Summary:",self.modelSummary)
  
    # Plotting Accuracy Charts
    def PlotAccuracy(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) < 4:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")
        elif self.model is None:
            QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        elif self.modelHistory is None:
            QMessageBox.warning(None,"Model not Trained","First Train the Model!")  
        else:
            cv2.destroyAllWindows()
            plt.close("all")
            plt.figure("Accuracy Plot")

            history_dict = self.modelHistory.history
            # Extract the accuracy and validation accuracy
            acc_values = history_dict['accuracy']
            val_acc_values = history_dict['val_accuracy']
            # Extract the loss and validation losses
            loss_values = history_dict['loss']
            val_loss_values = history_dict['val_loss']

            epochs = range(1, len(loss_values) + 1)

            line1 = plt.plot(epochs, val_acc_values, label='Validation/Test Accuracy')
            line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
            plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
            plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
            plt.xlabel('Epochs') 
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.show()

     # Plot Accuracy
   
    # Plotting Loss Charts
    def PlotLoss(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) < 4:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")
        elif self.model is None:
            QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        elif self.modelHistory is None:
            QMessageBox.warning(None,"Model not Trained","First Train the Model!")  
        else:
            cv2.destroyAllWindows()
            plt.close("all")
            plt.figure("Loss Plot")

            # Use the History object to get saved performance results
            history_dict = self.modelHistory.history

            # Extract the loss and validation losses
            loss_values = history_dict['loss']
            val_loss_values = history_dict['val_loss']

            # Get the number of epochs and create an array up to that number using range()
            epochs = range(1, len(loss_values) + 1)

            # Plot line charts for both Validation and Training Loss
            line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
            line2 = plt.plot(epochs, loss_values, label='Training Loss')
            plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
            plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
            plt.xlabel('Epochs') 
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.show()

     # Save Trained Model
   
    # Save Trained Model
    def SaveTrainedModel(self):
        if len(self.x_train) ==0 or len(self.y_train) ==0 or len(self.x_test) ==0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) < 4:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")
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
            self.TrainedModel.save("resources/models/SimpleCNN.keras", overwrite= True,include_optimizer=True)
            # Obtain accuracy score by evalute function
            score = self.TrainedModel.evaluate(self.x_test, self.y_test, verbose=1)
            QMessageBox.information(None,"Training Model Saved Successfully","Saving Path = resources/models folder in the root." + 
                                    "\nTest Loss: " +str(score[0]) + "\nTest Accuracy: " + str(score[1]) + 
                                    "\n In Colors and images manipulation page using this model for image to numbers operation.") 

    # Train the Model in another Thread
    def TrainModel(self,total_epochs):
        if len(self.x_train) == 0 or len(self.y_train) == 0 or len(self.x_test) == 0  or len(self.y_test) ==0:
            QMessageBox.warning(None,"No Data","First Load MNIST Dataset!")
        elif len(self.x_train.shape) < 4:
            QMessageBox.warning(None,"Data is not Ready","First Prepare Data!")
        elif self.model == None:
             QMessageBox.warning(None,"Model not Exist","First Create a Model!")  
        else:
            self.total_epochs = total_epochs
            self.batch_size = 32
            self.steps_per_epoch = int(len(self.x_train) / self.batch_size)
            self._is_running = True         
            try:
                # Running the Training Model in a seperate Thread to ba able cancel long running process and Inversion of Control (IOC)
                self.training_thread = TrainingThread(self._is_running,self.signal_emitter, self.total_epochs, self.steps_per_epoch, self.batch_size,self.model,self.x_train,self.y_train,self.x_test,self.y_test)
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
           
# Training Thread
class TrainingThread(QThread):
    def __init__(self,_is_running, signal_emitter, total_epochs,steps_per_epoch, batch_size, model,x_train,y_train,x_test,y_test):
        super().__init__()
        # Store Results for Creating Plots
        # In fit function specify datsets (x_train & y_train)
        # batch size (typically 16 to 128 depending on RAM of the System) 
        # number of epochs (usually 10 to 100) is number of train
        # validation datasets (x_test & y_test)
        # verbose = 1, sets our training to output performance metrics every epoch
        self.signal_emitter = signal_emitter
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test 
        self.y_test = y_test
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
                self.modelHistory = self.model.fit(self.x_train,
                                                    self.y_train,
                                                    batch_size = batch_size,
                                                    epochs = epochs,
                                                    verbose = 1,
                                                    validation_data = (self.x_test, self.y_test),
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
               
# Signal emitter for GUI Updates
class SignalEmitter(QObject):
    # Update Realtime Progress of Training - Content Message - Epoch
    batch_signal = pyqtSignal(str, int) 
    # Update Summary Result of Training of each Epoch - Content Message - Epoch
    epoch_signal = pyqtSignal(str, int) 
    # Cancel Training - Exit Message 
    exit_signal = pyqtSignal(str)  
    # Assigning results of Training including: History Log - Trained Model      
    modelTrainedSignal = pyqtSignal(object,object) 

# Custom callback to Simulating Keras Console Output Log
class ConsoleCallback(Callback):
    def __init__(self, _is_running,signal_emitter, steps_per_epoch, total_epochs):
        super().__init__()
        self.signal_emitter = signal_emitter
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        self.batch_times = []
        # This is a flag to control Canceling un-wanted long running process of training anytime
        self._is_running = _is_running
        self.epoch = None
    # Start of Each Epoch
    def on_epoch_begin(self, epoch, logs=None):
         if self._is_running == True:
            self.epoch = epoch
            self.epoch_start_time = time.time()
            self.batch_times = []
         else:
            self.model.stop_training = True
            raise RuntimeError("Training Canceled!")
    
    # Realtime Progress Update
    def on_train_batch_end(self, batch,logs=None):
        if self._is_running == True:
            now = time.time()
            self.batch_times.append(now)

            elapsed = now - self.epoch_start_time 
            # Estimate total epoch duration
            if len(self.batch_times) > 1:
                avg_batch_time = (self.batch_times[-1] - self.batch_times[0]) / (len(self.batch_times) - 1)
                estimated_total = avg_batch_time * self.steps_per_epoch
                percent = min(int((elapsed / estimated_total) * 100), 100)
            else:
                percent = int(((batch + 1) / self.steps_per_epoch) * 100)
           
            bar = '=' * (percent // 5) + '-' * (20 - (percent // 5))

            total_time_ms = elapsed * 1000  # Convert seconds to milliseconds
            step_time = int(total_time_ms / 1875)

            msg = (     
                f"Epoch {self.epoch + 1}/{self.total_epochs}\n"       
                f"{batch + 1}/{self.steps_per_epoch} "
                f"[{bar}] {elapsed:.2f}s {step_time}ms/step - loss: {logs['loss']:.4f} - accuracy: {logs.get('accuracy', 0):.4f}"
            )
            self.signal_emitter.batch_signal.emit(msg, self.epoch)

        else:
            self.model.stop_training = True
            raise RuntimeError("Training Canceled!")
    
    # End of each Epoch 
    def on_epoch_end(self, epoch, logs=None):
        if self._is_running == True:
            summary = (
                f"- val_loss: {logs.get('val_loss',0):.4f} - val_accuracy: {logs.get('val_accuracy', 0):.4f}"
            )
            self.signal_emitter.epoch_signal.emit(summary, epoch)

        else:
            self.model.stop_training = True
            raise RuntimeError("Training Canceled!")

# Popup window for Training log
class TrainingLogPopupClass(QDialog):
    def __init__(self, total_epochs, training_thread):
        super().__init__()
        self.training_thread = training_thread
        self.setWindowTitle("Training Log")
        self.setFixedSize(800, 600)
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
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Add QTextEdits to the layout
        for e in range(total_epochs):
            textEdit = QTextEdit()
            textEdit.setObjectName(f"text_edit{e+1}")
            textEdit.setReadOnly(True)
            textEdit.setFixedHeight(50)
            layout.addWidget(textEdit)
        # Set the container as the scroll area's widget
        self.scroll_area.setWidget(container)
        # Final layout for the dialog
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll_area)
    
    # Sending Realtime Progress of the Model Training
    def Show_Batch_Progress(self, message, epoch):
        textEdit = self.findChild(QTextEdit, "text_edit" + str(epoch + 1))
        textEdit.setText(message)
    
    # Sending Result of the Model Training for Each Epoch
    def Show_Epoch_Summary(self, message, epoch):
        # Find the widget by its object name
        textEdit = self.findChild(QTextEdit, "text_edit" + str(epoch + 1))
        messagePlus = textEdit.toPlainText() + message 
        textEdit.setText(messagePlus)
        self.scroll_area.verticalScrollBar().setValue(textEdit.y())#(epoch + 1) * 60)

    # Stop Training On Close Popup
    def closeEvent(self, event):
        self.training_thread.stop()


  # List of Libraries and Packages required to Install for running above Functions:
    '''
    1) Tensorflow: compatible with your Operation System (OS)(Software) and Your Computer System (Hardware).
                   (check cpu gpu support of the System and OS, check Python version)
    2) Keras:      when installing tensorflow, it installs keras and tensorboard automatically.
    3) OpenCV:     compatible with Tensorflow and Keras.
    4) Numpy:      It will be installed During Tensorflow or OpenCV installation, compatible with both.
    '''
    '''
    import os
    # Put import of Tensorflow and Keras in Try Block to check Correct Instalation
    try:
        # Below to bypass unnecessary log info
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        import keras
        from keras.datasets import mnist 
        from keras.utils import to_categorical
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
        from keras import backend as K
        from keras import callbacks
        from keras.callbacks import Callback
        from keras.optimizers import SGD 
        print(tf.__version__)
        print(keras.__version__)
    except:
        print("Check instalation of Tensorflow and Keras for Compatibility with OS and HardWare!")
    import cv2
    import time
    import matplotlib.pyplot as plt
    from os.path import isfile, join
    '''

