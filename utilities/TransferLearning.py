'''
Transfer learning is a machine learning technique in which knowledge gained through one task or dataset is used to improve model performance on another related task
and/or different dataset. 1. In other words, transfer learning uses what has been learned in one setting to improve generalization in another setting.
'''
from utilities.DeepLearningFoundationOperations import LogEmitter, Downloader, DownloadLogPopup
from utilities.CreateSimpleCNN import SignalEmitter, ConsoleCallback, TrainingLogPopupClass
from utilities.ScrollableMessageBox import show_scrollable_message
import os
from os.path import isfile, join
import time
import json
import math
try:
    os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax", "torch"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    # print(tf.config.list_physical_devices('GPU'))
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import keras
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
    from keras import callbacks, Model, optimizers
    from keras.callbacks import Callback
    from keras.datasets import mnist
    from keras.utils import to_categorical
    from keras.models import Sequential, load_model
    from keras.layers import Dropout, Dense, GlobalAveragePooling2D, BatchNormalization
    from keras import backend as K
    from keras.optimizers import SGD
    from keras.datasets import cifar10
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

class TransferLearning(QObject):
    def __init__(self,DLOperationsHandler,CreateSimpleCNNHandler, parent=None):
        super().__init__()
        # Internal Variable to Access Data inside All Functions in the Class
        self.input_size = (48,48)
        self.batch_size = 32
        self.epochs = None
        self.learning_rate = 1e-4
        self.model = None
        self.base_model = None
        self.class_labels = []
        ################
        self.models = {}
        self.accuracy = 50
        self.DLOperationsHandler = DLOperationsHandler
        self.CreateSimpleCNNHandler = CreateSimpleCNNHandler
        self.DownloadLogPopup = None
        self._is_running = False
        self.downloadResult = None
        self.log_emitter = LogEmitter()
        self.log_emitter.log_signal.connect(self.Append_Log)
        self.log_emitter.progressbar_signal.connect(self.Update_Progress)
        self.log_emitter.finished_signal.connect(self.On_Finished)
        self.LoadModelDetails()
        ##############
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.input_shape = ()
        self.numberOfClasses = 0
        self.numberOfPixels = 0
        self.modelSummary = ""
        self.modelHistory = None
        self.TrainedModel  = None
        self.training_thread = None
        self.steps_per_epoch = None
        self.signal_emitter = SignalEmitter()
        self.signal_emitter.batch_signal.connect(self.Show_Batch)
        self.signal_emitter.epoch_signal.connect(self.Show_Epoch_Summary)
        self.signal_emitter.exit_signal.connect(self.Exit_Message)
        self.signal_emitter.modelTrainedSignal.connect(self.Set_Model_Trained_Values)
        self.train_generator = []
        self.validation_generator = []

    # Consider|Attention:
    # Process Functions Contains Computer Vision Functions with Comments and Explanations
    # Rest of Functions are Pre-Processor and Helpers

    # Assigning Results of Training including: Trained Model and History Log
    def Set_Model_Trained_Values(self,modelHistoryObject,TrainedModelObject):
        self.modelHistory = modelHistoryObject
        self.TrainedModel = TrainedModelObject
        # Obtain accuracy score by evalute function
        if self.DownloadLogPopup is None:
               self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
        self.DownloadLogPopup.show()
        self.DownloadLogPopup.Append_Log("Training Finished Successfully.\n")
        self.EvaluateModel()

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
            self.BuildModel(modelType, filepath)

    # Updating ProgressBar
    def Update_Progress(self, percent):
        if self._is_running:
           self.DownloadLogPopup.Update_Progress(percent)

    # Updating Logs
    def Append_Log(self,message):
        if self._is_running:
            self.DownloadLogPopup.Append_Log(message)

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

    # Import Load Prepare Cifar10 Collection
    def Import_Load_Prepare_Cifar10(self):
        self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
        self.DownloadLogPopup.show()
        self.DownloadLogPopup.Append_Log("Waiting ...\nIt takes Seconds to Import Load Prepare Cifar10 Collection")
        """
        Load CIFAR-10 dataset, resize images, normalize, and one-hot encode labels.
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # Resize images to match input size
        self.x_train = np.array([cv2.resize(img, self.input_size) for img in self.x_train])
        self.x_test = np.array([cv2.resize(img, self.input_size) for img in self.x_test])

        # Normalize pixel values
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        # Dynamically determine number of classes
        self.numberOfClasses = len(np.unique(self.y_train))
        self.class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        # One-hot encode labels
        self.y_train = to_categorical(self.y_train, self.numberOfClasses)
        self.y_test = to_categorical(self.y_test, self.numberOfClasses)

        self.DownloadLogPopup.Append_Log("Success!\ncifar10 Collection Imported, Loaded and Prepared.")

    # Build Model
    def BuildModel(self, modelType, filepath):
        if self.numberOfClasses > 0:
            self.log_emitter.log_signal.emit("Loading model weights...")
            self.modelType = modelType
            match modelType:
                case "VGGNet16":
                    # Creating an empty VGGNet16 Model
                    self.base_model = VGG16(weights=None, include_top=False)

                case "VGGNet19":
                    # Creating an empty VGGNet19 Model
                    self.base_model = VGG19(weights=None, include_top=False)

                case "ResNet50":
                    # Creating an empty ResNet50 Model
                    self.base_model = ResNet50(weights=None, include_top=False)

                case "Xception":
                    # Creating an empty Xception Model
                    self.base_model = Xception(weights=None, include_top=False)

            # Loding Pre-Trained Weights into the Model
            self.base_model.load_weights(filepath, by_name=True, skip_mismatch=True)

            self.log_emitter.log_signal.emit("Pre-Trained Weight Loaded into the Base Model successfully.")

            # Freeze base model layers
            for layer in self.base_model.layers:
                layer.trainable = False

            # Add custom classification layers
            x = self.base_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dense(256, activation='relu')(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            top = Dense(self.numberOfClasses, activation='softmax')(x)

            # Combine base and top layers
            self.model = Model(inputs = self.base_model.input, outputs = top)

            # Compile model
            self.model.compile(loss='categorical_crossentropy',
                            optimizer=optimizers.Adam(learning_rate = self.learning_rate),
                            metrics=['accuracy'])

            # Capture Summary function to Display Model Layers and Parameters
            self.modelSummary = self.CreateSimpleCNNHandler.ModelSummaryCapture(self.model)
            show_scrollable_message("Model Summary:",self.modelSummary)

        else:
             QMessageBox.warning(None,"No Dataset","First, Load the Dataset.")

    # Check, Download, Load, Change, Create Model
    def CreateModel(self, modelType):
        # Open Log Popup
        self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
        self.DownloadLogPopup.show()

        self._is_running = True

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
                self.BuildModel(modelType, filepath)

        else:
            self.log_emitter.log_signal.emit("Error: 'models.json' not found or Not Contains Details for this Operation ( "+modelType+" ).\nPlease ensure the file exists in the root and contains Details for this Operation ( "+modelType+" ).")

    # Enhancing Data
    def Enhance_Dataset(self):
        if len(self.x_train) > 0:
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.show()
            self.DownloadLogPopup.Append_Log("Waiting ...\nIt takes Seconds to Enhance Dataset.")
            """
            Enhancing Data using data Generators.
            """
            train_datagen = ImageDataGenerator(horizontal_flip=False)
            val_datagen = ImageDataGenerator()

            self.train_generator = train_datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
            self.validation_generator = val_datagen.flow(self.x_test, self.y_test, batch_size=self.batch_size)
            self.DownloadLogPopup.Append_Log("Success!\nDataset Enhanced.")
        else:
            QMessageBox.warning(None,"No Dataset","First, Load the Dataset.")

    # Show Model Summary
    def ShowModelSummary(self):
        if self.model is not None:
            show_scrollable_message("Model Summary:",self.modelSummary)
        else:
             QMessageBox.warning(None,"No Model","First, Create Model.")

    # TrainingModel
    def TrainModel(self,total_epochs):
        if self.model == None or self.train_generator == None:
             QMessageBox.warning(None,"Model not Exist","First Create a Model and Enhance Dataset!")
        else:
            self.total_epochs = total_epochs
            self.batch_size = 32
            self._is_running = True
            self.number_of_train_samples = len(self.x_train)
            self.number_of_validation_samples = len(self.x_test)

            self.steps_per_epoch = math.ceil(self.number_of_train_samples / self.batch_size)
            self.validation_steps = math.ceil(self.number_of_validation_samples / self.batch_size)

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
            self.TrainedModel.save("resources/models/" + self.modelType + "_with_CustomClassifier_Cifar10.keras", overwrite= True,include_optimizer=True)
            # Obtain accuracy score by evalute function
            if self.DownloadLogPopup is None:
               self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.show()
            self.DownloadLogPopup.Append_Log("Training Model Saved Successfully.\nSaving Path = resources/models folder in the root.\n")
            self.EvaluateModel()

    # Evaluate Model
    def EvaluateModel(self):
        """
        Evaluate the trained model.
        """
        if self.modelHistory is None or self.TrainedModel is None or self.validation_generator is None:
            QMessageBox.warning(None,"Model not Trained","First, Load and Enhance Dataset and Train the Model!")
        else:
            if self.DownloadLogPopup is None:
               self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.show()
            self.DownloadLogPopup.Append_Log("Evaluation of the Model Started...\nIt takes several Minutes.\nWait...")
            scores = self.TrainedModel.evaluate(self.validation_generator,steps=self.validation_steps,verbose=1)
            self.DownloadLogPopup.Append_Log("Evaluation Result:\nTest Loss: "+str(scores[0])+"\nTest Accuracy: "+str(scores[1]))

    # Testing Model
    def TestingModel(self):
        if self.TrainedModel is not None:
            scale = 4.0
            """
            Predict and display random test images with predicted labels.
            """
            for i in range(min(10, len(self.x_test))):
                rand = np.random.randint(0, len(self.x_test))
                input_im = self.x_test[rand]
                imageL = cv2.resize(input_im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                input_im = input_im.reshape(1, self.input_size[0], self.input_size[1], 3)
               
                res = int(np.argmax(self.TrainedModel.predict(input_im, verbose=1)[0]))
             
                if res < len(self.class_labels):
                    pred = self.class_labels[res]
                else:
                    pred = "Unknown"

                BLACK = [0, 0, 0]
                expanded_image = cv2.copyMakeBorder(imageL, 0, 0, 0, imageL.shape[0]*2, cv2.BORDER_CONSTANT, value=BLACK)
                cv2.putText(expanded_image, pred, (300, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 0), 2)
                cv2.imshow("Prediction", expanded_image)
                cv2.waitKey(0)

            cv2.destroyAllWindows()

        else:
            _ = self.Import_Load_Prepare_Cifar10()
            _a = self.Enhance_Dataset()
            paths = ["resources/models/VGGNet16_with_CustomClassifier_Cifar10.keras",
                        "resources/models/VGGNet19_with_CustomClassifier_Cifar10.keras",
                        "resources/models/ResNet50_with_CustomClassifier_Cifar10.keras",
                        "resources/models/Xception_with_CustomClassifier_Cifar10.keras"]
            for i in paths:
                if os.path.exists(i):
                    modelType = (i.split("/")[2]).split("_")[0]
                    match modelType:
                        case "VGGNet16":
                            self.TrainedModel = keras.models.load_model(i)   
                            self.TrainedModel.load_weights(i)    
                            self.TestingModel()                        
                            break

                        case "VGGNet19":
                            self.TrainedModel = keras.models.load_model(i)   
                            self.TestingModel()    
                            break

                        case "ResNet50":
                            self.TrainedModel = keras.models.load_model(i)   
                            self.TestingModel()    
                            break

                        case "Xception":
                            self.TrainedModel = keras.models.load_model(i)    
                            self.TestingModel()    
                            break

                        case _ :
                            QMessageBox.warning(None,"No Model","First, Create and Train a Model.")
                            break
                       
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

                # Callback Function for Communication with Model during Training for: Canceling Training, Updating UI by Logs, Exporting Results of Training
                self.ConsoleCallback = ConsoleCallback(self._is_running,self.signal_emitter, self.steps_per_epoch, self.total_epochs)
                # Train Function returning History
                self.modelHistory = self.model.fit( self.train_generator,
                                                    steps_per_epoch = self.steps_per_epoch,
                                                    epochs = self.total_epochs,
                                                    validation_data = self.validation_generator,
                                                    verbose = 1,
                                                    validation_steps = self.validation_steps,
                                                    callbacks = [self.ConsoleCallback]
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
