import io
import sys
import contextlib
import os
from os.path import isfile, join
import time
import tkinter as tk
import threading
from utilities.DeepLearningFoundationOperations import DownloadLogPopup, LogEmitter
from utilities.DLbyPyTorch import EarlyStop, DLbyPyTorch
from utilities.ScrollableMessageBox import show_scrollable_message
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    # print(tf.config.list_physical_devices('GPU'))
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as T
    from torchvision.utils import make_grid
except:
    print("Check instalation of torch for Compatibility with OS and HardWare!")
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
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except:
    print("You Should Install matplotlib Library!")
try: 
    from  PyQt6.QtGui import  QTextCursor   
    from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt
    from PyQt6.QtWidgets import QMessageBox,QTextEdit, QWidget, QVBoxLayout, QPushButton, QLabel, QDialog, QTextEdit,QScrollArea,QMainWindow,QApplication
except:
    print("You Should Install PyQt6 Library!")

class SimpleGANs(QObject):
    # Constructor to initialize all attributes and setup environment
    def __init__(self, parent=None):
        # Call parent class constructor
        super().__init__()
        # Initialize empty list to hold training data
        self.train_data = []
        # Set number of data points to generate
        self.observations = 2048
        # Set batch size for training
        self.batch_size = 128
        # Placeholder for DataLoader object
        self.train_loader = None
        # Placeholder for Generator model (shape mode)
        self.model_Generator_shape = None
        # Placeholder for Discriminator model (shape mode)
        self.model_Discriminator_shape = None
        # Binary Cross Entropy loss for Discriminator
        self.loss_fn = nn.BCELoss()
        # Mean Squared Error loss for Generator
        self.mse = nn.MSELoss()
        # Learning rate for optimizers
        self.lr = 0.0005
        # Logger object to emit messages to UI
        self.log_emitter = LogEmitter()
        # Set manual seed for reproducibility
        torch.manual_seed(0)
        # Select device: GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Define transformation pipeline: convert to tensor and normalize
        self.transformer = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        # Placeholder for input noise (numbers mode)
        self.numbers = None
        # Placeholder for current batch (numbers mode)
        self.batch = None
        # Placeholder for Generator model (numbers mode)
        self.model_Generator_numbers = None
        # Placeholder for Discriminator model (numbers mode)
        self.model_Discriminator_numbers = None
        # Placeholder for train_loader
        self.train_loader_grayimages = None
        # Placeholder for train_set
        self.train_set = []
        # Placeholder for train_set
        self.batch_size_grayImage = 64 # 32 - 64 - 128
        # Placeholder for Generator model (gray image mode)
        self.model_Generator_grayimages = None
        # Placeholder for Discriminator model (gray image mode)
        self.model_Discriminator_grayimages = None

    # Method to create synthetic exponential dataset
    def CreateDataset_Shape(self):
        # Create a tensor of zeros with shape [observations, 2]
        self.train_data = torch.zeros((self.observations, 2))
        # Fill first column with random values scaled by 50
        self.train_data[:, 0] = 50 * torch.rand(self.observations)
        # Fill second column with exponential values based on first column
        self.train_data[:, 1] = 1.08 ** self.train_data[:, 0]
        # Show success message in UI
        QMessageBox.information(None, "Success", "train_data Created Successfully.")

    # Method to plot the generated dataset
    def PlotDataset_Shape(self):
        # Check if training data exists
        if len(self.train_data) > 0:
            # Create a new figure for plotting
            fig = plt.figure(dpi=100, figsize=(8, 6))
            # Plot x vs y values as red dots
            plt.plot(self.train_data[:, 0], self.train_data[:, 1], ".", c="r")
            # Label x-axis
            plt.xlabel("values of x", fontsize=15)
            # Label y-axis with exponential expression
            plt.ylabel("values of $y=1.08^x$", fontsize=15)
            # Add title to the plot
            plt.title("An exponential growth shape", fontsize=20)
            # Display the plot
            plt.show()
        else:
            # Show warning if data hasn't been created
            QMessageBox.warning(None, "No train_data", "First, Create train_data!")

    # Method to prepare DataLoader for training
    def PrepareDataset_Shape(self):
        # Check if training data exists
        if len(self.train_data) > 0:
            # If DataLoader hasn't been created yet
            if self.train_loader is None:
                # Create DataLoader with batch size and shuffling
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
                # Show success message
                QMessageBox.information(None, "Success", "Data Prepared Successfully.")
            else:
                # Inform user that data is already prepared
                QMessageBox.information(None, "Data Prepared", "Data Already Prepared.")
        else:
            # Show warning if data hasn't been created
            QMessageBox.warning(None, "No train_data", "First, Create train_data!")

    # Method to display a sample batch from the DataLoader
    def DisplayDataset_Shape(self):
        # Check if DataLoader is ready
        if self.train_loader is None:
            # Show warning if data isn't prepared
            QMessageBox.warning(None, "Data not Ready", "First, Prepare Data!")
        else:
            # Get first batch from DataLoader
            batch0 = next(iter(self.train_loader))
            # Display batch contents in scrollable message box
            show_scrollable_message("batch0", str(batch0))

    # Method to create Generator and Discriminator models for shape data
    def CreateModels_Shape(self):
        # Ensure data is prepared before creating models
        if self.train_loader is None:
            QMessageBox.warning(None, "Data not Ready", "First, Prepare Data!")
        else:
            # Check if models haven't been created yet
            if self.model_Generator_shape is None or self.model_Discriminator_shape is None:

                # Define Discriminator model
                self.model_Discriminator_shape = nn.Sequential(
                    # Input layer: 2 features → 256 neurons
                    nn.Linear(2, 256),
                    # Activation function
                    nn.ReLU(),
                    # Dropout for regularization
                    nn.Dropout(0.3),
                    # Hidden layer: 256 → 128 neurons
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    # Hidden layer: 128 → 64 neurons
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    # Output layer: 64 → 1 neuron (binary classification)
                    nn.Linear(64, 1),
                    # Sigmoid activation to produce probability
                    nn.Sigmoid()
                ).to(self.device)

                # Define Generator model
                self.model_Generator_shape = nn.Sequential(
                    # Input layer: 2D noise → 16 neurons
                    nn.Linear(2, 16),
                    # Activation function
                    nn.ReLU(),
                    # Hidden layer: 16 → 32 neurons
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    # Output layer: 32 → 2D output (same shape as real data)
                    nn.Linear(32, 2)
                ).to(self.device)

                # Show success message
                QMessageBox.information(None, "Success", "Models Created Successfully.")
            else:
                # Inform user that models already exist
                QMessageBox.information(None, "Models Created", "Models Already are created.")

    # Method to start training the models using a separate thread
    def TrainModels_Shape(self):
        # Check if data is ready
        if self.train_loader is None:
            QMessageBox.warning(None, "Data not Ready", "First, Prepare Data!")
        # Check if models are created
        elif self.model_Generator_shape is None or self.model_Discriminator_shape is None:
            QMessageBox.warning(None, "No Models", "First, Create Models!")
        else:
            # Create and show plot window
            self.plot_window = PlotWindow()
            self.plot_window.show()
            # Create and show log popup window
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.cancel_button.setEnabled(True)
            self.DownloadLogPopup.show()
            # Append initial log message
            self.DownloadLogPopup.Append_Log("Training Models!\nWait ...")
            # Create training thread with all required parameters
            self.training_thread = TrainingShapeThread(
                self.plot_window,
                self.DownloadLogPopup,
                self.train_data,
                self.batch_size,
                self.device,
                self.model_Discriminator_shape,
                self.model_Generator_shape,
                self.lr,
                self.train_loader,
                self.loss_fn,
                self.mse
            )
            # Connect log signal to log popup
            self.training_thread.log_signal.connect(self.DownloadLogPopup.Append_Log)
            # Connect plot signal to plot window
            self.training_thread.plot_signal.connect(lambda fake, ep: self.plot_window.add_plot(fake, self.train_data, ep))
            # Connect cancel button to stop training
            self.DownloadLogPopup.cancel_button.clicked.connect(self.training_thread.stop)
            # Start training thread
            self.training_thread.start()

    # Method to save the trained Generator model
    def SaveModel_Shape(self):
        # Check if models are created
        if self.model_Generator_shape is None or self.model_Discriminator_shape is None:
            QMessageBox.warning(None, "No Model", "First, Create and Train the Model!")
        else:
            # Create directory if it doesn't exist
            if not os.path.exists("resources/models"):
                os.makedirs("resources/models", exist_ok=True)
            if not os.path.exists("resources/models/ExponentialGenerator.pt"):
                # Convert model to TorchScript format
                scripted = torch.jit.script(self.model_Generator_shape).to(self.device)
                # Save scripted model to file
                scripted.save('resources/models/ExponentialGenerator.pt')
            # Load the saved model for confirmation
            Loaded_Generator_Model = torch.jit.load('resources/models/ExponentialGenerator.pt', map_location=self.device)
            # Show confirmation message with model structure
            show_scrollable_message("Generator Model Saved/Loaded", "Model Saved/Loaded Successfully:\n" + str(Loaded_Generator_Model.eval()))

    # Method to test the saved Generator model by generating new samples
    def TestModel_Shape(self):
        # Check if the saved Generator model file exists
        if os.path.exists("resources/models/ExponentialGenerator.pt"):
            # Load the saved Generator model from disk
            Loaded_Generator_Model = torch.jit.load('resources/models/ExponentialGenerator.pt', map_location=self.device)
            # Generate random noise input for the Generator
            noise = torch.randn((self.batch_size, 2)).to(self.device)
            # Use the Generator to produce new synthetic data
            new_data = Loaded_Generator_Model(noise)
            # Create a new figure for plotting
            fig = plt.figure(dpi=100)
            # Plot generated samples in green
            plt.plot(new_data.detach().cpu().numpy()[:, 0],
                     new_data.detach().cpu().numpy()[:, 1], "*", c="g", label="Generated Samples")
            # Plot real training data in red with low opacity
            plt.plot(self.train_data[:, 0], self.train_data[:, 1], ".", c="r", alpha=0.1, label="Real Samples")
            # Add title to the plot
            plt.title("Inverted-U Shape Generated by GANs")
            # Set x-axis limits
            plt.xlim(0, 50)
            # Set y-axis limits
            plt.ylim(0, 50)
            # Display legend
            plt.legend()
            # Show the plot
            plt.show()
        else:
            # Show warning if model file is missing
            QMessageBox.warning(None, "Model not Saved", "First, Create, Train and Save the Model!")

    # Encode an integer position into a one-hot vector of specified depth
    def onehot_encoder(self, position, depth):
        # Create a zero vector of length 'depth'
        onehot = torch.zeros((depth,))
        # Set the element at 'position' to 1
        onehot[position] = 1
        # Return the one-hot encoded vector
        return onehot

    # Convert an integer to a one-hot vector with fixed depth of 100
    def int_to_onehot(self, number):
        # Use onehot_encoder to encode the number
        onehot = self.onehot_encoder(number, 100)
        # Return the one-hot vector
        return onehot

    # Convert a one-hot vector back to its integer representation
    def onehot_to_int(self, onehot):
        # Find the index of the maximum value (which is 1)
        number = torch.argmax(onehot)
        # Return the index as a Python integer
        return number.item()

    # Generate a sequence of integers between 0 and 19, scaled by 5
    def gen_sequence(self):
        # Randomly sample 10 integers in range [0, 20)
        indices = torch.randint(0, 20, (10, 1))
        # Multiply each index by 5 to scale the values
        values = indices * 5
        # Return the resulting tensor
        return values

    # Generate a batch of one-hot encoded vectors from a sequence
    def gen_batch(self):
        # Generate a sequence of scaled integers
        sequence = self.gen_sequence()
        # Convert each integer to a one-hot vector and collect them
        batch = [self.int_to_onehot(i).numpy() for i in sequence]
        # Convert list of arrays to a NumPy array
        batch = np.array(batch)
        # Convert NumPy array to a PyTorch tensor
        return torch.tensor(batch)

    # Convert one-hot encoded data back to integer values
    def data_to_num(self, data):
        # Find the index of the maximum value along the last dimension
        number = torch.argmax(data, dim=-1)
        # Return the tensor of decoded integers
        return number
    
    # Method to create dataset of one-hot encoded numbers
    def CreateDataset_Numbers(self):
        # Generate a batch of one-hot encoded vectors
        self.batch = self.gen_batch()
        # Convert one-hot vectors to integer labels
        self.numbers = self.data_to_num(self.batch)
        # Show success message in UI
        QMessageBox.information(None, "Success", "Dataset of Numbers Created Successfully.")

    # Method to display the generated number dataset
    def ShowDataset_Numbers(self):
        # Check if numbers exist and are non-empty
        if self.numbers is not None and len(self.numbers) > 0:
            # Show the numbers in a message box
            QMessageBox.information(None, "train_data Created Successfully", str(self.numbers))
        else:
            # Show warning if dataset hasn't been created
            QMessageBox.warning(None, "Dataset of Numbers", "First, Create Dataset of Numbers!")

    # Method to create Generator and Discriminator models for one-hot encoded data
    def CreateModels_Numbers(self):
        # Check if models haven't been created yet
        if self.model_Generator_numbers is None or self.model_Discriminator_numbers is None:

            # Define Discriminator model
            self.model_Discriminator_numbers = nn.Sequential(
                # Input layer: 100-dim one-hot vector → 1 output neuron
                nn.Linear(100, 1),
                # Sigmoid activation to produce probability
                nn.Sigmoid()
            ).to(self.device)

            # Define Generator model
            self.model_Generator_numbers = nn.Sequential(
                # Input layer: 100-dim noise → 100 neurons
                nn.Linear(100, 100),
                # Activation function
                nn.ReLU()
            ).to(self.device)

            # Show success message
            QMessageBox.information(None, "Success", "Models Created Successfully.")
        else:
            # Inform user that models already exist
            QMessageBox.information(None, "Models Created", "Models Already are created.")

    # Method to train the number-based GAN models
    def TrainModels_Numbers(self):
        # Check if models are created
        if self.model_Generator_numbers is None or self.model_Discriminator_numbers is None:
            QMessageBox.warning(None, "No Models", "First, Create Models!")
        else:
            # Create and show log popup window
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.cancel_button.setEnabled(True)
            self.DownloadLogPopup.show()
            # Append initial log message
            self.DownloadLogPopup.Append_Log("Training Models!\nWait ...")
            # Create training thread with all required parameters
            self.training_thread = TrainingNumbersThread(
                self.DownloadLogPopup,
                self.batch,
                self.device,
                self.model_Discriminator_numbers,
                self.model_Generator_numbers,
                self.lr,
                self.loss_fn,
                self.mse
            )
            # Connect log signal to log popup
            self.training_thread.log_signal.connect(self.DownloadLogPopup.Append_Log)
            # Connect plot signal to plot window (uses shape data for visualization)
            self.training_thread.plot_signal.connect(lambda fake, ep: self.plot_window.add_plot(fake, self.train_data, ep))
            # Connect cancel button to stop training
            self.DownloadLogPopup.cancel_button.clicked.connect(self.training_thread.stop)
            # Start training thread
            self.training_thread.start()

    # Method to save the trained Generator model for numbers
    def SaveModel_Numbers(self):
        # Check if models are created
        if self.model_Generator_numbers is None or self.model_Discriminator_numbers is None:
            QMessageBox.warning(None, "No Model", "First, Create and Train the Model!")
        else:
            # Create directory if it doesn't exist
            if not os.path.exists("resources/models"):
                os.makedirs("resources/models", exist_ok=True)
            if not os.path.exists("resources/models/NumbersGenerator.pt"):
                # Convert model to TorchScript format
                scripted = torch.jit.script(self.model_Generator_numbers).to(self.device)
                # Save scripted model to file
                scripted.save('resources/models/NumbersGenerator.pt')
            # Load the saved model for confirmation
            Loaded_Generator_Model = torch.jit.load('resources/models/NumbersGenerator.pt', map_location=self.device)
            # Show confirmation message with model structure
            QMessageBox.information(None, "Generator Model Saved/Loaded", "Model Saved/Loaded Successfully:\n" + str(Loaded_Generator_Model.eval()))

    # Method to test the saved Generator model by generating new samples
    def TestModel_Numbers(self):
        # Check if the saved Generator model file exists
        if os.path.exists("resources/models/NumbersGenerator.pt"):
            # Load the saved Generator model from disk
            Loaded_Generator_Model = torch.jit.load('resources/models/NumbersGenerator.pt', map_location=self.device)
            # Generate random noise input for the Generator
            noise = torch.randn((10, 100)).to(self.device)
            # Use the Generator to produce new synthetic data
            new_data = Loaded_Generator_Model(noise)

            # Create a sine-shaped reference dataset for comparison
            train_data = torch.zeros((self.observations, 2))
            # Generate x values in range [-5, 5]
            train_data[:, 0] = 10 * (torch.rand(self.observations) - 0.5)
            # Compute sine of x values
            train_data[:, 1] = torch.sin(train_data[:, 0])

            # Create a new figure for plotting
            fig = plt.figure(dpi=100, figsize=(8, 6))
            # Plot sine-shaped reference data in red
            plt.plot(train_data[:, 0], train_data[:, 1], ".", c="r")
            # Label x-axis
            plt.xlabel("values of x", fontsize=15)
            # Label y-axis with sine expression
            plt.ylabel("values of $y=sin(x)$", fontsize=15)
            # Add title to the plot
            plt.title("A sine shape", fontsize=20)
            # Show the plot
            plt.show()
        else:
            # Show warning if model file is missing
            QMessageBox.warning(None, "Model not Saved", "First, Create, Train and Save the Model!")

    # Method to prepare the DataLoader for training using grayscale images
    def PrepareDataset_GrayImages(self, train_set):
        
        # Optional: number of worker threads for data loading (commented out for experimentation)
        # num_workers = 4  # experiment with values between 2–8

        # Check if the training dataset is not empty
        if len(train_set) > 0:

            # Store the training dataset in the class instance
            self.train_set = train_set

            # Create a DataLoader for the grayscale image dataset
            self.train_loader_grayimages = torch.utils.data.DataLoader(
                
                # Dataset to load into the DataLoader
                train_set,

                # Batch size for loading data, defined as a class attribute
                batch_size = self.batch_size_grayImage,

                # Shuffle the data at each epoch to improve training robustness
                shuffle = True
            )

            # Display a message box indicating successful dataset preparation
            QMessageBox.information(
                None,
                "FashionMNIST Dataset Prepared",
                "FashionMNIST Dataset Prepared Successfully, Ready for Trainig."
            )
        
        # If the dataset is empty or not loaded
        else:
            # Display a warning message to prompt dataset loading
            QMessageBox.warning(
                None,
                "FashionMNIST is not Ready",
                "First, Download/Load Fashion-MINIST Dataset."
            )

    # Method to plot the first 32 grayscale images from the Fashion-MNIST training dataset
    def PlotMINIST(self):

        # Check if the DataLoader has been initialized
        if self.train_loader_grayimages is None:

            # Display a warning if the dataset is not ready
            QMessageBox.warning(
                None,
                "Dataset is not Ready",
                "First, Prepare the Dataset."
            )
        
        # If the DataLoader is ready
        else:
            # Close any previously opened OpenCV windows
            cv2.destroyAllWindows()

            # Close any previously opened matplotlib plots
            plt.close()

            # Retrieve the first batch of images and labels from the DataLoader
            images, labels = next(iter(self.train_loader_grayimages))

            # Create a grid of the first 32 images (8 columns x 4 rows)
            # Normalize the images for better visualization: center around 0.5 and scale
            grid = make_grid(0.5 - images / 2, 8, 4)

            # Convert the grid tensor to a NumPy array and transpose for correct display
            plt.imshow(grid.numpy().transpose((1, 2, 0)), cmap = "gray_r")

            # Hide axis ticks and labels
            plt.axis("off")

            # Display the image grid
            plt.show()

    # Method to create Generator and Discriminator models for Gray Images Mode
    def CreateModels_GrayImages(self):

        # Check if the training dataset has been prepared
        if self.train_loader_grayimages is None:
            # Warn the user to prepare the dataset first
            QMessageBox.warning(None, "Dataset is not Ready", "First, Prepare the Dataset.")

        # Check if the models haven't been created yet
        elif self.model_Discriminator_grayimages is None or self.model_Generator_grayimages is None:

            # Define the Discriminator model as a binary classifier
            self.model_Discriminator_grayimages = nn.Sequential(

                # Fully connected layer: input 784 (28x28 image) → 1024 units
                nn.Linear(784, 1024),

                # Activation function: ReLU
                nn.ReLU(),

                # Dropout layer to reduce overfitting
                nn.Dropout(0.3),

                # Fully connected layer: 1024 → 512 units
                nn.Linear(1024, 512),

                # Activation function: ReLU
                nn.ReLU(),

                # Dropout layer
                nn.Dropout(0.3),

                # Fully connected layer: 512 → 256 units
                nn.Linear(512, 256),

                # Activation function: ReLU
                nn.ReLU(),

                # Dropout layer
                nn.Dropout(0.3),

                # Final layer: 256 → 1 output (probability of being real)
                nn.Linear(256, 1),

                # Sigmoid activation to output probability
                nn.Sigmoid()
            ).to(self.device)

            # Define the Generator model to produce fake images from noise
            self.model_Generator_grayimages = nn.Sequential(

                # Fully connected layer: input 100 (noise vector) → 256 units
                nn.Linear(100, 256),

                # Activation function: ReLU
                nn.ReLU(),

                # Fully connected layer: 256 → 512 units
                nn.Linear(256, 512),

                # Activation function: ReLU
                nn.ReLU(),

                # Fully connected layer: 512 → 1024 units
                nn.Linear(512, 1024),

                # Activation function: ReLU
                nn.ReLU(),

                # Final layer: 1024 → 784 (flattened 28x28 image)
                nn.Linear(1024, 784),

                # Tanh activation to scale output between -1 and 1
                nn.Tanh()
            ).to(self.device)

            # Visualize the generator output before training
            self.see_output("beforTraining")

            # Notify user of successful model creation
            QMessageBox.information(None, "Success", "Models Created Successfully.")

        # If models already exist
        else:
            # Inform user that models are already created
            QMessageBox.information(None, "Models Created", "Models Already are created.")

    # Method to visualize generated images from the Generator model
    def see_output(self, beforeORafter):

        # Check if the Generator model is initialized
        if self.model_Generator_grayimages is None:
            # Warn the user to create the Generator model first
            QMessageBox.warning(None, "Model is not Ready", "First, Create Gray Images Model Generator.")

        # If the Generator model is ready
        else:
            # Close any OpenCV windows
            cv2.destroyAllWindows()

            # Close any existing matplotlib plots
            plt.close()

            # Set the plot title based on training stage
            if beforeORafter == "beforTraining":
                title = "Generated Images before Training"
            else:
                title = "Generated Images after Training"

            # Generate random noise input for the Generator
            noise = torch.randn(32, 100).to(device=self.device)

            # Generate fake samples using the Generator model
            fake_samples = self.model_Generator_grayimages(noise).cpu().detach()

            # Create a matplotlib figure to display the generated images
            plt.figure(title, dpi=72, figsize=(20, 10))

            # Loop through each generated image
            for i in range(32):
                # Create subplot for each image
                ax = plt.subplot(4, 8, i + 1)

                # Reshape the flattened image to 28x28 and normalize to [0, 1]
                image = (fake_samples[i] / 2 + 0.5).reshape(28, 28)

                # Display the image
                plt.imshow(image)

                # Remove x and y axis ticks
                plt.xticks([])
                plt.yticks([])

            # Show the plot
            plt.show()

    # Method to start training the models using a separate thread
    def TrainModels_GrayImages(self):

        # Check if the training dataset has been prepared
        if self.train_loader_grayimages is None:
            # Warn the user to prepare the dataset first
            QMessageBox.warning(None, "Data not Ready", "First, Prepare the Data!")

        # Check if both Generator and Discriminator models are created
        elif self.model_Generator_grayimages is None or self.model_Discriminator_grayimages is None:
            # Warn the user to create the models first
            QMessageBox.warning(None, "No Models Exist", "First, Create Models!")

        # If data and models are ready
        else:
            # Create a window to visualize training progress
            self.plot_window = PlotWindow(self.model_Generator_grayimages, self.device)

            # Display the plot window
            self.plot_window.show()

            # Create a popup window to show training logs
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

            # Enable the cancel button to allow stopping training
            self.DownloadLogPopup.cancel_button.setEnabled(True)

            # Display the log popup window
            self.DownloadLogPopup.show()

            # Add an initial log message to indicate training has started
            self.DownloadLogPopup.Append_Log("Training Models!\nWait ...")

            # Create a separate thread to handle training asynchronously
            self.training_thread = TrainingGrayImagesThread(

                # Reference to the plot window for visual updates
                self.plot_window,

                # Reference to the log popup for status updates
                self.DownloadLogPopup,

                # Batch size used during training
                self.batch_size_grayImage,

                # Device to run training on (CPU or GPU)
                self.device,

                # Discriminator model for training
                self.model_Discriminator_grayimages,

                # Generator model for training
                self.model_Generator_grayimages,

                # Learning rate for the optimizer
                self.lr,

                # DataLoader containing the training data
                self.train_loader_grayimages,

                # Loss function used during training
                self.loss_fn,

                # Mean squared error metric for evaluation
                self.mse
            )

            # Connect the thread's log signal to the log popup's append method
            self.training_thread.log_signal.connect(self.DownloadLogPopup.Append_Log)

            # Connect the thread's display signal to the plot window's display method
            self.training_thread.display_signal.connect(self.plot_window.display_signal)

            # Connect the cancel button to the thread's stop method to allow interruption
            self.DownloadLogPopup.cancel_button.clicked.connect(self.training_thread.stop)

            # Start the training thread
            self.training_thread.start()

    # Method to save the trained Generator model for Gray Images
    def SaveModel_GrayImages(self):

        # Check if both Generator and Discriminator models are created
        if self.model_Generator_grayimages is None or self.model_Discriminator_grayimages is None:
            # Warn the user to create and train the models first
            QMessageBox.warning(None, "No Model", "First, Create and Train the Model!")

        # If models are ready
        else:
            # Check if the directory for saving models exists
            if not os.path.exists("resources/models"):
                # Create the directory if it doesn't exist
                os.makedirs("resources/models", exist_ok=True)

            # Check if the Generator model file doesn't already exist
            if not os.path.exists("resources/models/GrayImagesGenerator.pt"):
                # Convert the Generator model to TorchScript format for saving
                scripted = torch.jit.script(self.model_Generator_grayimages).to(self.device)

                # Save the scripted model to the specified file path
                scripted.save('resources/models/GrayImagesGenerator.pt')

            # Load the saved Generator model from disk for confirmation
            Loaded_Generator_Model = torch.jit.load('resources/models/GrayImagesGenerator.pt', map_location=self.device)

            # Display a confirmation message with the loaded model's structure
            QMessageBox.information(
                None,
                "Generator Model Saved/Loaded",
                "Model Saved/Loaded Successfully:\n" + str(Loaded_Generator_Model.eval())
            )

    # Method to test the saved Generator model by generating new samples
    def TestModel_GrayImages(self):

        # Check if the saved Generator model file exists
        if os.path.exists("resources/models/GrayImagesGenerator.pt"):

            # Load the saved Generator model from disk using TorchScript
            Loaded_Generator_Model = torch.jit.load(
                'resources/models/GrayImagesGenerator.pt',
                map_location=self.device
            )

            # Generate random noise input for the Generator
            noise = torch.randn((self.batch_size_grayImage, 100)).to(self.device)

            # Use the Generator to produce new synthetic data from the noise
            fake_samples = Loaded_Generator_Model(noise).cpu().detach()

            # Loop through the first 32 generated samples
            for i in range(32):

                # Create subplot for each image (4 rows × 8 columns)
                ax = plt.subplot(4, 8, i + 1)

                # Reshape and normalize the image for display
                plt.imshow((fake_samples[i] / 2 + 0.5).reshape(28, 28))

                # Remove x-axis ticks
                plt.xticks([])

                # Remove y-axis ticks
                plt.yticks([])

            # Adjust vertical spacing between subplots
            plt.subplots_adjust(hspace=-0.6)

            # Display the generated images
            plt.show()

        # If the model file is missing
        else:
            # Show warning to prompt model creation and saving
            QMessageBox.warning(
                None,
                "Model not Saved",
                "First, Create, Train and Save the Model!"
            )

# Class to display training progress using matplotlib plots inside a scrollable PyQt window
class PlotWindow(QMainWindow):
    # Constructor to initialize the plot window
    def __init__(self, model_Generator_grayimages = None, device = None):
        # Call the parent class constructor
        super().__init__()

        # Set the window title
        self.setWindowTitle("Training Progress")

        # Set the initial window size
        self.resize(800, 700)

        # Create a scrollable area to hold plots
        self.scroll = QScrollArea()

        # Create a container widget to hold the layout and plot canvases
        self.container = QWidget()

        # Create a vertical layout for stacking multiple plots
        self.layout = QVBoxLayout(self.container)

        # Set the container widget inside the scroll area
        self.scroll.setWidget(self.container)

        # Allow the scroll area to resize its contents dynamically
        self.scroll.setWidgetResizable(True)

        # Set the scroll area as the central widget of the window
        self.setCentralWidget(self.scroll)

        self.model_Generator_grayimages = model_Generator_grayimages

        self.device = device

    # Method to add a new plot showing generated vs real samples
    def add_plot(self, fake_samples, train_data, epoch):
        # Create a new matplotlib figure with specified size
        fig = Figure(figsize=(6, 4))

        # Create a canvas to render the figure inside the PyQt window
        canvas = FigureCanvas(fig)

        # Set a minimum height for the canvas to ensure visibility
        canvas.setMinimumHeight(400)

        # Add a subplot to the figure for plotting data
        ax = fig.add_subplot(111)

        # Plot generated samples (fake data) in green with star markers
        ax.plot(fake_samples[:, 0], fake_samples[:, 1], "*", c="g", label="generated samples")

        # Plot real samples (training data) in red with dot markers and reduced opacity
        ax.plot(train_data[:, 0], train_data[:, 1], ".", c="r", alpha=0.1, label="real samples")

        # Set the title of the plot to indicate the current epoch
        ax.set_title(f"Epoch {epoch}")

        # Set the x-axis limits to focus on the relevant range
        ax.set_xlim(0, 50)

        # Set the y-axis limits to match the expected output range
        ax.set_ylim(0, 50)

        # Display the legend to differentiate between real and generated samples
        ax.legend()

        # Adjust the subplot margins for better spacing
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        # Add the canvas to the vertical layout so it appears in the scrollable view
        self.layout.addWidget(canvas)

        # Automatically scroll to the bottom to show the latest plot
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
    
    def display_signal(self, epoch):
        # Generate noise and produce fake samples
        noise = torch.randn(32, 100).to(self.device)
        fake_samples = self.model_Generator_grayimages(noise).cpu().detach()

        # Create a new matplotlib figure
        fig = Figure(figsize=(20, 10), dpi=72)
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(600)

        # Create a grid of subplots
        for i in range(32):
            ax = fig.add_subplot(4, 8, i + 1)
            image = (fake_samples[i] / 2 + 0.5).reshape(28, 28)
            ax.imshow(image, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Generated Images after Epoch {epoch}", fontsize=16)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3)

        # Add the canvas to the layout
        self.layout.addWidget(canvas)

        # Scroll to the bottom to show the latest plot
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
        
# Thread class to train GAN models on exponential shape data
class TrainingShapeThread(QThread):
    # Signal to emit log messages to the UI
    log_signal = pyqtSignal(str)

    # Signal to emit generated samples for plotting
    plot_signal = pyqtSignal(np.ndarray, int)

    # Constructor to initialize training parameters and models
    def __init__(self, plot_window, DownloadLogPopup, train_data, batch_size, device, model_Discriminator_shape, model_Generator_shape, lr, train_loader, loss_fn, mse):
        # Call parent constructor
        super().__init__()

        # Store reference to plot window for visual updates
        self.plot_window = plot_window

        # Store reference to log popup for training feedback
        self.DownloadLogPopup = DownloadLogPopup

        # Store training data
        self.train_data = train_data

        # Set batch size for training
        self.batch_size = batch_size

        # Set device (CPU or GPU)
        self.device = device

        # Set learning rate
        self.lr = lr

        # Store discriminator model
        self.model_Discriminator_shape = model_Discriminator_shape

        # Store generator model
        self.model_Generator_shape = model_Generator_shape

        # Initialize optimizer for discriminator
        self.optimD = torch.optim.Adam(self.model_Discriminator_shape.parameters(), lr=self.lr)

        # Initialize optimizer for generator
        self.optimG = torch.optim.Adam(self.model_Generator_shape.parameters(), lr=self.lr)

        # Create real label tensor (1s)
        self.real_labels = torch.ones((self.batch_size, 1))

        # Move real labels to device
        self.real_labels = self.real_labels.to(self.device)

        # Create fake label tensor (0s)
        self.fake_labels = torch.zeros((self.batch_size, 1))

        # Move fake labels to device
        self.fake_labels = self.fake_labels.to(self.device)

        # Initialize early stopping mechanism
        self.stopper = EarlyStop(patience=1000, min_delta=0.01)

        # Store data loader for training batches
        self.train_loader = train_loader

        # Store loss function (Binary Cross Entropy)
        self.loss_fn = loss_fn

        # Generate fixed noise input for generator
        self.noise = torch.randn((self.batch_size, 2)).to(self.device)

        # Store MSE loss function for performance evaluation
        self.mse = mse

        # Flag to track manual stop requests
        self._stop_requested = False

    # Method to evaluate generator output using MSE against true exponential curve
    def performance(self, fake_samples):
        # Compute true y values using exponential formula
        real = 1.08 ** fake_samples[:, 0]

        # Compute mean squared error between generated and true values
        mseloss = self.mse(fake_samples[:, 1], real)

        # Return the loss value
        return mseloss

    # Method to train discriminator on real samples
    def train_D_on_real(self, real_samples, optimD, real_labels):
        # Move real samples to device
        real_samples = real_samples.to(self.device)

        # Zero gradients for discriminator
        optimD.zero_grad()

        # Forward pass through discriminator
        out_D = self.model_Discriminator_shape(real_samples)

        # Compute loss against real labels
        loss_D = self.loss_fn(out_D, real_labels)

        # Backpropagation
        loss_D.backward()

        # Update discriminator weights
        optimD.step()

        # Return loss value
        return loss_D

    # Method to train discriminator on fake samples
    def train_D_on_fake(self, optimD, fake_labels):
        # Generate fake samples from generator
        fake_samples = self.model_Generator_shape(self.noise)

        # Zero gradients for discriminator
        optimD.zero_grad()

        # Forward pass through discriminator
        out_D = self.model_Discriminator_shape(fake_samples)

        # Compute loss against fake labels
        loss_D = self.loss_fn(out_D, fake_labels)

        # Backpropagation
        loss_D.backward()

        # Update discriminator weights
        optimD.step()

        # Return loss value
        return loss_D

    # Method to train generator to fool discriminator
    def train_G(self, optimG, real_labels):
        # Zero gradients for generator
        optimG.zero_grad()

        # Generate fake samples from generator
        fake_samples = self.model_Generator_shape(self.noise)

        # Forward pass through discriminator
        out_G = self.model_Discriminator_shape(fake_samples)

        # Compute loss against real labels (goal: fool discriminator)
        loss_G = self.loss_fn(out_G, real_labels)

        # Backpropagation
        loss_G.backward()

        # Update generator weights
        optimG.step()

        # Return loss and generated samples
        return loss_G, fake_samples

    # Method to log and plot training progress at each epoch
    def test_epoch(self, epoch, gloss, dloss, n, fake_samples):
        # Compute average generator loss
        g = gloss.item() / n

        # Compute average discriminator loss
        d = dloss.item() / n

        # Log progress to UI
        if self.DownloadLogPopup:
            self.DownloadLogPopup.Append_Log(f"at epoch {epoch+1}, G loss: {g}, D loss {d}")
            self.DownloadLogPopup.scroll_area.verticalScrollBar().setValue(self.DownloadLogPopup.scroll_area.verticalScrollBar().maximum())
            self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
            self.DownloadLogPopup.log_output.ensureCursorVisible()
            QApplication.processEvents()

        # Emit plot signal every 25 epochs or at start
        if epoch == 0 or (epoch + 1) % 25 == 0:
            fake = fake_samples.detach().cpu().numpy()
            self.plot_signal.emit(fake, epoch + 1)

    # Method to manually stop training
    def stop(self):
        # Set stop flag
        self._stop_requested = True

        # Disable cancel button in UI
        self.DownloadLogPopup.cancel_button.setEnabled(False)

    # Main training loop
    def run(self):
        # Loop over epochs
        for epoch in range(10000):
            # Check for manual stop
            if self._stop_requested:
                self.log_signal.emit("Training stopped by user.")
                break

            # Initialize loss accumulators
            gloss = 0
            dloss = 0

            # Loop over training batches
            for n, real_samples in enumerate(self.train_loader):
                # Train discriminator on real data
                loss_D = self.train_D_on_real(real_samples, self.optimD, self.real_labels)
                dloss += loss_D

                # Train discriminator on fake data
                loss_D = self.train_D_on_fake(self.optimD, self.fake_labels)
                dloss += loss_D

                # Train generator
                loss_G, fake_samples = self.train_G(self.optimG, self.real_labels)
                gloss += loss_G

            # Log and plot results
            self.test_epoch(epoch, gloss, dloss, n, fake_samples)
            QApplication.processEvents()

            # Evaluate generator performance
            gdif = self.performance(fake_samples).item()

            # Check early stopping condition
            if self.stopper.stop(gdif):
                break

        # Emit final log messages
        self.log_signal.emit(f"Epoch {epoch+1} completed")
        self.log_signal.emit("Training Finished!")
        time.sleep(1)
        self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
        self.DownloadLogPopup.log_output.ensureCursorVisible()

# Thread class to train GAN models on one-hot encoded number data
class TrainingNumbersThread(QThread):
    # Signal to emit log messages to the UI
    log_signal = pyqtSignal(str)

    # Signal to emit generated samples for plotting (not used here but defined)
    plot_signal = pyqtSignal(np.ndarray, int)

    # Constructor to initialize training parameters and models
    def __init__(self, DownloadLogPopup, batch, device, model_Discriminator_numbers, model_Generator_numbers, lr, loss_fn, mse):
        # Call parent constructor
        super().__init__()

        # Store reference to log popup for training feedback
        self.DownloadLogPopup = DownloadLogPopup

        # Store batch of one-hot encoded training data
        self.batch = batch

        # Set device (CPU or GPU)
        self.device = device

        # Set learning rate
        self.lr = lr

        # Store discriminator model
        self.model_Discriminator_numbers = model_Discriminator_numbers

        # Store generator model
        self.model_Generator_numbers = model_Generator_numbers

        # Initialize optimizer for discriminator
        self.optimD = torch.optim.Adam(self.model_Discriminator_numbers.parameters(), lr=self.lr)

        # Initialize optimizer for generator
        self.optimG = torch.optim.Adam(self.model_Generator_numbers.parameters(), lr=self.lr)

        # Get batch size from input data
        batch_size = self.batch.size(0)

        # Create real label tensor (1s) and move to device
        self.real_labels = torch.ones((batch_size, 1)).to(self.device)

        # Create fake label tensor (0s) and move to device
        self.fake_labels = torch.zeros((batch_size, 1)).to(self.device)

        # Initialize early stopping mechanism
        self.stopper = EarlyStop(patience=800, min_delta=0.01)

        # Store loss function (Binary Cross Entropy)
        self.loss_fn = loss_fn

        # Store MSE loss function for penalty calculation
        self.mse = mse

        # Flag to track manual stop requests
        self._stop_requested = False

    # Method to train both discriminator and generator in one step
    def train_D_G(self):
        # Generate random noise input for generator
        noise = torch.randn(10, 100).to(self.device)

        # Move real data to device
        true_data = self.batch.to(self.device)

        # Forward pass through discriminator with real data
        preds = self.model_Discriminator_numbers(true_data)

        # Compute loss for real data
        loss_D1 = self.loss_fn(preds, self.real_labels.reshape(10, 1))

        # Zero gradients and update discriminator
        self.optimD.zero_grad()
        loss_D1.backward()
        self.optimD.step()

        # Generate fake data from generator
        generated_data = self.model_Generator_numbers(noise)

        # Forward pass through discriminator with fake data
        preds = self.model_Discriminator_numbers(generated_data)

        # Compute loss for fake data
        loss_D2 = self.loss_fn(preds, self.fake_labels.reshape(10, 1))

        # Zero gradients and update discriminator again
        self.optimD.zero_grad()
        loss_D2.backward()
        self.optimD.step()

        # Generate fake data again for generator training
        generated_data = self.model_Generator_numbers(noise)

        # Forward pass through discriminator
        preds = self.model_Discriminator_numbers(generated_data)

        # Compute generator loss (goal: fool discriminator)
        loss_G = self.loss_fn(preds, self.real_labels.reshape(10, 1))

        # Compute penalty for non-divisible-by-5 outputs
        generated_indices = torch.argmax(generated_data, dim=-1)

        # Compute the remainder when each generated index is divided by 5
        remainders = torch.remainder(generated_indices, 5)

        # Convert remainders to float and reshape to [10, 1]
        remainders = remainders.to(torch.float).view(-1, 1)

        # Create a float tensor of zeros with the same shape
        target = torch.zeros_like(remainders).to(torch.float)

        # Compute the mean squared error between remainders and the zero target
        penalty = self.mse(remainders, target)

        # Apply penalty with weight
        penalty_weight = 0.1
        loss_G += penalty * penalty_weight

        # Zero gradients and update generator
        self.optimG.zero_grad()
        loss_G.backward()
        self.optimG.step()

        # Return generated data for evaluation
        return generated_data
   
   # Method to convert one-hot encoded data to integer labels
    def data_to_num(self, data):
        # Get index of highest value in each one-hot vector (returns LongTensor)
        number = torch.argmax(data, dim=-1)
        return number

    # Method to compute distance metric for early stopping
    def distance(self, generated_data):
        # Convert generated data to integer labels
        nums = self.data_to_num(generated_data)

        # Compute remainder when divided by 5 (still LongTensor)
        remainders = nums % 5

        # Convert remainders to float for MSE loss
        remainders = remainders.to(torch.float).view(-1, 1)

        # Create a float tensor of zeros with the same shape
        ten_zeros = torch.zeros_like(remainders).to(torch.float)

        # Compute MSE between remainders and zeros
        mseloss = self.mse(remainders, ten_zeros)
        return mseloss

    # Method to manually stop training
    def stop(self):
        # Set stop flag
        self._stop_requested = True

        # Disable cancel button in UI
        self.DownloadLogPopup.cancel_button.setEnabled(False)

    # Main training loop
    def run(self):
        # Flag to track if training completed normally
        check = True

        # Loop over epochs
        for i in range(10000):
            # Check for manual stop
            if self._stop_requested:
                self.log_signal.emit("Training stopped by user.")
                check = False
                break

            # Train models and get generated data
            generated_data = self.train_D_G()

            # Compute distance metric for early stopping
            dis = self.distance(generated_data)

            # Check early stopping condition
            if self.stopper.stop(dis):
                break

            # Log generated numbers every 50 epochs
            if i % 50 == 0:
                self.log_signal.emit(str(self.data_to_num(generated_data)))
                time.sleep(0.01)
                self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
                self.DownloadLogPopup.log_output.ensureCursorVisible()

        # Final log message if training completed normally
        if check:
            self.log_signal.emit("Training finished.")
            time.sleep(0.01)
            self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
            self.DownloadLogPopup.log_output.ensureCursorVisible()
  
# Thread class to train GAN models on grayscale image data
class TrainingGrayImagesThread(QThread):
    # Signal to emit log messages to the UI
    log_signal = pyqtSignal(str)

    # Signal to emit generated samples for display
    display_signal = pyqtSignal(int)

    # Constructor to initialize training parameters and models
    def __init__(self, plot_window, DownloadLogPopup, batch_size_grayImage , device, model_Discriminator_grayimages, model_Generator_grayimages, lr, train_loader_grayimages, loss_fn, mse):
        # Call the parent QThread constructor
        super().__init__()

        # Store reference to the plot window for visual updates
        self.plot_window = plot_window

        # Store reference to the log popup for training feedback
        self.DownloadLogPopup = DownloadLogPopup

        # Set the batch size for training
        self.batch_size = batch_size_grayImage

        # Set the device (CPU or GPU) for computation
        self.device = device

        # Set the learning rate (overrides passed-in value)
        self.lr = 0.0001

        # Store the discriminator model
        self.model_Discriminator_grayimages = model_Discriminator_grayimages

        # Store the generator model
        self.model_Generator_grayimages = model_Generator_grayimages
        
        # Initialize optimizer for discriminator using Adam
        self.optimD = torch.optim.Adam(self.model_Discriminator_grayimages.parameters(), lr=self.lr)

        # Initialize optimizer for generator using Adam
        self.optimG = torch.optim.Adam(self.model_Generator_grayimages.parameters(), lr=self.lr)

        # Create tensor of real labels (1s) and move to device
        self.real_labels = torch.ones((batch_size_grayImage, 1)).to(device)

        # Create tensor of fake labels (0s) and move to device
        self.fake_labels = torch.zeros((batch_size_grayImage, 1)).to(device)

        # Initialize early stopping mechanism with patience and delta
        self.stopper = EarlyStop(patience=1000, min_delta=0.01)

        # Store the data loader for training batches
        self.train_loader_grayimages = train_loader_grayimages

        # Set the loss function to Binary Cross Entropy
        self.loss_fn = nn.BCELoss()

        # Generate fixed noise input for generator and move to device
        self.noise = torch.randn((self.batch_size, 2)).to(device)

        # Store MSE loss function for evaluation purposes
        self.mse = mse

        # Flag to track manual stop requests
        self._stop_requested = False

    # Train discriminator on real samples
    def train_D_on_real(self, real_samples):
        # Reshape real samples to flat vectors and move to device
        r = real_samples.reshape(-1, 28*28).to(self.device)

        # Forward pass through discriminator with real samples
        out_D = self.model_Discriminator_grayimages(r)

        # Create label tensor filled with 1s (real labels)
        labels = torch.ones((r.shape[0], 1)).to(self.device)

        # Compute Binary Cross Entropy loss between predictions and real labels
        loss_D = self.loss_fn(out_D, labels)

        # Clear previous gradients for discriminator
        self.optimD.zero_grad()

        # Backpropagate the loss to compute gradients
        loss_D.backward()

        # Update discriminator weights using optimizer
        self.optimD.step()

        # Return the computed loss value
        return loss_D

    # Train discriminator on fake samples
    def train_D_on_fake(self):
        # Generate random noise input for generator and move to device
        noise = torch.randn(self.batch_size, 100).to(self.device)

        # Generate fake samples using generator
        generated_data = self.model_Generator_grayimages(noise)

        # Forward pass through discriminator with fake samples
        preds = self.model_Discriminator_grayimages(generated_data)

        # Compute loss between predictions and fake labels (0s)
        loss_D = self.loss_fn(preds, self.fake_labels)

        # Clear previous gradients for discriminator
        self.optimD.zero_grad()

        # Backpropagate the loss to compute gradients
        loss_D.backward()

        # Update discriminator weights using optimizer
        self.optimD.step()

        # Return the computed loss value
        return loss_D

    # Train generator to fool the discriminator
    def train_G(self):
        # Generate random noise input for generator and move to device
        noise = torch.randn(self.batch_size, 100).to(self.device)

        # Generate fake samples using generator
        generated_data = self.model_Generator_grayimages(noise)

        # Forward pass through discriminator with generated samples
        preds = self.model_Discriminator_grayimages(generated_data)

        # Compute loss between predictions and real labels (goal: fool discriminator)
        loss_G = self.loss_fn(preds, self.real_labels)

        # Clear previous gradients for generator
        self.optimG.zero_grad()

        # Backpropagate the loss to compute gradients
        loss_G.backward()

        # Update generator weights using optimizer
        self.optimG.step()

        # Return the computed loss value
        return loss_G

    # Method to manually stop training
    def stop(self):
        # Set stop flag to True
        self._stop_requested = True

        # Disable cancel button in UI to prevent further interaction
        self.DownloadLogPopup.cancel_button.setEnabled(False)

    # Main training loop executed in thread
    def run(self):
        try:
            # Emit message indicating training has started
            self.log_signal.emit("Training thread started.")

            # Emit message with number of batches in train loader
            self.log_signal.emit(f"Train loader has {len(self.train_loader_grayimages)} batches.")

            # Loop over epochs
            for i in range(50):
                # Check for manual stop request
                if self._stop_requested:
                    # Break out of epoch loop if stop requested
                    break

                # Emit message indicating current epoch
                self.log_signal.emit(f"Epoch {i+1} started.")

                # Initialize loss accumulators for generator and discriminator
                gloss = 0
                dloss = 0

                # Loop over training batches
                for n, (real_samples, _) in enumerate(self.train_loader_grayimages):
                    # Check for manual stop request
                    if self._stop_requested:
                        # Emit message and break out of batch loop
                        self.log_signal.emit("Training stopped by user.")
                        break

                    # Emit progress message every 100 batches
                    if n % 100 == 0:
                        self.log_signal.emit(f"Processing batch {n+1}/{len(self.train_loader_grayimages)}")

                    # Train discriminator on real samples and accumulate loss
                    loss_D = self.train_D_on_real(real_samples)
                    dloss += loss_D.item()

                    # Train discriminator on fake samples and accumulate loss
                    loss_D = self.train_D_on_fake()
                    dloss += loss_D.item()

                    # Train generator and accumulate loss
                    loss_G = self.train_G()
                    gloss += loss_G.item()

                # Compute average generator loss over batches
                gloss = gloss / n

                # Compute average discriminator loss over batches
                dloss = dloss / n

                # Emit display signal every 5 epochs
                if i % 5 == 0:
                    self.display_signal.emit(i + 1)

                # Emit loss summary for current epoch
                self.log_signal.emit(f"At epoch {i+1}, Discriminator Model loss: {dloss}, Generator Model loss: {gloss}")

                # Scroll log output to bottom in UI
                self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)

                # Ensure cursor is visible in log output
                self.DownloadLogPopup.log_output.ensureCursorVisible()

                # Process UI events to keep interface responsive
                QApplication.processEvents()

            # Emit message indicating training has finished
            self.log_signal.emit("Training Finished.")

        # Catch and log any exceptions that occur during training
        except Exception as e:
            self.log_signal.emit(f"Error during training: {str(e)}")