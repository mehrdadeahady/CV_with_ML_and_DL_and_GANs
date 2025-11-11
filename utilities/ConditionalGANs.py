import io
import sys
import contextlib
import os
from os.path import isfile, join
import time
import shutil
import random
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
    from torchvision.datasets import ImageFolder
except:
    print("Check instalation of torch for Compatibility with OS and HardWare!")
try:
    import numpy as np
except:
    print("You Should Install numpy Library")
try:
    import PIL
    from PIL import Image
except:
    print("You Should Install pillow Library")
try:
    import pandas as pd
except:
    print("You Should Install pandas Library")
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
    from PyQt6.QtGui import  QTextCursor   
    from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt
    from PyQt6.QtWidgets import QMessageBox,QTextEdit, QWidget, QVBoxLayout, QPushButton, QLabel, QDialog, QTextEdit,QScrollArea,QMainWindow,QApplication
except:
    print("You Should Install PyQt6 Library!")

# Define the ConditionalGANs class, which inherits from QObject to support Qt signals and slots
class ConditionalGANs(QObject):

    # Constructor method to initialize the ConditionalGANs instance
    def __init__(self, parent=None):
        # parent: optional reference to a parent QObject, used in Qt applications

        # Call the constructor of the parent QObject class
        super().__init__()

        # Set a fixed random seed for reproducibility in training and generation
        torch.manual_seed(0)

        # Initialize placeholder for training data (will be loaded later)
        self.train = None

        # Define path to dataset containing images with eyeglasses
        self.G = 'kagglehub/glasses/G/'

        # Define path to dataset containing images without eyeglasses
        self.NoG = 'kagglehub/glasses/NoG/'

        # Set the number of images to process in each training batch
        self.batch_size = 16

        # Set the target image size (height and width) for preprocessing
        self.imagesSize = 256

        # Define a transformation pipeline to preprocess images:
        # - Resize to the target dimensions
        # - Convert to tensor format
        # - Normalize pixel values to the range [-1, 1]
        self.transform = T.Compose([
            # Resize each image to (256, 256)
            T.Resize((self.imagesSize, self.imagesSize)),
            # Convert image to PyTorch tensor
            T.ToTensor(),
            # Normalize RGB channels to mean=0, std=1 (range [-1, 1])
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Initialize placeholder for the dataset object
        self.data_set = None

        # Create an empty list to hold combined image-label pairs
        self.combined_data = []

        # Initialize placeholder for the DataLoader used during training
        self.data_loader = None

        # Create a custom log emitter to send messages to the UI or console
        self.log_emitter = LogEmitter()

        # Set the device for computation: use GPU if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize placeholder for the generator model (will be defined later)
        self.generator = None

        # Initialize placeholder for the critic/discriminator model (will be defined later)
        self.critic = None

        # Set the dimensionality of the latent noise vector used for image generation
        self.z_dim = 100

        # Placeholder for males with glasses
        self.z_male_g = None
        # Placeholder for females with glasses
        self.z_female_g = None
        # Placeholder for males without glasses
        self.z_male_ng = None
        # Placeholder for females without glasses
        self.z_female_ng = None
   
    # Method to organize the eyeglasses dataset into two folders: with glasses and without glasses
    def ArrangeEyeGlassesDataset(self):
        # Check if the dataset files and folders exist and contain enough files
        if os.path.exists("kagglehub/train.csv") and os.path.exists("kagglehub/faces") and self.CountFilesInPath("kagglehub/glasses") + self.CountFilesInPath("kagglehub/faces") >= 5000:
            if self.train is None:
                # Load the training metadata from CSV
                self.train = pd.read_csv("kagglehub/train.csv")
            # Check if dataset needs to be arranged or is incomplete
            if not os.path.exists("kagglehub/glasses/G") or not os.path.exists("kagglehub/glasses/NoG") or self.CountFilesInPath("kagglehub/glasses/NoG") < 2000 or self.CountFilesInPath("kagglehub/glasses/G") < 2000:
                # Create a popup window to show training logs
                self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

                # Enable the cancel button to allow stopping training
                self.DownloadLogPopup.cancel_button.setEnabled(False)

                # Display the log popup window
                self.DownloadLogPopup.show()

                # Add an initial log message to indicate training has started
                self.DownloadLogPopup.Append_Log("Arranging the Dataset!\nWait ...")
                
                # Set the 'id' column as the index for easy lookup
                self.train.set_index('id', inplace=True)
                # Create folders for glasses and no-glasses images if they don't exist
                os.makedirs(self.G, exist_ok=True)
                os.makedirs(self.NoG, exist_ok=True)
                # Define the source folder containing face images
                folder = 'kagglehub/faces'
                # Loop through image IDs and sort them based on the 'glasses' label
                for i in range(1, 4501):
                    # Construct the original image path
                    oldpath = f"{folder}face-{i}.png"
                    # Determine the destination path based on the glasses label
                    if self.train.loc[i]['glasses'] == 0:
                        newpath = f"{self.NoG}face-{i}.png"
                    elif self.train.loc[i]['glasses'] == 1:
                        newpath = f"{self.G}face-{i}.png"
                    # Move the image if it exists
                    if os.path.exists(oldpath):
                        shutil.move(oldpath, newpath)
                # Show a message explaining that the dataset is arranged but may need manual cleanup
                self.DownloadLogPopup.Append_Log("Dataset Arranged\n" +
                    "The classification column *glasses* in the file *train.csv* is not perfect.\n" +
                    "In subfolder G , most images have glasses, but about 10% of them have no glasses.\n" +
                    "Similarly, in subfolder NoG, about 10% of them actually have glasses. You need to manually fix this.\n" +
                    "This is important for training so :\n" +
                    "Manually move images in the two folders so that one contains only images with glasses and the other images without glasses.\n" +
                    "Fixing data problems is part of daily routine of data scientist!")
            # If dataset is already arranged, show confirmation
            else:
                QMessageBox.information(None, "Dataset aaranged", "Dataset Already Arranged.")
        # If required files are missing, prompt the user to download and copy the dataset
        else:
            QMessageBox.information(None, "No Dataset", "First, Download and Copy the Dataset to the root of Project.")

    # Method to count the number of files in a given directory path
    def CountFilesInPath(self, path):
        # Check if the specified path exists
        if os.path.exists(path):
            # Initialize a counter for files
            count = 0
            # Traverse the directory tree recursively
            for root_dir, cur_dir, files in os.walk(path):
                # Add the number of files in the current directory
                count += len(files)
            # Return the total count of files
            return count
        # If the path doesn't exist, return zero
        else:
            return 0

    # Method to display a sample of images with or without glasses based on the sender button
    def ShowEyeGlassesImages(self, sender):
        # Initialize the directory variable
        directory = None
        # Match the sender to determine which folder to display
        match sender:
            # If the button is for displaying images with glasses
            case "pushButton_DisplayImagesWithGlasses_ConditionalGANs":
                directory = self.G
            # If the button is for displaying images without glasses
            case "pushButton_DisplayImagesWithoutGlasses_ConditionalGANs":
                directory = self.NoG

        # List all image filenames in the selected directory
        imgs = os.listdir(directory)
        # Set a fixed seed for reproducible sampling
        random.seed(42)
        # Randomly select 16 image filenames from the directory
        samples = random.sample(imgs, 16)
        # Create a figure to display the images
        fig = plt.figure(dpi=100, figsize=(8, 2))
        # Loop through the selected samples and display each image
        for i in range(16):
            # Create a subplot for each image
            ax = plt.subplot(2, 8, i + 1)
            # Open the image file
            img = Image.open(f"{directory}{samples[i]}")
            # Display the image
            plt.imshow(img)
            # Remove x and y axis ticks
            plt.xticks([])
            plt.yticks([])
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=-0.01, hspace=-0.01)
        # Show the final image grid
        plt.show()
   
    # Method to add one-hot and tensor labels to each image in the dataset
    def AddLabels(self):
        # Check if training metadata and image folders are available and contain enough images
        if self.train is not None and os.path.exists("kagglehub/glasses/G") and os.path.exists("kagglehub/glasses/NoG") and self.CountFilesInPath("kagglehub/glasses/NoG") > 2000 and self.CountFilesInPath("kagglehub/glasses/G") > 2000:
            # Proceed only if labels haven't already been added
            if len(self.combined_data) <= 0:
                # Load images from both folders using torchvision's ImageFolder
                self.data_set = torchvision.datasets.ImageFolder(root=r"kagglehub/glasses", transform=self.transform)

                # Initialize a popup window to show log messages during label processing
                self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

                # Enable the cancel button in the popup
                self.DownloadLogPopup.cancel_button.setEnabled(True)

                # Display the popup window
                self.DownloadLogPopup.show()

                # Log the start of the labeling process
                self.DownloadLogPopup.Append_Log("Adding Labels Started!\nIt takes several minutes.\nWait ...")
                                    
                # Iterate over the dataset to assign labels and prepare augmented image data
                for i, (img, label) in enumerate(self.data_set):

                    # Create a one-hot encoded label vector of size 2 (e.g., [1, 0] for class 0, [0, 1] for class 1)
                    onehot = torch.zeros((2))
                    
                    # Set the appropriate index in the one-hot vector to 1 based on the label
                    onehot[label] = 1

                    # Initialize a tensor to hold two label-specific channels, same spatial size as the image
                    channels = torch.zeros((2, self.imagesSize, self.imagesSize))

                    # If the label is 0, fill the first channel with ones
                    if label == 0:
                        channels[0, :, :] = 1

                    # Otherwise, fill the second channel with ones
                    else:
                        channels[1, :, :] = 1

                    # Concatenate the original image with the label channels along the channel dimension
                    img_and_label = torch.cat([img, channels], dim=0)

                    # Append the original image, label, one-hot vector, and concatenated image-label tensor to the combined dataset
                    self.combined_data.append((img, label, onehot, img_and_label))

                # Log the completion of the labeling process
                self.DownloadLogPopup.Append_Log("Adding Labels finished.\nNow prepare the Dataset.")
            # If labels are already added, show a message
            else:
                QMessageBox.information(None, "Labels are Ready", "Labels already added.")
        # If dataset is not ready, prompt the user to prepare it first
        else:
            QMessageBox.information(None, "Dataset is not Ready", "First, Download and Copy the Dataset to the root of Project and Arrange it.")

    # Method to prepare the dataset for training by creating a DataLoader
    def PrepareDataset(self):
        # Proceed only if labeled data is available
        if len(self.combined_data) > 0:
            # Create a DataLoader to batch and shuffle the labeled data
            self.data_loader = torch.utils.data.DataLoader(
                self.combined_data,
                batch_size = self.batch_size,
                shuffle = True
            )
            # Notify the user that the dataset is ready for training
            QMessageBox.information(None, "Dataset Prepared", "Dataset Prepared, ready for training.")
        # If labels are not added yet, prompt the user to add them first
        else:
            QMessageBox.information(None, "Labels are not Ready", "First, Add Labels.")

    # Method to initialize weights of model layers using custom rules
    def weights_init(self, m):
        # m: a layer/module from the model passed during initialization

        # Get the class name of the layer to identify its type
        classname = m.__class__.__name__

        # If the layer is a convolutional layer, initialize weights with normal distribution (mean=0, std=0.02)
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        # If the layer is a batch normalization layer, initialize weights and biases
        elif classname.find('BatchNorm') != -1:
            # Initialize weights with normal distribution (mean=1, std=0.02)
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            # Set bias values to zero
            nn.init.constant_(m.bias.data, 0)

    # Method to create generator and critic models and initialize their weights
    def CreateModels_InitializeWeights(self):
        # Check if models haven't been created yet
        if self.generator is None or self.critic is None:
            # Set number of image channels (RGB = 3)
            img_channels = 3

            # Set base number of features for model layers
            features = 16

            # Create the generator model with input size = latent vector + label channels
            self.generator = Generator(self.z_dim + 2, img_channels, features).to(self.device)

            # Create the critic (discriminator) model with input size = image channels + label channels
            self.critic = Critic(img_channels + 2, features).to(self.device)

            # Initialize weights of the generator using the custom method
            self.weights_init(self.generator)

            # Initialize weights of the critic using the custom method
            self.weights_init(self.critic)

            # Show a message indicating models were created and initialized
            QMessageBox.information(None, "Models Created", "Models created and weights initialized.")
        # If models already exist, notify the user
        else:
            QMessageBox.information(None, "Models Exist", "Models already created and weights initialized.")

    # Method to start training the Conditional GAN model
    def TrainModel(self):
        # Check if the training dataset has been prepared
        if self.data_loader is None:
            # Warn the user to prepare the dataset first
            QMessageBox.warning(None, "Data is not Ready", "First, Prepare the Dataset!")

        # Check if both Generator and Discriminator models are created
        elif self.generator is None or self.critic is None:
            # Warn the user to create the models first
            QMessageBox.warning(None, "No Models Exist", "First, Create Models!")

        # If data and models are ready
        else:
            # Create a window to visualize training progress
            self.plot_window = PlotWindow(self.device, self.generator, self.z_dim)

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
            self.training_thread = TrainingConditionalGANsThread(
                # Reference to the plot window for visual updates
                self.plot_window,

                # Reference to the log popup for status updates
                self.DownloadLogPopup,

                # Batch size used during training
                self.batch_size,

                # Device to run training on (CPU or GPU)
                self.device,

                # Discriminator model for training
                self.critic,

                # Generator model for training
                self.generator,

                # DataLoader containing the training data
                self.data_loader,

                # Dimensionality of the latent noise vector
                self.z_dim
            )

            # Connect the thread's log signal to the log popup's append method
            self.training_thread.log_signal.connect(self.DownloadLogPopup.Append_Log)

            # Connect the thread's display signal to the plot window's display method
            self.training_thread.display_signal.connect(self.plot_window.plot_epoch)

            # Connect the cancel button to the thread's stop method to allow interruption
            self.DownloadLogPopup.cancel_button.clicked.connect(self.training_thread.stop)

            # Start the training thread
            self.training_thread.start()

    # Method to load a previously trained Generator model from disk
    def LoadTrainedModel(self):
        # Check if the generator model has been created
        if self.generator is None:
            # Show a warning message prompting the user to create the model first
            QMessageBox.warning(
                None,                          # No parent widget
                "Model does not exist",          # Title of the warning dialog
                "First, Create the Model!"     # Message body
            )
            # Return False to indicate loading failed
            return False
        else:
            # Check if the saved Generator model file exists on disk
            if os.path.exists("resources/models/conditional_gan.pth"):
                # Load the model weights into the generator
                self.generator.load_state_dict(torch.load("resources/models/conditional_gan.pth",
                                                          map_location=self.device))
                # Set the generator to evaluation mode
                self.generator.eval()
                # Return True to indicate successful loading
                return True
            # If the model file does not exist
            else:
                # Show a warning message prompting the user to train and save the model first
                QMessageBox.warning(
                    None,                          # No parent widget
                    "Model not Saved",             # Title of the warning dialog
                    "First, Train and Save the Model!"  # Message body
                )
                # Return False to indicate loading failed
                return False

    # Handles image generation and visualization based on the selected UI button.
    def GenerateAndDisplayImages(self, sender):
        # Checks if the trained model is loaded; exits early if not
        if not self.LoadTrainedModel():
            # Stops execution if model loading failed
            return
               
        # Generates random noise for the "genuine" image batch with shape (32, z_dim, 1, 1)
        noise_g = torch.randn(32, self.z_dim, 1, 1)
        
        # Initializes a tensor of zeros for the "genuine" image labels with shape (32, 2, 1, 1)
        labels_g = torch.zeros(32, 2, 1, 1)
        
        # Sets the first label channel to 1 for all "genuine" images
        labels_g[:,0,:,:] = 1
        
        # Generates random noise for the "non-genuine" image batch with shape (32, z_dim, 1, 1)
        noise_ng = torch.randn(32, self.z_dim, 1, 1)
        
        # Initializes a tensor of zeros for the "non-genuine" image labels with shape (32, 2, 1, 1)
        labels_ng = torch.zeros(32, 2, 1, 1)
        
        # Sets the second label channel to 1 for all "non-genuine" images
        labels_ng[:,1,:,:] = 1
        
        # Defines a list of weights for interpolation or blending purposes
        weights = [0, 0.25, 0.5, 0.75, 1]


        # Match sender input to determine generation logic
        match sender:
            # Case 1: Generate 32 images with glasses
            case "pushButton_SelectImagesWithEyeGlasses_ConditionalGANs":   
                # Concatenates noise and labels along the channel dimension and moves the tensor to the selected device
                noise_and_labels = torch.cat([noise_g, labels_g], dim=1).to(self.device)
                
                # Passes the combined tensor through the generator to produce fake images, then moves them to CPU and detaches from computation graph
                fake = self.generator(noise_and_labels).cpu().detach()
                
                # Stores the first noise vector (assumed to represent a male image) for later use
                self.z_male_g = noise_g[0]
                
                # Stores the 15th noise vector (assumed to represent a female image) for later use
                self.z_female_g = noise_g[14]
                
                # Initializes a matplotlib figure with specified size and resolution
                plt.figure(figsize=(20, 10), dpi=50)
                
                # Loops through all 32 generated images
                for i in range(32):
                    # Creates a subplot in a 4x8 grid for each image
                    ax = plt.subplot(4, 8, i + 1)
                    
                    # Normalizes the image tensor and rearranges dimensions for display (channels last)
                    img = (fake[i] / 2 + 0.5).permute(1, 2, 0)
                    
                    # Displays the image using matplotlib
                    plt.imshow(img.numpy())
                    
                    # Removes x-axis ticks for cleaner visualization
                    plt.xticks([])
                    
                    # Removes y-axis ticks for cleaner visualization
                    plt.yticks([])
                
                # Adjusts spacing between subplots to reduce whitespace
                plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                
                # Renders the final image grid to the screen
                plt.show()

            # Case 2: Generate 32 images without glasses
            case "pushButton_SelectImagesWithoutEyeGlasses_ConditionalGANs":
                # Concatenates noise and labels along the channel dimension and moves the tensor to the selected device
                noise_and_labels = torch.cat([noise_ng, labels_ng], dim=1).to(self.device)
                
                # Passes the combined tensor through the generator to produce fake images, then moves them to CPU and detaches from computation graph
                fake = self.generator(noise_and_labels).cpu().detach()
                
                # Stores the 9th noise vector (assumed to represent a male image) for later use
                self.z_male_ng = noise_ng[8]
                
                # Stores the 32nd noise vector (assumed to represent a female image) for later use
                self.z_female_ng = noise_ng[31]
                
                # Initializes a matplotlib figure with specified size and resolution
                plt.figure(figsize=(20, 10), dpi=50)
                
                # Loops through all 32 generated images
                for i in range(32):
                    # Creates a subplot in a 4x8 grid for each image
                    ax = plt.subplot(4, 8, i + 1)
                    
                    # Normalizes the image tensor and rearranges dimensions for display (channels last)
                    img = (fake[i] / 2 + 0.5).permute(1, 2, 0)
                    
                    # Displays the image using matplotlib
                    plt.imshow(img.numpy())  # Optional: could be upscaled with .repeat if uncommented
                    
                    # Removes x-axis ticks for cleaner visualization
                    plt.xticks([])
                    
                    # Removes y-axis ticks for cleaner visualization
                    plt.yticks([])
                
                # Adjusts spacing between subplots to reduce whitespace
                plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                
                # Renders the final image grid to the screen
                plt.show()

            # Case 3: Transition female with glasses → without glasses
            case "pushButton_TransitionFemalesWithEyeGlassesToWithoutEyeGlasses_ConditionalGANs":
                # Checks if the female image with glasses has been previously generated
                if self.z_female_g is not None:
                    
                    # Initializes a matplotlib figure with specified size and resolution
                    plt.figure(figsize=(20, 4), dpi=50)
                    
                    # Loops through 5 interpolation steps between glasses and no-glasses labels
                    for i in range(5):
                        
                        # Creates a subplot in a 1x5 grid for each transition image
                        ax = plt.subplot(1, 5, i + 1)
                        
                        # Computes a weighted blend of the no-glasses and glasses labels
                        label = weights[i] * labels_ng[0] + (1 - weights[i]) * labels_g[0]
                        
                        # Concatenates the reshaped noise vector and interpolated label, then moves to device
                        noise_and_labels = torch.cat(
                            [self.z_female_g.reshape(1, self.z_dim, 1, 1),
                            label.reshape(1, 2, 1, 1)], dim=1).to(self.device)
                        
                        # Generates a fake image from the interpolated input, moves to CPU, and detaches from graph
                        fake = self.generator(noise_and_labels).cpu().detach()
                        
                        # Normalizes and rearranges image dimensions for display
                        img = (fake[0] / 2 + 0.5).permute(1, 2, 0)
                        
                        # Displays the image using matplotlib
                        plt.imshow(img.numpy())
                        
                        # Removes x-axis ticks for cleaner visualization
                        plt.xticks([])
                        
                        # Removes y-axis ticks for cleaner visualization
                        plt.yticks([])
                    
                    # Adjusts spacing between subplots to reduce whitespace
                    plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                    
                    # Renders the final transition image grid to the screen
                    plt.show()
                
                # If the female image with glasses hasn't been selected, show a warning message
                else:
                    QMessageBox.warning(None, "Follow Periority", "First, Select Images With EyeGlasses!")

            # Case 4: Transition male with glasses → without glasses
            case "pushButton_TransitionMalesWithEyeGlassesToWithoutEyeGlasses_ConditionalGANs":
                # Checks if the male image with glasses has been previously generated
                if self.z_male_g is not None:
                    
                    # Initializes a matplotlib figure with specified size and resolution
                    plt.figure(figsize=(20, 4), dpi=50)
                    
                    # Loops through 5 interpolation steps between glasses and no-glasses labels
                    for i in range(5):
                        
                        # Creates a subplot in a 1x5 grid for each transition image
                        ax = plt.subplot(1, 5, i + 1)
                        
                        # Computes a weighted blend of the no-glasses and glasses labels
                        label = weights[i] * labels_ng[0] + (1 - weights[i]) * labels_g[0]
                        
                        # Concatenates the reshaped noise vector and interpolated label, then moves to device
                        noise_and_labels = torch.cat(
                            [self.z_male_g.reshape(1, self.z_dim, 1, 1),
                            label.reshape(1, 2, 1, 1)], dim=1).to(self.device)
                        
                        # Generates a fake image from the interpolated input, moves to CPU, and detaches from graph
                        fake = self.generator(noise_and_labels).cpu().detach()
                        
                        # Normalizes and rearranges image dimensions for display
                        img = (fake[0] / 2 + 0.5).permute(1, 2, 0)
                        
                        # Displays the image using matplotlib
                        plt.imshow(img.numpy())
                        
                        # Removes x-axis ticks for cleaner visualization
                        plt.xticks([])
                        
                        # Removes y-axis ticks for cleaner visualization
                        plt.yticks([])
                    
                    # Adjusts spacing between subplots to reduce whitespace
                    plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                    
                    # Renders the final transition image grid to the screen
                    plt.show()
                
                # If the male image with glasses hasn't been selected, show a warning message
                else:
                    QMessageBox.warning(None, "Follow Periority", "First, Select Images With EyeGlasses!")

            # Case 5: Transition male → female with glasses
            case "pushButton_TransitionMaleToFemalesWithoutEyeGlasses_ConditionalGANs":
                # Checks if both male and female noise vectors (without glasses) have been generated
                if self.z_female_ng is not None and self.z_male_ng is not None:
                    
                    # Initializes a matplotlib figure with specified size and resolution
                    plt.figure(figsize=(20, 4), dpi=50)
                    
                    # Loops through 5 interpolation steps between male and female latent vectors
                    for i in range(5):
                        
                        # Creates a subplot in a 1x5 grid for each transition image
                        ax = plt.subplot(1, 5, i + 1)
                        
                        # Interpolates between male and female noise vectors using the current weight
                        z = weights[i] * self.z_female_ng + (1 - weights[i]) * self.z_male_ng
                        
                        # Concatenates the interpolated noise vector and the no-glasses label, then moves to device
                        noise_and_labels = torch.cat(
                            [z.reshape(1, self.z_dim, 1, 1),
                            labels_ng[0].reshape(1, 2, 1, 1)], dim=1).to(self.device)
                        
                        # Generates a fake image from the interpolated input, moves to CPU, and detaches from graph
                        fake = self.generator(noise_and_labels).cpu().detach()
                        
                        # Normalizes and rearranges image dimensions for display
                        img = (fake[0] / 2 + 0.5).permute(1, 2, 0)
                        
                        # Displays the image using matplotlib
                        plt.imshow(img.numpy())
                        
                        # Removes x-axis ticks for cleaner visualization
                        plt.xticks([])
                        
                        # Removes y-axis ticks for cleaner visualization
                        plt.yticks([])
                    
                    # Adjusts spacing between subplots to reduce whitespace
                    plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                    
                    # Renders the final transition image grid to the screen
                    plt.show()
                
                # If either the male or female noise vector is missing, show a warning message
                else:
                    QMessageBox.warning(None, "Follow Periority", "First, Select Images Without EyeGlasses!")

            # Case 6: Transition male → female without glasses
            case "pushButton_TransitionMaleToFemalesWithEyeGlasses_ConditionalGANs":
                # Checks if both male and female noise vectors (without glasses) have been generated
                if self.z_female_ng is not None and self.z_male_ng is not None:
                    
                    # Initializes a matplotlib figure with specified size and resolution
                    plt.figure(figsize=(20, 4), dpi=50)
                    
                    # Loops through 5 interpolation steps between male and female latent vectors
                    for i in range(5):
                        
                        # Creates a subplot in a 1x5 grid for each transition image
                        ax = plt.subplot(1, 5, i + 1)
                        
                        # Interpolates between male and female noise vectors using the current weight
                        z = weights[i] * self.z_female_ng + (1 - weights[i]) * self.z_male_ng
                        
                        # Concatenates the interpolated noise vector and the glasses label, then moves to device
                        noise_and_labels = torch.cat(
                            [z.reshape(1, self.z_dim, 1, 1),
                            labels_g[0].reshape(1, 2, 1, 1)], dim=1).to(self.device)
                        
                        # Generates a fake image from the interpolated input, moves to CPU, and detaches from graph
                        fake = self.generator(noise_and_labels).cpu().detach()
                        
                        # Normalizes and rearranges image dimensions for display
                        img = (fake[0] / 2 + 0.5).permute(1, 2, 0)
                        
                        # Displays the image using matplotlib
                        plt.imshow(img.numpy())
                        
                        # Removes x-axis ticks for cleaner visualization
                        plt.xticks([])
                        
                        # Removes y-axis ticks for cleaner visualization
                        plt.yticks([])
                    
                    # Adjusts spacing between subplots to reduce whitespace
                    plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                    
                    # Renders the final transition image grid to the screen
                    plt.show()
                
                # If either the male or female noise vector is missing, show a warning message
                else:
                    QMessageBox.warning(None, "Follow Periority", "First, Select Images Without EyeGlasses!")

            # Case 7: 2x2 grid of gender and glasses transitions
            case "pushButton_TransitionMaleToFemalesWithEyeGlassesToWithoutEyeGlasses_ConditionalGANs":
                # Checks if both male and female noise vectors (with glasses) have been generated
                if self.z_female_g is not None and self.z_male_g is not None:
                    
                    # Initializes a matplotlib figure with specified size and resolution
                    plt.figure(figsize=(20, 5), dpi=50)
                    
                    # Loops through 4 combinations to fill the 2x2 grid
                    for i in range(4):
                        
                        # Creates a subplot in a 1x4 grid for each image
                        ax = plt.subplot(1, 4, i + 1)
                        
                        # Determines gender: 0 for male, 1 for female
                        p = i // 2
                        
                        # Determines glasses status: 0 for glasses, 1 for no glasses
                        q = i % 2
                        
                        # Selects the appropriate latent vector based on gender
                        z = self.z_female_g * p + self.z_male_g * (1 - p)
                        
                        # Selects the appropriate label based on glasses status
                        label = labels_ng[0] * q + labels_g[0] * (1 - q)
                        
                        # Concatenates the latent vector and label, then moves to device
                        noise_and_labels = torch.cat(
                            [z.reshape(1, self.z_dim, 1, 1),
                            label.reshape(1, 2, 1, 1)], dim=1).to(self.device)
                        
                        # Generates a fake image from the input, moves to CPU, and detaches from graph
                        fake = self.generator(noise_and_labels)
                        
                        # Normalizes and rearranges image dimensions for display
                        img = (fake.cpu().detach()[0] / 2 + 0.5).permute(1, 2, 0)
                        
                        # Displays the image using matplotlib
                        plt.imshow(img.numpy())
                        
                        # Removes x-axis ticks for cleaner visualization
                        plt.xticks([])
                        
                        # Removes y-axis ticks for cleaner visualization
                        plt.yticks([])
                    
                    # Adjusts spacing between subplots to reduce whitespace
                    plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                    
                    # Renders the final image grid to the screen
                    plt.show()
                
                # If either the male or female noise vector is missing, show a warning message
                else:
                    QMessageBox.warning(None, "Follow Periority", "First, Select Images With EyeGlasses!")

            # Case 8: 6x6 grid of interpolated gender and glasses transitions
            case "pushButton_TransitionMaleToFemalesWithEyeGlassesToWithoutEyeGlasses2_ConditionalGANs":
                # Checks if both male and female noise vectors (without glasses) have been generated
                if self.z_female_ng is not None and self.z_male_ng is not None:
                    
                    # Initializes a matplotlib figure with specified size and resolution
                    plt.figure(figsize=(20, 20), dpi=50)
                    
                    # Loops through 36 combinations to fill the 6x6 grid
                    for i in range(36):
                        
                        # Creates a subplot in a 6x6 grid for each image
                        ax = plt.subplot(6, 6, i + 1)
                        
                        # Determines the row index (controls gender interpolation)
                        p = i // 6
                        
                        # Determines the column index (controls glasses interpolation)
                        q = i % 6
                        
                        # Interpolates between male and female noise vectors using row index
                        z = self.z_female_ng * p / 5 + self.z_male_ng * (1 - p / 5)
                        
                        # Interpolates between no-glasses and glasses labels using column index
                        label = labels_ng[0] * q / 5 + labels_g[0] * (1 - q / 5)
                        
                        # Concatenates the interpolated noise vector and label, then moves to device
                        noise_and_labels = torch.cat(
                            [z.reshape(1, self.z_dim, 1, 1),
                            label.reshape(1, 2, 1, 1)], dim=1).to(self.device)
                        
                        # Generates a fake image from the input
                        fake = self.generator(noise_and_labels)
                        
                        # Normalizes and rearranges image dimensions for display
                        img = (fake.cpu().detach()[0] / 2 + 0.5).permute(1, 2, 0)
                        
                        # Displays the image using matplotlib
                        plt.imshow(img.numpy())
                        
                        # Removes x-axis ticks for cleaner visualization
                        plt.xticks([])
                        
                        # Removes y-axis ticks for cleaner visualization
                        plt.yticks([])
                    
                    # Adjusts spacing between subplots to reduce whitespace
                    plt.subplots_adjust(wspace=-0.08, hspace=-0.01)
                    
                    # Renders the final image grid to the screen
                    plt.show()
                
                # If either the male or female noise vector is missing, show a warning message
                else:
                    QMessageBox.warning(None, "Follow Periority", "First, Select Images Without EyeGlasses!")

# Define the Critic (Discriminator) class for the Conditional GAN
class Critic(nn.Module):
    # Constructor to initialize the Critic model
    def __init__(self, img_channels, features):
        # img_channels: number of input image channels (e.g., 3 for RGB)
        # features: base number of feature maps used in convolutional layers

        # Call the parent class constructor
        super().__init__()

        # Define the sequential model architecture
        self.net = nn.Sequential(
            # First convolutional layer: downsample input image
            nn.Conv2d(img_channels, features, kernel_size=4, stride=2, padding=1),

            # Apply LeakyReLU activation for non-linearity
            nn.LeakyReLU(0.2),

            # Downsampling block 1: features → features * 2
            self.block(features, features * 2, 4, 2, 1),

            # Downsampling block 2: features * 2 → features * 4
            self.block(features * 2, features * 4, 4, 2, 1),

            # Downsampling block 3: features * 4 → features * 8
            self.block(features * 4, features * 8, 4, 2, 1),

            # Downsampling block 4: features * 8 → features * 16
            self.block(features * 8, features * 16, 4, 2, 1),

            # Downsampling block 5: features * 16 → features * 32
            self.block(features * 16, features * 32, 4, 2, 1),

            # Final convolution: reduce to single output score
            nn.Conv2d(features * 32, 1, kernel_size=4, stride=2, padding=0)
        )

    # Helper method to define a convolutional block with normalization and activation
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        # in_channels: number of input channels
        # out_channels: number of output channels
        # kernel_size, stride, padding: convolution parameters

        return nn.Sequential(
            # Convolutional layer without bias
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),

            # Instance normalization to stabilize training
            nn.InstanceNorm2d(out_channels, affine=True),

            # LeakyReLU activation for non-linearity
            nn.LeakyReLU(0.2)
        )

    # Forward pass through the Critic network
    def forward(self, x):
        return self.net(x)

# Define the Generator class for the Conditional GAN
class Generator(nn.Module):
    # Constructor to initialize the Generator model
    def __init__(self, noise_channels, img_channels, features):
        # noise_channels: number of input channels (latent vector + label channels)
        # img_channels: number of output image channels (e.g., 3 for RGB)
        # features: base number of feature maps used in transposed convolutions

        # Call the parent class constructor
        super(Generator, self).__init__()

        # Define the sequential model architecture
        self.net = nn.Sequential(
            # First upsampling block: noise → features * 64
            self.block(noise_channels, features * 64, 4, 1, 0),

            # Upsampling block 1: features * 64 → features * 32
            self.block(features * 64, features * 32, 4, 2, 1),

            # Upsampling block 2: features * 32 → features * 16
            self.block(features * 32, features * 16, 4, 2, 1),

            # Upsampling block 3: features * 16 → features * 8
            self.block(features * 16, features * 8, 4, 2, 1),

            # Upsampling block 4: features * 8 → features * 4
            self.block(features * 8, features * 4, 4, 2, 1),

            # Upsampling block 5: features * 4 → features * 2
            self.block(features * 4, features * 2, 4, 2, 1),

            # Final transposed convolution: features * 2 → image channels
            nn.ConvTranspose2d(features * 2, img_channels, kernel_size=4, stride=2, padding=1),

            # Tanh activation to scale output pixels to [-1, 1]
            nn.Tanh()
        )

    # Helper method to define an upsampling block with normalization and activation
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        # in_channels: number of input channels
        # out_channels: number of output channels
        # kernel_size, stride, padding: transposed convolution parameters

        return nn.Sequential(
            # Transposed convolutional layer without bias
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),

            # Batch normalization to stabilize training
            nn.BatchNorm2d(out_channels),

            # ReLU activation for non-linearity
            nn.ReLU()
        )

    # Forward pass through the Generator network
    def forward(self, x):
        return self.net(x)

# Class to visualize GAN training progress using matplotlib plots inside a PyQt scrollable window
class PlotWindow(QMainWindow):

    # Constructor to initialize the plot window and its components
    def __init__(self, device=None, model_Generator=None, z_dim=None):
        # device: computation device (CPU or GPU)
        # model_Generator: trained generator model used to produce images
        # z_dim: dimensionality of the latent noise vector

        # Call the base QMainWindow constructor
        super().__init__()

        # Set the title of the main window
        self.setWindowTitle("Training Progress")

        # Set the initial size of the window
        self.resize(800, 700)

        # Create a scrollable area to hold multiple plot canvases
        self.scroll = QScrollArea()

        # Create a container widget to hold the layout and plots
        self.container = QWidget()

        # Create a vertical layout to stack plots vertically inside the container
        self.layout = QVBoxLayout(self.container)

        # Set the container widget as the content of the scroll area
        self.scroll.setWidget(self.container)

        # Enable dynamic resizing of the scroll area’s contents
        self.scroll.setWidgetResizable(True)

        # Set the scroll area as the central widget of the main window
        self.setCentralWidget(self.scroll)

        # Store reference to the generator model for generating images
        self.generator = model_Generator

        # Store the device (CPU or GPU) used for computation
        self.device = device

        # Store the dimensionality of the latent noise vector
        self.z_dim = z_dim

    # Method to generate and display images after each training epoch
    def plot_epoch(self, epoch):
        # epoch: current training epoch number

        # Internal function to generate and plot images for a specific label
        def generate_and_plot(label_index, title_suffix):
            # label_index: index of the label (0 for glasses, 1 for no glasses)
            # title_suffix: text to append to the plot title

            # Generate random noise vectors for 32 samples
            noise = torch.randn(32, self.z_dim, 1, 1)

            # Create label tensor with shape [32, 2, 1, 1]
            labels = torch.zeros(32, 2, 1, 1)

            # Set the specified label index to 1 for all samples
            labels[:, label_index, :, :] = 1

            # Concatenate noise and labels to form the input to the generator
            noise_and_labels = torch.cat([noise, labels], dim=1).to(self.device)

            # Generate fake images using the generator
            fake = self.generator(noise_and_labels).cpu().detach()

            # Create a matplotlib figure to hold the image grid
            fig = Figure(figsize=(20, 10), dpi=72)

            # Create a canvas to render the figure inside the PyQt window
            canvas = FigureCanvas(fig)

            # Set a minimum height for the canvas to ensure visibility
            canvas.setMinimumHeight(600)

            # Loop through each generated image and add it to the figure
            for i in range(32):
                # Create a subplot for each image
                ax = fig.add_subplot(4, 8, i + 1)

                # Normalize image pixels to [0, 1] and rearrange dimensions
                image = (fake[i] / 2 + 0.5).permute(1, 2, 0)

                # Display the image in the subplot
                ax.imshow(image)

                # Remove axis ticks for cleaner display
                ax.set_xticks([])
                ax.set_yticks([])

            # Add a title to the figure indicating the epoch and label type
            fig.suptitle(f"Generated Images after Epoch {epoch} ({title_suffix})", fontsize=16)

            # Adjust spacing between subplots
            fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3)

            # Add the canvas to the layout so it appears in the scrollable window
            self.layout.addWidget(canvas)

            # Scroll to the bottom to show the latest plot
            self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

        # Generate and plot images with glasses
        generate_and_plot(label_index=0, title_suffix="With Glasses")

        # Generate and plot images without glasses
        generate_and_plot(label_index=1, title_suffix="Without Glasses")

# Class to handle Conditional GAN training in a separate thread using PyQt
class TrainingConditionalGANsThread(QThread):
    # Define a PyQt signal to send log messages to the UI
    log_signal = pyqtSignal(str)

    # Define a PyQt signal to trigger image display updates in the UI
    display_signal = pyqtSignal(int)

    # Constructor method to initialize training configuration and models
    def __init__(self, plot_window, DownloadLogPopup, batch_size, device, criticModel, generatorModel, data_loader, z_dim):
        # Call the base class constructor to initialize QThread
        super().__init__()

        # Reference to the plotting window for visualizing generated images
        self.plot_window = plot_window

        # Reference to the log popup window for displaying training logs
        self.DownloadLogPopup = DownloadLogPopup

        # Set the batch size used during training iterations
        self.batch_size = batch_size

        # Set the device (CPU or GPU) for model computation
        self.device = device

        # Override the learning rate with a fixed value
        self.lr = 0.0001

        # Store the dimensionality of the latent noise vector
        self.z_dim = z_dim

        # Store the discriminator model used to classify real vs fake images
        self.critic = criticModel

        # Store the generator model used to synthesize fake images
        self.generator = generatorModel

        # Define optimizer for the generator using Adam
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.0, 0.9))

        # Define optimizer for the critic using Adam
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.0, 0.9))

        # Initialize early stopping mechanism to halt training if no improvement
        self.stopper = EarlyStop(patience=1000, min_delta=0.01)

        # Store the data loader that provides batches of training images
        self.data_loader = data_loader

        # Generate fixed noise input for generator evaluation and move to device
        self.noise = torch.randn((self.batch_size, 2)).to(device)

        # Flag to indicate whether a manual stop has been requested
        self._stop_requested = False

    # Method to compute gradient penalty for WGAN-GP
    def GP(self, critic, real, fake):
        # Get batch size and image dimensions
        B, C, H, W = real.shape

        # Generate random interpolation weights
        alpha = torch.rand((B, 1, 1, 1)).repeat(1, C, H, W).to(self.device)

        # Create interpolated images between real and fake
        interpolated_images = real * alpha + fake * (1 - alpha)

        # Get critic scores for interpolated images
        critic_scores = critic(interpolated_images)

        # Use PyTorch's autograd to compute gradients
        # This is often used in GANs (e.g., WGAN-GP) to calculate gradient penalties
        # Compute gradients of critic scores with respect to interpolated images
        gradient = torch.autograd.grad(

            # Specify the input tensor for which gradients are computed
            inputs=interpolated_images,

            # Specify the output tensor whose gradients are needed
            outputs=critic_scores,

            # Provide gradient of outputs w.r.t. themselves (∂output/∂output = 1)
            grad_outputs=torch.ones_like(critic_scores),

            # Retain computation graph for higher-order gradients
            create_graph=True,

            # Retain graph for possible reuse in further backward passes
            retain_graph=True
        )[0]  # Extract the first element from the returned tuple (the actual gradient tensor)

        # Flatten gradients for norm computation
        gradient = gradient.view(gradient.shape[0], -1)

        # Compute L2 norm of gradients
        gradient_norm = gradient.norm(2, dim=1)

        # Compute gradient penalty term
        gp = torch.mean((gradient_norm - 1) ** 2)

        # Return gradient penalty
        return gp

    # Method to train the GAN on a single batch
    def train_batch(self, onehots, img_and_labels, epoch):
        # Move real images to device
        real = img_and_labels.to(self.device)

        # Get batch size
        B = real.shape[0]

        # Train the critic multiple times per batch
        for _ in range(5):
            # Generate random noise
            noise = torch.randn(B, self.z_dim, 1, 1)

            # Reshape one-hot labels to match input format
            onehots = onehots.reshape(B, 2, 1, 1)

            # Concatenate noise and labels
            noise_and_labels = torch.cat([noise, onehots], dim=1).to(self.device)

            # Generate fake images from generator
            fake_img = self.generator(noise_and_labels).to(self.device)

            # Extract label tensors from real data
            fakelabels = img_and_labels[:, 3:, :, :].to(self.device)

            # Concatenate fake images with their labels
            fake = torch.cat([fake_img, fakelabels], dim=1).to(self.device)

            # Get critic scores for real and fake images
            critic_real = self.critic(real).reshape(-1)
            critic_fake = self.critic(fake).reshape(-1)

            # Compute gradient penalty
            gp = self.GP(self.critic, real, fake)

            # Compute critic loss using WGAN-GP formula
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + 10 * gp

            # Backpropagate and update critic
            self.opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            self.opt_critic.step()

        # Train the generator once per batch
        gen_fake = self.critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)

        # Backpropagate and update generator
        self.opt_gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

        # Return losses for logging
        return loss_critic, loss_gen

    # Method to manually stop the training process
    def stop(self):
        # Set the stop flag to True
        self._stop_requested = True

        # Disable the cancel button in the UI to prevent further interaction
        self.DownloadLogPopup.cancel_button.setEnabled(False)

    # Main method that runs the training loop in a separate thread
    def run(self):
        try:
            # Emit signal to indicate training has started
            self.log_signal.emit("Training thread started.")

            # Emit signal with number of batches in the training loader
            self.log_signal.emit(f"Train loader has {len(self.data_loader)} batches.")

            # Loop through training epochs
            for epoch in range(1, 101):
                # Check if stop was requested
                if self._stop_requested:
                    # Exit the epoch loop
                    break

                # Emit signal indicating current epoch
                self.log_signal.emit(f"Epoch {epoch} started.")

                # Initialize accumulators for generator and discriminator loss
                gloss = 0
                dloss = 0

                # Loop through each batch in the training data
                for batch_idx, (_, _, onehots, img_and_labels) in enumerate(self.data_loader):
                    # Check for manual stop request
                    if self._stop_requested:
                        self.log_signal.emit("Training stopped by user.")
                        break

                    # Log progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        self.log_signal.emit(f"Processing batch {batch_idx + 1}/{len(self.data_loader)}")

                    # Train on current batch and accumulate losses
                    loss_critic, loss_gen = self.train_batch(onehots, img_and_labels, epoch)
                    dloss += loss_critic.detach() / len(self.data_loader)
                    gloss += loss_gen.detach() / len(self.data_loader)

                # Emit display signal every 2 epochs to update visuals
                if epoch % 2 == 0:
                    self.display_signal.emit(epoch)

                # Emit loss summary for the current epoch
                self.log_signal.emit(f"At epoch {epoch}, Critic(Discriminator) Model loss: {dloss}, Generator Model loss: {gloss}")

                # Scroll log output to the bottom in the UI
                self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)

                # Ensure the cursor is visible in the log output
                self.DownloadLogPopup.log_output.ensureCursorVisible()

                # Process UI events to keep the interface responsive
                QApplication.processEvents()

            # Save the trained generator model to disk
            torch.save(self.generator.state_dict(), 'resources/models/conditional_gan.pth')

            # Emit signal indicating training has completed
            self.log_signal.emit("Training Finished.")

        # Catch and log any exceptions that occur during training
        except Exception as e:
            self.log_signal.emit(f"Error during training: {str(e)}")