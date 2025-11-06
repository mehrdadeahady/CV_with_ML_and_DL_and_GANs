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
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    import torchvision.transforms as T
    from torchvision.utils import make_grid, save_image
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
    from tqdm import tqdm
except:
    print("You Should Install tqdm Library")
try:
    from contextlib import nullcontext
except:
    print("You Should Install contextlib Library")
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
    import albumentations
    from albumentations.pytorch import ToTensorV2
except:
    print("You Should Install albumentations Library with below flag to avoid installing opencv headless causing confilict.\npip install albumentations --no-deps\nthen install one of its dependencies:\npip install albucore==0.0.24  --no-deps")
try: 
    from PyQt6.QtGui import  QTextCursor   
    from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt
    from PyQt6.QtWidgets import QMessageBox,QTextEdit, QWidget, QVBoxLayout, QPushButton, QLabel, QDialog, QTextEdit,QScrollArea,QMainWindow,QApplication
except:
    print("You Should Install PyQt6 Library!")

# Define the CycleGANs class, which inherits from QObject to support Qt signals and slots
class CycleGANs(QObject):

    # Constructor method to initialize the CycleGANs instance
    def __init__(self, parent=None):
        # parent: optional reference to a parent QObject, used in Qt applications for signal-slot management
        # Call the constructor of the parent QObject class to enable Qt signal-slot functionality
        super().__init__()

        # Set a fixed random seed to ensure reproducibility across training runs and model outputs
        torch.manual_seed(0)

        # Placeholder for training data; will be assigned later during data loading
        self.train = None

        # Placeholder for a DataFrame to hold metadata or structured information about the dataset
        self.dataframe = None

        # Instantiate a custom log emitter to handle logging messages via Qt signals
        self.log_emitter = LogEmitter()

        # Define a sequence of image transformations using Albumentations for preprocessing
        self.transforms = albumentations.Compose(
            [
                # Resize all images to 256x256 pixels for uniform input dimensions
                albumentations.Resize(width=256, height=256),

                # Apply horizontal flip augmentation with a 50% probability to increase data diversity
                albumentations.HorizontalFlip(p=0.5),

                # Normalize image pixel values to the range [-1, 1] using mean and std deviation
                albumentations.Normalize(
                    mean=[0.5, 0.5, 0.5],              # Mean normalization for RGB channels
                    std=[0.5, 0.5, 0.5],               # Standard deviation for RGB channels
                    max_pixel_value=255                # Maximum pixel value in the input images
                ),

                # Convert image and associated data to PyTorch tensors
                ToTensorV2()
            ],
            # Specify additional target for transformation, e.g., a second image input
            additional_targets={"image0": "image"}
        )

        # Placeholder for the data loader that will feed batches of data during training
        self.loader = None

        # Placeholder for the dataset object that will be used to store and manage image data
        self.dataset = None

        # Set the device for computation: use GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Placeholder for Generator A (e.g., transforms domain A to domain B)
        self.generator_A = None

        # Placeholder for Generator B (e.g., transforms domain B to domain A)
        self.generator_B = None

        # Placeholder for Discriminator A (evaluates realism of images in domain A)
        self.discriminator_A = None

        # Placeholder for Discriminator B (evaluates realism of images in domain B)
        self.discriminator_B = None

    # Method to organize the CelebA dataset into two folders based on hair color: black and blond
    def ArrangeCelebFacesDataset(self):
        # Check if all required metadata files exist and the total number of image files exceeds 200,000
        if os.path.exists("kagglehub/list_attr_celeba.csv") and \
        os.path.exists("kagglehub/list_bbox_celeba.csv") and \
        os.path.exists("kagglehub/list_eval_partition.csv") and \
        os.path.exists("kagglehub/list_landmarks_align_celeba.csv") and \
        self.CountFilesInPath("kagglehub/img_align_celeba") + \
        self.CountFilesInPath("kagglehub/black") + \
        self.CountFilesInPath("kagglehub/blond") > 200000:

            # Load the attribute CSV into a DataFrame if it hasn't been loaded yet
            if self.dataframe is None:
                self.dataframe = pd.read_csv("kagglehub/list_attr_celeba.csv")

            # Check if the target folders exist and contain enough images; if not, proceed to arrange
            if not os.path.exists("kagglehub/black") or \
            not os.path.exists("kagglehub/blond") or \
            self.CountFilesInPath("kagglehub/black") < 10000 or \
            self.CountFilesInPath("kagglehub/blond") < 10000:

                # Create a popup window to display dataset arrangement logs
                self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

                # Disable the cancel button during dataset arrangement to prevent interruption
                self.DownloadLogPopup.cancel_button.setEnabled(False)

                # Show the popup window to the user
                self.DownloadLogPopup.show()

                # Append a log message indicating that dataset arrangement has started
                self.DownloadLogPopup.Append_Log("Arranging the Dataset!\nWait ...")

                # Create directories for black and blond hair categories if they don't exist
                os.makedirs("kagglehub/black", exist_ok=True)
                os.makedirs("kagglehub/blond", exist_ok=True)

                # Define the source folder containing the original aligned CelebA images
                folder = "kagglehub/img_align_celeba"

                # Iterate over each row in the attribute DataFrame
                for i in range(len(self.dataframe)):
                    # Extract the current row (image attributes and ID)
                    dfi = self.dataframe.iloc[i]

                    # Check if the image is labeled as having black hair
                    if dfi['Black_Hair'] == 1:
                        try:
                            # Construct the source and destination paths for the image
                            oldpath = f"{folder}/{dfi['image_id']}"
                            newpath = f"kagglehub/black/{dfi['image_id']}"

                            # Move the image from the source to the black hair folder
                            shutil.move(oldpath, newpath)
                        except:
                            # Silently ignore any errors during file move
                            pass

                    # Check if the image is labeled as having blond hair
                    elif dfi['Blond_Hair'] == 1:
                        try:
                            # Construct the source and destination paths for the image
                            oldpath = f"{folder}/{dfi['image_id']}"
                            newpath = f"kagglehub/blond/{dfi['image_id']}"

                            # Move the image from the source to the blond hair folder
                            shutil.move(oldpath, newpath)
                        except:
                            # Silently ignore any errors during file move
                            pass

                # Append a log message indicating successful dataset arrangement
                self.DownloadLogPopup.Append_Log("Dataset Arranged successfully.")

            # If dataset is already arranged, show an informational message to the user
            else:
                QMessageBox.information(None, "Dataset arranged", "Dataset Already Arranged.")

        # If required files are missing or dataset is incomplete, prompt the user to download it
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

    # Method to display a sample of CelebA images based on hair color, triggered by a UI button
    def ShowCelebFacesImages(self, sender):
        # Check if both black and blond hair folders exist and contain sufficient images
        if os.path.exists("kagglehub/black") and \
        os.path.exists("kagglehub/blond") and \
        self.CountFilesInPath("kagglehub/black") > 10000 and \
        self.CountFilesInPath("kagglehub/blond") > 10000:

            # Initialize variables to hold the selected folder path and display title
            trainGroup = None
            title = ""

            # Determine which button was pressed to select the appropriate image group
            match sender:
                # If the button for black hair images was pressed
                case "pushButton_DisplayImagesWithDarkHair_CycleGANs":
                    trainGroup = "kagglehub/black/"
                    title = "Images with Black Hair"

                # If the button for blond hair images was pressed
                case "pushButton_DisplayImagesWithBlondHair_CycleGANs":
                    trainGroup = "kagglehub/blond/"
                    title = "Images with Blond Hair"

            # List all image filenames in the selected folder
            imgs = os.listdir(trainGroup)

            # Set a fixed seed for random sampling to ensure consistent results
            random.seed(42)

            # Randomly select 8 image filenames from the folder
            samples = random.sample(imgs, 8)

            # Prepare folder and image lists for indexed access
            fs = [trainGroup]  # List of folder paths (only one in this case)
            ps = [imgs]        # List of image filename lists (only one in this case)

            # Create a matplotlib figure with a title and specified resolution and size
            fig = plt.figure(title, dpi=100, figsize=(1.78 * 8, 2.18 * 2))

            # Loop through the 8 selected images to display them in a subplot grid
            for i in range(8):
                # Create a subplot in a 1-row, 8-column layout
                ax = plt.subplot(1, 8, i + 1)

                # Determine folder index (always 0 here since only one folder is used)
                folder = i // 8

                # Determine image index within the selected sample
                p = i % 8

                # Open the image file using PIL
                img = Image.open(fr"{fs[folder]}{ps[folder][p]}")

                # Display the image in the subplot
                plt.imshow(img)

                # Remove x-axis ticks for a cleaner display
                plt.xticks([])

                # Remove y-axis ticks for a cleaner display
                plt.yticks([])

            # Adjust spacing between subplots to minimize gaps
            plt.subplots_adjust(wspace=-0.01, hspace=-0.1)

            # Show the final image grid to the user
            plt.show()

        # If dataset folders are missing or insufficiently populated, show an error message
        else:
            QMessageBox.information(
                None,
                "No Dataset",
                "First, Download and Copy the Dataset to the root of Project and Arrange it."
            )

    # Method to prepare the CelebA dataset for training by creating a PyTorch DataLoader
    def PrepareDataset(self):
        # Check if both black and blond hair folders exist and contain sufficient images
        if os.path.exists("kagglehub/black") and \
        os.path.exists("kagglehub/blond") and \
        self.CountFilesInPath("kagglehub/black") > 10000 and \
        self.CountFilesInPath("kagglehub/blond") > 10000:

            # Proceed only if the DataLoader hasn't already been initialized
            if self.loader is None:
                # Create a custom dataset object using the black and blond hair image folders
                self.dataset = LoadData(
                    root_A=["kagglehub/black"],     # Source domain A: images with black hair
                    root_B=["kagglehub/blond"],     # Target domain B: images with blond hair
                    transform=self.transforms       # Apply predefined image transformations
                )

                # Wrap the dataset in a PyTorch DataLoader for efficient batch loading
                self.loader = DataLoader(
                    self.dataset,                   # Dataset object containing paired images
                    batch_size=1,                   # Load one image pair per batch
                    shuffle=True,                   # Shuffle data to improve training robustness
                    pin_memory=self.device == "cuda"  # Optimize memory transfer if using GPU
                )

                # Notify the user that the dataset is ready for training
                QMessageBox.information(None, "Dataset Prepared", "Dataset Prepared, ready for training.")

            # If the DataLoader is already initialized, inform the user
            else:
                QMessageBox.information(None, "Dataset Prepared", "Dataset already Prepared.")

        # If dataset folders are missing or insufficiently populated, show an error message
        else:
            QMessageBox.information(
                None,
                "No Dataset",
                "First, Download and Copy the Dataset to the root of Project and Arrange it."
            )

    # Method to initialize weights of model layers using custom rules for different layer types
    def weights_init(self, m):
        # Retrieve the class name of the layer/module passed as argument
        name = m.__class__.__name__

        # Check if the layer is a convolutional or linear (fully connected) layer
        if name.find('Conv') != -1 or name.find('Linear') != -1:
            # Initialize weights with a normal distribution (mean=0.0, std=0.02)
            nn.init.normal_(m.weight.data, 0.0, 0.02)

            # Initialize biases to zero
            nn.init.constant_(m.bias.data, 0)

        # Check if the layer is a 2D normalization layer (e.g., BatchNorm2d)
        elif name.find('Norm2d') != -1:
            # Initialize weights to 1 for normalization scaling
            nn.init.constant_(m.weight.data, 1)

            # Initialize biases to zero for normalization offset
            nn.init.constant_(m.bias.data, 0)

    # Method to create discriminator models for both domains and initialize their weights
    def CreateDiscriminators(self):
        # Check if discriminator models haven't been instantiated yet
        if self.discriminator_A is None or self.discriminator_B is None:
            # Instantiate Discriminator A (e.g., for domain A) and move it to the selected device (CPU/GPU)
            self.discriminator_A = Discriminator().to(self.device)

            # Instantiate Discriminator B (e.g., for domain B) and move it to the selected device (CPU/GPU)
            self.discriminator_B = Discriminator().to(self.device)

            # Apply custom weight initialization to Discriminator A
            self.weights_init(self.discriminator_A)

            # Apply custom weight initialization to Discriminator B
            self.weights_init(self.discriminator_B)

            # Display a message box to inform the user that discriminators were created and initialized
            QMessageBox.information(
                None,
                "Discriminator models Created",
                "Discriminator models created and weights initialized."
            )

        # If discriminator models already exist, notify the user to avoid redundant creation
        else:
            QMessageBox.information(
                None,
                "Discriminator models Exist",
                "Discriminator models already created and weights initialized."
            )

    # Method to create generator models for both domains and initialize their weights
    def CreateGenerators(self):
        # Check if generator models haven't been instantiated yet
        if self.generator_A is None or self.generator_B is None:
            # Instantiate Generator A (e.g., transforms images from domain A to domain B)
            # Set input image channels to 3 (RGB) and use 9 residual blocks for deeper learning
            self.generator_A = Generator(img_channels=3, num_residuals=9).to(self.device)

            # Instantiate Generator B (e.g., transforms images from domain B to domain A)
            # Same configuration as Generator A
            self.generator_B = Generator(img_channels=3, num_residuals=9).to(self.device)

            # Apply custom weight initialization to Generator A
            self.weights_init(self.generator_A)

            # Apply custom weight initialization to Generator B
            self.weights_init(self.generator_B)

            # Display a message box to inform the user that generators were created and initialized
            QMessageBox.information(
                None,
                "Generator models Created",
                "Generator models created and weights initialized."
            )

        # If generator models already exist, notify the user to avoid redundant creation
        else:
            QMessageBox.information(
                None,
                "Generator models Exist",
                "Generator models already created and weights initialized."
            )

    # Method to start training the CycleGAN model using prepared data and initialized models
    def TrainModel(self):
        # Check if the training dataset has been prepared
        if self.loader is None:
            # Warn the user to prepare the dataset before starting training
            QMessageBox.warning(None, "Data is not Ready", "First, Prepare the Dataset!")

        # Check if both Generator and Discriminator models have been created
        elif self.generator_A is None or \
            self.generator_B is None or \
            self.discriminator_A is None or \
            self.discriminator_B is None:
            # Warn the user to create the models before starting training
            QMessageBox.warning(None, "No Models Exist", "First, Create Models!")

        # If both data and models are ready, proceed with training
        else:
            # Create a window to visualize training progress in real time
            self.plot_window = PlotWindow()

            # Display the plot window to the user
            self.plot_window.show()

            # Create a popup window to show training logs and allow cancellation
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

            # Enable the cancel button to allow user to stop training if needed
            self.DownloadLogPopup.cancel_button.setEnabled(True)

            # Display the log popup window
            self.DownloadLogPopup.show()

            # Add an initial log message to indicate that training has started
            self.DownloadLogPopup.Append_Log("Training Models!\nWait ...")

            # Create a separate thread to run the training process asynchronously
            self.training_thread = TrainingCycleGANsThread(
                self.plot_window,             # Reference to the plot window for visual updates
                self.DownloadLogPopup,        # Reference to the log popup for status updates
                self.generator_A,             # Generator A model
                self.generator_B,             # Generator B model
                self.discriminator_A,         # Discriminator A model
                self.discriminator_B,         # Discriminator B model
                self.loader,                  # DataLoader containing training data
                self.device                   # Device to run training on (CPU or GPU)
            )

            # Connect the training thread's log signal to the log popup's append method
            self.training_thread.log_signal.connect(self.DownloadLogPopup.Append_Log)

            # Connect the training thread's display signal to the plot window's plot method
            self.training_thread.display_signal.connect(self.plot_window.plot)

            # Connect the cancel button to the thread's stop method to allow user interruption
            self.DownloadLogPopup.cancel_button.clicked.connect(self.training_thread.stop)

            # Start the training thread to begin model training
            self.training_thread.start()

    # Method to load previously trained Generator models from disk and prepare them for inference
    def LoadTrainedModel(self):
        # Check if the training dataset has been prepared
        if self.loader is None:
            # Warn the user to prepare the dataset before attempting to load models
            QMessageBox.warning(None, "Data is not Ready", "First, Prepare the Dataset!")
            # Return False to indicate that loading cannot proceed
            return False

        # Check if both Generator models have been instantiated
        elif self.generator_A is None or self.generator_B is None:
            # Warn the user to create the generator models before loading weights
            QMessageBox.warning(None, "No Models Exist", "First, Create Generator Models!")
            # Return False to indicate that loading cannot proceed
            return False

        # Proceed if dataset and models are ready
        else:
            # Check if both saved Generator model files exist on disk
            if os.path.exists("resources/models/gen_black.pth") and \
            os.path.exists("resources/models/gen_blond.pth"):

                # Load the saved weights into Generator A (black hair model)
                self.generator_A.load_state_dict(
                    torch.load("resources/models/gen_black.pth", map_location=self.device)
                )

                # Load the saved weights into Generator B (blond hair model)
                self.generator_B.load_state_dict(
                    torch.load("resources/models/gen_blond.pth", map_location=self.device)
                )

                # Move Generator A to the appropriate device and set to evaluation mode
                self.generator_A.to(self.device).eval()

                # Move Generator B to the appropriate device and set to evaluation mode
                self.generator_B.to(self.device).eval()

                # Return True to indicate successful model loading
                return True

            # If either model file is missing, notify the user
            else:
                QMessageBox.warning(
                    None,                          # No parent widget specified
                    "Models not Saved",            # Title of the warning dialog
                    "First, Train and Save the Models!"  # Message body prompting user action
                )
                # Return False to indicate loading failed
                return False

    # Handles image generation and visualization based on the selected UI button
    def GenerateAndDisplayImages(self, sender):
        # Check if the trained generator models are loaded; exit early if not
        if not self.LoadTrainedModel():
            return

        # Ensure the dataset contains at least 8 samples for visualization
        if len(self.dataset) < 8:
            print("Not enough samples in dataset to generate 8 images.")
            return

        # Randomly select 8 samples from the dataset
        indices = random.sample(range(len(self.dataset)), 8)
        selected_samples = [self.dataset[i] for i in indices]

        # Initialize input batch and generator reference
        input_batch = []
        generator = None

        # Determine which generator to use based on the sender button
        match sender:
            # If the button for generating images with dark hair was pressed
            case "pushButton_ShowImagesWithDarkHair_CycleGANs":
                # Extract black hair images from the selected samples
                input_batch = [black.unsqueeze(0) for black, _ in selected_samples]
                generator = self.generator_A

            # If the button for generating images with blond hair was pressed
            case "pushButton_ShowImagesWithBlondHair_CycleGANs":
                # Extract blond hair images from the selected samples
                input_batch = [blond.unsqueeze(0) for _, blond in selected_samples]
                generator = self.generator_B

            # If the sender is unrecognized, print an error and exit
            case _:
                print(f"Unknown sender: {sender}")
                return

        # Concatenate the input images into a single batch tensor and move to device
        input_tensor = torch.cat(input_batch, dim=0).to(self.device)

        # Set the generator to evaluation mode to disable dropout and batchnorm updates
        generator.eval()

        # Run inference without tracking gradients
        with torch.no_grad():
            # Generate output images from the input batch
            output_batch = generator(input_tensor).cpu()

        # Create a 2×4 grid for displaying the generated images
        fig, axes = plt.subplots(2, 4, figsize=(8, 4), dpi=100)

        # Set the window title for the plot
        fig.canvas.manager.set_window_title("Generated Images")

        # Loop through each subplot and display the corresponding image
        for ax, img in zip(axes.flatten(), output_batch):
            # Denormalize image from [-1, 1] to [0, 1] for visualization
            img = (img + 1) / 2

            # Rearrange tensor dimensions from [C, H, W] to [H, W, C] and clamp values
            img = img.permute(1, 2, 0).clamp(0, 1)

            # Display the image in the subplot
            ax.imshow(img.numpy())

            # Hide axis ticks for a cleaner look
            ax.axis("off")

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        # Optimize layout to prevent overlap
        plt.tight_layout()

        # Show the final plot window with generated images
        plt.show()

    # Method to inform the user how to adapt the CycleGAN pipeline for eyeglasses-based image translation
    def Implementing_Last_cGAN_for_EyeGlasses_by_CycleGAN(self):
        # Display an informational message box explaining how to prepare the dataset for eyeglasses training
        QMessageBox.information(
            None,  # No parent widget specified
            "Attention:",  # Title of the message box
            "Creating models and training them is the same as described on this page.\n\n"
            "However, when preparing the dataset before training, make sure to point to the eyeglasses images as shown below:\n\n"
            "dataset = LoadData(\n"
            "    root_A = ['kagglehub/glasses/G/'],\n"
            "    root_B = ['kagglehub/glasses/NoG/'],\n"
            "    transform = transforms\n"
            ")\n"
            "loader = DataLoader(\n"
            "    dataset,\n"
            "    batch_size = 1,\n"
            "    shuffle = True,\n"
            "    pin_memory = True\n"
            ")"
        )

# Class to visualize GAN training progress using matplotlib plots inside a PyQt scrollable window
class PlotWindow(QMainWindow):

    # Constructor to initialize the plot window and its components
    def __init__(self):
        # Initialize the base QMainWindow class
        super().__init__()

        # Set the title of the main window
        self.setWindowTitle("Training Progress")

        # Set the initial dimensions of the window
        self.resize(800, 700)

        # Create a scrollable area to hold multiple plot canvases
        self.scroll = QScrollArea()

        # Create a container widget that will hold the layout and plots
        self.container = QWidget()

        # Create a vertical layout to stack plots vertically inside the container
        self.layout = QVBoxLayout(self.container)

        # Assign the container widget to the scroll area
        self.scroll.setWidget(self.container)

        # Enable dynamic resizing of the scroll area’s contents
        self.scroll.setWidgetResizable(True)

        # Set the scroll area as the central widget of the main window
        self.setCentralWidget(self.scroll)

    # Helper method to convert a PyTorch tensor to a NumPy image array
    def plot_tensor(self, tensor):
        """Convert a tensor to a NumPy image array."""
        # Detach the tensor from the computation graph and move it to CPU
        img = tensor.detach().cpu().numpy()

        # Rearrange dimensions from [C, H, W] to [H, W, C] for visualization
        img = img.transpose(1, 2, 0)

        # Rescale pixel values from [-1, 1] to [0, 1] for display
        img = (img + 1) / 2

        # Clip values to ensure they stay within [0, 1] range
        return img.clip(0, 1)

    # Method to plot a batch of real and generated images during training
    def plot(self, i, A, B, fake_A, fake_B):
        # Convert the first image in each batch tensor to NumPy format
        real_A_img = self.plot_tensor(A[0])       # Real image from domain A
        real_B_img = self.plot_tensor(B[0])       # Real image from domain B
        fake_A_img = self.plot_tensor(fake_A[0])  # Fake image generated from domain B
        fake_B_img = self.plot_tensor(fake_B[0])  # Fake image generated from domain A

        # Create a matplotlib figure with 1 row and 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Define titles for each subplot
        titles = ["Real A", "Fake B from A", "Real B", "Fake A from B"]

        # Group the images to be displayed
        images = [real_A_img, fake_B_img, real_B_img, fake_A_img]

        # Loop through each subplot and display the corresponding image
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)         # Display the image
            ax.set_title(title)    # Set the title for the subplot
            ax.axis("off")         # Hide axis ticks for cleaner display

        # Add a title to the entire figure indicating the batch number
        fig.suptitle(f"Batch {i+1}", fontsize=14)

        # Adjust layout to prevent overlap
        fig.tight_layout()

        # Embed the matplotlib figure into the PyQt layout using a canvas
        canvas = FigureCanvas(fig)
        self.layout.addWidget(canvas)

        # Automatically scroll to the bottom of the scroll area to show the latest plot
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

# Class to handle CycleGAN training in a separate thread using PyQt for UI responsiveness
class TrainingCycleGANsThread(QThread):
    # Signal to emit log messages to the UI
    log_signal = pyqtSignal(str)

    # Signal to emit image batches for visualization (index, real A, real B, fake A, fake B)
    display_signal = pyqtSignal(int, object, object, object, object)

    # Constructor to initialize training thread with models, data, and UI hooks
    def __init__(self, plot_window, DownloadLogPopup, gen_A, gen_B, disc_A, disc_B, loader, device):
        super().__init__()

        # UI components for plotting and logging
        self.plot_window = plot_window
        self.DownloadLogPopup = DownloadLogPopup

        # Device to run training on (CPU or GPU)
        self.device = device

        # Loss functions: L1 for cycle consistency, MSE for adversarial loss
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        # Initialize gradient scalers for mixed precision training if using CUDA
        if device.type == "cuda":
            self.g_scaler = torch.cuda.amp.GradScaler()
            self.d_scaler = torch.cuda.amp.GradScaler()
        else:
            self.g_scaler = None
            self.d_scaler = None

        # Learning rate for both optimizers
        self.lr = 0.00001

        # Optimizer for both discriminators
        self.opt_disc = torch.optim.Adam(
            list(disc_A.parameters()) + list(disc_B.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999)
        )

        # Optimizer for both generators
        self.opt_gen = torch.optim.Adam(
            list(gen_A.parameters()) + list(gen_B.parameters()),
            lr=self.lr,
            betas=(0.5, 0.999)
        )

        # Store model references
        self.disc_A = disc_A
        self.disc_B = disc_B
        self.gen_A = gen_A
        self.gen_B = gen_B

        # Early stopping utility (not used in this snippet but initialized)
        self.stopper = EarlyStop(patience=1000, min_delta=0.01)

        # DataLoader for training data
        self.loader = loader

        # Flag to allow stopping training manually
        self._stop_requested = False

    # Method to train one epoch of CycleGAN
    def train_epoch(self, disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen,
                    l1, mse, d_scaler, g_scaler, device, log_signal):

        # Create a progress bar for the training loop
        loop = tqdm(loader, leave=True)

        for i, (A, B) in enumerate(loop):
            # Stop training if requested
            if self._stop_requested:
                self.log_signal.emit("Training stopped by user.")
                break

            # Log the start of the current batch
            log_signal.emit(f"Batch {i+1} started.")

            # Move real images from both domains to the training device
            A = A.to(device)
            B = B.to(device)

            # Use mixed precision context if on CUDA
            context = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
            with context:
                # Generate fake images
                fake_A = gen_A(B)
                fake_B = gen_B(A)

                # Discriminator A loss
                D_A_real = disc_A(A)
                D_A_fake = disc_A(fake_A.detach())
                D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
                D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
                D_A_loss = D_A_real_loss + D_A_fake_loss

                # Discriminator B loss
                D_B_real = disc_B(B)
                D_B_fake = disc_B(fake_B.detach())
                D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
                D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
                D_B_loss = D_B_real_loss + D_B_fake_loss

                # Total discriminator loss
                D_loss = (D_A_loss + D_B_loss) / 2

            # Backpropagation for discriminators
            opt_disc.zero_grad()
            if d_scaler:
                d_scaler.scale(D_loss).backward()
                d_scaler.step(opt_disc)
                d_scaler.update()
            else:
                D_loss.backward()
                opt_disc.step()

            # Generator training
            with context:
                # Re-evaluate fake images for generator loss
                D_A_fake = disc_A(fake_A)
                D_B_fake = disc_B(fake_B)

                # Adversarial loss for generators
                loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
                loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

                # Cycle consistency loss
                cycle_B = gen_B(fake_A)
                cycle_A = gen_A(fake_B)
                cycle_B_loss = l1(B, cycle_B)
                cycle_A_loss = l1(A, cycle_A)

                # Total generator loss
                G_loss = loss_G_A + loss_G_B + cycle_A_loss * 10 + cycle_B_loss * 10

            # Backpropagation for generators
            opt_gen.zero_grad()
            if g_scaler:
                g_scaler.scale(G_loss).backward()
                g_scaler.step(opt_gen)
                g_scaler.update()
            else:
                G_loss.backward()
                opt_gen.step()

            # Emit log message with current loss values
            log_signal.emit(f"Batch {i+1}, D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}")

            # Scroll log output to the bottom
            self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
            self.DownloadLogPopup.log_output.ensureCursorVisible()
            QApplication.processEvents()

            # Emit images for visualization every 10 batches
            if i % 10 == 0:
                self.display_signal.emit(i, A, B, fake_A, fake_B)

            # Update progress bar with current loss values
            loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    # Method to request stopping the training loop
    def stop(self):
        self._stop_requested = True
        self.DownloadLogPopup.cancel_button.setEnabled(False)

    # Main method executed when the thread starts
    def run(self):
        try:
            # Emit initial log messages
            self.log_signal.emit("Training thread started.")
            self.log_signal.emit(f"Train loader has {len(self.loader)} batches.")

            # Start training for one epoch
            self.train_epoch(
                self.disc_A, self.disc_B, self.gen_A, self.gen_B,
                self.loader, self.opt_disc, self.opt_gen,
                self.l1, self.mse, self.d_scaler, self.g_scaler,
                self.device, self.log_signal
            )

            # Save trained generator models to disk
            torch.save(self.gen_A.state_dict(), "resources/models/gen_black_.pth")
            torch.save(self.gen_B.state_dict(), "resources/models/gen_blond_.pth")

            # Emit completion message
            self.log_signal.emit("Training Finished.")
            self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
            self.DownloadLogPopup.log_output.ensureCursorVisible()
            QApplication.processEvents()

        except Exception as e:
            # Emit error message if training fails
            self.log_signal.emit(f"Error during training: {str(e)}")

# A modular convolutional block used in CycleGAN architecture
# Supports both downsampling and upsampling with optional activation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        # Initialize the base nn.Module
        super().__init__()

        # Define the convolutional block using nn.Sequential
        # If downsampling is requested, use a standard Conv2d layer
        # Otherwise, use ConvTranspose2d for upsampling
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,               # Number of input channels
                out_channels,              # Number of output channels
                padding_mode="reflect",    # Use reflect padding to reduce edge artifacts
                **kwargs                   # Additional arguments like kernel_size, stride, padding
            ) if down else nn.ConvTranspose2d(
                in_channels,               # Number of input channels
                out_channels,              # Number of output channels
                **kwargs                   # Additional arguments for transposed convolution
            ),

            # Apply instance normalization to stabilize training and reduce style variance
            nn.InstanceNorm2d(out_channels),

            # Apply ReLU activation if use_act is True; otherwise, use identity (no activation)
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    # Forward pass through the convolutional block
    def forward(self, x):
        # Pass input tensor through the sequential layers
        return self.conv(x)

# Residual block used in the generator network of CycleGAN
# Helps preserve input features while enabling deeper learning
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        # Initialize the base nn.Module
        super().__init__()

        # Define the residual block as a sequence of two convolutional blocks
        self.block = nn.Sequential(
            # First ConvBlock with activation (ReLU by default)
            ConvBlock(
                channels,        # Number of input and output channels (same for residual)
                channels,        # Keeps spatial dimensions unchanged
                kernel_size=3,   # Standard kernel size for feature extraction
                padding=1        # Padding to preserve input size
            ),

            # Second ConvBlock without activation (identity)
            ConvBlock(
                channels,        # Same number of channels
                channels,        # Output matches input for residual addition
                use_act=False,   # Skip activation to preserve linearity in residual path
                kernel_size=3,   # Same kernel size
                padding=1        # Same padding to maintain dimensions
            )
        )

    # Forward pass through the residual block
    def forward(self, x):
        # Add the input tensor to the output of the block (residual connection)
        return x + self.block(x)

# Generator network for CycleGAN
# Translates images from one domain to another using an encoder-residual-decoder architecture
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        # Initialize the base nn.Module
        super().__init__()

        # Initial convolutional layer:
        # Uses a 7x7 kernel with reflection padding to reduce edge artifacts
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,         # Number of input channels (e.g., 3 for RGB)
                num_features,         # Number of output feature maps
                kernel_size=7,        # Large receptive field to capture global structure
                stride=1,             # Preserve spatial resolution
                padding=3,            # Padding to maintain input size
                padding_mode="reflect"  # Reflection padding to avoid border artifacts
            ),
            nn.InstanceNorm2d(num_features),  # Normalize across each instance
            nn.ReLU(inplace=True)             # Non-linear activation
        )

        # Downsampling blocks:
        # Reduce spatial dimensions while increasing feature depth
        self.down_blocks = nn.ModuleList([
            ConvBlock(
                num_features,             # Input channels
                num_features * 2,         # Double the channels
                kernel_size=3,
                stride=2,                 # Downsample by factor of 2
                padding=1
            ),
            ConvBlock(
                num_features * 2,         # Input from previous block
                num_features * 4,         # Double again
                kernel_size=3,
                stride=2,
                padding=1
            )
        ])

        # Residual blocks:
        # Preserve spatial dimensions and allow deeper feature learning
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4)  # Keep channel depth constant
              for _ in range(num_residuals)]   # Stack multiple residual blocks
        )

        # Upsampling blocks:
        # Restore spatial dimensions while reducing feature depth
        self.up_blocks = nn.ModuleList([
            ConvBlock(
                num_features * 4,         # Input channels
                num_features * 2,         # Halve the channels
                down=False,               # Use ConvTranspose2d for upsampling
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1          # Ensure correct output size
            ),
            ConvBlock(
                num_features * 2,
                num_features * 1,
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        ])

        # Final output layer:
        # Maps features back to image space with same number of channels as input
        self.last = nn.Conv2d(
            num_features * 1,       # Input channels
            img_channels,           # Output channels (e.g., 3 for RGB)
            kernel_size=7,          # Large kernel for smooth output
            stride=1,
            padding=3,
            padding_mode="reflect"  # Maintain spatial size and reduce artifacts
        )

    # Forward pass through the generator
    def forward(self, x):
        # Initial convolution
        x = self.initial(x)

        # Downsampling layers
        for layer in self.down_blocks:
            x = layer(x)

        # Residual transformation
        x = self.res_blocks(x)

        # Upsampling layers
        for layer in self.up_blocks:
            x = layer(x)

        # Final output with tanh activation to scale output to [-1, 1]
        return torch.tanh(self.last(x))

# A basic convolutional block used in the discriminator network of CycleGAN
# Performs downsampling with normalization and activation
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        # Initialize the base nn.Module
        super().__init__()

        # Define the convolutional block using nn.Sequential
        self.conv = nn.Sequential(
            # Convolutional layer:
            # Uses a 4x4 kernel, specified stride, and padding of 1
            # Reflection padding helps reduce edge artifacts
            nn.Conv2d(
                in_channels,         # Number of input channels
                out_channels,        # Number of output channels
                kernel_size=4,       # Kernel size for spatial reduction
                stride=stride,       # Controls downsampling rate
                padding=1,           # Padding to maintain spatial alignment
                padding_mode="reflect"  # Use reflection padding for smoother borders
            ),

            # Instance normalization to stabilize training and reduce style variance
            nn.InstanceNorm2d(out_channels),

            # LeakyReLU activation to allow small gradients for negative values
            nn.LeakyReLU(0.2, inplace=True)  # Slope of 0.2 for negative inputs
        )

    # Forward pass through the block
    def forward(self, x):
        # Pass input tensor through the convolutional block
        return self.conv(x)

# Discriminator network for CycleGAN
# Evaluates whether an image is real or generated using a PatchGAN architecture
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        # Initialize the base nn.Module
        super().__init__()

        # Initial convolutional layer:
        # Applies a 4x4 kernel with stride 2 for downsampling
        # Uses reflection padding to reduce edge artifacts
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,         # Number of input channels (e.g., 3 for RGB)
                features[0],         # First feature map size (typically 64)
                kernel_size=4,       # Kernel size for spatial reduction
                stride=2,            # Downsample by factor of 2
                padding=1,           # Padding to maintain spatial alignment
                padding_mode="reflect"  # Use reflection padding for smoother borders
            ),
            nn.LeakyReLU(0.2, inplace=True)  # LeakyReLU activation with slope 0.2
        )

        # Create additional convolutional blocks for deeper feature extraction
        layers = []
        in_channels = features[0]  # Start with output channels from initial layer

        # Iterate through remaining feature sizes to build the network
        for feature in features[1:]:
            # Use stride 1 for the last layer to preserve spatial resolution
            # Use stride 2 for earlier layers to continue downsampling
            layers.append(Block(
                in_channels,         # Input channels from previous layer
                feature,             # Output channels for current layer
                stride=1 if feature == features[-1] else 2  # Conditional stride
            ))
            in_channels = feature   # Update input channels for next block

        # Final convolutional layer:
        # Outputs a single-channel prediction map (PatchGAN)
        layers.append(nn.Conv2d(
            in_channels,            # Input channels from last block
            1,                      # Output channel for binary classification
            kernel_size=4,          # Kernel size for final decision
            stride=1,               # No further downsampling
            padding=1,              # Padding to maintain spatial size
            padding_mode="reflect"  # Use reflection padding
        ))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    # Forward pass through the discriminator
    def forward(self, x):
        # Pass input through initial layer, then through the rest of the model
        out = self.model(self.initial(x))

        # Apply sigmoid activation to convert logits to probabilities
        return torch.sigmoid(out)

# Custom PyTorch Dataset class to load and pair images from two domains (A and B)
class LoadData(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        # Initialize the base Dataset class
        super().__init__()

        # Store the root directories for domain A and domain B
        self.root_A = root_A
        self.root_B = root_B

        # Store the transformation function (e.g., from Albumentations or torchvision)
        self.transform = transform

        # Initialize list to hold image paths from domain A
        self.A_images = []

        # Loop through each directory in root_A
        for r in root_A:
            # List all files in the directory
            files = os.listdir(r)

            # Add valid image files to A_images list
            self.A_images += [
                os.path.join(r, i) for i in files
                if i.endswith(".jpg") or i.endswith(".png") or
                   i.endswith(".jpeg") or i.endswith(".gif")
            ]

        # Initialize list to hold image paths from domain B
        self.B_images = []

        # Loop through each directory in root_B
        for r in root_B:
            # List all files in the directory
            files = os.listdir(r)

            # Add valid image files to B_images list
            self.B_images += [
                os.path.join(r, i) for i in files
                if i.endswith(".jpg") or i.endswith(".png") or
                   i.endswith(".jpeg") or i.endswith(".gif")
            ]

        # Determine the length of the dataset based on the larger domain
        self.len_data = max(len(self.A_images), len(self.B_images))

        # Store individual lengths for modulo indexing
        self.A_len = len(self.A_images)
        self.B_len = len(self.B_images)

    # Return the total number of samples in the dataset
    def __len__(self):
        return self.len_data

    # Retrieve a paired sample from domain A and domain B
    def __getitem__(self, index):
        # Use modulo to cycle through images if lengths are unequal
        A_img = self.A_images[index % self.A_len]
        B_img = self.B_images[index % self.B_len]

        # Load and convert images to RGB format
        A_img = np.array(Image.open(A_img).convert("RGB"))
        B_img = np.array(Image.open(B_img).convert("RGB"))

        # Apply transformations if provided
        if self.transform:
            augmentations = self.transform(image=B_img, image0=A_img)
            B_img = augmentations["image"]
            A_img = augmentations["image0"]

        # Return the transformed image pair
        return A_img, B_img
