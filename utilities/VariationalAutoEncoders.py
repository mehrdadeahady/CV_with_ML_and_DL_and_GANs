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
from utilities.DLbyPyTorch import EarlyStop, DLbyPyTorch, PopupStream
from utilities.ScrollableMessageBox import show_scrollable_message
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    # print(tf.config.list_physical_devices('GPU'))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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

# Define the VariationalAutoEncoders class, representing a Variational Autoencoder (VAE) architecture
# Inherits from QObject to support Qt signal-slot mechanisms for GUI integration or logging
class VariationalAutoEncoders(QObject):

    # Constructor method to initialize the VariationalAutoEncoders instance
    def __init__(self, parent=None):
        # parent: optional reference to a parent QObject, used in Qt applications for object hierarchy and signal-slot management

        # Call the constructor of the base QObject class to enable Qt features
        super().__init__()

        # Set a fixed random seed for PyTorch to ensure reproducibility across training runs
        torch.manual_seed(0)

        # Initialize a custom log emitter for sending messages or updates (likely via Qt signals)
        self.log_emitter = LogEmitter()

        # Initialize an empty list to hold the training dataset
        self.train_set = []

        # Initialize an empty list to hold the testing dataset
        self.test_set = []

        # Define a transformation pipeline to convert input images to tensors
        self.transform = T.Compose([T.ToTensor()])

        # Set the batch size for training and testing data loaders
        self.batch_size = 32

        # Placeholder for the training data loader (to be configured later)
        self.train_loader = None

        # Placeholder for the testing data loader (to be configured later)
        self.test_loader = None

        # Placeholder for the AutoEncoder model instance
        self.model_AE = None

        # Placeholder for the optimizer used during model training
        self.optimizer = None

        # Select the computation device: use GPU if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the input dimensionality (e.g., 28x28 images flattened to 784)
        self.input_dim = 784

        # Set the dimensionality of the latent space (z-vector)
        self.z_dim = 20

        # Set the size of the hidden layer in the encoder/decoder network
        self.h_dim = 200

        # Set the learning rate for the optimizer
        self.lr = 0.00025

        # Placeholder for a generic data loader (can be used flexibly)
        self.loader = None

        # Placeholder for raw input data
        self.data = None

        # Placeholder for the Variational AutoEncoder model instance
        self.model_VAE = None

        # Set the dimensionality of the latent space for VAE-GAN hybrid architecture (if used)
        self.latent_dims = 100

        # Placeholder for generated male images (e.g., from decoder or GAN)
        self.men_g = None

        # Placeholder for generated female images
        self.women_g = None

        # Placeholder for non-generated (real) male images
        self.men_ng = None

        # Placeholder for non-generated (real) female images
        self.women_ng = None

        # Placeholder for reconstructed generated male images
        self.men_g_recon = None

        # Placeholder for reconstructed generated female images
        self.women_g_recon = None

        # Placeholder for reconstructed non-generated male images
        self.men_ng_recon = None

        # Placeholder for reconstructed non-generated female images
        self.women_ng_recon = None

        # Placeholder for encoded latent representations of generated male images
        self.men_g_encoding = None

        # Placeholder for encoded latent representations of non-generated male images
        self.men_ng_encoding = None

        # Placeholder for encoded latent representations of generated female images
        self.women_g_encoding = None

        # Placeholder for encoded latent representations of non-generated female images
        self.women_ng_encoding = None

    # Define a method to download or load the MNIST dataset, with optional download flag
    def DownloadMINIST(self, download=True):
        
        # Check if the MNIST directory exists and has sufficient size; if not, trigger download
        if not os.path.exists("temp/MNIST") or self.get_dir_size("temp/MNIST") < 66000000:
            download = True
        else:
            download = False

        # Create a popup window to display logs related to dataset download or loading
        self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

        # Display the popup window to the user
        self.DownloadLogPopup.show()

        # If download is required, notify the user that the dataset is being downloaded
        if download:
            self.DownloadLogPopup.Append_Log("Downloading MINIST Dataset.\nIt takes Minutes.\nWait ...\n")
        # Otherwise, notify the user that the dataset is being loaded from disk
        else:
            self.DownloadLogPopup.Append_Log("Loading MINIST Dataset.\nIt takes Seconds.\nWait ...\n")

        # Backup the original standard output stream to restore later
        original_stdout = sys.stdout

        # Backup the original standard error stream to restore later
        original_stderr = sys.stderr

        # Redirect standard output to the popup log so printed messages appear in the UI
        sys.stdout = PopupStream(self.DownloadLogPopup.Append_Log)

        # Redirect standard error to the popup log so error messages appear in the UI
        sys.stderr = PopupStream(self.DownloadLogPopup.Append_Log)

        try:
            # Load the MNIST training dataset, downloading if necessary
            self.train_set = torchvision.datasets.MNIST(
                root="temp",              # Directory to store or load the dataset
                train=True,               # Load the training portion of the dataset
                download=download,        # Download if not already present
                transform=self.transform  # Apply preprocessing transformations
            )

            # Load the MNIST test dataset, downloading if necessary
            self.test_set = torchvision.datasets.MNIST(
                root="temp",               # Same directory for consistency
                train=False,               # Load the testing portion of the dataset
                download=download,         # Download if not already present
                transform=self.transform   # Apply preprocessing transformations
            )

            # Log success message based on whether download occurred
            if download:
                self.DownloadLogPopup.Append_Log("Download Finished.\nClose the Log Window and Test the Dataset.")
            else:
                self.DownloadLogPopup.Append_Log("Loading Finished.\nClose the Log Window and Test the Dataset.")

        # Catch and log any exceptions that occur during dataset loading or download
        except Exception as e:
            self.DownloadLogPopup.Append_Log(f"Download Failed ❌: {e}")

        finally:
            # Restore the original standard output stream
            sys.stdout = sys.__stdout__

            # Restore the original standard error stream
            sys.stderr = sys.__stderr__

    # Define a method to recursively calculate the total size of a directory in bytes
    def get_dir_size(self, path):
        
        # Initialize a variable to accumulate the total size
        total = 0

        # Iterate over all entries (files and subdirectories) in the specified directory
        with os.scandir(path) as it:
            for entry in it:
                # If the entry is a file, add its size to the total
                if entry.is_file():
                    total += entry.stat().st_size
                # If the entry is a directory, recursively calculate its size and add to the total
                elif entry.is_dir():
                    total += self.get_dir_size(entry.path)

        # Return the total size of the directory in bytes
        return total  # Bytes

    # Method to test and display a random image from the MNIST training dataset
    def TestMINIST(self):

        # Check if the training dataset is empty
        if len(self.train_set) < 1:
            try:
                # Attempt to load the dataset without triggering a new download
                _ = self.DownloadMINIST(download=False)

                # Close the download log popup after loading completes
                self.DownloadLogPopup.close()
            except:
                # Show a warning message if dataset loading fails
                QMessageBox.warning(None, "No Dataset", "First, Download MINIST Dataset")

        # Proceed only if the training dataset is successfully loaded
        if len(self.train_set) > 0:

            # Close any OpenCV windows that might still be open
            cv2.destroyAllWindows()

            # Close any open matplotlib figures to avoid overlap
            plt.close("all")

            # Generate a random index to select a sample from the training dataset
            random_num = torch.randint(0, len(self.train_set), (1,)).item()

            # Retrieve the image tensor and its label using the random index
            image_tensor, label = self.train_set[random_num]

            # Remove the channel dimension and convert the tensor to a NumPy array
            image_np = image_tensor.squeeze().numpy()

            # Resize the image to 4x its original size for better visibility
            image_scaled = cv2.resize(image_np, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

            # Convert pixel values from [0,1] to [0,255] and cast to uint8 for OpenCV compatibility
            image_scaled = (image_scaled * 255).astype(np.uint8)

            # Display the scaled image in a window titled "Random MNIST Dataset Sample"
            cv2.imshow("Random MNIST Dataset Sample", image_scaled)

            # Wait for a key press before closing the image window
            cv2.waitKey(0)

            # Close the OpenCV window after the image is viewed
            cv2.destroyAllWindows()

        # If the dataset is still unavailable, show a warning to the user
        else:
            QMessageBox.warning(None, "No MNIST", "First, Download MINIST Dataset")

    # Method to prepare the MNIST dataset for AutoEncoder training by initializing data loaders
    def PrepareAEDataset(self):

        # Check if the training dataset is empty
        if len(self.train_set) < 1:
            try:
                # Attempt to load the dataset from disk without downloading again
                _ = self.DownloadMINIST(download=False)

                # Close the download log popup after loading completes
                self.DownloadLogPopup.close()
            except:
                # Show a warning message if dataset loading fails
                QMessageBox.warning(None, "No MNIST Dataset", "First, Download MINIST Dataset")

        # Proceed only if the training dataset is successfully loaded
        if len(self.train_set) > 0:

            # Check if data loaders have not already been initialized
            if self.train_loader is None or self.test_loader is None:

                # Create a new popup window to log the data preparation process
                self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

                # Display the popup window to the user
                self.DownloadLogPopup.show()

                # Log the start of data preparation in the popup
                self.DownloadLogPopup.Append_Log("Preparing Data.\nIt takes Seconds.\nWait ...")

                # Initialize the training data loader with batching and shuffling
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_set,
                    batch_size=self.batch_size,
                    shuffle=True
                )

                # Initialize the testing data loader with batching and shuffling
                self.test_loader = torch.utils.data.DataLoader(
                    self.test_set,
                    batch_size=self.batch_size,
                    shuffle=True
                )

                # Log a success message once data loaders are ready
                self.DownloadLogPopup.Append_Log("Data Prepared Successfully.\nClose Log Window and Create the Model.")

            # If data loaders are already initialized, notify the user
            else:
                QMessageBox.warning(None, "Data Prepared", "Data Already prepared.")

        # If the training dataset is still unavailable, show a warning
        else:
            QMessageBox.warning(None, "No MNIST", "First, Download MINIST Dataset.")

    # Method to create and initialize the AutoEncoder model for training
    def CreateAEModel(self):

        # Check if the training and testing data loaders have been prepared
        if self.train_loader is None or self.test_loader is None:
            # Warn the user that the dataset must be prepared before creating the model
            QMessageBox.warning(None, "Dataset is not Ready", "First, Prepare the Dataset.")

        # Proceed only if the model hasn't been created yet
        elif self.model_AE is None:
            # Instantiate the AutoEncoder model with input, latent, and hidden dimensions
            self.model_AE = AE(self.input_dim, self.z_dim, self.h_dim).to(self.device)

            # Initialize the optimizer (Adam) with model parameters and learning rate
            self.optimizer = torch.optim.Adam(self.model_AE.parameters(), lr=self.lr)

            # Plot the initial digit reconstructions before training begins
            self.plot_digits("Plot Before Training")

            # Notify the user that the model was created successfully
            QMessageBox.information(None, "Success", "Model Created Successfully.")

        # If the model already exists
        else:
            # Inform the user that the model has already been created
            QMessageBox.information(None, "Model Created", "Model is Already created.")

    # Method to visualize original and reconstructed digit images using the AutoEncoder
    def plot_digits(self, beforeORafter):

        # Check if the training and testing data loaders have been initialized
        if self.train_loader is None or self.test_loader is None:
            # Warn the user that the dataset must be prepared before plotting
            QMessageBox.warning(None, "Dataset is not Ready", "First, Prepare the Dataset.")
        else:
            # Initialize a list to store one sample image for each digit (0–9)
            originals = []
            idx = 0

            # Iterate through the test set to collect one image per digit
            for img, label in self.test_set:
                # If the label matches the current digit index, add the image to the list
                if label == idx:
                    originals.append(img)
                    idx += 1
                # Stop once all 10 digits have been collected
                if idx == 10:
                    break

            # Initialize a list to store reconstructed images
            reconstructed = []

            # Reconstruct each digit image using the AutoEncoder
            for idx in range(10):
                with torch.no_grad():
                    # Flatten the image and move it to the computation device
                    img = originals[idx].reshape((1, self.input_dim))
                    # Pass the image through the AutoEncoder to get the output and latent mean
                    out, mu = self.model_AE(img.to(self.device))
                # Append the reconstructed image to the list
                reconstructed.append(out)

            # Combine original and reconstructed images for visualization
            imgs = originals + reconstructed

            # Create a matplotlib figure to display the images
            fig = plt.figure(figsize=(10, 2), dpi=50)

            # Set the window title to indicate whether it's before or after training
            fig.canvas.manager.set_window_title(beforeORafter)

            # Plot each image in a 2-row grid (originals on top, reconstructions below)
            for i in range(20):
                ax = plt.subplot(2, 10, i + 1)
                # Convert the image tensor to a NumPy array for plotting
                img = (imgs[i]).detach().cpu().numpy()
                # Reshape and display the image using a binary colormap
                plt.imshow(img.reshape(28, 28), cmap="binary")
                # Remove axis ticks for cleaner visualization
                plt.xticks([])
                plt.yticks([])

            # Show the plot window
            plt.show()

    # Method to start training the AutoEncoder model and visualize progress
    def TrainAEMode(self):

        # Check if the training and testing datasets have been prepared
        if self.train_loader is None or self.test_loader is None:
            # Warn the user to prepare the dataset first
            QMessageBox.warning(None, "Dataset Not Ready", "Please prepare the dataset first.")

        # Check if the AutoEncoder model has been created
        elif self.model_AE is None:
            # Warn the user to create the model before training
            QMessageBox.warning(None, "Model Not Found", "Please create the model first.")

        # If both data and model are ready, proceed with training
        else:
            # Create a window to visualize training progress and reconstructions
            self.plot_window = PlotWindow(
                device=self.device,             # Computation device (CPU or GPU)
                input_dim=self.input_dim,       # Input dimensionality
                model_AE=self.model_AE,         # AutoEncoder model instance
                test_set=self.test_loader.dataset,  # Dataset for visualization
                latent_dims=None,               # Latent dimensions (unused here)
                model_VAE=None                  # VAE model (unused here)
            )

            # Display the plot window to the user
            self.plot_window.show()

            # Create a popup window to show training logs and allow cancellation
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

            # Enable the cancel button to allow the user to stop training
            self.DownloadLogPopup.cancel_button.setEnabled(True)

            # Display the log popup window
            self.DownloadLogPopup.show()

            # Add an initial log message to indicate training has started
            self.DownloadLogPopup.Append_Log("Training model...\nPlease wait.")

            # Create a separate thread to handle training asynchronously
            self.training_thread = TrainingAutoEncoderThread(
                self.plot_window,               # Reference to the plot window for updates
                self.DownloadLogPopup,         # Reference to the log popup for status messages
                self.train_loader,             # Training data loader
                self.input_dim,                # Input dimensionality
                self.model_AE,                 # AutoEncoder model
                self.optimizer,                # Optimizer instance
                self.device                    # Computation device
            )

            # Connect the thread's log signal to the popup's log appending method
            self.training_thread.log_signal.connect(self.DownloadLogPopup.Append_Log)

            # Connect the thread's display signal to the plot window's update method
            self.training_thread.display_signal.connect(self.plot_window.plot_AE)

            # Connect the cancel button to the thread's stop method to allow interruption
            self.DownloadLogPopup.cancel_button.clicked.connect(self.training_thread.stop)

            # Start the training thread
            self.training_thread.start()

    # Method to test the saved AutoEncoder model by loading it and visualizing its output
    def TestAEModel(self):

        # Check if the saved AutoEncoder model file exists on disk
        if os.path.exists("resources/models/AEdigits.pt"):

            # Load the AutoEncoder model using TorchScript for efficient inference
            self.model_AE = torch.jit.load(
                'resources/models/AEdigits.pt',  # Path to the saved model file
                map_location=self.device         # Load the model onto the appropriate device (CPU or GPU)
            )

            # Visualize the model's reconstruction performance after training
            self.plot_digits("After Training")

        # If the model file does not exist
        else:
            # Show a warning message prompting the user to create, train, and save the model first
            QMessageBox.warning(
                None,                            # No parent widget specified
                "Model Not Found",               # Title of the warning dialog
                "Please create, train, and save the model first."  # Message body with refined grammar
            )

    # Method to prepare the dataset for training a Variational AutoEncoder (VAE)
    def PrepareVAEDataset(self):

        # Check if required files and folders exist and contain enough data
        if os.path.exists("kagglehub/train.csv") and \
        os.path.exists("kagglehub/faces") and \
        self.CountFilesInPath("kagglehub/glasses") + self.CountFilesInPath("kagglehub/faces") >= 5000:

            # Proceed only if the data loader hasn't been initialized yet
            if self.loader is None:
                # Define a transformation pipeline: resize images and convert to tensors
                transform = T.Compose([
                    T.Resize(256),
                    T.ToTensor(),
                ])

                # Load images from the 'glasses' folder using the defined transformations
                self.data = torchvision.datasets.ImageFolder(
                    root="kagglehub/glasses",
                    transform=transform
                )

                # Set the batch size for loading images
                batch_size = 16

                # Initialize the data loader with batching and shuffling
                self.loader = torch.utils.data.DataLoader(
                    self.data,
                    batch_size=batch_size,
                    shuffle=True
                )

                # Inform the user that the dataset has been successfully prepared
                QMessageBox.information(None, "Dataset Prepared", "The dataset has been prepared successfully.")

            # If the dataset is already loaded, notify the user
            else:
                QMessageBox.information(None, "Dataset Prepared", "The dataset is already prepared.")

        # If required files or sufficient data are missing, prompt the user to set up the dataset
        else:
            QMessageBox.information(
                None,
                "Dataset Missing",
                "Please download and copy the dataset to the project root, then arrange it properly."
            )

    # Method to create and initialize the Variational AutoEncoder (VAE) model
    def CreateVAEModel(self):

        # Check if the dataset has been prepared and the data loader is available
        if self.loader is None:
            # Warn the user to prepare the dataset before creating the model
            QMessageBox.warning(None, "Dataset Not Ready", "Please prepare the dataset first.")

        # Proceed only if the VAE model has not been created yet
        elif self.model_VAE is None:
            # Instantiate the VAE model and move it to the appropriate device (CPU or GPU)
            self.model_VAE = VAE(device=self.device).to(self.device)

            # Notify the user that the model was created successfully
            QMessageBox.information(None, "Success", "The model has been created successfully.")

        # If the model already exists
        else:
            # Inform the user that the model is already created
            QMessageBox.information(None, "Model Already Exists", "The model has already been created.")

    # Method to train the Variational AutoEncoder (VAE) model and visualize training progress
    def TrainVAEModel(self):

        # Check if the dataset has been prepared
        if self.loader is None:
            # Warn the user to prepare the dataset first
            QMessageBox.warning(None, "Dataset Not Ready", "Please prepare the dataset first.")

        # Check if the VAE model has been created
        elif self.model_VAE is None:
            # Warn the user to create the model before training
            QMessageBox.warning(None, "Model Not Found", "Please create the model first.")

        # If both dataset and model are ready, proceed with training
        else:
            # Create a window to visualize training progress
            self.plot_window = PlotWindow(
                # Device to run computations on (CPU or GPU)
                device=self.device,

                # Not used for VAE, so set to None
                input_dim=None,

                # Not used for VAE, so set to None
                model_AE=None,

                # Not used for VAE, so set to None
                test_set=None,

                # Dimensionality of the latent space
                latent_dims=self.latent_dims,

                # VAE model instance to be trained
                model_VAE=self.model_VAE
            )

            # Show the plot window to the user
            self.plot_window.show()

            # Create a popup window to display training logs
            self.DownloadLogPopup = DownloadLogPopup(
                # Log emitter for streaming messages to the popup
                self.log_emitter
            )

            # Enable the cancel button to allow the user to stop training
            self.DownloadLogPopup.cancel_button.setEnabled(True)

            # Show the log popup window
            self.DownloadLogPopup.show()

            # Add an initial log message to indicate training has started
            self.DownloadLogPopup.Append_Log("Training model...\nPlease wait.")

            # Create a separate thread to handle training asynchronously
            self.training_thread = TrainingVariationalAutoEncoderThread(
                # Reference to the plot window for visual updates
                self.plot_window,

                # Reference to the log popup for status updates
                self.DownloadLogPopup,

                # Data loader for training images
                self.loader,

                # VAE model to be trained
                self.model_VAE,

                # Device to run training on
                self.device,

                # Dimensionality of the latent space
                self.latent_dims
            )

            # Connect the thread's log signal to the popup's log appending method
            self.training_thread.log_signal.connect(self.DownloadLogPopup.Append_Log)

            # Connect the thread's display signal to the plot window's VAE plotting method
            self.training_thread.display_signal.connect(self.plot_window.plot_VAE)

            # Connect the cancel button to the thread's stop method to allow interruption
            self.DownloadLogPopup.cancel_button.clicked.connect(self.training_thread.stop)

            # Start the training thread
            self.training_thread.start()

    # Method to load a previously trained Variational AutoEncoder (VAE) model from disk
    def LoadTrainedModel(self):

        # Check if the VAE model has been instantiated
        if self.model_VAE is None:
            # Show a warning message prompting the user to create the model first
            QMessageBox.warning(
                None,                          # No parent widget
                "Model Does Not Exist",        # Title of the warning dialog
                "Please create the model first."  # Message body with refined grammar
            )
            # Return False to indicate that model loading failed
            return False

        else:
            # Check if the saved model file exists on disk
            if os.path.exists("resources/models/VAEglasses.pth"):

                # Load the model weights from the file
                self.model_VAE.load_state_dict(
                    torch.load(
                        "resources/models/VAEglasses.pth",  # Path to the saved model file
                        map_location=self.device            # Load weights onto the correct device (CPU/GPU)
                    )
                )

                # Set the model to evaluation mode for inference
                self.model_VAE.eval()

                # Return True to indicate successful loading
                return True

            # If the model file does not exist
            else:
                # Show a warning message prompting the user to train and save the model first
                QMessageBox.warning(
                    None,                          # No parent widget
                    "Model Not Saved",             # Title of the warning dialog
                    "Please train and save the model first."  # Message body with refined grammar
                )
                # Return False to indicate that model loading failed
                return False

    # Define a method to test the VAE model by visualizing generated and reconstructed images
    def TestVAEModel(self):

        # Check if the trained VAE model is loaded successfully
        if not self.LoadTrainedModel():
            # Exit early if model loading failed
            return

        # Disable gradient computation for inference to improve performance
        with torch.no_grad():

            # Generate random noise vectors for the latent space
            noise = torch.randn(
                18,                      # Number of samples to generate
                self.latent_dims         # Dimensionality of the latent space
            ).to(self.device)            # Move the noise tensor to the appropriate device

            # Decode the noise vectors into images using the VAE decoder
            gen_imgs = self.model_VAE.decoder(noise).cpu()

            # Arrange generated images into a grid for visualization
            gen_imgs = torchvision.utils.make_grid(
                gen_imgs,                # Tensor of generated images
                6,                       # Number of images per row
                3                        # Padding between images
            ).numpy()

            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(
                1,                       # Number of rows
                2,                       # Number of columns
                figsize=(12, 5),         # Size of the figure in inches
                dpi=100                  # Resolution of the figure
            )

            # Set the window title for the figure
            fig.canvas.manager.set_window_title("VAE Model Output")

            # Display the generated images in the first subplot
            ax1.imshow(np.transpose(gen_imgs, (1, 2, 0)))  # Convert from CHW to HWC format
            ax1.set_title("Generated Images")              # Title for the first subplot
            ax1.axis("off")                                # Hide axis ticks and labels

            # Load a batch of real images from the data loader
            imgs, _ = next(iter(self.loader))

            # Move images to the appropriate device (CPU/GPU)
            imgs = imgs.to(self.device)

            # Pass real images through the VAE to get reconstructions
            mu, std, out = self.model_VAE(imgs)

            # Concatenate original and reconstructed images for side-by-side comparison
            real_recon = torch.cat(
                [imgs, out],            # Original and reconstructed tensors
                dim=0                   # Stack along the batch dimension
            ).detach().cpu()

            # Arrange the comparison images into a grid
            real_recon = torchvision.utils.make_grid(
                real_recon,             # Tensor of real and reconstructed images
                8,                      # Number of images per row
                4                       # Padding between images
            ).numpy()

            # Display the real and reconstructed images in the second subplot
            ax2.imshow(np.transpose(real_recon, (1, 2, 0)))  # Convert from CHW to HWC format
            ax2.set_title("Real & Reconstructed Images")     # Title for the second subplot
            ax2.axis("off")                                  # Hide axis ticks and labels

            # Adjust layout to prevent overlap and improve spacing
            plt.tight_layout()

            # Show the combined figure with both subplots
            plt.show()

    # Method to display a grid of images labeled as wearing glasses and select samples for further use
    def DisplayImagesWithGlasses(self):

        # Check if the dataset has been prepared
        if self.data is None:
            # Warn the user to prepare the dataset first
            QMessageBox.warning(None, "Dataset Not Ready", "Please prepare the dataset first.")

        else:
            # Initialize a list to store selected images
            glasses = []

            # Create a new matplotlib figure for displaying images
            fig = plt.figure()

            # Set the window title for the figure
            fig.canvas.manager.set_window_title("Images with Glasses")

            # Loop through the first 25 images in the dataset
            for i in range(25):
                # Retrieve the image and its label from the dataset
                img, label = self.data[i]

                # Add the image to the list
                glasses.append(img)

                # Create a subplot in a 5x5 grid
                plt.subplot(5, 5, i + 1)

                # Convert the image tensor to a NumPy array and display it
                plt.imshow(img.numpy().transpose((1, 2, 0)))

                # Hide axis ticks and labels
                plt.axis("off")

            # Show the complete grid of images
            plt.show()

            # Select three sample images labeled as men wearing glasses
            self.men_g = [glasses[0], glasses[3], glasses[14]]

            # Select three sample images labeled as women wearing glasses
            self.women_g = [glasses[9], glasses[15], glasses[21]]

    # Method to display a grid of images without glasses and select samples for further use
    def DisplayImagesWithoutGlasses(self):

        # Check if the dataset has been prepared
        if self.data is None:
            # Warn the user to prepare the dataset first
            QMessageBox.warning(None, "Dataset Not Ready", "Please prepare the dataset first.")

        else:
            # Initialize a list to store selected images without glasses
            noglasses = []

            # Create a new matplotlib figure for displaying images
            fig = plt.figure()

            # Set the window title for the figure
            fig.canvas.manager.set_window_title("Images Without Glasses")

            # Loop through the last 25 images in the dataset
            for i in range(25):
                # Retrieve the image and its label from the dataset in reverse order
                img, label = self.data[-i - 1]

                # Add the image to the list
                noglasses.append(img)

                # Create a subplot in a 5x5 grid
                plt.subplot(5, 5, i + 1)

                # Convert the image tensor to a NumPy array and display it
                plt.imshow(img.numpy().transpose((1, 2, 0)))

                # Hide axis ticks and labels
                plt.axis("off")

            # Show the complete grid of images
            plt.show()

            # Select three sample images labeled as men without glasses
            self.men_ng = [noglasses[1], noglasses[7], noglasses[22]]

            # Select three sample images labeled as women without glasses
            self.women_ng = [noglasses[4], noglasses[9], noglasses[19]]

    # Method to prepare grouped image batches and compute their average latent encodings using the VAE
    def PrepareImages(self):

        # Group 1: Men with glasses
        men_g_batch = torch.cat((
            self.men_g[0].unsqueeze(0),  # Add batch dimension to first image
            self.men_g[1].unsqueeze(0),  # Add batch dimension to second image
            self.men_g[2].unsqueeze(0)   # Add batch dimension to third image
        ), dim=0).to(self.device)        # Concatenate and move to device

        # Group 2: Women with glasses
        women_g_batch = torch.cat((
            self.women_g[0].unsqueeze(0),  # Add batch dimension to first image
            self.women_g[1].unsqueeze(0),  # Add batch dimension to second image
            self.women_g[2].unsqueeze(0)   # Add batch dimension to third image
        ), dim=0).to(self.device)          # Concatenate and move to device

        # Group 3: Men without glasses
        men_ng_batch = torch.cat((
            self.men_ng[0].unsqueeze(0),  # Add batch dimension to first image
            self.men_ng[1].unsqueeze(0),  # Add batch dimension to second image
            self.men_ng[2].unsqueeze(0)   # Add batch dimension to third image
        ), dim=0).to(self.device)         # Concatenate and move to device

        # Group 4: Women without glasses
        women_ng_batch = torch.cat((
            self.women_ng[0].unsqueeze(0),  # Add batch dimension to first image
            self.women_ng[1].unsqueeze(0),  # Add batch dimension to second image
            self.women_ng[2].unsqueeze(0)   # Add batch dimension to third image
        ), dim=0).to(self.device)           # Concatenate and move to device

        # Encode each group and compute the average latent representation

        # Encode men with glasses
        _, _, men_g_encodings = self.model_VAE.encoder(men_g_batch)
        self.men_g_encoding = men_g_encodings.mean(dim=0)  # Average across batch

        # Encode women with glasses
        _, _, women_g_encodings = self.model_VAE.encoder(women_g_batch)
        self.women_g_encoding = women_g_encodings.mean(dim=0)

        # Encode men without glasses
        _, _, men_ng_encodings = self.model_VAE.encoder(men_ng_batch)
        self.men_ng_encoding = men_ng_encodings.mean(dim=0)

        # Encode women without glasses
        _, _, women_ng_encodings = self.model_VAE.encoder(women_ng_batch)
        self.women_ng_encoding = women_ng_encodings.mean(dim=0)

        # Decode the average encodings to reconstruct representative images

        self.men_g_recon = self.model_VAE.decoder(self.men_g_encoding.unsqueeze(0))      # Men with glasses
        self.women_g_recon = self.model_VAE.decoder(self.women_g_encoding.unsqueeze(0))  # Women with glasses
        self.men_ng_recon = self.model_VAE.decoder(self.men_ng_encoding.unsqueeze(0))    # Men without glasses
        self.women_ng_recon = self.model_VAE.decoder(self.women_ng_encoding.unsqueeze(0))# Women without glasses

    # Method to visualize a sequence of reconstructed images:
    # Man with glasses → Woman with glasses → Man without glasses → Woman without glasses
    def DoubleTransitionCase1(self):

        # Check if the required image groups are available
        if self.women_g is not None and self.men_ng is not None:

            # Check if reconstructions have already been generated
            if self.men_g_recon is None or self.women_g_recon is None or \
            self.men_ng_recon is None or self.women_ng_recon is None:

                # Attempt to load the trained model if reconstructions are missing
                if not self.LoadTrainedModel():
                    # Exit early if model loading failed
                    return

                # Prepare image encodings and reconstructions
                self.PrepareImages()

            # Concatenate all reconstructed images into a single batch
            imgs = torch.cat((
                self.men_g_recon,     # Man with glasses
                self.women_g_recon,   # Woman with glasses
                self.men_ng_recon,    # Man without glasses
                self.women_ng_recon   # Woman without glasses
            ), dim=0)

            # Arrange the images into a grid for visualization
            imgs = torchvision.utils.make_grid(
                imgs,     # Tensor of images
                4,        # Number of images per row
                1         # Padding between images
            ).cpu().numpy()

            # Convert image format from CHW to HWC for display
            imgs = np.transpose(imgs, (1, 2, 0))

            # Create a matplotlib figure and axis
            fig, ax = plt.subplots(
                figsize=(8, 2),  # Size of the figure in inches
                dpi=100          # Resolution of the figure
            )

            # Set the window title for the figure
            fig.canvas.manager.set_window_title(
                "Man/Woman With and Without Glasses - All Reconstructions"
            )

            # Display the image grid
            plt.imshow(imgs)

            # Set the plot title with refined grammar and formatting
            plt.title(
                "Man with glasses → Woman with glasses → Man without glasses → Woman without glasses",
                fontsize=10,
                color="red"
            )

            # Hide axis ticks and labels
            plt.axis("off")

            # Show the plot window
            plt.show()

        # If required image groups are missing, prompt the user to prepare them
        else:
            QMessageBox.warning(
                None,  # No parent widget
                "Images Not Ready",  # Title of the warning dialog
                "Please display both sets of images—With Glasses and Without Glasses—before preparing."  # Refined message
            )

    # Method to visualize a semantic transition using latent space arithmetic:
    # (Man with glasses) - (Woman with glasses) + (Woman without glasses) ≈ (Man without glasses)
    def DoubleTransitionCase2(self):

        # Check if required image groups are available
        if self.women_g is not None and self.men_ng is not None:

            # Check if reconstructions are already prepared
            if self.men_g_recon is None or self.women_g_recon is None or \
            self.men_ng_recon is None or self.women_ng_recon is None:

                # Attempt to load the trained model if reconstructions are missing
                if not self.LoadTrainedModel():
                    # Exit early if model loading failed
                    return

                # Prepare image encodings and reconstructions
                self.PrepareImages()

            # Perform latent space arithmetic to estimate "man without glasses"
            z = self.men_g_encoding - self.women_g_encoding + self.women_ng_encoding

            # Decode the resulting latent vector into an image
            out = self.model_VAE.decoder(z.unsqueeze(0))  # Add batch dimension

            # Concatenate all relevant images for visualization
            imgs = torch.cat((
                self.men_g_recon,     # Man with glasses
                self.women_g_recon,   # Woman with glasses
                self.women_ng_recon,  # Woman without glasses
                out                   # Estimated man without glasses
            ), dim=0)

            # Arrange the images into a grid
            imgs = torchvision.utils.make_grid(
                imgs,     # Tensor of images
                4,        # Number of images per row
                1         # Padding between images
            ).cpu().numpy()

            # Convert image format from CHW to HWC for display
            imgs = np.transpose(imgs, (1, 2, 0))

            # Create a matplotlib figure and axis
            fig, ax = plt.subplots(
                figsize=(8, 2),  # Size of the figure in inches
                dpi=100          # Resolution of the figure
            )

            # Set the window title for the figure
            fig.canvas.manager.set_window_title(
                "Latent Transition: Man Without Glasses"
            )

            # Display the image grid
            plt.imshow(imgs)

            # Set the plot title with refined grammar and formatting
            plt.title(
                "Man with glasses − Woman with glasses + Woman without glasses ≈ Man without glasses",
                fontsize=10,
                color="red"
            )

            # Hide axis ticks and labels
            plt.axis("off")

            # Show the plot window
            plt.show()

        # If required image groups are missing, prompt the user to prepare them
        else:
            QMessageBox.warning(
                None,  # No parent widget
                "Images Not Ready",  # Title of the warning dialog
                "Please display both sets of images—With Glasses and Without Glasses—before preparing."  # Refined message
            )

    # Method to visualize a semantic transition using latent space arithmetic:
    # (Man with glasses) − (Man without glasses) + (Woman without glasses) ≈ (Woman with glasses)
    def DoubleTransitionCase3(self):

        # Check if required image groups are available
        if self.women_g is not None and self.men_ng is not None:

            # Check if reconstructions have already been generated
            if self.men_g_recon is None or self.women_g_recon is None or \
            self.men_ng_recon is None or self.women_ng_recon is None:

                # Attempt to load the trained model if reconstructions are missing
                if not self.LoadTrainedModel():
                    # Exit early if model loading failed
                    return

                # Prepare image encodings and reconstructions
                self.PrepareImages()

            # Perform latent space arithmetic to estimate "woman with glasses"
            z = self.men_g_encoding - self.men_ng_encoding + self.women_ng_encoding

            # Decode the resulting latent vector into an image
            out = self.model_VAE.decoder(z.unsqueeze(0))  # Add batch dimension

            # Concatenate all relevant images for visualization
            imgs = torch.cat((
                self.men_g_recon,     # Man with glasses
                self.men_ng_recon,    # Man without glasses
                self.women_ng_recon,  # Woman without glasses
                out                   # Estimated woman with glasses
            ), dim=0)

            # Arrange the images into a grid
            imgs = torchvision.utils.make_grid(
                imgs,     # Tensor of images
                4,        # Number of images per row
                1         # Padding between images
            ).cpu().numpy()

            # Convert image format from CHW to HWC for display
            imgs = np.transpose(imgs, (1, 2, 0))

            # Create a matplotlib figure and axis
            fig, ax = plt.subplots(
                figsize=(8, 2),  # Size of the figure in inches
                dpi=100          # Resolution of the figure
            )

            # Set the window title for the figure
            fig.canvas.manager.set_window_title(
                "Latent Transition: Woman With Glasses"
            )

            # Display the image grid
            plt.imshow(imgs)

            # Set the plot title with refined grammar and formatting
            plt.title(
                "Man with glasses − Man without glasses + Woman without glasses ≈ Woman with glasses",
                fontsize=10,
                color="red"
            )

            # Hide axis ticks and labels
            plt.axis("off")

            # Show the plot window
            plt.show()

        # If required image groups are missing, prompt the user to prepare them
        else:
            QMessageBox.warning(
                None,  # No parent widget
                "Images Not Ready",  # Title of the warning dialog
                "Please display both sets of images—With Glasses and Without Glasses—before preparing."  # Refined message
            )

    # Method to visualize a semantic transition using latent space arithmetic:
    # (Man without glasses) − (Woman without glasses) + (Woman with glasses) ≈ (Man with glasses)
    def DoubleTransitionCase4(self):

        # Check if required image groups are available
        if self.women_g is not None and self.men_ng is not None:

            # Check if reconstructions have already been generated
            if self.men_g_recon is None or self.women_g_recon is None or \
            self.men_ng_recon is None or self.women_ng_recon is None:

                # Attempt to load the trained model if reconstructions are missing
                if not self.LoadTrainedModel():
                    # Exit early if model loading failed
                    return

                # Prepare image encodings and reconstructions
                self.PrepareImages()

            # Perform latent space arithmetic to estimate "man with glasses"
            z = self.men_ng_encoding - self.women_ng_encoding + self.women_g_encoding

            # Decode the resulting latent vector into an image
            out = self.model_VAE.decoder(z.unsqueeze(0))  # Add batch dimension

            # Concatenate all relevant images for visualization
            imgs = torch.cat((
                self.men_ng_recon,    # Man without glasses
                self.women_ng_recon,  # Woman without glasses
                self.women_g_recon,   # Woman with glasses
                out                   # Estimated man with glasses
            ), dim=0)

            # Arrange the images into a grid
            imgs = torchvision.utils.make_grid(
                imgs,     # Tensor of images
                4,        # Number of images per row
                1         # Padding between images
            ).cpu().numpy()

            # Convert image format from CHW to HWC for display
            imgs = np.transpose(imgs, (1, 2, 0))

            # Create a matplotlib figure and axis
            fig, ax = plt.subplots(
                figsize=(8, 2),  # Size of the figure in inches
                dpi=100          # Resolution of the figure
            )

            # Set the window title for the figure
            fig.canvas.manager.set_window_title(
                "Latent Transition: Man With Glasses"
            )

            # Display the image grid
            plt.imshow(imgs)

            # Set the plot title with refined grammar and formatting
            plt.title(
                "Man without glasses − Woman without glasses + Woman with glasses ≈ Man with glasses",
                fontsize=10,
                color="red"
            )

            # Hide axis ticks and labels
            plt.axis("off")

            # Show the plot window
            plt.show()

        # If required image groups are missing, prompt the user to prepare them
        else:
            QMessageBox.warning(
                None,  # No parent widget
                "Images Not Ready",  # Title of the warning dialog
                "Please display both sets of images—With Glasses and Without Glasses—before preparing."  # Refined message
            )

    # Method to visualize a semantic transition using latent space arithmetic:
    # (Woman without glasses) − (Man without glasses) + (Man with glasses) ≈ (Woman with glasses)
    def DoubleTransitionCase5(self):

        # Check if required image groups are available
        if self.women_g is not None and self.men_ng is not None:

            # Check if reconstructions have already been generated
            if self.men_g_recon is None or self.women_g_recon is None or \
            self.men_ng_recon is None or self.women_ng_recon is None:

                # Attempt to load the trained model if reconstructions are missing
                if not self.LoadTrainedModel():
                    # Exit early if model loading failed
                    return

                # Prepare image encodings and reconstructions
                self.PrepareImages()

            # Perform latent space arithmetic to estimate "woman with glasses"
            z = self.women_ng_encoding - self.men_ng_encoding + self.men_g_encoding

            # Decode the resulting latent vector into an image
            out = self.model_VAE.decoder(z.unsqueeze(0))  # Add batch dimension

            # Concatenate all relevant images for visualization
            imgs = torch.cat((
                self.women_ng_recon,  # Woman without glasses
                self.men_ng_recon,    # Man without glasses
                self.men_g_recon,     # Man with glasses
                out                   # Estimated woman with glasses
            ), dim=0)

            # Arrange the images into a grid
            imgs = torchvision.utils.make_grid(
                imgs,     # Tensor of images
                4,        # Number of images per row
                1         # Padding between images
            ).cpu().numpy()

            # Convert image format from CHW to HWC for display
            imgs = np.transpose(imgs, (1, 2, 0))

            # Create a matplotlib figure and axis
            fig, ax = plt.subplots(
                figsize=(8, 2),  # Size of the figure in inches
                dpi=100          # Resolution of the figure
            )

            # Set the window title for the figure
            fig.canvas.manager.set_window_title(
                "Latent Transition: Woman With Glasses"
            )

            # Display the image grid
            plt.imshow(imgs)

            # Set the plot title with refined grammar and formatting
            plt.title(
                "Woman without glasses − Man without glasses + Man with glasses ≈ Woman with glasses",
                fontsize=10,
                color="red"
            )

            # Hide axis ticks and labels
            plt.axis("off")

            # Show the plot window
            plt.show()

        # If required image groups are missing, prompt the user to prepare them
        else:
            QMessageBox.warning(
                None,  # No parent widget
                "Images Not Ready",  # Title of the warning dialog
                "Please display both sets of images—With Glasses and Without Glasses—before preparing."  # Refined message
            )

    # Method to visualize smooth transitions between latent encodings using interpolation
    def Transition(self, sender):

        # Check if required image groups are available
        if self.women_g is not None and self.men_ng is not None:

            # Check if reconstructions have already been generated
            if self.men_g_recon is None or self.women_g_recon is None or \
            self.men_ng_recon is None or self.women_ng_recon is None:

                # Attempt to load the trained model if reconstructions are missing
                if not self.LoadTrainedModel():
                    # Exit early if model loading failed
                    return

                # Prepare image encodings and reconstructions
                self.PrepareImages()

            # Initialize list to store interpolated outputs
            results = []

            # Initialize title for the visualization
            title = ""

            # Interpolate between encodings using weights from 0 to 1
            for w in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:

                # Determine the transition type based on the sender button
                match sender:

                    # Transition: Woman with glasses → Woman without glasses
                    case "pushButton_WomanWithGlassesToWomanWithoutGlasses_VAE":
                        z = w * self.women_ng_encoding + (1 - w) * self.women_g_encoding
                        title = "Woman with Glasses → Woman without Glasses"

                    # Transition: Man with glasses → Man without glasses
                    case "pushButton_ManWithGlassesToManWithoutGlasses_VAE":
                        z = w * self.men_ng_encoding + (1 - w) * self.men_g_encoding
                        title = "Man with Glasses → Man without Glasses"

                    # Transition: Woman without glasses → Man without glasses
                    case "pushButton_WomanWithoutGlassesToManWithoutGlasses_VAE":
                        z = w * self.men_ng_encoding + (1 - w) * self.women_ng_encoding
                        title = "Woman without Glasses → Man without Glasses"

                    # Transition: Woman with glasses → Man with glasses
                    case "pushButton_WomanWithGlassesToManWithGlasses_VAE":
                        z = w * self.men_g_encoding + (1 - w) * self.women_g_encoding
                        title = "Woman with Glasses → Man with Glasses"

                # Decode the interpolated latent vector into an image
                out = self.model_VAE.decoder(z.unsqueeze(0))  # Add batch dimension
                results.append(out)

            # Concatenate all interpolated images into a single batch
            imgs = torch.cat(results, dim=0)

            # Arrange the images into a grid for visualization
            imgs = torchvision.utils.make_grid(
                imgs,     # Tensor of images
                6,        # Number of images per row
                1         # Padding between images
            ).cpu().numpy()

            # Convert image format from CHW to HWC for display
            imgs = np.transpose(imgs, (1, 2, 0))

            # Create a matplotlib figure and axis
            fig, ax = plt.subplots(dpi=100)

            # Set the window title for the figure
            fig.canvas.manager.set_window_title(title)

            # Display the image grid
            plt.imshow(imgs)

            # Set the plot title
            plt.title(title)

            # Hide axis ticks and labels
            plt.axis("off")

            # Show the plot window
            plt.show()

        # If required image groups are missing, prompt the user to prepare them
        else:
            QMessageBox.warning(
                None,  # No parent widget
                "Images Not Ready",  # Title of the warning dialog
                "Please display both sets of images—With Glasses and Without Glasses—before preparing."  # Refined message
            )

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

# Define a simple Autoencoder (AE) architecture
class AE(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim):
        super().__init__()

        # Linear layer to project input into a hidden representation
        self.common = nn.Linear(input_dim, h_dim)

        # Linear layer to encode the hidden representation into latent space
        self.encoded = nn.Linear(h_dim, z_dim)

        # Linear layer to project latent vector back to hidden space
        self.l1 = nn.Linear(z_dim, h_dim)

        # Linear layer to decode hidden representation back to input space
        self.decode = nn.Linear(h_dim, input_dim)

    # Encoder: maps input x to latent vector mu
    def encoder(self, x):
        common = F.relu(self.common(x))  # Apply ReLU to hidden layer
        mu = self.encoded(common)        # Get latent representation
        return mu

    # Decoder: reconstructs input from latent vector z
    def decoder(self, z):
        out = F.relu(self.l1(z))         # Project latent vector to hidden space
        out = torch.sigmoid(self.decode(out))  # Reconstruct input with sigmoid activation
        return out

    # Forward pass: encode input and decode reconstruction
    def forward(self, x):
        mu = self.encoder(x)             # Encode input
        out = self.decoder(mu)           # Decode reconstruction
        return out, mu                   # Return reconstruction and latent vector

# Define a class to visualize AutoEncoder (AE) and Variational AutoEncoder (VAE) outputs
# using embedded matplotlib plots inside a scrollable PyQt window
class PlotWindow(QMainWindow):

    # Constructor to initialize the plot window and its components
    def __init__(self, device, input_dim, model_AE, test_set, latent_dims, model_VAE):
        super().__init__()

        # Set the title of the main window
        self.setWindowTitle("Training Progress")

        # Store model and visualization parameters
        self.device = device                  # Device for tensor operations (CPU/GPU)
        self.input_dim = input_dim            # Input dimensionality of the model
        self.model_AE = model_AE              # Trained AutoEncoder model
        self.test_set = test_set              # Dataset for visualization
        self.latent_dims = latent_dims        # Dimensionality of latent space
        self.model_VAE = model_VAE            # Trained Variational AutoEncoder model

        # Set initial window size
        self.resize(800, 700)

        # Create scrollable area to hold dynamic plots
        self.scroll = QScrollArea()
        self.container = QWidget()            # Container widget for layout
        self.layout = QVBoxLayout(self.container)  # Vertical layout for stacking plots

        # Configure scroll behavior
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)

        # Set scroll area as the central widget of the main window
        self.setCentralWidget(self.scroll)

    # Method to visualize original and reconstructed images from the AutoEncoder
    def plot_AE(self, epoch):
        originals = []  # Store one sample image per digit (0–9)
        idx = 0

        # Collect one image per digit from the test set
        for img, label in self.test_set:
            if label == idx:
                originals.append(img)
                idx += 1
            if idx == 10:
                break

        reconstructed = []  # Store reconstructed images

        # Generate reconstructions using the AutoEncoder
        for idx in range(10):
            with torch.no_grad():
                img = originals[idx].reshape((1, self.input_dim)).to(self.device)
                out, _ = self.model_AE(img)
            reconstructed.append(out.squeeze(0).cpu())

        # Combine originals and reconstructions for display
        imgs = originals + reconstructed

        # Create a matplotlib figure to display the images
        fig = plt.figure(figsize=(10, 2), dpi=50)
        fig.suptitle(f"Epoch: {epoch}")
        fig.canvas.manager.set_window_title("After Training")

        # Add each image to the figure as a subplot
        for i in range(20):
            ax = fig.add_subplot(2, 10, i + 1)
            img = imgs[i].detach().cpu().numpy().reshape(28, 28)
            ax.imshow(img, cmap="binary")
            ax.axis("off")

        # Embed the matplotlib figure into the PyQt window
        canvas = FigureCanvas(fig)
        self.layout.addWidget(canvas)
        canvas.draw()

    # Method to visualize generated samples from the VAE decoder
    def plot_VAE(self, epoch):
        with torch.no_grad():
            # Generate random latent vectors
            noise = torch.randn(18, self.latent_dims).to(self.device)

            # Decode latent vectors into images
            imgs = self.model_VAE.decoder(noise).cpu()

            # Arrange images into a grid
            grid = torchvision.utils.make_grid(imgs, nrow=6, padding=2)
            np_grid = grid.numpy()

            # Create a matplotlib figure to display the grid
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            fig.suptitle(f"VAE Samples - Epoch: {epoch}")
            ax.imshow(np.transpose(np_grid, (1, 2, 0)))
            ax.axis("off")

            # Embed the matplotlib figure into the PyQt window
            canvas = FigureCanvas(fig)
            self.layout.addWidget(canvas)
            canvas.draw()

# Class to handle AutoEncoder training in a separate thread using PyQt for UI responsiveness
class TrainingAutoEncoderThread(QThread):

    # Signal to emit log messages to the UI
    log_signal = pyqtSignal(str)

    # Signal to trigger visualization updates (e.g., after each epoch)
    display_signal = pyqtSignal(int)

    # Constructor to initialize training thread with models, data, and UI hooks
    def __init__(self, plot_window, DownloadLogPopup, train_loader, input_dim, model_AE, optimizer, device):
        super().__init__()

        # UI components for plotting and logging
        self.plot_window = plot_window
        self.DownloadLogPopup = DownloadLogPopup

        # AutoEncoder model and optimizer
        self.model_AE = model_AE
        self.optimizer = optimizer

        # Device to run training on (CPU or GPU)
        self.device = device

        # Input dimensionality of the data
        self.input_dim = input_dim

        # Early stopping utility (initialized but not used in this snippet)
        self.stopper = EarlyStop(patience=1000, min_delta=0.01)

        # DataLoader for training data
        self.train_loader = train_loader

        # Flag to allow manual interruption of training
        self._stop_requested = False

    # Method to request stopping the training loop
    def stop(self):
        self._stop_requested = True
        self.DownloadLogPopup.cancel_button.setEnabled(False)

    # Main method executed when the thread starts
    def run(self):
        try:
            # Emit initial log messages
            self.log_signal.emit("Training thread started.")
            self.log_signal.emit(f"Train loader has {len(self.train_loader)} batches.")

            # Training loop over epochs
            for epoch in range(10):
                if self._stop_requested:
                    break  # Exit early if stop requested

                self.log_signal.emit(f"Epoch {epoch + 1} started.")
                tloss = 0  # Track total loss for the epoch

                # Loop over training batches
                for imgs, labels in self.train_loader:
                    if self._stop_requested:
                        self.log_signal.emit("Training stopped by user.")
                        break

                    # Flatten and move images to device
                    imgs = imgs.to(self.device).view(-1, self.input_dim)

                    # Forward pass through AutoEncoder
                    out, mu = self.model_AE(imgs)

                    # Compute reconstruction loss (Mean Squared Error)
                    loss = ((out - imgs) ** 2).sum()

                    # Backpropagation and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    tloss += loss.item()

                # Emit epoch summary
                self.log_signal.emit(f"At epoch {epoch + 1}, total loss = {tloss}")

                # Update UI log output
                self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
                self.DownloadLogPopup.log_output.ensureCursorVisible()

                # Keep UI responsive during training
                QApplication.processEvents()

                # Trigger visualization update
                self.display_signal.emit(epoch + 1)

            # Save trained model using TorchScript
            scripted = torch.jit.script(self.model_AE)
            scripted.save('resources/models/_AEdigits.pt')

            # Emit training completion message
            self.log_signal.emit("Training Finished.")
            self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
            self.DownloadLogPopup.log_output.ensureCursorVisible()
            QApplication.processEvents()

        except Exception as e:
            # Emit error message if training fails
            self.log_signal.emit(f"Error during training: {str(e)}")

# Class to handle Variational AutoEncoder (VAE) training in a separate thread
# using PyQt for UI responsiveness and real-time visualization
class TrainingVariationalAutoEncoderThread(QThread):

    # Signal to emit log messages to the UI
    log_signal = pyqtSignal(str)

    # Signal to trigger visualization updates (e.g., after each epoch)
    display_signal = pyqtSignal(int)

    # Constructor to initialize training thread with models, data, and UI hooks
    def __init__(self, plot_window, DownloadLogPopup, loader, model_VAE, device, latent_dims):
        super().__init__()

        # UI components for plotting and logging
        self.plot_window = plot_window
        self.DownloadLogPopup = DownloadLogPopup

        # VAE model and optimizer
        self.model_VAE = model_VAE
        self.lr = 1e-4  # Learning rate
        self.optimizer = torch.optim.Adam(
            self.model_VAE.parameters(), lr=self.lr, weight_decay=1e-5
        )

        # Device to run training on (CPU or GPU)
        self.device = device

        # Dimensionality of the latent space
        self.latent_dims = latent_dims

        # DataLoader for training data
        self.loader = loader

        # Flag to allow manual interruption of training
        self._stop_requested = False

    # Method to train the model for one epoch
    def train_epoch(self, epoch):
        self.log_signal.emit(f"Epoch {epoch + 1} started.")
        self.model_VAE.train()
        epoch_loss = 0.0

        # Loop over training batches
        for imgs, _ in self.loader:
            if self._stop_requested:
                self.log_signal.emit("Training stopped by user.")
                break

            # Move images to device
            imgs = imgs.to(self.device)

            # Forward pass through VAE
            mu, std, out = self.model_VAE(imgs)

            # Reconstruction loss (Mean Squared Error)
            reconstruction_loss = ((imgs - out) ** 2).sum()

            # KL divergence loss for regularizing latent space
            kl = ((std ** 2) / 2 + (mu ** 2) / 2 - torch.log(std) - 0.5).sum()

            # Total loss = reconstruction + KL divergence
            loss = reconstruction_loss + kl

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        # Emit epoch summary
        self.log_signal.emit(f"At epoch {epoch + 1}, loss = {epoch_loss}")

        # Update UI log output
        self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
        self.DownloadLogPopup.log_output.ensureCursorVisible()

        # Keep UI responsive during training
        QApplication.processEvents()

        # Trigger visualization update
        self.display_signal.emit(epoch + 1)

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

            # Training loop over epochs
            for epoch in range(10):
                if self._stop_requested:
                    break
                self.train_epoch(epoch)

            # Save trained VAE model to disk
            torch.save(self.model_VAE.state_dict(), "resources/models/_VAEglasses.pth")

            # Emit training completion message
            self.log_signal.emit("Training Finished.")
            self.DownloadLogPopup.log_output.moveCursor(QTextCursor.MoveOperation.End)
            self.DownloadLogPopup.log_output.ensureCursorVisible()
            QApplication.processEvents()

        except Exception as e:
            # Emit error message if training fails
            self.log_signal.emit(f"Error during training: {str(e)}")

# Encoder module for a Variational AutoEncoder (VAE)
class Encoder(nn.Module):
    def __init__(self, latent_dims=100, device="cpu"):
        super().__init__()

        # Input: RGB image of size 256x256x3
        # First convolutional layer: reduces spatial size to 128x128, outputs 8 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)

        # Second convolutional layer: reduces to 64x64, outputs 16 feature maps
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)

        # Batch normalization to stabilize training and improve convergence
        self.batch2 = nn.BatchNorm2d(num_features=16)

        # Third convolutional layer: reduces to 31x31, outputs 32 feature maps
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0)

        # Fully connected layer to project flattened features to hidden dimension
        self.linear1 = nn.Linear(in_features=31 * 31 * 32, out_features=1024)

        # Linear layer to produce mean (μ) of the latent distribution
        self.linear2 = nn.Linear(in_features=1024, out_features=latent_dims)

        # Linear layer to produce log-variance (σ) of the latent distribution
        self.linear3 = nn.Linear(in_features=1024, out_features=latent_dims)

        # Standard normal distribution for sampling latent vector z
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

        # Store device for tensor operations
        self.device = device

    # Forward pass through the encoder
    def forward(self, x):
        x = x.to(self.device)                          # Move input to device
        x = F.relu(self.conv1(x))                      # Apply first conv layer + ReLU
        x = F.relu(self.batch2(self.conv2(x)))         # Apply second conv + batch norm + ReLU
        x = F.relu(self.conv3(x))                      # Apply third conv layer + ReLU
        x = torch.flatten(x, start_dim=1)              # Flatten feature maps
        x = F.relu(self.linear1(x))                    # Project to hidden layer

        mu = self.linear2(x)                           # Compute mean of latent distribution
        std = torch.exp(self.linear3(x))               # Compute std (via exponentiation of log-variance)

        # Reparameterization trick: sample z from N(mu, std)
        z = mu + std * self.N.sample(mu.shape)

        return mu, std, z                              # Return latent parameters and sampled vector

# Decoder module for a Variational AutoEncoder (VAE)
class Decoder(nn.Module):
    def __init__(self, latent_dims=100):
        super().__init__()

        # Fully connected layers to expand latent vector into spatial feature maps
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 1024),         # Project latent vector to hidden dimension
            nn.ReLU(True),                        # Non-linear activation
            nn.Linear(1024, 31 * 31 * 32),        # Expand to match shape before convolutional decoding
            nn.ReLU(True)                         # Activation before reshaping
        )

        # Reshape flattened tensor into 3D feature maps (channels, height, width)
        self.unflatten = nn.Unflatten(
            dim=1,                                # Unflatten along channel dimension
            unflattened_size=(32, 31, 31)         # Match encoder's output shape
        )

        # Transposed convolutional layers to upsample and reconstruct the image
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(                   # Upsample: 31x31 → ~63x63
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                output_padding=1
            ),
            nn.BatchNorm2d(16),                   # Normalize feature maps
            nn.ReLU(True),                        # Activation

            nn.ConvTranspose2d(                   # Upsample: ~63x63 → ~127x127
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.BatchNorm2d(8),                    # Normalize feature maps
            nn.ReLU(True),                        # Activation

            nn.ConvTranspose2d(                   # Final upsample: ~127x127 → ~256x256
                in_channels=8,
                out_channels=3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
            # Output: reconstructed RGB image of shape (3, 256, 256)
        )

    # Forward pass through the decoder
    def forward(self, x):
        x = self.decoder_lin(x)                   # Expand latent vector
        x = self.unflatten(x)                     # Reshape to feature maps
        x = self.decoder_conv(x)                  # Upsample through transposed convolutions
        x = torch.sigmoid(x)                      # Normalize output to [0, 1] range
        return x                                  # Return reconstructed image

# Variational AutoEncoder (VAE) wrapper class
class VAE(nn.Module):
    def __init__(self, latent_dims=100, device="cpu"):
        super().__init__()

        # Encoder network: maps input images to latent distribution (mu, std)
        self.encoder = Encoder(latent_dims, device)

        # Decoder network: reconstructs images from sampled latent vector z
        self.decoder = Decoder(latent_dims)

        # Device for computation (CPU or GPU)
        self.device = device

    # Forward pass through the VAE
    def forward(self, x):
        x = x.to(self.device)                  # Move input to the correct device
        mu, std, z = self.encoder(x)           # Encode input to latent distribution and sample z
        recon = self.decoder(z)                # Decode sampled z into reconstructed image
        return mu, std, recon                  # Return latent parameters and reconstruction