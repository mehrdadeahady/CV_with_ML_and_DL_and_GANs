
import io
import sys
import contextlib
import os
from os.path import isfile, join
import time
import tkinter as tk
import threading
from utilities.DeepLearningFoundationOperations import DownloadLogPopup, LogEmitter
try:
   # os.environ["KERAS_BACKEND"] = "torch" # "tensorflow"  # or "jax", "torch"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' # '0' or '1' 1 activate intel speed support
    # print(tf.config.list_physical_devices('GPU'))
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
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
except:
    print("You Should Install matplotlib Library!")
try:
    from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt
    from PyQt6.QtWidgets import QMessageBox,QTextEdit, QWidget, QVBoxLayout, QPushButton, QLabel, QDialog, QTextEdit,QScrollArea
except:
    print("You Should Install PyQt6 Library!")

class DLbyPyTorch(QObject):
    def __init__(self,parent=None):
        super().__init__()
        # Internal Variable to Access Data inside All Functions in the Class 
        self.train_set = []
        self.test_set = []
        self.binary_train_set = []
        self.binary_test_set = []
        self.epochs = 50
        self.batch_size = 64
        self.binary_train_loader = None
        self.binary_test_loader = None
        self.binary_model = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loss_fn =  nn.CrossEntropyLoss()
        self.stopper = EarlyStop()
        self.log_emitter = LogEmitter()
        # Assign Seed for Torch for getting same results in the Test with same parameters
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"   
        self.transform = T.Compose([T.ToTensor(),T.Normalize([0.5],[0.5])])
        # Source Link: https://github.com/pranay414/Fashion-MNIST-Pytorch 
        self.text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']                          
    
    # Method to download the Fashion-MNIST dataset and log the process in a popup
    def DownloadMINIST(self, download=True):
        # Create a popup window to display download logs
        self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

        # Show the popup window to the user
        self.DownloadLogPopup.show()

        # Append an initial message to the popup log indicating the download has started
        self.DownloadLogPopup.Append_Log("Downloading Fashion-MINIST Dataset.\nIt takes Minutes.\nWait ...\n")

        # Save the original standard output stream (stdout)
        original_stdout = sys.stdout

        # Save the original standard error stream (stderr)
        original_stderr = sys.stderr

        # Redirect stdout to the popup log so print statements appear in the popup
        sys.stdout = PopupStream(self.DownloadLogPopup.Append_Log)

        # Redirect stderr to the popup log so error messages appear in the popup
        sys.stderr = PopupStream(self.DownloadLogPopup.Append_Log)

        try:
            # Download and load the Fashion-MNIST training dataset
            self.train_set = torchvision.datasets.FashionMNIST(
                root="temp",              # Directory to store the dataset
                train=True,               # Load the training portion of the dataset
                download=download,        # Download the dataset if not already present
                transform=self.transform  # Apply any preprocessing transformations
            )

            # Download and load the Fashion-MNIST test dataset
            self.test_set = torchvision.datasets.FashionMNIST(
                root="temp",              # Same directory for consistency
                train=True,               # This should likely be 'False' to load test data
                download=download,        # Download if needed
                transform=self.transform  # Apply the same transformations
            )

            # Log a success message once the download is complete
            self.DownloadLogPopup.Append_Log("Download Finished.\nClose the Log Window and Test the Dataset.")

        # Handle any exceptions that occur during download
        except Exception as e:
            # Log the error message in the popup
            self.DownloadLogPopup.Append_Log(f"Download Failed ❌: {e}")

        finally:
            # Restore the original stdout stream
            sys.stdout = sys.__stdout__

            # Restore the original stderr stream
            sys.stderr = sys.__stderr__
  
    # Method to test and display a random image from the Fashion-MNIST training dataset
    def TestMINIST(self):
        # Check if the training dataset is empty
        if len(self.train_set) < 1:
            try:
                # Attempt to load the dataset without downloading again
                _ = self.DownloadMINIST(download=False)

                # Close the download log popup after loading
                self.DownloadLogPopup.close()
            except:
                # Show a warning message if loading fails
                QMessageBox.warning(None, "No FashionMNIST", "First, Download Fashion-MINIST")

        # Proceed only if the training dataset is available
        if len(self.train_set) > 0:

            # Close any OpenCV windows that might be open
            cv2.destroyAllWindows()

            # Close any matplotlib plots that might be open
            plt.close("all")

            # Generate a random index to select a sample from the training set
            random_num = torch.randint(0, len(self.train_set), (1,)).item()

            # Retrieve the image tensor and label at the random index
            image_tensor, label = self.train_set[random_num]

            # Remove the channel dimension and convert the tensor to a NumPy array
            image_np = image_tensor.squeeze().numpy()

            # Resize the image to make it larger for display (4x scale)
            image_scaled = cv2.resize(image_np, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

            # Convert pixel values from [0,1] to [0,255] and cast to uint8 for OpenCV
            image_scaled = (image_scaled * 255).astype(np.uint8)

            # Display the image in a window titled "Random FashionMNIST Sample"
            cv2.imshow("Random FashionMNIST Sample", image_scaled)

            # Wait for a key press to close the window
            cv2.waitKey(0)

            # Close the OpenCV window after viewing
            cv2.destroyAllWindows()

        # If the dataset is still not available, show a warning
        else:
            QMessageBox.warning(None, "No MNIST", "First, Download Fashion-MINIST")
 
    # Method to plot the first 24 images from the Fashion-MNIST training dataset
    def PlotMINIST(self):
        # Check if the training dataset is empty
        if len(self.train_set) < 1:
            try:
                # Attempt to load the dataset without downloading again
                _ = self.DownloadMINIST(download=False)

                # Close the download log popup after loading
                self.DownloadLogPopup.close()
            except:
                # Show a warning message if loading fails
                QMessageBox.warning(None, "No FashionMNIST", "First, Download Fashion-MINIST")

        # Proceed only if the training dataset is available
        if len(self.train_set) > 0:

            # Close any OpenCV windows that might be open
            cv2.destroyAllWindows()

            # Close any matplotlib plots that might be open
            plt.close("all")

            # Create a new figure with specified resolution and size
            plt.figure(dpi=100, figsize=(8, 4))

            # Loop through the first 24 images in the training set
            for i in range(24):

                # Create a subplot in a 4x6 grid
                ax = plt.subplot(4, 6, i + 1)

                # Extract the image tensor from the dataset
                image = self.train_set[i][0]

                # Unnormalize the image (assuming it was normalized during transform)
                image = image / 2 + 0.5

                # Reshape the image to 28x28 pixels for display
                image = image.reshape(28, 28)

                # Display the image using a binary (black-and-white) colormap
                plt.imshow(image, cmap="binary")

                # Hide axis ticks and labels
                plt.axis('off')

                # Display the class label as the title of the subplot
                plt.title(self.text_labels[self.train_set[i][1]], fontsize=8)

            # Show the complete figure with all subplots
            plt.show()

        # If the dataset is still not available, show a warning
        else:
            QMessageBox.warning(None, "No MNIST", "First, Download Fashion-MINIST")
   
    # Method to prepare Fashion-MNIST data for binary classification using classes 0 and 9
    def PrepareDataForBinaryClassification(self):
        # Check if the training dataset is empty
        if len(self.train_set) < 1:
            try:
                # Attempt to load the dataset without downloading again
                _ = self.DownloadMINIST(download=False)

                # Close the download log popup after loading
                self.DownloadLogPopup.close()
            except:
                # Show a warning message if loading fails
                QMessageBox.warning(None, "No FashionMNIST", "First, Download Fashion-MINIST")

        # Proceed only if the training dataset is available
        if len(self.train_set) > 0:

            # Check if binary training data has not already been prepared
            if len(self.binary_train_set) < 1:

                # Create a new popup window to log the data preparation process
                self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

                # Show the popup window
                self.DownloadLogPopup.show()

                # Log the start of binary data preparation
                self.DownloadLogPopup.Append_Log("Preparing Data for Binary Classification.\nIt takes Minutes.\nWait ...")

                # Filter the training set to include only samples with labels 0 and 9
                self.binary_train_set = [x for x in self.train_set if x[1] in [0, 9]]

                # Filter the test set to include only samples with labels 0 and 9
                self.binary_test_set = [x for x in self.test_set if x[1] in [0, 9]]

                # Create a DataLoader for the binary training set with shuffling
                self.binary_train_loader = torch.utils.data.DataLoader(
                    self.binary_train_set, self.batch_size, shuffle=True
                )

                # Create a DataLoader for the binary test set with shuffling
                self.binary_test_loader = torch.utils.data.DataLoader(
                    self.binary_test_set, self.batch_size, shuffle=True
                )

                # Log that the binary data preparation is complete
                self.DownloadLogPopup.Append_Log("Data Prepared Successfully.\nClose Log Window and Create the Model.")

            # If binary data is already prepared, show a warning
            else:
                QMessageBox.warning(None, "Data Prepared", "Data Already prepared for Binary Classification.")

        # If the training dataset is still unavailable, show a warning
        else:
            QMessageBox.warning(None, "No MNIST", "First, Download Fashion-MINIST")
   
    # Method to create a binary classification model using a feedforward neural network
    def CreateBiinaryClassificationModel(self):
        # Check if the model has not already been created
        if self.binary_model is None:

            # Create a popup window to log model creation progress
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

            # Show the popup window
            self.DownloadLogPopup.show()

            # Log the start of model creation
            self.DownloadLogPopup.Append_Log("Creating Model!")

            # Define the model architecture using nn.Sequential
            self.binary_model = nn.Sequential(

                # Input layer: 784 features (28x28 image) → 256 neurons
                nn.Linear(28*28, 256),

                # Activation function: ReLU
                nn.ReLU(),

                # Hidden layer: 256 → 128 neurons
                nn.Linear(256, 128),

                # Activation function: ReLU
                nn.ReLU(),

                # Hidden layer: 128 → 32 neurons
                nn.Linear(128, 32),

                # Activation function: ReLU
                nn.ReLU(),

                # Output layer: 32 → 1 neuron (binary output)
                nn.Linear(32, 1),

                # Dropout layer to reduce overfitting
                nn.Dropout(p=0.25),

                # Sigmoid activation to produce output between 0 and 1
                nn.Sigmoid()
            ).to(self.device)  # Move model to the appropriate device (CPU or GPU)

            # Log that the model was successfully created
            self.DownloadLogPopup.Append_Log("Model created successfully.\n Ready for Training.")

        # If the model already exists, show a warning
        else:
            QMessageBox.warning(None, "Binary Model Exist", "Binary Model Already Exist for Training.")

    # Method to train the binary classification model
    def TrainBiinaryClassificationModel(self):

        # Check if training and test data loaders are ready
        if self.binary_train_loader is None or self.binary_test_loader is None:
            QMessageBox.warning(None, "Data Not Ready", "First, Prepare Data for Binary Classification.")
        else:
            # Create a popup window to log training progress
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

            # Show the popup window
            self.DownloadLogPopup.show()

            # Log the start of training
            self.DownloadLogPopup.Append_Log("Training Model!\nWait ...")

            # Set learning rate
            lr = 0.001

            # Define optimizer (Adam) for updating model weights
            optimizer = torch.optim.Adam(self.binary_model.parameters(), lr=lr)

            # Define loss function for binary classification
            loss_fn = nn.BCELoss()

            # Loop through training epochs
            for i in range(1, self.epochs + 1):

                # Initialize total loss for the epoch
                tloss = 0

                # Loop through each batch in the training data
                for n, (images, labels) in enumerate(self.binary_train_loader):

                    # Flatten images from (batch_size, 1, 28, 28) to (batch_size, 784)
                    images = images.reshape(-1, 28*28)

                    # Move images to the appropriate device
                    images = images.to(self.device)

                    # Convert labels to binary: 0 stays 0, all others become 1
                    labels = torch.FloatTensor([x if x == 0 else 1 for x in labels])

                    # Reshape labels to match output shape and move to device
                    labels = labels.reshape(-1, 1).to(self.device)

                    # Forward pass: get predictions from the model
                    preds = self.binary_model(images)

                    # Compute loss between predictions and actual labels
                    loss = loss_fn(preds, labels)

                    # Clear previous gradients
                    optimizer.zero_grad()

                    # Backpropagation: compute gradients
                    loss.backward()

                    # Update model weights
                    optimizer.step()

                    # Accumulate loss
                    tloss += loss

                # Compute average loss for the epoch
                tloss = tloss / n

                # Log the loss for the current epoch
                self.DownloadLogPopup.Append_Log(f"At epoch {i}, loss is {tloss}")

            # Log that training is complete
            self.DownloadLogPopup.Append_Log("Model Trained!")

    # Method to calculate accuracy of the binary classification model on the test set
    def CalculateBiinaryClassificationModelAccuracy(self):
        # Check if test data is ready
        if self.binary_train_loader is None or self.binary_test_loader is None:
            QMessageBox.warning(None, "Data Not Ready", "First, Prepare Data for Binary Classification.")
        else:
            # Create a popup window to log accuracy calculation
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)

            # Show the popup window
            self.DownloadLogPopup.show()

            # Log the start of accuracy calculation
            self.DownloadLogPopup.Append_Log("Calculating Biinary Classification Accuracy.\nWait ...")

            # List to store accuracy results for each batch
            results = []

            # Loop through test data batches
            for images, labels in self.binary_test_loader:

                # Flatten images and move to device
                images = images.reshape(-1, 28*28).to(self.device)

                # Convert labels to binary format and move to device
                labels = torch.FloatTensor([x if x == 0 else 1 for x in labels])
                labels = labels.reshape(-1, 1).to(self.device)

                # Get predictions from the model
                preds = self.binary_model(images)

                # Convert probabilities to binary predictions using threshold 0.5
                pred10 = torch.where(preds > 0.5, 1, 0)

                # Compare predictions with actual labels
                correct = (pred10 == labels)

                # Compute batch accuracy and store it
                results.append(correct.detach().cpu().numpy().mean())

            # Compute overall accuracy across all batches
            accuracy = np.array(results).mean()

            # Log the final accuracy
            self.DownloadLogPopup.Append_Log(f"The Accuracy of the Predictions is {accuracy}")

    # Method to prepare data for multi-category classification
    def PrepareDataForMultiCategoryClassification(self):
        # Check if training data is available
        if len(self.train_set) < 1:
            try:
                # Attempt to load dataset without downloading again
                _ = self.DownloadMINIST(download=False)
                self.DownloadLogPopup.close()
            except:
                # Show warning if loading fails
                QMessageBox.warning(None, "No FashionMNIST", "First, Download Fashion-MNIST")
        # Proceed if training data is available
        if len(self.train_set) > 0:
            # Check if data loaders are not already prepared
            if self.train_loader is None or self.val_loader is None:
                # Create and show popup for logging
                self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
                self.DownloadLogPopup.show()
                self.DownloadLogPopup.Append_Log("Preparing Data for Multi Category Classification.\nIt takes Minutes.\nWait ...")
                # Split training set into 50,000 training and 10,000 validation samples
                Multi_train_set, Multi_val_set = torch.utils.data.random_split(self.train_set, [50000, 10000])
                # Create DataLoader for training set
                self.train_loader = torch.utils.data.DataLoader(Multi_train_set, batch_size=self.batch_size, shuffle=True)
                # Create DataLoader for validation set
                self.val_loader = torch.utils.data.DataLoader(Multi_val_set, batch_size=self.batch_size, shuffle=True)
                # Create DataLoader for test set
                self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
                # Log success message
                self.DownloadLogPopup.Append_Log("Data Prepared Successfully.\nClose Log Window and Create the Model.")
            else:
                # Warn if data is already prepared
                QMessageBox.warning(None, "Data Prepared", "Data Already prepared for Multi Category Classification.")
        else:
            # Warn if dataset is unavailable
            QMessageBox.warning(None, "No MNIST", "First, Download Fashion-MNIST")

    # Method to create a multi-category classification model
    def CreateMultiCategoryClassificationModel(self):
        # Check if model is not already created
        if self.model is None:
            # Create and show popup for logging
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.show()
            self.DownloadLogPopup.Append_Log("Creating Model!")
            # Define a feedforward neural network for 10-class classification
            self.model = nn.Sequential(
                nn.Linear(28*28, 256),  # Input layer: 784 → 256
                nn.ReLU(),
                nn.Linear(256, 128),    # Hidden layer: 256 → 128
                nn.ReLU(),
                nn.Linear(128, 64),     # Hidden layer: 128 → 64
                nn.ReLU(),
                nn.Linear(64, 10)       # Output layer: 64 → 10 (for 10 classes)
            ).to(self.device)
            # Log success message
            self.DownloadLogPopup.Append_Log("Model created successfully.\n Ready for Training.")
        else:
            # Warn if model already exists
            QMessageBox.warning(None, "Multi Model Exist", "Multi Model Already Exist for Training.")

    # Method to train the model for one epoch
    def train_epoch(self):
        # Set learning rate
        lr = 0.001
        # Initialize total loss
        tloss = 0
        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Loop through training batches
        for n, (images, labels) in enumerate(self.train_loader):
            # Flatten images and move to device
            images = images.reshape(-1, 28*28).to(self.device)
            labels = labels.reshape(-1,).to(self.device)
            # Forward pass
            preds = self.model(images)
            # Compute loss
            loss = self.loss_fn(preds, labels)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate loss
            tloss += loss
        # Return average loss
        return tloss / n

    # Method to evaluate the model on the validation set
    def val_epoch(self):
        # Initialize validation loss
        vloss = 0
        # Loop through validation batches
        for n, (images, labels) in enumerate(self.val_loader):
            # Flatten images and move to device
            images = images.reshape(-1, 28*28).to(self.device)
            labels = labels.reshape(-1,).to(self.device)
            # Forward pass
            preds = self.model(images)
            # Compute loss
            loss = self.loss_fn(preds, labels)
            # Accumulate loss
            vloss += loss
        # Return average validation loss
        return vloss / n

    # Method to train the multi-category classification model
    def TrainMultiCategoryClassificationModel(self):
        # Check if data is ready
        if self.train_loader is None or self.val_loader is None:
            QMessageBox.warning(None, "Data Not Ready", "First, Prepare Data for Multi Category Classification.")
        else:
            # Create and show popup for logging
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.show()
            self.DownloadLogPopup.Append_Log("Training Model!\nWait ...")
            # Training loop
            for i in range(1, self.epochs + 1):
                # Train and validate for one epoch
                tloss = self.train_epoch()
                vloss = self.val_epoch()
                # Log epoch results
                self.DownloadLogPopup.Append_Log(f"At epoch {i}, train loss is {tloss}, value loss is {vloss}")
                # Early stopping condition
                if self.stopper.stop(vloss):
                    break
            # Log training completion
            self.DownloadLogPopup.Append_Log("Model Trained!")

    # Method to calculate and display model accuracy on test data
    def CalculateMultiCategoryClassificationModelAccuracy(self):
        # Check if data is ready
        if self.train_loader is None or self.val_loader is None:
            QMessageBox.warning(None, "Data Not Ready", "First, Prepare Data for Multi Category Classification.")
        else:
            # Close any open windows
            cv2.destroyAllWindows()
            plt.close("all")
            # Create and show popup for logging
            self.DownloadLogPopup = DownloadLogPopup(self.log_emitter)
            self.DownloadLogPopup.show()
            # Plot 30 test images with labels
            plt.figure(dpi=100, figsize=(5, 6))
            for i in range(30):
                ax = plt.subplot(5, 6, i + 1)
                img = self.test_set[i][0]
                label = self.test_set[i][1]
                img = img / 2 + 0.5
                img = img.reshape(28, 28)
                plt.imshow(img, cmap="binary")
                plt.axis('off')
                plt.title(self.text_labels[label] + f"; {label}", fontsize=8)
            # Predict and log labels for the same 30 images
            for i in range(30):
                img, label = self.test_set[i]
                img = img.reshape(-1, 28*28).to(self.device)
                pred = self.model(img)
                index_pred = torch.argmax(pred, dim=1)
                idx = index_pred.item()
                self.DownloadLogPopup.Append_Log(f"The Label is {label}; the Prediction is {idx}")
            # Log start of accuracy calculation
            self.DownloadLogPopup.Append_Log("Calculating Multi Category Classification Accuracy.\nWait ...")
            # Evaluate accuracy on test set
            results = []
            for images, labels in self.test_loader:
                images = images.reshape(-1, 28*28).to(self.device)
                labels = labels.reshape(-1,).to(self.device)
                preds = self.model(images)
                pred10 = torch.argmax(preds, dim=1)
                correct = (pred10 == labels)
                results.append(correct.detach().cpu().numpy().mean())
            # Compute and log overall accuracy
            accuracy = np.array(results).mean()
            self.DownloadLogPopup.Append_Log(f"The Accuracy of the Predictions is {accuracy}")
            # Show the plotted images
            plt.show()

# A stream wrapper class that redirects output to a logging callback
class PopupStream:
    # Initialize with a callback function to handle log messages
    def __init__(self, log_callback):
        self.log_callback = log_callback

    # Write method to intercept messages and send them to the callback
    def write(self, message):
        # Only log non-empty messages (skip blank lines)
        if message.strip():
            self.log_callback(message)

    # Flush method required for compatibility with standard streams
    def flush(self):
        pass

# A utility class for early stopping during model training
class EarlyStop:
    # Initialize with a patience value (number of epochs to wait before stopping)
    def __init__(self, patience=10):
        self.patience = patience      # Maximum allowed steps without improvement
        self.steps = 0                # Counter for consecutive non-improving steps
        self.min_loss = float('inf') # Track the minimum validation loss seen so far

    # Method to check whether training should stop based on validation loss
    def stop(self, val_loss):
        # If current loss is better than previous best, reset counter
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.steps = 0
        # If not improved, increment the counter
        elif val_loss >= self.min_loss:
            self.steps += 1
        # Return True if patience threshold is exceeded
        if self.steps >= self.patience:
            return True
        else:
            return False