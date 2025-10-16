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

class SimpleGANs(QObject):
    def __init__(self,parent=None):
        super().__init__()
        # Internal Variable to Access Data inside All Functions in the Class 
        self.train_set = []
        self.test_set = []
        self.binary_train_set = []
        self.binary_test_set = []
        self.epochs = None
        self.batch_size = 64
        self.binary_train_loader = None
        self.binary_test_loader = None
        self.binary_model = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loss_fn =  nn.CrossEntropyLoss()
        #self.stopper = EarlyStop()
        self.log_emitter = LogEmitter()
        # Assign Seed for Torch for getting same results in the Test with same parameters
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"   
        self.transform = T.Compose([T.ToTensor(),T.Normalize([0.5],[0.5])])
                          
    