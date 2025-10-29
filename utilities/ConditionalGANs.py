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
    from torchvision.datasets import ImageFolder
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

class ConditionalGANs(QObject):
    # Constructor to initialize all attributes and setup environment
    def __init__(self, parent=None):
        # Call parent class constructor
        super().__init__()
        # Initialize empty list to hold training data
       