# -*- Encoding: utf-8 -*- #
"""
@Author | Developer: Mehrdad Ahady
"""
from utilities.ImagesAndColorsManipulationsAndOprations import ImagesAndColorsManipulationsAndOprations
from utilities.CreateSimpleCNN import CreateSimpleCNN
from utilities.DeepLearningFoundationOperations import DeepLearningFoundationOperations
from utilities.CreateHandGestureRecognitionCNN import CreateHandGestureRecognitionCNN
from utilities.UI_MainWindow import UI_MainWindow
from utilities.FaceRecognitionOperation import FaceRecognitionOperation
from utilities.TransferLearning import TransferLearning
from utilities.NeuralStyleTransfer import NeuralStyleTransfer
from utilities.DLbyPyTorch import DLbyPyTorch
from utilities.SimpleGANs import SimpleGANs
from utilities.ConditionalGANs import ConditionalGANs
from utilities.CycleGANs import CycleGANs
from utilities.VariationalAutoEncoders import VariationalAutoEncoders
from utilities.ScrollableMessageBox import show_scrollable_message
import os
from os import path, listdir
from os.path import isfile, join
import shutil
import inspect
from functools import partial
import time
try:
    import numpy as np
except:
    print("You Should Install numpy Library")
try:
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
except:
    print("You Should Install OpenCV-Python and cv2-enumerate-cameras Libraries")
try:
    import PyQt6
    import PyQt6.QtCore
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtWidgets import QVBoxLayout,QMenu, QMainWindow, QApplication, QWidget, QMessageBox, QFileDialog
    from PyQt6.QtPdf import QPdfDocument
    from PyQt6.QtPdfWidgets import QPdfView
    from PyQt6.QtGui import QDesktopServices, QCloseEvent,QFont
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage
    from PyQt6.QtCore import QUrl, Qt
    from PyQt6.QtWebEngineCore import QWebEngineProfile
    from utilities.CustomPDFView import CustomPdfView
except:
    print("You Should Install PyQt6 Library!")

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.ui = UI_MainWindow()
        self.ui.setupUi(self)    
        self.ManualSetup()
        self.SetupUi(self)
        self.ConnectActions()

    def closeEvent(self, event):     
        cv2.destroyAllWindows()
        event.accept()
    
    def SetupUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.menu_Mathematics.setTitle(_translate("MainWindow", "🧮 Mathematics"))
        self.action_LinearAlgebraAndCalculus.setText(_translate("MainWindow", "📉 Linear Algebra and Calculus"))
        self.action_ProbabilityAndStatistics.setText(_translate("MainWindow", "🎲 Probability and Statistics"))
        self.menu_PythonProgramming.setTitle(_translate("MainWindow", "🐍 Python Programming"))
        self.action_PythonProgramming.setText(_translate("MainWindow", "🐍 Base of Python Programming"))
        self.action_Numpy.setText(_translate("MainWindow", "🔢 Numpy Library Sheet"))
        self.action_Pandas.setText(_translate("MainWindow", "🥨 Pandas Library Sheet"))
        self.action_MatPlotLib.setText(_translate("MainWindow", "📊 MatPlotLib Library Sheet"))
        self.action_SeaBorn.setText(_translate("MainWindow", "📊 SeaBorn Library Sheet"))
        self.menu_CoreMachineLearningPrinciples.setTitle(_translate("MainWindow", "🧠 Core Machine Learning Principles"))
        self.action_MLBigPicture.setText(_translate("MainWindow", "🖼️ ML Big Picture"))
        self.action_CategorizingByLearningParadigm.setText(_translate("MainWindow", "🗂️ Categorizing by Learning Paradigm"))
        self.action_FromFundamentalsToAdvanced.setText(_translate("MainWindow", "🔥 From Fundamentals to Advanced"))
        self.action_MLModelOverview.setText(_translate("MainWindow", "🌌 ML Model Overview"))
        self.action_CoreMLModelFormatSpecification.setText(_translate("MainWindow", "📚 Core ML Model Format Specification"))
        self.action_SupervisedMLProcess.setText(_translate("MainWindow", "🎵 Supervised ML Process"))
        self.action_CodeSamplesByLearningParadigm.setText(_translate("MainWindow", "📜 Code Samples by Learning Paradigm"))
        self.action_DeeperCodeSamplesWithDefinitions.setText(_translate("MainWindow", "🔍 Deeper Code Samples with Definitions"))
        self.action_TheoreticalFoundationsOfComputerVision.setText(_translate("MainWindow", "👀 Theoretical"))
        self.menu_PracticalFoundationsOfComputerVision.setTitle(_translate("MainWindow", "🛠 Practical"))
        self.action_ImagesAndColorsManipulationsAndOprations.setText(_translate("MainWindow", "🎨 Images and Colors Manipulations And Oprations"))
        self.action_CreateSimpleCNNConvolutionalNeuralNetwork.setText(_translate("MainWindow", "🕸️ Create Simple CNN(ConvolutionalNeuralNetwork)"))
        self.action_TheoreticalDeepLearningFoundation.setText(_translate("MainWindow", "֎ Theoretical"))
        self.menu_PracticalDeepLearningFoundations.setTitle(_translate("MainWindow","🛠 Practical"))
        self.action_DeepLearningFoundationOperations.setText(_translate("MainWindow","✳️ Deep Learning Foundation Operations"))
        self.action_CreateHandGestureRecognItionCNN.setText(_translate("MainWindow","✋🏻 Create Hand Gesture RecognItion CNN"))
        self.action_FaceRecognitionOperation.setText(_translate("MainWindow","🧑🏻‍🦱 Face Recognition Operation"))
        self.action_TransferLearning.setText(_translate("MainWindow","🔂 Transfer Learning"))
        self.action_NeuralStyleTransfer.setText(_translate("MainWindow","🏊 Neural Style Transfer"))
        self.menu_PracticalGANsDeploymentOptimization.setTitle(_translate("MainWindow","🛠 Practical"))
        self.menu_PracticalGANs.setTitle(_translate("MainWindow","🛠 Practical"))
        self.menu_TheoreticalGANsDeploymentOptimization.setTitle(_translate("MainWindow","📖 Theoretical"))
        self.menu_TheoreticalGANs.setTitle(_translate("MainWindow","📖 Theoretical"))
        self.action_TheoreticalGANsMainSource.setText(_translate("MainWindow","✅ GANs MainSource"))
        self.action_TheoreticalGANsSource1.setText(_translate("MainWindow","🧱 GANs Architecture Source1"))
        self.action_TheoreticalGANsSource2.setText(_translate("MainWindow","🧱 GANs Architecture Source2"))
        self.action_TheoreticalGANsSource3.setText(_translate("MainWindow","🧱 GANs Architecture Source3"))
        self.action_TheoreticalGANsSource4.setText(_translate("MainWindow","🧱 GANs Architecture Source4"))
        self.action_DLbyPyTorchBinaryAndMultiCategoryClassifications.setText(_translate("MainWindow","☯ DL by PyTorch - Binary and Multi Category Classifications"))
        self.action_SimpleGANs.setText(_translate("MainWindow","🔰 Creating 4 Simple GANs"))
        self.action_ConditionalGANs.setText(_translate("MainWindow","🐦‍🔥 Creating Conditional GANs (cGAN, wGAN)"))
        self.action_CycleGANs.setText(_translate("MainWindow","🎭 Creating Cycle GANs"))
        self.action_VariationalAutoEncoders.setText(_translate("MainWindow","🧩 Creating Variational AutoEncoders GANs"))

    def PrepareCancelTraining(self):
        self.CreateSimpleCNNHandler.CancelTraining()

    def PrepareTrainModel(self):
        total_epochs = int(self.ui.comboBox_Epochs_Step4CreateSimpleCNN.currentText().strip())
        self.CreateSimpleCNNHandler.TrainModel(total_epochs)

    def LoadMNISTRawDataOrPreparedData(self,type):
        if type == 0:
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTrainingDataValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTrainingDataValue.setText(str(self.CreateSimpleCNNHandler.x_train.shape))
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTrainingDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTrainingDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_train.shape))
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTestDataValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTestDataValue.setText(str(self.CreateSimpleCNNHandler.x_test.shape))
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTestDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info1_ShapeOfTestDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_test.shape))
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataValue.setText(str(len(self.CreateSimpleCNNHandler.x_train)))
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataLabelsValue.setText(str(len(self.CreateSimpleCNNHandler.y_train)))
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTestDataValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTestDataValue.setText(str(len(self.CreateSimpleCNNHandler.x_test)))
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTestDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info2_NumberOfSamplesInTestDataLabelsValue.setText(str(len(self.CreateSimpleCNNHandler.y_test)))
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataValue.setText(str(self.CreateSimpleCNNHandler.x_train[0].shape))
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_train[0].shape))
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataValue.setText(str(self.CreateSimpleCNNHandler.x_test[0].shape))
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step1CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_test[0].shape))
        elif type == 1:
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTrainingDataValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTrainingDataValue.setText(str(self.CreateSimpleCNNHandler.x_train.shape))
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTrainingDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTrainingDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_train.shape))
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTestDataValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTestDataValue.setText(str(self.CreateSimpleCNNHandler.x_test.shape))
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTestDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info1_ShapeOfTestDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_test.shape))
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataValue.setText(str(len(self.CreateSimpleCNNHandler.x_train)))
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTrainingDataLabelsValue.setText(str(len(self.CreateSimpleCNNHandler.y_train)))
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTestDataValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTestDataValue.setText(str(len(self.CreateSimpleCNNHandler.x_test)))
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTestDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info2_NumberOfSamplesInTestDataLabelsValue.setText(str(len(self.CreateSimpleCNNHandler.y_test)))
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataValue.setText(str(self.CreateSimpleCNNHandler.x_train[0].shape))
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTrainingDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_train[0].shape))
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataValue.setText(str(self.CreateSimpleCNNHandler.x_test[0].shape))
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataLabelsValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info3_ShapeOf1SampleInTestDataLabelsValue.setText(str(self.CreateSimpleCNNHandler.y_test[0].shape))
            self.ui.label_Step2CreateSimpleCNN_Info4NumberOfClassesValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info4NumberOfClassesValue.setText(str(self.CreateSimpleCNNHandler.numberOfClasses))
            self.ui.label_Step2CreateSimpleCNN_Info4NumberOfPixelsValue.setStyleSheet("color:red")
            self.ui.label_Step2CreateSimpleCNN_Info4NumberOfPixelsValue.setText(str(self.CreateSimpleCNNHandler.numberOfPixels))

    def PrepareOpticalCharacterRecognition(self,text):
        if str(text).strip() != "":
           self.ImagesAndColorsHandler.OpticalCharacterRecognition(text.strip())

    def PrepareObjectDetection(self,text):
        if str(text).strip() != "":
               self.ImagesAndColorsHandler.ObjectDetection(text.strip())

    def PrepareSegmentationAndContours(self,text):
        if str(text).strip() != "":
           self.ImagesAndColorsHandler.SegmentationAndContours(text.strip())

    def PrepareOperations(self,operation):
        if str(operation).strip() != "":
           self.ImagesAndColorsHandler.Operations(operation.strip())

    def PrepareFilters(self,filter):
        if str(filter).strip() != "":
           self.ImagesAndColorsHandler.Filters(filter.strip())

    def PrepareDilationErosionEdgeDetection(self,operation):
        if str(operation).strip() != "":
           self.ImagesAndColorsHandler.DilationErosionEdgeDetection(operation.strip())

    def PrepareDrawShape(self,shape):
        if str(shape).strip() != "":
           self.ImagesAndColorsHandler.DrawShape(shape)
        # else:
        #      QMessageBox.warning(None, "No Shape Selected", "First, Select a Shape!")

    def PrepareAddText(self):
        text = self.ui.textEdit_AddText.toPlainText().strip()
        if text != "":
             self.ImagesAndColorsHandler.AddText(text)
        else:
             QMessageBox.warning(None, "Empty Text", "First, Write a Text!")

    def PrepareCrop(self):
        # print(value,self.sender().objectName())
        if self.ui.label_ImageShapeValue.text().strip() != ""  and self.ImagesAndColorsHandler.image is not None:
            name = self.sender().objectName().split("_")[1]
            if name == "CropTopLefCoefficient":
                 self.ui.label_CropTopLefCoefficientValue.setText(str(self.ui.horizontalSlider_CropTopLefCoefficient.value()) +" %")
                 QMessageBox.warning(None, "Top Lef Coefficient is Set", "Set Bottom Right Coefficient to Continue!")
            else:
                TopLeft= self.ui.horizontalSlider_CropTopLefCoefficient.value()
                BottomRight = self.ui.horizontalSlider_CropBottomRightCoefficient.value()
                if TopLeft == 0 or TopLeft == 100 or BottomRight == 0 or BottomRight == 100:
                   QMessageBox.critical(None, "Coefficient Error", "Coefficient Can't be 0 % or 100 %!")
                else:
                    self.lower()
                    cv2.destroyAllWindows()
                    self.ui.label_CropBottomRightCoefficientValue.setText(str(BottomRight) + " %")
                    self.ImagesAndColorsHandler.Crop(TopLeft/100,BottomRight/100)

    def PrepareTranspose(self):
        if self.ui.label_ImageShapeValue.text().strip() != "" and self.ImagesAndColorsHandler.image is not None:
            self.lower()
            cv2.destroyAllWindows()
            self.ImagesAndColorsHandler.Transpose()

    def PrepareFlip(self,check):
        if self.ui.label_ImageShapeValue.text().strip() != "" and self.ImagesAndColorsHandler.image is not None:
            self.lower()
            cv2.destroyAllWindows()
            name = self.sender().objectName().split("_")[1]
            self.ImagesAndColorsHandler.Flip(name)

    def PrepareRotationByAngle(self,angle):
          if self.ui.label_ImageShapeValue.text().strip() != "" and self.ImagesAndColorsHandler.image is not None:
             if angle == 0:
                QMessageBox.critical(None, "Value Error", "Angle Must be Greater than 0!")
             else:
                self.lower()
                cv2.destroyAllWindows()
                self.ui.label_RotationDegreeValue.setText(str(angle) + " degree")
                self.ImagesAndColorsHandler.RotationByAngle(angle)

    def PrepareScaleByCoefficient(self,coefficient):
          if self.ui.label_ImageShapeValue.text().strip() != "" and self.ImagesAndColorsHandler.image is not None:
             if coefficient == 0:
                QMessageBox.critical(None, "Value Error", "Coefficient Must be Greater than 0!")
             else:
                self.lower()
                cv2.destroyAllWindows()
                self.ui.label_RorationScaleValue.setText(str(coefficient) + " times")
                self.ImagesAndColorsHandler.ScaleByCoefficient(coefficient)

    def PrepareTranslateImage(self,value):
        if  self.ImagesAndColorsHandler.image is not None:
            self.lower()
            cv2.destroyAllWindows()
            name = self.sender().objectName().split("_")[1]
            Diff_Array = [
                 [self.ui.dial_X1.value(),self.ui.dial_X2.value()],
                 [self.ui.dial_Y1.value(),self.ui.dial_Y2.value()],
                 [self.ui.dial_Z1.value(),self.ui.dial_Z2.value()]
            ]
            for TranslationLabelValue in self.TranslationLabelValues:
                 dialName = "dial_" + TranslationLabelValue.objectName().split("_")[1]
                 dial =  [dial for dial in self.TranslationDials if dial.objectName() == dialName]
                 TranslationLabelValue.setText(str(dial[0].value()))
            self.ImagesAndColorsHandler.TranslateImage(name,value,np.float32(Diff_Array))

    def PrepareSkewImage(self,value):
        #print(value,type(value))
        if self.ui.label_ImageShapeValue.text().strip() != "" and self.ImagesAndColorsHandler.image is not None:
            self.lower()
            cv2.destroyAllWindows()
            name = self.sender().objectName().split("_")[1]
            #value = self.sender().value()
            #time.sleep(0.1)
            self.ImagesAndColorsHandler.SkewImage(name,value)

    def PrepareResizeImage(self,value):
        if self.ui.label_ImageShapeValue.text().strip() != "" and self.ImagesAndColorsHandler.image is not None:
            self.lower()
            cv2.destroyAllWindows()
            name = self.sender().objectName().split("_")[1]
            self.ImagesAndColorsHandler.ResizeImage(name,value)

    def PreparePyrUpDown(self):
        if self.ui.label_ImageShapeValue.text().strip() != "" and self.ImagesAndColorsHandler.image is not None:
            self.lower()
            cv2.destroyAllWindows()
            name = self.sender().objectName().split("_")[1]
            self.ImagesAndColorsHandler.PyrUpDown(name)

    def PrepareColorChannelRemove(self,text,check):
        if self.ImagesAndColorsHandler.image is not None and self.ImagesAndColorsHandler.imageName is not None:
             channels = {}
             for channel in self.ColorChannelChangeCheckBoxes:
                 channels[channel.objectName().split("_")[1]] = channel.isChecked()
                 if channel.isChecked():pass
                 else:
                      channel.setDisabled(True)
                      channel.setEnabled(False)

             self.ImagesAndColorsHandler.ColorChannelRemove(channels)

        else:
             QMessageBox.warning(None, "No Image Selected", "First, Select an Image!")

    def PrepareConvertColorSpace(self,text):
        if text.strip() != "":
            self.ui.comboBox_ArithmeticAndBitwiseOperations.setCurrentIndex(0)
            self.ui.comboBox_Filters.setCurrentIndex(0)
            self.ui.comboBox_DilationErosionEdgeDetection.setCurrentIndex(0)
            self.ui.comboBox_SegmentationAndContours.setCurrentIndex(0)
            self.ui.comboBox_ObjectDetection.setCurrentIndex(0)
            self.ui.comboBox_OCR.setCurrentIndex(0)
            self.ui.comboBox_DrawShape.setCurrentIndex(0)
            self.ui.textEdit_AddText.clear()

            self.ImagesAndColorsHandler.ConvertColorSpace(text)

    def ImageSizeChanged(self, text):
            shape = self.ImagesAndColorsHandler.image.shape
            height = self.ImagesAndColorsHandler.image.shape[0]
            width = self.ImagesAndColorsHandler.image.shape[1]

            self.ui.horizontalSlider_ResizeHeight.blockSignals(True)
            self.ui.horizontalSlider_ResizeWidth.blockSignals(True)
            self.ui.horizontalSlider_SkewHeight.blockSignals(True)
            self.ui.horizontalSlider_SkewWidth.blockSignals(True)
            self.ui.horizontalSlider_ResizeHeight.setValue(height)
            self.ui.horizontalSlider_ResizeWidth.setValue(width)
            self.ui.horizontalSlider_SkewHeight.setValue(height)
            self.ui.horizontalSlider_SkewWidth.setValue(width)
            self.ui.horizontalSlider_ResizeHeight.blockSignals(False)
            self.ui.horizontalSlider_ResizeWidth.blockSignals(False)
            self.ui.horizontalSlider_SkewHeight.blockSignals(False)
            self.ui.horizontalSlider_SkewWidth.blockSignals(False)

            self.ui.label_ImageShapeValue.setText(str(shape))
            self.ui.label_ImageHeightValue.setText(str(height))
            self.ui.label_ImageWidthValue.setText(str(width))
            if self.ImagesAndColorsHandler.imageConversion not in ["BGR2GRAY","RGB2GRAY"]:
                depth = self.ImagesAndColorsHandler.image.shape[2]
                self.ui.label_ImageDepthValue.setText(str(depth))

    def SetImageInfo(self,text):
        if  self.ImagesAndColorsHandler.image is not None: #self.comboBox_SelectImage.currentText().strip() != "" and
            shape = self.ImagesAndColorsHandler.image.shape
            height = self.ImagesAndColorsHandler.image.shape[0]
            width = self.ImagesAndColorsHandler.image.shape[1]

            self.ui.horizontalSlider_ResizeHeight.setValue(height)
            self.ui.horizontalSlider_ResizeWidth.setValue(width)
            self.ui.horizontalSlider_SkewHeight.setValue(height)
            self.ui.horizontalSlider_SkewWidth.setValue(width)

            self.ui.label_ImageShapeValue.setText(str(shape))
            self.ui.label_ImageHeightValue.setText(str(height))
            self.ui.label_ImageWidthValue.setText(str(width))
            if self.ImagesAndColorsHandler.imageConversion not in ["BGR2GRAY","RGB2GRAY"]:
                depth = self.ImagesAndColorsHandler.image.shape[2]
                self.ui.label_ImageDepthValue.setText(str(depth))

            if self.ImagesAndColorsHandler.imageConversion is not None:
               match self.ImagesAndColorsHandler.imageConversion:
                    case "BGR2GRAY"|"RGB2GRAY":
                        for counter, option in enumerate(self.ColorChannelChangeCheckBoxes):
                                option.setDisabled(True)
                                option.setEnabled(False)
                                option.setChecked(False)

                    case "BGR2RGB"|"RGB2BGR"|"HSV2BGR"|"HSV2RGB":
                        for counter, option in enumerate(self.ColorChannelChangeCheckBoxes):
                                if counter in [0,1,2]:
                                    option.setDisabled(False)
                                    option.setEnabled(True)
                                    option.setChecked(True)
                                else:
                                    option.setDisabled(True)
                                    option.setEnabled(False)
                                    option.setChecked(False)

                    case "BGR2HSV"|"RGB2HSV":
                        for counter, option in enumerate(self.ColorChannelChangeCheckBoxes):
                                if counter in [3,4,5]:
                                    option.setDisabled(False)
                                    option.setEnabled(True)
                                    option.setChecked(True)
                                else:
                                    option.setDisabled(True)
                                    option.setEnabled(False)
                                    option.setChecked(False)

            else:
                for counter, option in enumerate(self.ColorChannelChangeCheckBoxes):
                                    if counter in [0,1,2]:
                                        option.setDisabled(False)
                                        option.setEnabled(True)
                                        option.setChecked(True)
                                    else:
                                        option.setDisabled(True)
                                        option.setEnabled(False)
                                        option.setChecked(False)

        else:
            self.lower()
            cv2.destroyAllWindows()
            self.ui.comboBox_ColorSpaceConversion.setCurrentIndex(0)
            self.ui.label_ImageShapeValue.clear()
            self.ui.label_ImageHeightValue.clear()
            self.ui.label_ImageWidthValue.clear()
            self.ui.label_ImageDepthValue.clear()
            self.ui.horizontalSlider_ResizeHeight.setValue(50)
            self.ui.horizontalSlider_ResizeWidth.setValue(50)
            self.ui.horizontalSlider_SkewHeight.setValue(50)
            self.ui.horizontalSlider_SkewWidth.setValue(50)
            for counter, option in enumerate(self.ColorChannelChangeCheckBoxes):
                    option.setChecked(False)
                    option.setDisabled(True)
                    option.setEnabled(False)

    def PrepareSelectVideo(self,comboBox,VideoName):
        self.ResetComboBoxSelections(comboBox)
        self.ResetParams("SelectVideo")
        if self.Is_Valid_Extension(VideoName.strip(),"video"):
           self.ImagesAndColorsHandler.ReadVideo(VideoName)
        else:
             if VideoName.strip() != "":
                QMessageBox.critical(None, "Video Extension Error", "Valid Extensions: " + " avi , mp4 , mpg , mpeg , mov , WMV , MKV , FLV ")
           
    def PrepareSelectImage(self, comboBox,text):
        self.ResetParams("SelectImage")
        self.ResetComboBoxSelections(comboBox)
        if self.Is_Valid_Extension(text.strip(),"image"):
           self.ImagesAndColorsHandler.ReadShowImage(text)
        else:
             if text.strip() != "":
                QMessageBox.critical(None, "Image Extension Error", "Valid Extensions: " + " jpg , jpeg , png , gif , bmp , psd ")

    def PrepareSelectCamera(self,comboBox,text):
        self.ResetComboBoxSelections(comboBox)
        if text != "":
           self.DLOperationsHandler.SelectDeepLearningCamera(text)

    def PrepareSelectDeepLearningOperations(self,comboBox,operation):
        self.ResetComboBoxSelections(comboBox)
        if operation != "":
            imagePath = "resources/images/" + self.ui.comboBox_SelectImage_DeepLearningFoundation.currentText().strip()
            if "VGGNet16" in operation:
                self.LoadFramePdf("VGGNet16.pdf")  
            if "VGGNet19" in operation:
                self.LoadFramePdf("VGGNet19.pdf")
            if "ResNet50" in operation:
                self.LoadFramePdf("ResNet50.pdf")
            if "Inception_v3" in operation:
                self.LoadFramePdf("Inception_v3.pdf")
            if "Xception" in operation:
                self.LoadFramePdf("Xception.pdf")
            if "Mobilenet SSD" in operation:
                self.LoadFramePdf("MobilenetSSD.pdf")
            if "MaskRCNN" in operation:
                self.LoadFramePdf("MaskRCNN.pdf")
            if "Tiny YOLO" in operation:
                self.LoadFramePdf("TinyYOLO.pdf")
            if "YOLO" in operation and not "Tiny YOLO" in operation and not "Optimized YOLO" in operation:
                self.LoadFramePdf("YOLO.pdf")
            if "Optimized YOLO" in operation:
                self.LoadFramePdf("OptimizedYOLO.pdf")
            
            accuracy = float(self.ui.comboBox_FilterAccuracy_DeepLearningFoundation.currentText().split("%")[0])/100
            self.DLOperationsHandler.SelectDeepLearningOperations(operation,imagePath, accuracy)                     

    def PrepareRecordHandGesture(self):
        if self.ImagesAndColorsHandler.camera is not None and self.ImagesAndColorsHandler.Check_Camera_Availability(self.ImagesAndColorsHandler.camera):
            if self.ui.textEdit_InsertSampleNameStep2CreateSimpleCNN2.toPlainText().strip() != "":
                sender = self.sender().objectName()
                GestureName = self.ui.textEdit_InsertSampleNameStep2CreateSimpleCNN2.toPlainText().strip()
                match sender:
                    case "pushButton_RecordTrainStep3CreateSimpleCNN2":
                        self.CreateHandGestureRecognitionCNNHandler.RecordHandGesture("train",GestureName)
                    case "pushButton_RecordTestStep4CreateSimpleCNN2":
                        self.CreateHandGestureRecognitionCNNHandler.RecordHandGesture("test",GestureName)

            else:
                QMessageBox.warning(None,"No Name Inserted","First, Insert a Name for Gesture!")
        else:
            QMessageBox.warning(None,"No Camera Selected","First, Select a Camera!")

    def PrepareTestHandGestureModel(self):
        if self.ImagesAndColorsHandler.camera is not None and self.ImagesAndColorsHandler.Check_Camera_Availability(self.ImagesAndColorsHandler.camera):
            HandGestureModelPath = os.path.normpath("./resources/models/SimpleHandGestureCNN.keras")
            if os.path.exists(HandGestureModelPath):
                self.CreateHandGestureRecognitionCNNHandler.TestHandGestureModel()
            else:
                QMessageBox.warning(None,"Model not Found","First, Create a Hand Gesture Model!")
        else:
            QMessageBox.warning(None,"No Camera Selected","First, Select a Camera!")

    def PrepareEnhanceDataset(self):
        if self.ui.textEdit_InsertSampleNameStep2CreateSimpleCNN2.toPlainText().strip() != "":
           GestureName = self.ui.textEdit_InsertSampleNameStep2CreateSimpleCNN2.toPlainText().strip()  
           GestureTrainPath = "./temp/handgesture/train/" + GestureName
           GestureTestPath = "./temp/handgesture/test/" + GestureName
           checkTrain = os.path.exists(GestureTrainPath) and os.path.isdir(GestureTrainPath)
           checkTest = os.path.exists(GestureTestPath) and os.path.isdir(GestureTestPath)
           if not checkTrain:
              QMessageBox.warning(None,"No Train Samples","First, Record Train Samples for Gesture!")
           elif not checkTest:
              QMessageBox.warning(None,"No Test Samples","First, Record Test Samples for Gesture!")
           else:
                self.CreateHandGestureRecognitionCNNHandler.EnhanceDataset(GestureName)
        else:
            QMessageBox.warning(None,"No Name Inserted","First, Insert a Name for Gesture!")
    
    def PrepareTrainHandGestureModel(self):
        total_epochs = int(self.ui.comboBox_Epochs_Step7CreateSimpleCNN2.currentText())
        self.CreateHandGestureRecognitionCNNHandler.TrainModel(total_epochs)

    def PrepareUploadFaceImage(self):
        name = self.ui.plainTextEdit_SelectName_Step1_FaceRecognitionOperation.toPlainText().strip()
        self.Upload_Files(name)
            
    def PrepareCompareVerifySimilarity(self):
        face1 = self.ui.comboBox_SelectFaceOne_Step3_FaceRecognitionOperation.currentText().strip()
        face2 = self.ui.comboBox_SelectFaceTwo_Step3_FaceRecognitionOperation.currentText().strip()
        if face1 == "" or face2 == "":
           QMessageBox.warning(None,"2 Faces Required", "First, Select 2 Faces for Comparison.")
        else:
            self.FaceRecognitionOperationHandler.VerifySimilarity(face1,face2)

    def PrepareFaceRecognitionOnCamera(self):
        if self.ImagesAndColorsHandler.camera is not None and self.ImagesAndColorsHandler.Check_Camera_Availability(self.ImagesAndColorsHandler.camera):
           self.FaceRecognitionOperationHandler.FaceRecognitionOnCamera()
        else:
            QMessageBox.warning(None,"No Camera Selected","First, Select a Camera!")

    def PrepareFaceRecognitionOnVideo(self):
        if self.ImagesAndColorsHandler.video is not None:
           video = self.ui.comboBox_SelectVideo_Step5_FaceRecognitionOperation.currentText().strip()
           if video != "":
              videoPath = os.path.normpath("resources/videos/"+video)
              self.FaceRecognitionOperationHandler.FaceRecognitionOnVideo(videoPath)
           else:
              QMessageBox.warning(None,"No Video Selected","First, Select a Video!")

        else:
            QMessageBox.warning(None,"No Video Selected","First, Select a Video!")

    def PrepareSelectModel_TransferLearning(self,text):
        if text.strip() != "":
           self.TransferLearningHandler.CreateModel(text)

    def PrepareTrainModelTransferLearning(self):
        total_epochs = int(self.ui.comboBox_Epochs_Step4TransferLearning.currentText().strip())
        print(total_epochs)
        self.TransferLearningHandler.TrainModel(total_epochs)

    def UploadFaceImage(self,path,new_path):
        if os.path.exists("resources/haarcascades/haarcascade_frontalface_default.xml"):
            face_detector = cv2.CascadeClassifier('resources/haarcascades/haarcascade_frontalface_default.xml')
            if isfile(path):
                person_image = cv2.imread(path)
                cv2.imshow("Original Image", person_image)
                face_info = face_detector.detectMultiScale(person_image, 1.3, 5)
                if len(face_info) > 0:
                    for (x,y,w,h) in face_info:
                        face = person_image[y:y+h, x:x+w]
                        file_name = os.path.basename(new_path)
                        roi = cv2.resize(face, (128, 128), interpolation = cv2.INTER_CUBIC)
                   
                    cv2.imwrite(new_path, roi)
                    self.LoadResources()
                    cv2.imshow(file_name, roi)        
                else:
                    QMessageBox.warning(None,"No Face Detected","Couldn't Detect any Face on this Image.")
                    
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        else:
            QMessageBox.warning(None,"Haarcascade not found","haarcascade_frontalface_default.xml File not found in: resources/haarcascades Path")

    def SyncSize_NeuralStyleTransfer(self, value):
        self.ui.label_SyncSize_NeuralStyleTransfer.setText(str(value))
        if self.NeuralStyleTransferHandler.image is not None and self.NeuralStyleTransferHandler.style is not None:
            self.NeuralStyleTransferHandler.SyncImageStyleSize(value)
        else:
            QMessageBox.warning(None, "No Image/Style","First, Select an Image and a Style.")

    def dial_RedValue_Changed_NeuralStyleTransfer(self, value):
        self.ui.label_RedValue_NeuralStyleTransfer.setText(str(value))

    def dial_GreenValue_Changed_NeuralStyleTransfer(self, value):
        self.ui.label_GreenValue_NeuralStyleTransfer.setText(str(value))

    def dial_BlueValue_Changed_NeuralStyleTransfer(self, value):
        self.ui.label_BlueValue_NeuralStyleTransfer.setText(str(value))
    
    def SelectImage_NeuralStyleTransfer(self, ImageName):
        if ImageName.strip() != "":
            self.ui.comboBox_SelectStyle_NeuralStyleTransfer.setCurrentIndex(0)
            self.NeuralStyleTransferHandler.SelectShowImage(ImageName)

    def PrepareTransferStyle(self, ModelName):
        if ModelName.strip() != "":
           if self.NeuralStyleTransferHandler.image is not None and self.NeuralStyleTransferHandler.style is not None:
                RedValue = self.ui.dial_Red_NeuralStyleTransfer.value()
                GreenValue = self.ui.dial_Green_NeuralStyleTransfer.value()
                BlueValue = self.ui.dial_Blue_NeuralStyleTransfer.value()
                self.NeuralStyleTransferHandler.TransferStyle(ModelName,RedValue,GreenValue,BlueValue)
           else:
                QMessageBox.warning(None, "No Image/Style","First, Select an Image and a Style.")

    def PrepareDownloadFashionMINIST(self):
        if not os.path.exists("temp/FashionMNIST") or self.get_dir_size("temp/FashionMNIST") < 84000000:
            _ = self.CheckCreateDefaultFolders()
            self.DLbyPyTorchHandler.DownloadFashionMINIST()
        else:
            self.DLbyPyTorchHandler.DownloadFashionMINIST(download=False)

    def PrepareDownloadMINIST(self):
        if not os.path.exists("temp/MNIST") or self.get_dir_size("temp/MNIST") < 84000000:
            _ = self.CheckCreateDefaultFolders()
            self.VAEHandler.DownloadMINIST()
        else:
            self.VAEHandler.DownloadMINIST(download = False)

    def DownloadEyeGlassesDataset(self):
        if  self.CountFilesInPath("kagglehub/glasses") + self.CountFilesInPath("kagglehub/faces") < 5002:
            show_scrollable_message("Download Information:","Manually Download eyeglasses Dataset from Kaggle, Below Link:" +
                                    "\nhttps://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses/data" +
                                    "\nIt contains: 1) Image folder (faces-spring-2020) 2) train.csv 3) test.csv" +
                                    "\nThere are 5000 images in the folder /faces-spring-2020/" +
                                    "\nAlso you can adjust the Download in the Code same as Below:" +
                                    "\nimport kagglehub" +
                                    "\n# Download latest version" +
                                    "\npath = kagglehub.dataset_download('jeffheaton/glasses-or-no-glasses')" +
                                    "\nprint('Path to dataset files:', path)" +
                                    "\nAfter Download:" +
                                    "\nUN-Zip the File" +
                                    "\nRename folder files to kagglehub" +
                                    "\nRename folder faces-spring-2020 to faces" +
                                    "\nCopy it into Root of your Project.")
        else:
            QMessageBox.information(None,"Dataset exist","Dataset Already Downloaded.")

    def DownloadCelebFacesDataset(self):
        if  self.CountFilesInPath("kagglehub/img_align_celeba") + self.CountFilesInPath("kagglehub/black") + self.CountFilesInPath("kagglehub/blond") < 200000:
            show_scrollable_message("Download Information:","Manually Download CelebFaces Dataset from Kaggle, Below Link:" +
                                    "\nhttps://www.kaggle.com/datasets/jessicali9530/celeba-dataset" +
                                    "\nIt contains: 1) list_attr_celeba.csv 2) list_bbox_celeba.csv 3) list_eval_partition.csv 4) list_landmarks_align_celeba.csv 5) img_align_celeba(Images Folder)" +
                                    "\nThere are 202,599 images with 10,177 unique identities in the folder /img_align_celeba/" +
                                    "\nimg_align_celeba folder located in several nested Folders, cut it and paste it in kagglehub beside .csv files" +
                                    "\nAlso you can adjust the Download in the Code same as Below:" +
                                    "\nimport kagglehub" +
                                    "\nfrom kagglehub import KaggleDatasetAdapter" +
                                    "\n# Set the path to the file you'd like to load" +
                                    "\nfile_path = ''" +
                                    "\n# Load the latest version" +
                                    "\ndf = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,'jessicali9530/celeba-dataset',file_path," +
                                    # Provide any additional arguments like 
                                    # sql_query or pandas_kwargs. See the 
                                    # documenation for more information:
                                    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
                                    ")" +                          
                                    "\nCopy it into Root of your Project.")
        else:
            QMessageBox.information(None,"Dataset exist","Dataset Already Downloaded.")

    def PrepareDataset_GrayImages_SimpleGANs(self):
        if len(self.DLbyPyTorchHandler.train_set) > 0:
            self.SimpleGANsHandler.PrepareDataset_GrayImages(self.DLbyPyTorchHandler.train_set)
        else:
            QMessageBox.warning(None,"FashionMNIST not Ready","First, Download/Load Fashion-MINIST Dataset.")

    def PrepareShowEyeGlassesImages(self):
        sender = self.sender().objectName()
        self.ConditionalGANsHandler.ShowEyeGlassesImages(sender)
 
    def PrepareShowCelebFacesImages(self):
        sender = self.sender().objectName()
        self.CycleGANsHandler.ShowCelebFacesImages(sender)

    def Epochs_DLbyPyTorch_Change(self):
        total_epochs = int(self.ui.comboBox_Epochs_DLbyPyTorch.currentText().strip())
        self.DLbyPyTorchHandler.epochs = total_epochs

    def LoadFramePdf(self, filename):
        pdfpath = "pages/" + filename
        self.pdf_path = os.path.relpath(pdfpath)
        self.frame_pdf_document.load(self.pdf_path)
        self.frame_pdf_view.pdf_path = self.pdf_path
        self.frame_pdf_view.setDocument(self.frame_pdf_document)
        self.frame_pdf_view.pdf_document = self.frame_pdf_document
        self.frame_pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.frame_pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)

    def CountFilesInPath(self, path):
        # Check if the specified path exists
        if os.path.exists(path):
            # Initialize a counter for files
            count = 0
            # Traverse the directory tree
            for root_dir, cur_dir, files in os.walk(path):
                # Add the number of files in the current directory
                count += len(files)
            # Return the total count of files
            return count
        # If the path doesn't exist, return zero
        else:
            return 0

    def get_dir_size(self,path):
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self.get_dir_size(entry.path)
        return total # Bytes

    def ResetComboBoxSelections(self, comboBox):
        if not comboBox.objectName().__contains__("comboBox_SelectOperationDeepLearningFoundation"):
            self.lower()
            cv2.destroyAllWindows()
            self.comboboxes = self.findChildren(QtWidgets.QComboBox)
            for combo in self.comboboxes:
                if combo is not comboBox and not combo.objectName().__contains__("comboBox_FilterAccuracy_DeepLearningFoundation"):
                    combo.blockSignals(True)
                    combo.setCurrentIndex(0)
                    combo.blockSignals(False)            
            
    def messageBox(self,type,title,contents):
          match type:
              case "red":
                  QMessageBox.critical(self, title, contents)
              case "blue":
                  QMessageBox.information(self, title, contents)
              case "yellow":
                  QMessageBox.warning(self, title, contents)

    def closeWindow(self):
        match self.sender().objectName():
             case "action_CloseOtherWindows":
                  self.lower()
                  cv2.destroyAllWindows()
             case "action_CloseMainWindow":
                  self.close()
                  self.destroy()
             case "action_CloseAllWindows":
                  self.lower()
                  cv2.destroyAllWindows()
                  self.close()
                  self.destroy()

    def Load_Html_File(self,file_path):
        with open(file_path, 'r') as f: #, encoding='utf-8'
            return f.read()

    def On_Cert_Error(self,e):
            # print(f"cert error: {e.description()}")
            # print(f"type: {e.type()}")
            # print(f"overridable: {e.isOverridable()}")
            # print(f"url: {e.url()}")
            # for c in e.certificateChain():
            #     print(c.toText())
            e.acceptCertificate()
            e.ignoreCertificateError()
            return True

    def CheckCreateDefaultFolders(self):
            base = os.path.normpath("resources")
            if os.path.isdir(base):
                pass
            else:
                os.makedirs(base, exist_ok=True)
            images = os.path.normpath(join("resources","images"))
            models = os.path.normpath(join("resources","models"))
            styles = os.path.normpath(join("resources","styles"))
            style_transfer_models = os.path.normpath(join("resources","style_transfer_models"))
            videos = os.path.normpath(join("resources","videos"))
            temp = os.path.normpath("temp")
            faces = os.path.normpath("resources/images/faces")
            haarcascades = os.path.normpath(join("resources","haarcascades"))
            if os.path.isdir(images):
                pass
            else:
                os.makedirs(images, exist_ok=True)
            if os.path.isdir(models):
                pass
            else:
                os.makedirs(models, exist_ok=True)
            if os.path.isdir(styles):
                pass
            else:
                os.makedirs(styles, exist_ok=True)
            if os.path.isdir(videos):
                pass
            else:
                os.makedirs(videos, exist_ok=True)
            if os.path.isdir(haarcascades):
                pass
            else:
                os.makedirs(haarcascades, exist_ok=True)
            if os.path.isdir(temp):
                pass
            else:
                os.makedirs(temp, exist_ok=True)
            if os.path.isdir(faces):
                pass
            else:
                os.makedirs(faces, exist_ok=True)
            if os.path.isdir(style_transfer_models):
                pass
            else:
                os.makedirs(style_transfer_models, exist_ok=True)

    def Upload_Files(self,name):
          self.CheckCreateDefaultFolders()
          destination_folder = os.path.normpath("resources")
          sender = self.sender().objectName() 
          file_paths, _ = QFileDialog.getOpenFileNames(self, "Select File", "", "All Files (*);;Text Files (*.txt)")
          if file_paths:
               # Copy each file
               for path in file_paths:
                    if not path.__contains__(destination_folder):
                        file_name = os.path.basename(path)

                        if sender.__contains__("UploadModels"):
                            if self.Is_Valid_Extension(file_name.strip(),"model"):
                               destination_folder = os.path.normpath(join("resources","models"))
                            else:
                                QMessageBox.critical(None, "Model Extension Error: " + file_name, "Valid Extensions: " + " keras , h5 ")
                                continue
                            
                        if sender.__contains__("UploadImages"):
                            if self.Is_Valid_Extension(file_name.strip(),"image"):
                               destination_folder = os.path.normpath(join("resources","images"))
                            else:
                                QMessageBox.critical(None, "Image Extension Error: " + file_name, "Valid Extensions: " + " jpg , jpeg , png , gif , bmp , psd ")
                                continue

                        if sender.__contains__("UploadStyles"):
                            destination_folder = os.path.normpath(join("resources","styles"))

                        if sender.__contains__("UploadVideos"):
                            if self.Is_Valid_Extension(file_name.strip(),"video"):
                               destination_folder = os.path.normpath(join("resources","Videos"))
                            else:
                                QMessageBox.critical(None, "Video Extension Error: " + file_name, "Valid Extensions: " + " avi , mp4 , mpg , mpeg , mov , wmv , mkv , flv ")
                                continue

                        if sender.__contains__("FaceRecognitionOperation"):
                            if self.Is_Valid_Extension(file_name.strip(),"image"):
                               new_name = ""
                               if name.strip() != "":
                                  new_name = name +  file_name[file_name.rindex("."):]
                               else:
                                   new_name = file_name

                               new_path = os.path.normpath("resources/images/faces/" + new_name)
                               self.UploadFaceImage(path,new_path)
                               return
                            else:
                                QMessageBox.critical(None, "Image Extension Error: " + file_name, "Valid Extensions: " + " jpg , jpeg , png , gif , bmp , psd ")
                                continue

                        if sender.__contains__("UploadStyles"):
                            if self.Is_Valid_Extension(file_name.strip(),"style"):
                               destination_folder = os.path.normpath(join("resources","styles"))
                            else:
                                QMessageBox.critical(None, "Image Extension Error: " + file_name, "Valid Extensions: " + " jpg , jpeg , png , gif , bmp , psd ")
                                continue
                      
                        if sender.__contains__("UploadStyleTransferModels"):
                            if self.Is_Valid_Extension(file_name.strip(),"style_transfer_model"):
                                destination_folder = os.path.normpath(join("resources","style_transfer_models"))
                        else:
                            QMessageBox.critical(None, "Image Extension Error: " + file_name, "Valid Extensions: " + " t7 ")
                            continue

                        dest_path = os.path.join(destination_folder, file_name)
                        if destination_folder != os.path.normpath("resources"):
                           shutil.copy2(path, dest_path)

               self.LoadResources()
               # print(f"Selected file: {file_paths}")

    def Html_In_Window(self,path):
        #  path = os.path.abspath(path)
        self.webView.setUrl(QUrl(path))
        #self.webView.load(QUrl.fromLocalFile(path))
        self.webView.show()

    def Is_Valid_Extension(self,file_name,file_type):
        match file_type:
             case "image":
                  valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp','.psd'}
             case "video":
                  valid_extensions = {'.avi','.mp4','.mpg','.mpeg','.mov','.WMV','.MKV','.FLV'}
             case "haarcascade":
                  valid_extensions = {'.xml'}
             case "model":
                  valid_extensions = {'.h5','.keras', ".caffemodel",".pb",".prototxt",".pbtxt",".cfg",".weights",".t7"}
             case "style":
                  valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp','.psd'}
             case "style_transfer_model":
                  valid_extensions = {'.t7'}
        
        return any(file_name.lower().endswith(extension) for extension in valid_extensions)
    
    def Pdf_In_Browser(self,pdf_path,local):
        if local == True:
           pdf_path = os.path.relpath(pdf_path)
           QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(pdf_path))
        else:
            QDesktopServices.openUrl(QUrl(pdf_path))

    def LoadResources(self):
        Base_Video_Path = os.path.normpath(join("resources","videos"))
        for f in listdir(Base_Video_Path):
            if isfile(join(Base_Video_Path, f)) and self.Is_Valid_Extension(f.strip(),"video"):
               if self.ui.comboBox_SelectVideo.findText(f) == -1 :
                  self.ui.comboBox_SelectVideo.addItem(f)
               if self.ui.comboBox_SelectVideo_DeepLearningFoundation.findText(f) == -1 :
                  self.ui.comboBox_SelectVideo_DeepLearningFoundation.addItem(f)
               if self.ui.comboBox_SelectVideo_Step5_FaceRecognitionOperation.findText(f) == -1 :
                  self.ui.comboBox_SelectVideo_Step5_FaceRecognitionOperation.addItem(f)

        Base_Image_Path = os.path.normpath(join("resources","images"))
        for f in listdir(Base_Image_Path):
            if isfile(join(Base_Image_Path, f)) and self.Is_Valid_Extension(f.strip(),"image"):
               if self.ui.comboBox_SelectImage.findText(f) == -1 :
                  self.ui.comboBox_SelectImage.addItem(f)
               if self.ui.comboBox_SelectImage_DeepLearningFoundation.findText(f) == -1 :
                  self.ui.comboBox_SelectImage_DeepLearningFoundation.addItem(f)
               if self.ui.comboBox_SelectImage_NeuralStyleTransfer.findText(f) == -1 :
                  self.ui.comboBox_SelectImage_NeuralStyleTransfer.addItem(f)

        Base_FaceImage_Path = os.path.normpath("resources/images/faces")
        for f in listdir(Base_FaceImage_Path):
            if isfile(join(Base_FaceImage_Path, f)) and self.Is_Valid_Extension(f.strip(),"image"):
               if self.ui.comboBox_SelectFaceOne_Step3_FaceRecognitionOperation.findText(f) == -1 :
                  self.ui.comboBox_SelectFaceOne_Step3_FaceRecognitionOperation.addItem(f)
               if self.ui.comboBox_SelectFaceTwo_Step3_FaceRecognitionOperation.findText(f) == -1 :
                  self.ui.comboBox_SelectFaceTwo_Step3_FaceRecognitionOperation.addItem(f)

        Base_Style_Path = os.path.normpath(join("resources","styles"))
        for f in listdir(Base_Style_Path):
            if isfile(join(Base_Style_Path, f)) and self.Is_Valid_Extension(f.strip(),"style"):
               if self.ui.comboBox_SelectStyle_NeuralStyleTransfer.findText(f) == -1 :
                  self.ui.comboBox_SelectStyle_NeuralStyleTransfer.addItem(f)

        for camera_info in enumerate_cameras(): # cv2.CAP_MSMF  param for windows
             cap = f"Index: {camera_info.index}, Name: {camera_info.name}, Backend: {camera_info.backend}"
             if self.ui.comboBox_SelectCameraDeepLearningFoundation.findText(cap) == -1 :
                  self.ui.comboBox_SelectCameraDeepLearningFoundation.addItem(cap)
             if self.ui.comboBox_SelectCameraStep1CreateSimpleCNN2.findText(cap) == -1 :
                  self.ui.comboBox_SelectCameraStep1CreateSimpleCNN2.addItem(cap)
             if self.ui.comboBox_SelectCamera_Step4_FaceRecognitionOperation.findText(cap) == -1 :
                  self.ui.comboBox_SelectCamera_Step4_FaceRecognitionOperation.addItem(cap)

    def FillCode(self, function, textBrowser, LineStart):
        function_code = inspect.getsource(function)
        lines = function_code.splitlines()[LineStart:]
        commentCount = 0
        ChangedContent = ""
        for index,line in enumerate(lines):
            #print(index)
            stripedLine = line.strip()
            if lines[index].strip().startswith("'''"): commentCount += 1
            if stripedLine.startswith("#") or (commentCount % 2 != 0 or lines[index].strip().startswith("'''")):
                line = "<span style='color: green'>" + line +"</span>" #.strip()
            ChangedContent += line +"\n"
        textBrowser.setHtml(("<pre>" + ChangedContent ).strip())
        textBrowser.show()

    def SaveCode(self, textBrowser):
        # Choose file location
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;HTML Files (*.html);;All Files (*)")
        if file_path:
            content = ""
            # Choose between plain text or HTML
            if(file_path.endswith("html") or file_path.endswith("htm")):
                content = textBrowser.toHtml()
            else:
                content = textBrowser.toPlainText()
            with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
    
    def changePage(self):
        cv2.destroyAllWindows()
        selectedPage = self.ui.pages.findChild(QtWidgets.QWidget,"page_" + self.sender().objectName().split("_")[1])
        if selectedPage != None:
           self.ui.pages.setCurrentWidget(selectedPage)
  
    def changePDFPage(self,index):
        cv2.destroyAllWindows()
        self.ui.pages.setCurrentWidget(self.pdf_view)
        self.pdf_path = ""
        match index:
             case 0:
                  self.pdf_path = os.path.relpath("pages/BigPicture.pdf")
             case 1:
                  self.pdf_path = os.path.relpath("pages/UniversityCurriculum.pdf")
             case 2:
                  self.pdf_path = os.path.relpath("pages/RoadMap.pdf")
             case 3:
                  self.pdf_path = os.path.relpath("pages/StudyPlan.pdf")
             case 4:
                  self.pdf_path = os.path.relpath("pages/HeadingResearch.pdf")
             case 5:
                  self.pdf_path = os.path.relpath("pages/UserGuide.pdf")
             case 6:
                  self.pdf_path = os.path.relpath("pages/ML_BigPicture.pdf")
             case 7:
                  self.pdf_path = os.path.relpath("pages/CategorizingByLearningParadigm.pdf")
             case 8:
                  self.pdf_path = os.path.relpath("pages/FromFundamentalsToAdvanced.pdf")
             case 9:
                  self.pdf_path = os.path.relpath("pages/CodeSamplesByLearningParadigm.pdf")
             case 10:
                  self.pdf_path = os.path.relpath("pages/DeeperCodeSamplesWithDefinitions.pdf")
             case 11:
                  self.pdf_path = os.path.relpath("pages/TheoreticalFoundationsOfComputerVision.pdf")
             case 12:
                  self.pdf_path = os.path.relpath("pages/Numpy_Sheet.pdf")
             case 13:
                  self.pdf_path = os.path.relpath("pages/Pandas_Sheet.pdf")
             case 14:
                  self.pdf_path = os.path.relpath("pages/MatPlotLib_Sheet.pdf")
             case 15:
                  self.pdf_path = os.path.relpath("pages/SeaBorn_Sheet.pdf")
             case 16:
                  self.pdf_path = os.path.relpath("pages/SupervisedML_Process.pdf")
             case 17:
                  self.pdf_path = os.path.relpath("pages/TheoreticalDeepLearningFoundation.pdf")
             case 18:
                  self.pdf_path = os.path.relpath("pages/GANs1.pdf")
             case 19:
                  self.pdf_path = os.path.relpath("pages/GANs2.pdf")
             case 20:
                  self.pdf_path = os.path.relpath("pages/GANs3.pdf")
             case 21:
                  self.pdf_path = os.path.relpath("pages/GANs4.pdf")
             case 22:
                  self.pdf_path = os.path.relpath("pages/GANs.pdf")
             case 23:
                  self.pdf_path = os.path.relpath("pages/CoreCVTasks.pdf")
        
        self.pdf_document.load(self.pdf_path)
        self.pdf_view.pdf_path = self.pdf_path
        self.pdf_view.setDocument(self.pdf_document)
        self.pdf_view.pdf_document = self.pdf_document
        self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)

    def ResetParams(self,text):
        self.lower()
        cv2.destroyAllWindows()

        if text != "SelectImage":
            self.ui.comboBox_SelectImage.blockSignals(True)
            self.ui.comboBox_SelectImage.setCurrentIndex(0)
            self.ui.comboBox_SelectImage.blockSignals(False)
            self.ui.comboBox_SelectImage_DeepLearningFoundation.blockSignals(True)
            self.ui.comboBox_SelectImage_DeepLearningFoundation.setCurrentIndex(0)
            self.ui.comboBox_SelectImage_DeepLearningFoundation.blockSignals(False)

        if text != "SelectVideo" and self.ImagesAndColorsHandler.image is not None:
           self.ui.comboBox_SelectVideo.blockSignals(True)
           self.ui.comboBox_SelectVideo.setCurrentIndex(0)
           self.ui.comboBox_SelectVideo.blockSignals(False)
           self.ui.comboBox_SelectVideo_DeepLearningFoundation.blockSignals(True)
           self.ui.comboBox_SelectVideo_DeepLearningFoundation.setCurrentIndex(0)
           self.ui.comboBox_SelectVideo_DeepLearningFoundation.blockSignals(False)

        self.ImagesAndColorsHandler.image = None
        self.ImagesAndColorsHandler.imageName = None
        self.ImagesAndColorsHandler.imageConversion = None
        self.ImagesAndColorsHandler.tempImage = None
        self.ImagesAndColorsHandler.tempImageName = None
        # if text == "ResetParams":
        #    self.ImagesAndColorsHandler.video = None
        if self.ImagesAndColorsHandler.videoCapturer is not None:
           self.ImagesAndColorsHandler.videoCapturer.release()
           #self.ImagesAndColorsHandler.videoCapturer = None

        self.ui.comboBox_ColorSpaceConversion.blockSignals(True)
        self.ui.comboBox_ColorSpaceConversion.setCurrentIndex(0)
        self.ui.comboBox_ColorSpaceConversion.blockSignals(False)
        self.ui.comboBox_ArithmeticAndBitwiseOperations.blockSignals(True)
        self.ui.comboBox_ArithmeticAndBitwiseOperations.setCurrentIndex(0)
        self.ui.comboBox_ArithmeticAndBitwiseOperations.blockSignals(False)
        self.ui.comboBox_Filters.blockSignals(True)
        self.ui.comboBox_Filters.setCurrentIndex(0)
        self.ui.comboBox_Filters.blockSignals(False)
        self.ui.comboBox_DilationErosionEdgeDetection.blockSignals(True)
        self.ui.comboBox_DilationErosionEdgeDetection.setCurrentIndex(0)
        self.ui.comboBox_DilationErosionEdgeDetection.blockSignals(False)
        self.ui.comboBox_DrawShape.blockSignals(True)
        self.ui.comboBox_DrawShape.setCurrentIndex(0)
        self.ui.comboBox_DrawShape.blockSignals(False)
        self.ui.comboBox_SegmentationAndContours.blockSignals(True)
        self.ui.comboBox_SegmentationAndContours.setCurrentIndex(0)
        self.ui.comboBox_SegmentationAndContours.blockSignals(False)
        self.ui.comboBox_ObjectDetection.blockSignals(True)
        self.ui.comboBox_ObjectDetection.setCurrentIndex(0)
        self.ui.comboBox_ObjectDetection.blockSignals(False)
        self.ui.comboBox_OCR.blockSignals(True)
        self.ui.comboBox_OCR.setCurrentIndex(0)
        self.ui.comboBox_OCR.blockSignals(False)

        self.ui.textEdit_AddText.clear()
        self.ui.label_ImageShapeValue.clear()
        self.ui.label_ImageHeightValue.clear()
        self.ui.label_ImageWidthValue.clear()
        self.ui.label_ImageDepthValue.clear()

        for counter, option in enumerate(self.ColorChannelChangeCheckBoxes):
                option.blockSignals(True)
                option.setChecked(False)
                option.setDisabled(True)
                option.setEnabled(False)
                option.blockSignals(False)

        self.ui.label_X1_Value.setText("10")
        self.ui.label_X2_Value.setText("100")
        self.ui.label_Y1_Value.setText("200")
        self.ui.label_Y2_Value.setText("50")
        self.ui.label_Z1_Value.setText("100")
        self.ui.label_Z2_Value.setText("250")
        self.ui.dial_X1.setValue(10)
        self.ui.dial_X2.setValue(100)
        self.ui.dial_Y1.setValue(200)
        self.ui.dial_Y2.setValue(50)
        self.ui.dial_Z1.setValue(100)
        self.ui.dial_Z2.setValue(250)

        self.ui.label_CropBottomRightCoefficientValue.setText("100 %")
        self.ui.label_CropTopLefCoefficientValue.setText("0 %")
        self.ui.horizontalSlider_CropBottomRightCoefficient.setValue(100)
        self.ui.horizontalSlider_CropTopLefCoefficient.setValue(0)

        self.ui.label_RotationDegreeValue.setText("0 degree")
        self.ui.label_RorationScaleValue.setText("0 times")
        self.ui.dial_RotationDegree.setValue(0)
        self.ui.dial_RotationScale.setValue(0)

        self.ui.comboBox_DrawShape.blockSignals(True)
        self.ui.comboBox_DrawShape.setCurrentIndex(0)
        self.ui.comboBox_DrawShape.blockSignals(False)

        self.ui.horizontalSlider_ResizeHeight.blockSignals(True)
        self.ui.horizontalSlider_ResizeWidth.blockSignals(True)
        self.ui.horizontalSlider_SkewHeight.blockSignals(True)
        self.ui.horizontalSlider_SkewWidth.blockSignals(True)
        self.ui.horizontalSlider_ResizeHeight.setValue(50)
        self.ui.horizontalSlider_ResizeWidth.setValue(50)
        self.ui.horizontalSlider_SkewHeight.setValue(50)
        self.ui.horizontalSlider_SkewWidth.setValue(50)
        self.ui.horizontalSlider_ResizeHeight.blockSignals(False)
        self.ui.horizontalSlider_ResizeWidth.blockSignals(False)
        self.ui.horizontalSlider_SkewHeight.blockSignals(False)
        self.ui.horizontalSlider_SkewWidth.blockSignals(False)

        self.ui.dial_X1.blockSignals(True)
        self.ui.dial_X2.blockSignals(True)
        self.ui.dial_Y1.blockSignals(True)
        self.ui.dial_Y2.blockSignals(True)
        self.ui.dial_Z1.blockSignals(True)
        self.ui.dial_Z2.blockSignals(True)
        self.ui.dial_X1.setValue(10)
        self.ui.dial_X2.setValue(100)
        self.ui.dial_Y1.setValue(200)
        self.ui.dial_Y2.setValue(50)
        self.ui.dial_Z1.setValue(100)
        self.ui.dial_Z2.setValue(250)
        self.ui.dial_X1.blockSignals(False)
        self.ui.dial_X2.blockSignals(False)
        self.ui.dial_Y1.blockSignals(False)
        self.ui.dial_Y2.blockSignals(False)
        self.ui.dial_Z1.blockSignals(False)
        self.ui.dial_Z2.blockSignals(False)

    def PrepareSelectImagesCGANs(self):
        sender = self.sender().objectName()
        self.ConditionalGANsHandler.GenerateAndDisplayImages(sender)

    def ConnectActions(self):
        self.ui.pushButton_ManWithGlassesToManWithoutGlasses_VAE.clicked.connect(partial(self.VAEHandler.Transition, self.ui.pushButton_ManWithGlassesToManWithoutGlasses_VAE.objectName()))
        self.ui.pushButton_WomanWithoutGlassesToManWithoutGlasses_VAE.clicked.connect(partial(self.VAEHandler.Transition, self.ui.pushButton_WomanWithoutGlassesToManWithoutGlasses_VAE.objectName()))
        self.ui.pushButton_WomanWithGlassesToManWithGlasses_VAE.clicked.connect(partial(self.VAEHandler.Transition, self.ui.pushButton_WomanWithGlassesToManWithGlasses_VAE.objectName()))
        self.ui.pushButton_WomanWithGlassesToWomanWithoutGlasses_VAE.clicked.connect(partial(self.VAEHandler.Transition, self.ui.pushButton_WomanWithGlassesToWomanWithoutGlasses_VAE.objectName()))
        self.ui.pushButton_DoubleTransitionCase5_VAE.clicked.connect(self.VAEHandler.DoubleTransitionCase5)
        self.ui.pushButton_DoubleTransitionCase4_VAE.clicked.connect(self.VAEHandler.DoubleTransitionCase4)
        self.ui.pushButton_DoubleTransitionCase3_VAE.clicked.connect(self.VAEHandler.DoubleTransitionCase3)
        self.ui.pushButton_DoubleTransitionCase2_VAE.clicked.connect(self.VAEHandler.DoubleTransitionCase2)
        self.ui.pushButton_DoubleTransitionCase1_VAE.clicked.connect(self.VAEHandler.DoubleTransitionCase1)
        self.ui.pushButton_DisplayImagesWithoutGlasses_VAE.clicked.connect(self.VAEHandler.DisplayImagesWithoutGlasses)
        self.ui.pushButton_DisplayImagesWithGlasses_VAE.clicked.connect(self.VAEHandler.DisplayImagesWithGlasses)
        self.ui.pushButton_TestModel_VAE.clicked.connect(self.VAEHandler.TestVAEModel)
        self.ui.pushButton_TrainModel_VAE.clicked.connect(self.VAEHandler.TrainVAEModel)
        self.ui.pushButton_CreateModel_VAE.clicked.connect(self.VAEHandler.CreateVAEModel)
        self.ui.pushButton_PrepareDataset_VAE.clicked.connect(self.VAEHandler.PrepareVAEDataset)
        self.ui.pushButton_ArrangeDataset_VAE.clicked.connect(self.ConditionalGANsHandler.ArrangeEyeGlassesDataset)
        self.ui.pushButton_DownloadDataset_VAE.clicked.connect(self.DownloadEyeGlassesDataset)
        self.ui.pushButton_TestModel_AE.clicked.connect(self.VAEHandler.TestAEModel)
        self.ui.pushButton_TrainModel_AE.clicked.connect(self.VAEHandler.TrainAEMode)
        self.ui.pushButton_CreateModel_AE.clicked.connect(self.VAEHandler.CreateAEModel)
        self.ui.pushButton_PrepareDataset_AE.clicked.connect(self.VAEHandler.PrepareAEDataset)
        self.ui.pushButton_TestDataset_AE.clicked.connect(self.VAEHandler.TestMINIST)
        self.ui.pushButton_DownloadDataset_AE.clicked.connect(self.PrepareDownloadMINIST)
        self.ui.pushButton_ImplementingLast_cGAN_for_EyeGlasses_by_CycleGAN_CycleGANs.clicked.connect(self.CycleGANsHandler.Implementing_Last_cGAN_for_EyeGlasses_by_CycleGAN)
        self.ui.pushButton_ShowImagesWithBlondHair_CycleGANs.clicked.connect(partial(self.CycleGANsHandler.GenerateAndDisplayImages,self.ui.pushButton_ShowImagesWithBlondHair_CycleGANs.objectName()))
        self.ui.pushButton_ShowImagesWithDarkHair_CycleGANs.clicked.connect(partial(self.CycleGANsHandler.GenerateAndDisplayImages,self.ui.pushButton_ShowImagesWithDarkHair_CycleGANs.objectName()))
        self.ui.pushButton_TrainModels_CycleGANs.clicked.connect(self.CycleGANsHandler.TrainModel)
        self.ui.pushButton_Create2GeneratorModels_CycleGANs.clicked.connect(self.CycleGANsHandler.CreateGenerators)
        self.ui.pushButton_Create2DiscriminatorModels_CycleGANs.clicked.connect(self.CycleGANsHandler.CreateDiscriminators)
        self.ui.pushButton_PrepareDataset_CycleGANs.clicked.connect(self.CycleGANsHandler.PrepareDataset)
        self.ui.pushButton_DisplayImagesWithBlondHair_CycleGANs.clicked.connect(self.PrepareShowCelebFacesImages)
        self.ui.pushButton_DisplayImagesWithDarkHair_CycleGANs.clicked.connect(self.PrepareShowCelebFacesImages)
        self.ui.pushButton_ArrangeDataset_CycleGANs.clicked.connect(self.CycleGANsHandler.ArrangeCelebFacesDataset)
        self.ui.pushButton_DownloadDataset_CycleGANs.clicked.connect(self.DownloadCelebFacesDataset)
        self.ui.pushButton_TransitionMaleToFemalesWithEyeGlassesToWithoutEyeGlasses2_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)
        self.ui.pushButton_TransitionMaleToFemalesWithEyeGlassesToWithoutEyeGlasses_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)
        self.ui.pushButton_TransitionMaleToFemalesWithEyeGlasses_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)
        self.ui.pushButton_TransitionMaleToFemalesWithoutEyeGlasses_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)
        self.ui.pushButton_TransitionMalesWithEyeGlassesToWithoutEyeGlasses_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)
        self.ui.pushButton_TransitionFemalesWithEyeGlassesToWithoutEyeGlasses_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)

        self.ui.pushButton_SelectImagesWithoutEyeGlasses_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)
        self.ui.pushButton_SelectImagesWithEyeGlasses_ConditionalGANs.clicked.connect(self.PrepareSelectImagesCGANs)
        self.ui.pushButton_TrainModels_ConditionalGANs.clicked.connect(self.ConditionalGANsHandler.TrainModel)
        self.ui.pushButton_CreateModels_ConditionalGANs.clicked.connect(self.ConditionalGANsHandler.CreateModels_InitializeWeights)
        self.ui.pushButton_PrepareDataset_ConditionalGANs.clicked.connect(self.ConditionalGANsHandler.PrepareDataset)
        self.ui.pushButton_AddLabels_ConditionalGANs.clicked.connect(self.ConditionalGANsHandler.AddLabels)
        self.ui.pushButton_DisplayImagesWithoutGlasses_ConditionalGANs.clicked.connect(self.PrepareShowEyeGlassesImages)
        self.ui.pushButton_DisplayImagesWithGlasses_ConditionalGANs.clicked.connect(self.PrepareShowEyeGlassesImages)
        self.ui.pushButton_ArrangeDataset_ConditionalGANs.clicked.connect(self.ConditionalGANsHandler.ArrangeEyeGlassesDataset)
        self.ui.pushButton_DownloadDataset_ConditionalGANs.clicked.connect(self.DownloadEyeGlassesDataset)
        self.ui.pushButton_TestModel_ColoredImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TestModel_ColoredImages)
        self.ui.pushButton_SaveModel_ColoredImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.SaveModel_ColoredImages)
        self.ui.pushButton_TrainModels_ColoredImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TrainModels_ColoredImages)
        self.ui.pushButton_CreateModels_ColoredImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.CreateModels_ColoredImages)
        self.ui.pushButton_PlotDataset_ColoredImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.PlotDataset_ColoredImages)
        self.ui.pushButton_PrepareDataset_ColoredImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.PrepareDataset_ColoredImages)
        self.ui.pushButton_DownloadDataset_ColoredImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.DownloadDataset_ColoredImages)
        self.ui.pushButton_TestModel_GrayImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TestModel_GrayImages)
        self.ui.pushButton_SaveModel_GrayImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.SaveModel_GrayImages)
        self.ui.pushButton_TrainModels_GrayImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TrainModels_GrayImages)
        self.ui.pushButton_CreateModels_GrayImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.CreateModels_GrayImages)
        self.ui.pushButton_PlotDataset_GrayImages_SimpleGANs.clicked.connect(self.SimpleGANsHandler.PlotMINIST)
        self.ui.pushButton_PrepareDataset_GrayImages_SimpleGANs.clicked.connect(self.PrepareDataset_GrayImages_SimpleGANs)
        self.ui.pushButton_DownloadDataset_GrayImages_SimpleGANs.clicked.connect(self.PrepareDownloadFashionMINIST)
        self.ui.pushButton_TestModel_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TestModel_Shape)
        self.ui.pushButton_TestModel_Numbers_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TestModel_Numbers)
        self.ui.pushButton_SaveModel_Numbers_SimpleGANs.clicked.connect(self.SimpleGANsHandler.SaveModel_Numbers)
        self.ui.pushButton_SaveModel_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.SaveModel_Shape)
        self.ui.pushButton_CreateModels_Numbers_SimpleGANs.clicked.connect(self.SimpleGANsHandler.CreateModels_Numbers)
        self.ui.pushButton_TrainModels_Numbers_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TrainModels_Numbers)
        self.ui.pushButton_CreateModels_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.CreateModels_Shape)
        self.ui.pushButton_TrainModels_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.TrainModels_Shape)
        self.ui.pushButton_DisplayDataset_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.DisplayDataset_Shape)
        self.ui.pushButton_PrepareDataset_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.PrepareDataset_Shape)
        self.ui.pushButton_ShowDataset_Numbers_Step1_SimpleGANs.clicked.connect(self.SimpleGANsHandler.ShowDataset_Numbers)
        self.ui.pushButton_PlotDataset_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.PlotDataset_Shape)
        self.ui.pushButton_CreateDataset_Numbers_Step1_SimpleGANs.clicked.connect(self.SimpleGANsHandler.CreateDataset_Numbers)
        self.ui.pushButton_CreateDataset_Shape_SimpleGANs.clicked.connect(self.SimpleGANsHandler.CreateDataset_Shape)
        self.ui.comboBox_Epochs_DLbyPyTorch.currentIndexChanged.connect(self.Epochs_DLbyPyTorch_Change)
        self.ui.pushButton_CreateModelStep3DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.CreateBiinaryClassificationModel)
        self.ui.pushButton_TrainModelStep3DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.TrainBiinaryClassificationModel)
        self.ui.pushButton_CalculateAccuracyStep3DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.CalculateBiinaryClassificationModelAccuracy)
        self.ui.pushButton_CreateModelStep5DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.CreateMultiCategoryClassificationModel)
        self.ui.pushButton_TrainModelStep5DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.TrainMultiCategoryClassificationModel)
        self.ui.pushButton_CalculateAccuracyStep5DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.CalculateMultiCategoryClassificationModelAccuracy)
        self.ui.pushButton_PrepareDataStep2DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.PrepareDataForBinaryClassification)
        self.ui.pushButton_PrepareDataStep4DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.PrepareDataForMultiCategoryClassification)
        self.ui.pushButton_PlotMNISTStep1DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.PlotMINIST)
        self.ui.pushButton_TestMNISTStep1DLbyPyTorch.clicked.connect(self.DLbyPyTorchHandler.TestMINIST)
        self.ui.pushButton_DownloadMNISTStep1DLbyPyTorch.clicked.connect(self.PrepareDownloadFashionMINIST)
        self.ui.comboBox_TransferStyle_NeuralStyleTransfer.currentTextChanged.connect(self.PrepareTransferStyle)
        self.ui.comboBox_SelectStyle_NeuralStyleTransfer.currentTextChanged.connect(self.NeuralStyleTransferHandler.SelectShowStyle)
        self.ui.comboBox_SelectImage_NeuralStyleTransfer.currentTextChanged.connect(self.SelectImage_NeuralStyleTransfer)
        self.ui.horizontalSlider_Sync_NeuralStyleTransfer.valueChanged.connect(self.SyncSize_NeuralStyleTransfer)
        self.ui.dial_Red_NeuralStyleTransfer.valueChanged.connect(self.dial_RedValue_Changed_NeuralStyleTransfer)
        self.ui.dial_Green_NeuralStyleTransfer.valueChanged.connect(self.dial_GreenValue_Changed_NeuralStyleTransfer)
        self.ui.dial_Blue_NeuralStyleTransfer.valueChanged.connect(self.dial_BlueValue_Changed_NeuralStyleTransfer)

        self.ui.action_BigPicture.triggered.connect(partial(self.changePDFPage,0))
        self.ui.action_UniversityCurriculum.triggered.connect(partial(self.changePDFPage,1))
        self.ui.action_RoadMap.triggered.connect(partial(self.changePDFPage,2))
        self.ui.action_StudyPlan.triggered.connect(partial(self.changePDFPage,3))
        self.ui.action_HeadingResearch.triggered.connect(partial(self.changePDFPage,4))
        self.ui.action_UserGuide.triggered.connect(partial(self.changePDFPage,5))
        self.action_MLBigPicture.triggered.connect(partial(self.changePDFPage,6))
        self.action_CategorizingByLearningParadigm.triggered.connect(partial(self.changePDFPage,7))
        self.action_FromFundamentalsToAdvanced.triggered.connect(partial(self.changePDFPage,8))
        self.action_CodeSamplesByLearningParadigm.triggered.connect(partial(self.changePDFPage,9))
        self.action_DeeperCodeSamplesWithDefinitions.triggered.connect(partial(self.changePDFPage,10))
        self.action_TheoreticalFoundationsOfComputerVision.triggered.connect(partial(self.changePDFPage,11))
        self.action_Numpy.triggered.connect(partial(self.changePDFPage,12))
        self.action_Pandas.triggered.connect(partial(self.changePDFPage,13))
        self.action_MatPlotLib.triggered.connect(partial(self.changePDFPage,14))
        self.action_SeaBorn.triggered.connect(partial(self.changePDFPage,15))
        self.action_SupervisedMLProcess.triggered.connect(partial(self.changePDFPage,16))
        self.action_TheoreticalDeepLearningFoundation.triggered.connect(partial(self.changePDFPage,17))
        self.action_TheoreticalGANsSource1.triggered.connect(partial(self.changePDFPage,18))
        self.action_TheoreticalGANsSource2.triggered.connect(partial(self.changePDFPage,19))
        self.action_TheoreticalGANsSource3.triggered.connect(partial(self.changePDFPage,20))
        self.action_TheoreticalGANsSource4.triggered.connect(partial(self.changePDFPage,21))
        self.action_TheoreticalGANsMainSource.triggered.connect(partial(self.changePDFPage,22))
        self.ui.action_Core_CV_Computer_Vision_Tasks.triggered.connect(partial(self.changePDFPage,23))

        self.ui.action_AboutTool.triggered.connect(self.changePage)
        self.ui.action_AboutAuthorDeveloper.triggered.connect(self.changePage)
        self.action_PythonProgramming.triggered.connect(self.changePage)
        self.action_LinearAlgebraAndCalculus.triggered.connect(self.changePage)
        self.action_ProbabilityAndStatistics.triggered.connect(self.changePage)
        self.action_ImagesAndColorsManipulationsAndOprations.triggered.connect(self.changePage)
        self.action_CreateSimpleCNNConvolutionalNeuralNetwork.triggered.connect(self.changePage)
        self.action_DeepLearningFoundationOperations.triggered.connect(self.changePage)
        self.action_CreateHandGestureRecognItionCNN.triggered.connect(self.changePage)
        self.action_FaceRecognitionOperation.triggered.connect(self.changePage)
        self.action_TransferLearning.triggered.connect(self.changePage)
        self.action_NeuralStyleTransfer.triggered.connect(self.changePage)
        self.action_DLbyPyTorchBinaryAndMultiCategoryClassifications.triggered.connect(self.changePage)
        self.action_SimpleGANs.triggered.connect(self.changePage)
        self.action_ConditionalGANs.triggered.connect(self.changePage)
        self.action_CycleGANs.triggered.connect(self.changePage)
        self.action_VariationalAutoEncoders.triggered.connect(self.changePage)

        self.ui.action_CloseOtherWindows.triggered.connect(self.closeWindow)
        self.ui.action_CloseMainWindow.triggered.connect(self.closeWindow)
        self.ui.action_CloseAllWindows.triggered.connect(self.closeWindow)

        self.ui.action_CreateDefaultFolders.triggered.connect(self.CheckCreateDefaultFolders)

        self.ui.action_UploadImages.triggered.connect(self.Upload_Files)
        self.ui.action_UploadVideos.triggered.connect(self.Upload_Files)
        self.ui.action_UploadModels.triggered.connect(self.Upload_Files)
        self.ui.pushButton_UploadImages.clicked.connect(self.Upload_Files)
        self.ui.pushButton_UploadVideos.clicked.connect(self.Upload_Files)
        self.ui.pushButton_UploadImages_DeepLearningFoundation.clicked.connect(self.Upload_Files)
        self.ui.pushButton_UploadVideos_DeepLearningFoundation.clicked.connect(self.Upload_Files)  
        self.ui.pushButton_UploadImages_NeuralStyleTransfer.clicked.connect(self.Upload_Files)
        self.ui.pushButton_UploadStyles_NeuralStyleTransfer.clicked.connect(self.Upload_Files)
        self.ui.action_UploadStyleTransferModels.triggered.connect(self.Upload_Files)

        self.ui.pushButton_SaveCode_TransferLearning.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_TransferLearning))
        self.ui.pushButton_SaveCode.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_ImageAndColors))
        self.ui.pushButton_SaveCode_CreateSimpleCNN.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_CreateSimpleCNN))
        self.ui.pushButton_SaveCode_DeepLearningFoundation.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_DeepLearningFoundation))
        self.ui.pushButton_SaveCode_CreateSimpleCNN2.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_CreateSimpleCNN2))
        self.ui.pushButton_SaveCode__FaceRecognitionOperation.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_FaceRecognitionOperation))
        self.ui.pushButton_SaveCode_NeuralStyleTransfer.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_NeuralStyleTransfer))
        self.ui.pushButton_SaveCode_DLbyPyTorch.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_DLbyPyTorch))
        self.ui.pushButton_SaveCode_SimpleGANs.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_SimpleGANs))
        self.ui.pushButton_SaveCode_ConditionalGANs.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_ConditionalGANs))
        self.ui.pushButton_SaveCode_CycleGANs.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_CycleGANs))
        self.ui.pushButton_SaveCode_VAE.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_VAE))

        self.ui.comboBox_ColorSpaceConversion.currentTextChanged.connect(self.PrepareConvertColorSpace)
        self.ui.pushButton_SaveImage.clicked.connect(self.ImagesAndColorsHandler.SaveImage)
        self.ui.pushButton_SaveImage_NeuralStyleTransfer.clicked.connect(self.NeuralStyleTransferHandler.SaveImage)
        self.ui.horizontalSlider_SkewHeight.valueChanged.connect(self.PrepareSkewImage)
        self.ui.horizontalSlider_SkewWidth.valueChanged.connect(self.PrepareSkewImage)
        self.ui.horizontalSlider_ResizeHeight.valueChanged.connect(self.PrepareResizeImage)
        self.ui.horizontalSlider_ResizeWidth.valueChanged.connect(self.PrepareResizeImage)
        self.ui.pushButton_LargerPyrUp.clicked.connect(self.PreparePyrUpDown)
        self.ui.pushButton_SmallerPyrDown.clicked.connect(self.PreparePyrUpDown)
        self.ui.dial_RotationScale.valueChanged.connect(self.PrepareScaleByCoefficient)
        self.ui.dial_RotationDegree.valueChanged.connect(self.PrepareRotationByAngle)
        self.ui.checkBox_FlipHorizantal.checkStateChanged.connect(self.PrepareFlip)
        self.ui.checkBox_FlipVertical.checkStateChanged.connect(self.PrepareFlip)
        self.ui.checkBox_SwapTranspose.checkStateChanged.connect(self.PrepareTranspose)
        self.ui.horizontalSlider_CropTopLefCoefficient.valueChanged.connect(self.PrepareCrop)
        self.ui.horizontalSlider_CropBottomRightCoefficient.valueChanged.connect(self.PrepareCrop)
        self.ui.pushButton_AddText.clicked.connect(self.PrepareAddText)
        self.ui.comboBox_DrawShape.currentTextChanged.connect(self.PrepareDrawShape)
        self.ui.comboBox_ArithmeticAndBitwiseOperations.currentTextChanged.connect(self.PrepareOperations)
        self.ui.comboBox_Filters.currentTextChanged.connect(self.PrepareFilters)
        self.ui.comboBox_DilationErosionEdgeDetection.currentTextChanged.connect(self.PrepareDilationErosionEdgeDetection)
        self.ui.comboBox_SegmentationAndContours.currentTextChanged.connect(self.PrepareSegmentationAndContours)
        self.ui.comboBox_ObjectDetection.currentTextChanged.connect(self.PrepareObjectDetection)
        self.ui.comboBox_OCR.currentTextChanged.connect(self.PrepareOpticalCharacterRecognition)
        self.ui.pushButton_LoadMNIST_Step1CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.LoadMNIST)
        self.ui.pushButton_TestMNIST_Step1CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.TestMNIST)
        self.ui.pushButton_PlotMNIST_Step1CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.PlotMNIST)
        self.ui.pushButton_PrepareData_Step2CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.PrepareData)
        self.ui.pushButton_EncodeMap_Step2CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.EncodeMap)
        self.ui.pushButton_ModelMap_Step3CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.ModelMap)
        self.ui.pushButton_CreateModel_Step3CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.CreateModel)
        self.ui.pushButton_ModelSummary_Step3CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.ModelSummaryFunction)
        self.ui.pushButton_TrainModel_Step4CreateSimpleCNN.clicked.connect(self.PrepareTrainModel)
        self.ui.pushButton_CancelTraining_Step4CreateSimpleCNN.clicked.connect(self.PrepareCancelTraining)
        self.ui.pushButton_PlotAccuracy_Step4CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.PlotAccuracy)
        self.ui.pushButton_PlotLoss_Step4CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.PlotLoss)
        self.ui.pushButton_SaveTrainedModel_Step4CreateSimpleCNN.clicked.connect(self.CreateSimpleCNNHandler.SaveTrainedModel)

        self.action_ProbabilityAndStatistics.triggered.connect(partial(self.Pdf_In_Browser,"https://mml-book.github.io/book/mml-book.pdf",False))
        self.action_PythonProgramming.triggered.connect(partial(self.Pdf_In_Browser,"https://www.w3schools.com/python/default.asp",False))
        self.action_LinearAlgebraAndCalculus.triggered.connect(partial(self.Pdf_In_Browser,"https://github.com/Ryota-Kawamura/Mathematics-for-Machine-Learning-and-Data-Science-Specialization",False))
        self.action_MLModelOverview.triggered.connect(partial(self.Pdf_In_Browser,"https://apple.github.io/coremltools/docs-guides/source/mlmodel.html",False))
        self.action_CoreMLModelFormatSpecification.triggered.connect(partial(self.Pdf_In_Browser,"https://apple.github.io/coremltools/mlmodel/index.html",False))       
      
        self.ImagesAndColorsHandler.valueChanged.connect(self.SetImageInfo)
        self.ImagesAndColorsHandler.ImageSizeChanged.connect(self.ImageSizeChanged)
        for channel in self.ColorChannelChangeCheckBoxes:
            channel.clicked.connect(partial(self.PrepareColorChannelRemove,channel.objectName()))#stateChanged
        for translation in self.TranslationDials:
             translation.valueChanged.connect(self.PrepareTranslateImage)       
        self.ImagesAndColorsHandler.ResetParams.connect(self.ResetParams) 
        self.CreateSimpleCNNHandler.LoadMNISTRawDataOrPreparedData.connect(self.LoadMNISTRawDataOrPreparedData)
        
        self.ui.comboBox_SelectImage.currentTextChanged.connect(partial(self.PrepareSelectImage,self.ui.comboBox_SelectImage ))
        self.ui.comboBox_SelectVideo.currentTextChanged.connect(partial(self.PrepareSelectVideo,self.ui.comboBox_SelectVideo))
        self.ui.comboBox_SelectImage_DeepLearningFoundation.currentTextChanged.connect(partial(self.PrepareSelectImage,self.ui.comboBox_SelectImage_DeepLearningFoundation ))
        self.ui.comboBox_SelectCameraDeepLearningFoundation.currentTextChanged.connect(partial(self.PrepareSelectCamera,self.ui.comboBox_SelectCameraDeepLearningFoundation))
        self.ui.comboBox_SelectVideo_DeepLearningFoundation.currentTextChanged.connect(partial(self.PrepareSelectVideo,self.ui.comboBox_SelectVideo_DeepLearningFoundation))
        self.ui.comboBox_SelectOperationDeepLearningFoundation.currentTextChanged.connect(partial(self.PrepareSelectDeepLearningOperations,self.ui.comboBox_SelectOperationDeepLearningFoundation))
        self.ui.comboBox_SelectCameraStep1CreateSimpleCNN2.currentTextChanged.connect(partial(self.PrepareSelectCamera,self.ui.comboBox_SelectCameraStep1CreateSimpleCNN2))
        self.ui.comboBox_SelectCamera_Step4_FaceRecognitionOperation.currentTextChanged.connect(partial(self.PrepareSelectCamera,self.ui.comboBox_SelectCamera_Step4_FaceRecognitionOperation))
        self.ui.comboBox_SelectVideo_Step5_FaceRecognitionOperation.currentTextChanged.connect(partial(self.PrepareSelectVideo,self.ui.comboBox_SelectVideo_Step5_FaceRecognitionOperation))

        self.ui.pushButton_RecordTrainStep3CreateSimpleCNN2.clicked.connect(self.PrepareRecordHandGesture)
        self.ui.pushButton_TestTrainedModel_Step8CreateSimpleCNN2.clicked.connect(self.PrepareTestHandGestureModel)
        self.ui.pushButton_RecordTestStep4CreateSimpleCNN2.clicked.connect(self.PrepareRecordHandGesture)
        self.ui.pushButton_EnhanceDatasetStep5CreateSimpleCNN2.clicked.connect(self.PrepareEnhanceDataset)
        self.ui.pushButton_CreateModel_Step6CreateSimpleCNN2.clicked.connect(self.CreateHandGestureRecognitionCNNHandler.CreateModel)
        self.ui.pushButton_ModelSummary_Step6CreateSimpleCNN2.clicked.connect(self.CreateHandGestureRecognitionCNNHandler.ModelSummaryFunction)
        self.ui.pushButton_TrainModel_Step7CreateSimpleCNN2.clicked.connect(self.PrepareTrainHandGestureModel)
        self.ui.pushButton_SaveTrainedModel_Step7CreateSimpleCNN2.clicked.connect(self.CreateHandGestureRecognitionCNNHandler.SaveTrainedModel)
        self.ui.pushButton_CancelTraining_Step7CreateSimpleCNN2.clicked.connect(self.CreateHandGestureRecognitionCNNHandler.CancelTraining)

        self.ui.pushButton_UploadImage_Step1_FaceRecognitionOperation.clicked.connect(self.PrepareUploadFaceImage)
        self.ui.pushButton_CreateUploadLoadModel_Step2_FaceRecognitionOperation.clicked.connect(self.FaceRecognitionOperationHandler.CheckVGGFaceModel)
        self.ui.pushButton_VerifaySimilarity_Step3_FaceRecognitionOperation.clicked.connect(self.PrepareCompareVerifySimilarity)
        self.ui.pushButton_FaceRecognitionCamera_Step4_FaceRecognitionOperation.clicked.connect(self.PrepareFaceRecognitionOnCamera)
        self.ui.pushButton_FaceRecognitionVideo_Step5_FaceRecognitionOperation.clicked.connect(self.PrepareFaceRecognitionOnVideo)

        self.ui.pushButton_LoadCIFER10_Step1TransferLearning.clicked.connect(self.TransferLearningHandler.Import_Load_Prepare_Cifar10)
        self.ui.comboBox_SelectModel_Step3TransferLearning.currentTextChanged.connect(self.PrepareSelectModel_TransferLearning)
        self.ui.pushButton_EnhancingDataset_Step2TransferLearning.clicked.connect(self.TransferLearningHandler.Enhance_Dataset)
        self.ui.pushButton_ShowSummary_Step3TransferLearning.clicked.connect(self.TransferLearningHandler.ShowModelSummary)
        self.ui.pushButton_CancelTraining_Step4TransferLearning.clicked.connect(self.TransferLearningHandler.CancelTraining)
        self.ui.pushButton_TrainingModel_Step4TransferLearning.clicked.connect(self.PrepareTrainModelTransferLearning)
        self.ui.pushButton_SaveTrainedModel_Step4TransferLearning.clicked.connect(self.TransferLearningHandler.SaveTrainedModel)
        self.ui.pushButton_EvaluateModel_Step5TransferLearning.clicked.connect(self.TransferLearningHandler.EvaluateModel)
        self.ui.pushButton_EvaluateModel_Step5TransferLearning.clicked.connect(self.TransferLearningHandler.EvaluateModel)
        self.ui.pushButton_TestingModel_Step6TransferLearning.clicked.connect(self.TransferLearningHandler.TestingModel)

    def ManualSetup(self):
        self.ImagesAndColorsHandler = ImagesAndColorsManipulationsAndOprations()
        self.CreateSimpleCNNHandler = CreateSimpleCNN()
        self.DLOperationsHandler = DeepLearningFoundationOperations(self.ImagesAndColorsHandler, self.CreateSimpleCNNHandler)
        self.CreateHandGestureRecognitionCNNHandler = CreateHandGestureRecognitionCNN(self.ImagesAndColorsHandler, self.CreateSimpleCNNHandler)
        self.FaceRecognitionOperationHandler = FaceRecognitionOperation(self.ImagesAndColorsHandler,self.DLOperationsHandler)
        self.TransferLearningHandler = TransferLearning(self.DLOperationsHandler,self.CreateSimpleCNNHandler)
        self.NeuralStyleTransferHandler = NeuralStyleTransfer()
        self.DLbyPyTorchHandler = DLbyPyTorch()
        self.SimpleGANsHandler = SimpleGANs()
        self.ConditionalGANsHandler = ConditionalGANs()
        self.CycleGANsHandler = CycleGANs()
        self.VAEHandler = VariationalAutoEncoders()
        
        self.ColorChannelChangeCheckBoxes = [
            self.ui.checkBox_BlueChannel,
            self.ui.checkBox_GreenChannel,
            self.ui.checkBox_RedChannel,
            self.ui.checkBox_HSVHueChannel,
            self.ui.checkBox_HSVSaturation,
            self.ui.checkBox_HSVValue
        ]
        for channel in self.ColorChannelChangeCheckBoxes:
                channel.setDisabled(True)
                channel.setEnabled(False)
        self.TranslationDials = [
             self.ui.dial_X1,
             self.ui.dial_Y1,
             self.ui.dial_Z1,
             self.ui.dial_X2,
             self.ui.dial_Y2,
             self.ui.dial_Z2
        ]
        self.ui.dial_X1.setValue(10)
        self.ui.dial_X2.setValue(100)
        self.ui.dial_Y1.setValue(200)
        self.ui.dial_Y2.setValue(50)
        self.ui.dial_Z1.setValue(100)
        self.ui.dial_Z2.setValue(250)
        self.TranslationLabelValues = [
             self.ui.label_X1_Value,
             self.ui.label_X2_Value,
             self.ui.label_Y1_Value,
             self.ui.label_Y2_Value,
             self.ui.label_Z1_Value,
             self.ui.label_Z2_Value
        ]
        for TranslationLabelValue in self.TranslationLabelValues:
             TranslationLabelValue.setStyleSheet("color:red")
        self.ui.label_CropTopLefCoefficientValue.setStyleSheet("color:red")
        self.ui.label_CropBottomRightCoefficientValue.setStyleSheet("color:red")
        self.ui.label_RorationScaleValue.setStyleSheet("color:red")
        self.ui.label_RotationDegreeValue.setStyleSheet("color:red")
        self.menu_PythonProgramming = QMenu(parent=self)
        self.menu_PythonProgramming.setObjectName("action_PythonProgramming")
        self.ui.menu_PreRequisites.addMenu(self.menu_PythonProgramming)
        self.action_PythonProgramming = QtGui.QAction(parent=self)
        self.action_PythonProgramming.setObjectName("action_PythonProgramming")
        self.menu_PythonProgramming.addAction(self.action_PythonProgramming)
        self.action_Numpy = QtGui.QAction(parent=self)
        self.action_Numpy.setObjectName("action_Numpy")
        self.menu_PythonProgramming.addAction(self.action_Numpy)
        self.action_Pandas = QtGui.QAction(parent=self)
        self.action_Pandas.setObjectName("action_Pandas")
        self.menu_PythonProgramming.addAction(self.action_Pandas)
        self.action_MatPlotLib = QtGui.QAction(parent=self)
        self.action_MatPlotLib.setObjectName("action_MatPlotLib")
        self.menu_PythonProgramming.addAction(self.action_MatPlotLib)
        self.action_SeaBorn = QtGui.QAction(parent=self)
        self.action_SeaBorn.setObjectName("action_SeaBorn")
        self.menu_PythonProgramming.addAction(self.action_SeaBorn)
        self.menu_Mathematics = QMenu(parent=self)
        self.menu_Mathematics.setObjectName("menu_Mathematics")
        self.ui.menu_PreRequisites.addMenu(self.menu_Mathematics)
        self.action_LinearAlgebraAndCalculus = QtGui.QAction(parent=self)
        self.action_LinearAlgebraAndCalculus.setObjectName("action_LinearAlgebraAndCalculus")
        self.menu_Mathematics.addAction(self.action_LinearAlgebraAndCalculus)
        self.action_ProbabilityAndStatistics = QtGui.QAction(parent=self)
        self.action_ProbabilityAndStatistics.setObjectName("action_ProbabilityAndStatistics")
        self.menu_Mathematics.addAction(self.action_ProbabilityAndStatistics)
        self.menu_CoreMachineLearningPrinciples = QMenu(parent=self)
        self.menu_CoreMachineLearningPrinciples.setObjectName("menu_CoreMachineLearningPrinciples")
        self.ui.menu_PreRequisites.addMenu(self.menu_CoreMachineLearningPrinciples)
        self.action_MLBigPicture = QtGui.QAction(parent=self)
        self.action_MLBigPicture.setObjectName("action_MLBigPicture")
        self.menu_CoreMachineLearningPrinciples.addAction(self.action_MLBigPicture)
        self.action_CategorizingByLearningParadigm = QtGui.QAction(parent=self)
        self.action_CategorizingByLearningParadigm.setObjectName("action_CategorizingByLearningParadigm")
        self.menu_CoreMachineLearningPrinciples.addAction(self.action_CategorizingByLearningParadigm)
        self.action_FromFundamentalsToAdvanced = QtGui.QAction(parent=self)
        self.action_FromFundamentalsToAdvanced.setObjectName("action_FromFundamentalsToAdvanced")
        self.menu_CoreMachineLearningPrinciples.addAction(self.action_FromFundamentalsToAdvanced)
        self.action_MLModelOverview = QtGui.QAction(parent=self)
        self.action_MLModelOverview.setObjectName("action_MLModelOverview")
        self.ui.menu_Machine_Learning_Model_Fundamentals.addAction(self.action_MLModelOverview)
        self.action_CoreMLModelFormatSpecification = QtGui.QAction(parent=self)
        self.action_CoreMLModelFormatSpecification.setObjectName("action_CoreMLModelFormatSpecification")
        self.ui.menu_Machine_Learning_Model_Fundamentals.addAction(self.action_CoreMLModelFormatSpecification)
        self.action_SupervisedMLProcess = QtGui.QAction(parent=self)
        self.action_SupervisedMLProcess.setObjectName("action_SupervisedMLProcess")
        self.ui.menu_Machine_Learning_Model_Fundamentals.addAction(self.action_SupervisedMLProcess)
        self.action_CodeSamplesByLearningParadigm = QtGui.QAction(parent=self)
        self.action_CodeSamplesByLearningParadigm.setObjectName("action_CodeSamplesByLearningParadigm")
        self.ui.menu_Machine_Learning_Model_Fundamentals.addAction(self.action_CodeSamplesByLearningParadigm)
        self.action_DeeperCodeSamplesWithDefinitions = QtGui.QAction(parent=self)
        self.action_DeeperCodeSamplesWithDefinitions.setObjectName("action_DeeperCodeSamplesWithDefinitions")
        self.ui.menu_Machine_Learning_Model_Fundamentals.addAction(self.action_DeeperCodeSamplesWithDefinitions)
        self.action_TheoreticalFoundationsOfComputerVision = QtGui.QAction(parent=self)
        self.action_TheoreticalFoundationsOfComputerVision.setObjectName("action_TheoreticalFoundationsOfComputerVision")
        self.ui.menu_FundamentalOfComputerVision.addAction(self.action_TheoreticalFoundationsOfComputerVision)
        self.menu_PracticalFoundationsOfComputerVision = QMenu(parent=self)
        self.menu_PracticalFoundationsOfComputerVision.setObjectName("menu_PracticalFoundationsOfComputerVision")
        self.ui.menu_FundamentalOfComputerVision.addMenu(self.menu_PracticalFoundationsOfComputerVision)
        self.action_ImagesAndColorsManipulationsAndOprations = QtGui.QAction(parent=self)
        self.action_ImagesAndColorsManipulationsAndOprations.setObjectName("action_ImagesAndColorsManipulationsAndOprations")
        self.menu_PracticalFoundationsOfComputerVision.addAction(self.action_ImagesAndColorsManipulationsAndOprations)
        self.action_CreateSimpleCNNConvolutionalNeuralNetwork = QtGui.QAction(parent=self)
        self.action_CreateSimpleCNNConvolutionalNeuralNetwork.setObjectName("action_CreateSimpleCNNConvolutionalNeuralNetwork")
        self.menu_PracticalFoundationsOfComputerVision.addAction(self.action_CreateSimpleCNNConvolutionalNeuralNetwork)
        self.action_TheoreticalDeepLearningFoundation = QtGui.QAction(parent=self)
        self.action_TheoreticalDeepLearningFoundation.setObjectName("action_TheoreticalDeepLearningFoundation")
        self.ui.menu_DeepLearningFoundations.addAction(self.action_TheoreticalDeepLearningFoundation)
        self.menu_PracticalDeepLearningFoundations = QMenu(parent=self)
        self.menu_PracticalDeepLearningFoundations.setObjectName("menu_PracticalDeepLearningFoundations")
        self.ui.menu_DeepLearningFoundations.addMenu(self.menu_PracticalDeepLearningFoundations)
        self.action_DeepLearningFoundationOperations = QtGui.QAction(parent=self)
        self.action_DeepLearningFoundationOperations.setObjectName("action_DeepLearningFoundationOperations")
        self.menu_PracticalDeepLearningFoundations.addAction(self.action_DeepLearningFoundationOperations)
        self.action_CreateHandGestureRecognItionCNN = QtGui.QAction(parent=self)
        self.action_CreateHandGestureRecognItionCNN.setObjectName("action_CreateHandGestureRecognItionCNN")
        self.menu_PracticalDeepLearningFoundations.addAction(self.action_CreateHandGestureRecognItionCNN)
        self.action_FaceRecognitionOperation = QtGui.QAction(parent=self)
        self.action_FaceRecognitionOperation.setObjectName("action_FaceRecognitionOperation")
        self.menu_PracticalDeepLearningFoundations.addAction(self.action_FaceRecognitionOperation)
        self.action_TransferLearning = QtGui.QAction(parent=self)
        self.action_TransferLearning.setObjectName("action_TransferLearning")
        self.menu_PracticalDeepLearningFoundations.addAction(self.action_TransferLearning)
        self.action_NeuralStyleTransfer = QtGui.QAction(parent=self)
        self.action_NeuralStyleTransfer.setObjectName("action_NeuralStyleTransfer")
        self.menu_PracticalDeepLearningFoundations.addAction(self.action_NeuralStyleTransfer)
        self.action_DLbyPyTorchBinaryAndMultiCategoryClassifications = QtGui.QAction(parent=self)
        self.action_DLbyPyTorchBinaryAndMultiCategoryClassifications.setObjectName("action_DLbyPyTorchBinaryAndMultiCategoryClassifications")
        self.menu_PracticalDeepLearningFoundations.addAction(self.action_DLbyPyTorchBinaryAndMultiCategoryClassifications)


        self.menu_TheoreticalGANs = QMenu(parent=self)
        self.menu_TheoreticalGANs.setObjectName("menu_TheoreticalGANs")
        self.ui.menu_Advanced_Generative_Models_Architectures.addMenu(self.menu_TheoreticalGANs)
        self.menu_PracticalGANs = QMenu(parent=self)
        self.menu_PracticalGANs.setObjectName("menu_PracticalGANs")
        self.ui.menu_Advanced_Generative_Models_Architectures.addMenu(self.menu_PracticalGANs)
        self.action_TheoreticalGANsMainSource = QtGui.QAction(parent=self)
        self.action_TheoreticalGANsMainSource.setObjectName("action_TheoreticalGANsMainSource")
        self.menu_TheoreticalGANs.addAction(self.action_TheoreticalGANsMainSource)
        self.action_TheoreticalGANsSource1 = QtGui.QAction(parent=self)
        self.action_TheoreticalGANsSource1.setObjectName("action_TheoreticalGANsSource1")
        self.menu_TheoreticalGANs.addAction(self.action_TheoreticalGANsSource1)
        self.action_TheoreticalGANsSource2 = QtGui.QAction(parent=self)
        self.action_TheoreticalGANsSource2.setObjectName("action_TheoreticalGANsSource2")
        self.menu_TheoreticalGANs.addAction(self.action_TheoreticalGANsSource2)
        self.action_TheoreticalGANsSource3 = QtGui.QAction(parent=self)
        self.action_TheoreticalGANsSource3.setObjectName("action_TheoreticalGANsSource3")
        self.menu_TheoreticalGANs.addAction(self.action_TheoreticalGANsSource3)
        self.action_TheoreticalGANsSource4 = QtGui.QAction(parent=self)
        self.action_TheoreticalGANsSource4.setObjectName("action_TheoreticalGANsSource4")
        self.menu_TheoreticalGANs.addAction(self.action_TheoreticalGANsSource4)

        self.action_SimpleGANs = QtGui.QAction(parent=self)
        self.action_SimpleGANs.setObjectName("action_SimpleGANs")
        self.menu_PracticalGANs.addAction(self.action_SimpleGANs)
        self.action_ConditionalGANs = QtGui.QAction(parent=self)
        self.action_ConditionalGANs.setObjectName("action_ConditionalGANs")
        self.menu_PracticalGANs.addAction(self.action_ConditionalGANs)
        self.action_CycleGANs = QtGui.QAction(parent=self)
        self.action_CycleGANs.setObjectName("action_CycleGANs")
        self.menu_PracticalGANs.addAction(self.action_CycleGANs)
        self.action_VariationalAutoEncoders = QtGui.QAction(parent=self)
        self.action_VariationalAutoEncoders.setObjectName("action_VariationalAutoEncoders")
        self.menu_PracticalGANs.addAction(self.action_VariationalAutoEncoders)


        self.menu_TheoreticalGANsDeploymentOptimization = QMenu(parent=self)
        self.menu_TheoreticalGANsDeploymentOptimization.setObjectName("menu_TheoreticalGANsDeploymentOptimization")
        self.ui.menu_Applications_Deployment_Optimization.addMenu(self.menu_TheoreticalGANsDeploymentOptimization)
        self.menu_PracticalGANsDeploymentOptimization = QMenu(parent=self)
        self.menu_PracticalGANsDeploymentOptimization.setObjectName("menu_PracticalGANsDeploymentOptimization")
        self.ui.menu_Applications_Deployment_Optimization.addMenu(self.menu_PracticalGANsDeploymentOptimization)

        self.pdf_view = CustomPdfView(self.ui.pages)
        self.pdf_document = QPdfDocument(self.pdf_view)
        self.ui.pages.addWidget(self.pdf_view)       
        self.framelayout = QVBoxLayout(self.ui.frame_DeepLearningFoundation)
        self.frame_pdf_view = CustomPdfView(self.ui.frame_DeepLearningFoundation)
        self.frame_pdf_document = QPdfDocument(self.frame_pdf_view)
        self.framelayout.addWidget(self.frame_pdf_view)
        AboutAuthorDeveloper = self.Load_Html_File(os.path.relpath("pages/Text_AboutAuthorDeveloper.html"))
        self.ui.textBrowser_AboutAuthorDeveloper.setHtml(AboutAuthorDeveloper)
        self.ui.textBrowser_AboutAuthorDeveloper.setStyleSheet("padding:10px")
        AboutTool = self.Load_Html_File(os.path.relpath("pages/Text_AboutTool.html"))
        self.ui.textBrowser_AboutTool.setHtml(AboutTool)
        self.ui.textBrowser_AboutTool.setStyleSheet("padding:10px")
        self.ui.textBrowser_AboutTool.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.ui.label_ImageDepthValue.setStyleSheet("color:red")
        self.ui.label_ImageShapeValue.setStyleSheet("color:red")
        self.ui.label_ImageWidthValue.setStyleSheet("color:red")
        self.ui.label_ImageHeightValue.setStyleSheet("color:red")
        self.ui.comboBox_Epochs_Step4CreateSimpleCNN.setCurrentIndex(1)
        self.ui.comboBox_Epochs_Step7CreateSimpleCNN2.setCurrentIndex(5)
        self.ui.comboBox_FilterAccuracy_DeepLearningFoundation.setCurrentIndex(12)
        self.ui.comboBox_Epochs_DLbyPyTorch.setCurrentIndex(9)
        self.ui.pages.setCurrentWidget(self.ui.page_AboutTool)
        self.CheckCreateDefaultFolders()
        self.LoadResources()
        self.FillCode(ImagesAndColorsManipulationsAndOprations,self.ui.textBrowser_ImageAndColors, 16)
        self.FillCode(CreateSimpleCNN,self.ui.textBrowser_CreateSimpleCNN, 25)
        self.FillCode(DeepLearningFoundationOperations,self.ui.textBrowser_DeepLearningFoundation, 16)
        self.FillCode(CreateHandGestureRecognitionCNN,self.ui.textBrowser_CreateSimpleCNN2, 26)
        self.FillCode(FaceRecognitionOperation,self.ui.textBrowser_FaceRecognitionOperation, 22)
        self.FillCode(TransferLearning,self.ui.textBrowser_TransferLearning, 46)
        self.FillCode(NeuralStyleTransfer,self.ui.textBrowser_NeuralStyleTransfer, 10)
        self.FillCode(DLbyPyTorch,self.ui.textBrowser_DLbyPyTorch, 27)
        self.FillCode(SimpleGANs,self.ui.textBrowser_SimpleGANs, 77)
        self.FillCode(ConditionalGANs,self.ui.textBrowser_ConditionalGANs, 73)
        self.FillCode(CycleGANs,self.ui.textBrowser_CycleGANs, 63)
        self.FillCode(VariationalAutoEncoders,self.ui.textBrowser_VAE, 102)

def LunchApp():
    import sys
    if sys.version_info.major < 3 or sys.version_info.minor < 10:
       print("You must use Python 3.10 or higher. Recommended version is Python 3.13")
       raise Exception("You must use Python 3.10 or higher. Recommended version is Python 3.13")
    else:
        app = QApplication(sys.argv)
        # app.setFont(QFont("Arial", 10))
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    LunchApp()