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
    print("You Should Install OpenCV-Python and cv2_enumerate_cameras Libraries")
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
                  valid_extensions = {'.h5','.keras', "caffemodel",".pb","prototxt","pbtxt","cfg","weights"}
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

        Base_FaceImage_Path = os.path.normpath("resources/images/faces")
        for f in listdir(Base_FaceImage_Path):
            if isfile(join(Base_FaceImage_Path, f)) and self.Is_Valid_Extension(f.strip(),"image"):
               if self.ui.comboBox_SelectFaceOne_Step3_FaceRecognitionOperation.findText(f) == -1 :
                  self.ui.comboBox_SelectFaceOne_Step3_FaceRecognitionOperation.addItem(f)
               if self.ui.comboBox_SelectFaceTwo_Step3_FaceRecognitionOperation.findText(f) == -1 :
                  self.ui.comboBox_SelectFaceTwo_Step3_FaceRecognitionOperation.addItem(f)

        for camera_info in enumerate_cameras(cv2.CAP_MSMF):
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

        self.pdf_document.load(self.pdf_path)
        self.pdf_view.pdf_path = self.pdf_path
        self.pdf_view.setDocument(self.pdf_document)
        self.pdf_view.pdf_document = self.pdf_document
        self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
      
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

    def LoadFramePdf(self, filename):
        pdfpath = "pages/" + filename
        self.pdf_path = os.path.relpath(pdfpath)
        self.frame_pdf_document.load(self.pdf_path)
        self.frame_pdf_view.pdf_path = self.pdf_path
        self.frame_pdf_view.setDocument(self.frame_pdf_document)
        self.frame_pdf_view.pdf_document = self.frame_pdf_document
        self.frame_pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.frame_pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)

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

    def ConnectActions(self):
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

        self.ui.pushButton_SaveCode.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_ImageAndColors))
        self.ui.pushButton_SaveCode_CreateSimpleCNN.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_CreateSimpleCNN))
        self.ui.pushButton_SaveCode_DeepLearningFoundation.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_DeepLearningFoundation))
        self.ui.pushButton_SaveCode_CreateSimpleCNN2.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_CreateSimpleCNN2))
        self.ui.pushButton_SaveCode__FaceRecognitionOperation.clicked.connect(partial(self.SaveCode,self.ui.textBrowser_FaceRecognitionOperation))

        self.ui.comboBox_ColorSpaceConversion.currentTextChanged.connect(self.PrepareConvertColorSpace)
        self.ui.pushButton_SaveImage.clicked.connect(self.ImagesAndColorsHandler.SaveImage)
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

    def ManualSetup(self):
        self.ImagesAndColorsHandler = ImagesAndColorsManipulationsAndOprations()
        self.CreateSimpleCNNHandler = CreateSimpleCNN()
        self.DLOperationsHandler = DeepLearningFoundationOperations(self.ImagesAndColorsHandler, self.CreateSimpleCNNHandler)
        self.CreateHandGestureRecognitionCNNHandler = CreateHandGestureRecognitionCNN(self.ImagesAndColorsHandler, self.CreateSimpleCNNHandler)
        self.FaceRecognitionOperationHandler = FaceRecognitionOperation(self.ImagesAndColorsHandler,self.DLOperationsHandler)
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
        self.ui.pages.setCurrentWidget(self.ui.page_AboutTool)
        self.CheckCreateDefaultFolders()
        self.LoadResources()
        self.FillCode(ImagesAndColorsManipulationsAndOprations,self.ui.textBrowser_ImageAndColors, 16)
        self.FillCode(CreateSimpleCNN,self.ui.textBrowser_CreateSimpleCNN, 25)
        self.FillCode(DeepLearningFoundationOperations,self.ui.textBrowser_DeepLearningFoundation, 16)
        self.FillCode(CreateHandGestureRecognitionCNN,self.ui.textBrowser_CreateSimpleCNN2, 26)
        self.FillCode(FaceRecognitionOperation,self.ui.textBrowser_FaceRecognitionOperation, 22)

def LunchApp():
    import sys
    app = QApplication(sys.argv)
    # app.setFont(QFont("Arial", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    LunchApp()