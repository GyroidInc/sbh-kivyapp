# -*- coding: utf-8 -*-

# Example for importing a UI without compiling the .ui to a .py

from PyQt5 import QtCore, QtGui,  uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QFileDialog
import matplotlib
from qtapp.gui.dynamicmplcanvas import DynamicMplCanvas
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import sys
import os



class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('interface2.ui', self)

        #setting up matplotlib element
        self.hbox = QtWidgets.QVBoxLayout()
        self.MplCanvas = DynamicMplCanvas()
        self.hbox.addWidget(self.MplCanvas)
        self.T1_GraphWindow.setLayout(self.hbox)

        # TODO graph doesnt update on slider button press/change
        # connecting graph refresh upon slider release
        self.T1_HorizontalSlider_MaxFrequency.sliderReleased.connect(self.MplCanvas.update_figure)
        self.T1_HorizontalSlider_MinFrequency.sliderReleased.connect(self.MplCanvas.update_figure)



        # adding slider behavior checks
        self.T1_SpinBox_MinFrequency.valueChanged.connect(self.T1_checkMinSlider)
        self.T1_SpinBox_MaxFrequency.valueChanged.connect(self.T1_checkMaxSlider)
        self.T1_Button_LoadDirectory.clicked.connect(self.openFolderNamesDialog)
        # set load file by behavior
        self.T1_ComboBox_LabelFilesBy.currentIndexChanged.connect(self.selectLoadFileType)




    def T1_checkMaxSlider(self):
        if (self.T1_HorizontalSlider_MaxFrequency.value() <= self.T1_HorizontalSlider_MinFrequency.value()):
            toSet= self.T1_HorizontalSlider_MaxFrequency.value() - 1
            if toSet < 0:
                toSet=0
                self.T1_HorizontalSlider_MaxFrequency.setValue(1)

            self.T1_HorizontalSlider_MinFrequency.setValue(toSet)

    def T1_checkMinSlider(self):
        if (self.T1_HorizontalSlider_MaxFrequency.value() <= self.T1_HorizontalSlider_MinFrequency.value()):
            toSet = self.T1_HorizontalSlider_MinFrequency.value() + 1

            if toSet >= self.T1_SpinBox_MaxFrequency.maximum():
                toSet = self.T1_SpinBox_MaxFrequency.maximum()
                self.T1_HorizontalSlider_MinFrequency.setValue(toSet - 1)

            self.T1_HorizontalSlider_MaxFrequency.setValue(toSet)

    def selectLoadFileType(self):
        try:
            self.T1_Button_LoadDirectory.clicked.disconnect()
        except Exception:
            print("button disconnect issue")

        if (self.T1_ComboBox_LabelFilesBy.currentText() == "CSV File"):
            self.T1_Button_LoadDirectory.clicked.connect(self.openFileNamesDialog)
        else:
            self.T1_Button_LoadDirectory.clicked.connect(self.openFolderNamesDialog)


    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"Load : Label by File", "",
                                                "All Files (*);;excel files (*.xlsx)", options=options)
        if files:
            #TODO do something with selected files
            print(files)

    def openFolderNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,
                    "Load : Label By Folder", os.path.expanduser("~"), options=options)
        if directory:
            #TODO do something with selected files
            print (directory)




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())