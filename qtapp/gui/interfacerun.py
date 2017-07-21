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



class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('interface2.ui', self)

        #self.T1_HorizontalSlider_MaxFrequency.sliderReleased.connect(self.T1_checkMaxSlider)
        #self.T1_HorizontalSlider_MinFrequency.sliderReleased.connect(self.T1_checkMinSlider)
        self.hbox = QtWidgets.QVBoxLayout()
        self.MplCanvas = DynamicMplCanvas()
        self.hbox.addWidget(self.MplCanvas)
        self.T1_GraphWindow.setLayout(self.hbox)

        self.T1_SpinBox_MinFrequency.valueChanged.connect(self.T1_checkMinSlider)
        self.T1_SpinBox_MaxFrequency.valueChanged.connect(self.T1_checkMaxSlider)
        self.T1_Button_LoadDirectory.clicked.connect(self.openFileNamesDialog)

    def replaceGfx(self):

        #self.title1.deleteLater()
        #self.gridLayout.addWidget(self.title2, 1, 5)
        pass




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
            #TODO fix this so that it doesn't crash
            #if toSet >= self.T1_SpinBox_MaxFrequency.maximum:
            #    toSet = self.T1_SpinBox_MaxFrequency.maximum
            #    self.T1_HorizontalSlider_MinFrequency.setValue(toSet - 1)

            self.T1_HorizontalSlider_MaxFrequency.setValue(toSet)

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;excel files (*.xlsx)", options=options)
        if files:
            #TODO do something with selected files
            print(files)





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())