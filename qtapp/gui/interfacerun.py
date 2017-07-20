# -*- coding: utf-8 -*-

# Example for importing a UI without compiling the .ui to a .py

from PyQt5 import QtCore, QtGui,  uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QFileDialog
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import sys


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure( dpi=dpi)
        self.axes = fig.add_subplot(111)
        # We want the axes cleared every time plot() is called

        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class DynamicMplCanvas(MplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = np.random.uniform(0, 10, size=4)
        self.clf()
        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.draw()


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('interface.ui', self)
        #self.T1_GraphWidget_Canvas.deletelater()
        #l = QtWidgets.QVBoxLayout(self.T1_GraphWidget_Canvas)
        #cv = DynamicMplCanvas(self.T1_GraphWidget_Canvas, dpi=200)
        #l.addWidget(cv)
        #self.T1_HorizontalSlider_MaxFrequency.sliderReleased.connect(self.T1_checkMaxSlider)
        #self.T1_HorizontalSlider_MinFrequency.sliderReleased.connect(self.T1_checkMinSlider)
        self.T1_SpinBox_MinFrequency.valueChanged.connect(self.T1_checkMinSlider)
        self.T1_SpinBox_MaxFrequency.valueChanged.connect(self.T1_checkMaxSlider)
        self.T1_Button_LoadDirectory.clicked.connect(self.openFileNamesDialog)





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