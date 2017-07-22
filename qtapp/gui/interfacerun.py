# -*- coding: utf-8 -*-

# Standard imports
from __future__ import division
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
from PyQt5 import QtCore, QtGui,  uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QTableWidgetItem, QFileDialog
import sys

# Imports from qtapp
from dynamicmplcanvas import DynamicMplCanvas
from utils import helper


class Ui(QtWidgets.QMainWindow):
    """ADD DESCRIPTION

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self):
        super(Ui, self).__init__()

        # Dynamically load .ui file
        uic.loadUi('interface.ui', self)

        # Create matplotlib widget
        self.hbox = QtWidgets.QVBoxLayout()
        self.MplCanvas = DynamicMplCanvas()
        self.hbox.addWidget(self.MplCanvas)
        self.T1_Frame_CanvasFrame.setLayout(self.hbox)

        # Clear table for files and labels (add rows to table dynamically)
        self.n_files = 0
        self.T1_TableWidget_Files.setRowCount(self.n_files)

        # Clear table for features/columns (add features/columns to list dynamically)
        self.T1_ListWidget_Features.clear()

        # Disable 'Load Labels File...' button until user selects Label Files By CSV File
        self.T1_Button_LoadLabelFiles.setDisabled(True)

        # TODO graph doesnt update on slider button press/change
        # connecting graph refresh upon slider release
        self.T1_HorizontalSlider_MaxFrequency.sliderReleased.connect(self.MplCanvas.update_figure)
        self.T1_HorizontalSlider_MinFrequency.sliderReleased.connect(self.MplCanvas.update_figure)

        # Connect frequency sliders
        self.T1_SpinBox_MinFrequency.valueChanged.connect(self.T1_checkMinSlider)
        self.T1_SpinBox_MaxFrequency.valueChanged.connect(self.T1_checkMaxSlider)

        # Connect 'Load Files...' and 'Load Directory...' buttons
        self.T1_Button_LoadFiles.clicked.connect(self.T1_openFiles)
        self.T1_Button_LoadDirectory.clicked.connect(self.T1_openDirectory)

        # set load file by behavior
        #self.T1_ComboBox_LabelFilesBy.currentIndexChanged.clicked.connect(self.selectLoadFileType)


    def T1_checkMaxSlider(self):
        """Checks maximum value of slider

        Parameters
        ----------

        Returns
        -------
        """
        if self.T1_HorizontalSlider_MaxFrequency.value() <= self.T1_HorizontalSlider_MinFrequency.value():
            toSet= self.T1_HorizontalSlider_MaxFrequency.value() - 1
            if toSet < 0:
                toSet = 0
                self.T1_HorizontalSlider_MaxFrequency.setValue(1)

            self.T1_HorizontalSlider_MinFrequency.setValue(toSet)


    def T1_checkMinSlider(self):
        """Checks minimum value of slider

        Parameters
        ----------

        Returns
        -------
        """
        if self.T1_HorizontalSlider_MaxFrequency.value() <= self.T1_HorizontalSlider_MinFrequency.value():
            toSet = self.T1_HorizontalSlider_MinFrequency.value() + 1

            if toSet >= self.T1_SpinBox_MaxFrequency.maximum():
                toSet = self.T1_SpinBox_MaxFrequency.maximum()
                self.T1_HorizontalSlider_MinFrequency.setValue(toSet - 1)

            self.T1_HorizontalSlider_MaxFrequency.setValue(toSet)


    def T1_selectLoadFileType(self):
        """ADD DESCRIPTION

        Parameters
        ----------

        Returns
        -------
        """
        if self.T1_ComboBox_LabelFilesBy.currentText() == "CSV File":
            self.T1_Button_LoadFiles.clicked.connect(self.openFileNamesDialog)
        else:
            self.T1_Button_LoadFiles.clicked.connect(self.openFolderNamesDialog)


    def T1_populateTable(self, labels, filenames):
        """Adds labels and filenames to table

        Parameters
        ----------
        labels : list
            A label for each item in filenames

        filenames : list
            A filename that corresponds to each item in labels

        Returns
        -------
        None
        """
        # MIGHT NEED TO CHECK THAT len(labels) == len(filenames)??
        # Only use basename for filenames to display in table (we do not need the absolute path!)
        filenames = [helper.get_base_filename(f) for f in filenames]
        self.T1_TableWidget_Files.setRowCount(1)
        for f in filenames:
            # POPULATES HERE BUT NOT CORRECTLY
            # Add current filename to table with label (if possible)
            self.T1_TableWidget_Files.setItem(0, 1, QTableWidgetItem(labels[0]))
            self.T1_TableWidget_Files.setItem(0, 2, QTableWidgetItem(f))


    def T1_openFiles(self):
        """Clicked action for 'Load Files...' button

        Parameters
        ----------

        Returns
        -------
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Load: SansEC experiment files", "",
                                                "All files (*);; *.xlsx files;; *.csv files", options=options)
        if files:
            # Update total number of files and add new ones with labels (if possible) to table
            self.n_files += len(files)
            #TODO do something with selected files
            print(files)

            for f in files:
                print(f)

    def T1_openDirectory(self):
        """Clicked action for 'Load Directory...' button

        Parameters
        ----------

        Returns
        -------
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,
                    "Load : SansEC directory with experiment files", os.path.expanduser("~"), options=options)

        if directory:
            print (directory)
            # Grab files that end with .xlsx, .csv, and .xls
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xlsx') or f.endswith('.csv') or f.endswith('xls')]
            print(files)

            # Update total number of files and add new ones with labels (if possible) to table
            self.n_files += len(files)
            for f in files:
                print(f)
                self.T1_populateTable(labels=helper._parse_label(f), filenames=f)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())