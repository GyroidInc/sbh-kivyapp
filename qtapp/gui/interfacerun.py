# -*- coding: utf-8 -*-

# Standard imports
from __future__ import division
from io import StringIO

import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import os
from PyQt5 import QtCore, QtGui,  uic, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox,
                             QWidget, QTableWidgetItem, QFileDialog)
import sys
import time
import traceback

# Imports from qtapp
from dynamicmplcanvas import DynamicMplCanvas
from utils import constants, helper


def excepthook(excType, excValue, tracebackobj):
    """
    Global function to catch unhandled exceptions.

    @param excType exception type
    @param excValue exception value
    @param tracebackobj traceback object
    """
    separator = '-' * 80
    logFile = "_errmsg_" + ".log"
    notice = \
        """An unhandled exception occurred. Please report the problem\n""" \
        """using the error reporting dialog or via email to <%s>.\n""" \
        """A log has been written to "%s".\n\nError information:\n""" % \
        ("contact@gyriod.io", "_errmsg_.log")
    versionInfo = "0.0.1"
    timeString = time.strftime("%Y-%m-%d, %H:%M:%S")

    tbinfofile = StringIO()
    traceback.print_tb(tracebackobj, None, tbinfofile)
    tbinfofile.seek(0)
    tbinfo = tbinfofile.read()
    errmsg = '%s: \n%s' % (str(excType), str(excValue))
    sections = [separator, timeString, separator, errmsg, separator, tbinfo]
    msg = '\n'.join(sections)
    try:
        f = open(logFile, "w")
        f.write(msg)
        f.write(versionInfo)
        f.close()
    except IOError:
        pass
    errorbox = QtWidgets.QMessageBox()
    errorbox.setText(str(notice) + str(msg) + str(versionInfo))
    errorbox.exec_()


sys.excepthook = excepthook



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

        # Create data structure to hold information about files
        self.data = {}

        # Create matplotlib widget
        self.hbox = QtWidgets.QVBoxLayout()
        self.MplCanvas = DynamicMplCanvas()
        self.hbox.addWidget(self.MplCanvas)
        self.T1_Frame_CanvasFrame.setLayout(self.hbox)

        # Clear table for files and labels (add rows to table dynamically)
        self.n_files = 0
        self.T1_TableWidget_Files.setRowCount(self.n_files)

        # Ensure each item in the ListWidget has an unchecked box next to it
        for index in range(self.T2_ListWidget_Models.count()):
            self.T2_ListWidget_Models.item(index).setCheckState(QtCore.Qt.Unchecked)

        # Clear table for features/columns (add features/columns to list dynamically)
        self.T1_ListWidget_Features.clear()

        # Disable 'Load Labels File...' button until user selects Label Files By CSV File
        self.T1_Button_LoadLabelFiles.clicked.connect(self.T1_openLabels)
        self.T1_Button_LoadLabelFiles.setDisabled(True)
        self.T1_ComboBox_LabelFilesBy.currentIndexChanged.connect(
            lambda: self.T1_Button_LoadLabelFiles.setEnabled(
                self.T1_ComboBox_LabelFilesBy.currentText() == "CSV File"))


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

        # Connect 'Ingest Files' button
        self.T1_Button_IngestFiles.clicked.connect(self.T1_ingestFiles)


    def T1_fileTable_createRow(self, label, file):
        """Adds new row to the file table

        Parameters
        ----------

        Returns
        -------
        """
        chkBoxItem = QtWidgets.QTableWidgetItem()
        chkBoxItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chkBoxItem.setCheckState(QtCore.Qt.Checked)
        file = QtWidgets.QTableWidgetItem(file)
        file.setFlags(QtCore.Qt.ItemIsEnabled)
        label = QtWidgets.QTableWidgetItem(label)
        inx = self.T1_TableWidget_Files.rowCount()
        self.T1_TableWidget_Files.insertRow(inx)
        self.T1_TableWidget_Files.setItem(inx, 0, chkBoxItem)
        self.T1_TableWidget_Files.setItem(inx, 1, label)
        self.T1_TableWidget_Files.setItem(inx, 2, file)


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


    def T1_openLabels(self):
        """Clicked action for 'Load LABELS...' button

        Parameters
        ----------

        Returns
        -------
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileNames(self, "Load: SansEC experiment files", "",
                                                "*.xlsx files (*.xlsx);; *.csv files (*.csv);;"
                                                " *.xls files (*xls);; All files (*)",
                                                options=options)
        if file:
            print(file)



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
                                                "*.xlsx files (*.xlsx);; *.csv files (*.csv);;"
                                                " *.xls files (*xls);; All files (*)",
                                                options=options)
        if files:
            # Add labels and files to table
            for f in files:
                basename = helper.get_base_filename(f)
                self.T1_fileTable_createRow(label=helper.parse_label(f), file=basename)
                self.data[basename] = {'absolute_path': f, 'features': None, 'label': None}


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
            # Grab files that end with .xlsx, .csv, and .xls
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xlsx')
                     or f.endswith('.csv') or f.endswith('.xls')]

            if files:
                # Add labels and files to table
                for f in files:
                    basename = helper.get_base_filename(f)
                    self.T1_fileTable_createRow(label=helper.parse_label(f), file=basename)
                    self.data[basename] = {'absolute_path': f, 'features': None, 'label': None}


    def T1_ingestFiles(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        n_files_selected = 0  # Keep track of this for warning message
        for i in range(self.T1_TableWidget_Files.rowCount()):

            # If checked, load file into memory
            if self.T1_TableWidget_Files.item(i, 0).checkState() == QtCore.Qt.Checked:

                n_files_selected += 1
                # Grab label and basename from table
                label, basename = self.T1_TableWidget_Files.item(i, 1).text(), self.T1_TableWidget_Files.item(i, 2).text()

                # Load data set and label
                self.data[basename]['features'] = helper.load(file=self.data[basename]['absolute_path'])
                self.data[basename]['label'] = label

            else:
                continue

        # Check for intersection of columns and frequencies
        columns = helper.find_unique_cols(self.data)
        freqs = helper.find_unique_freqs(self.data)

        # Remove columns that are usually constant
        for c in constants.COLUMNS_TO_DROP:
            columns.pop(columns.index(c))

        # Sanity check (delete if working correctly)
        print(columns)
        print(freqs)

        if len(freqs) > 0:
            min_freq, max_freq = np.ceil(min(freqs)), np.floor(max(freq))
        else:
            self.warningPopupMessage(message="No common frequencies found across %d selected files" % n_files_selected,
                                     informativeText="Check selected files and try again",
                                     windowTitle="Frequency Warning")

        #TODO: HERE
        # Add columns to T1_ListWidget_Features --> each item should be checkable with checkbox
        # EXAMPLE: QListWidget.addItems() --> add multiple items at once or .addItem()

        #TODO: HERE
        # Set slider bars based on min/max frequencies

    # TODO: CHECK FUNCTIONALITY OF WINDOW TITLE
    def warningPopupMessage(self, message, informativeText, windowTitle):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setInformativeText(informativeText)
        msg.setWindowTitle(windowTitle)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())