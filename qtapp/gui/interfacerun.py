# -*- coding: utf-8 -*-

# Standard imports
from __future__ import division
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import os
from PyQt5 import QtCore, QtGui,  uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QTableWidgetItem,\
    QFileDialog
from io import StringIO
import time
import sys, traceback

# Imports from qtapp
from dynamicmplcanvas import DynamicMplCanvas
from qtapp.utils import helper




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

        # Create matplotlib widget
        self.hbox = QtWidgets.QVBoxLayout()
        self.MplCanvas = DynamicMplCanvas()
        self.hbox.addWidget(self.MplCanvas)
        self.T1_Frame_CanvasFrame.setLayout(self.hbox)



        # Clear table for files and labels (add rows to table dynamically)
        self.n_files = 0
        self.T1_TableWidget_Files.setRowCount(self.n_files)
        #self.T1_TableWidget_Files.setFocusPolicy(QtCore.Qt.NoFocus)

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

        #TODO This is a simple test for createrow
        self.T1_fileTable_createRow(label="Label", file="test.xlsx")
        self.T1_fileTable_createRow(label="Label", file="test2.xlsx")
        self.T1_fileTable_createRow(label="Label", file="test3.xlsx")

    def T1_fileTable_createRow(self, label="", file="None"):
        """Adds new row to the file table

        Parameters
        ----------

        Returns
        -------
        """
        #chkBoxItem = QtWidgets.QTableWidgetItem()
        #chkBoxItem.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        #chkBoxItem.setCheckState(QtCore.Qt.Checked)
        #chkBoxItem.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        cell_widget = QWidget()
        chk_bx = QtWidgets.QCheckBox()
        chk_bx.setCheckState(QtCore.Qt.Checked)
        lay_out = QtWidgets.QHBoxLayout(cell_widget)
        lay_out.addWidget(chk_bx)
        lay_out.setAlignment(QtCore.Qt.AlignCenter)
        lay_out.setContentsMargins(0, 0, 0, 0)
        cell_widget.setLayout(lay_out)

        file = QtWidgets.QTableWidgetItem(file)
        file.setFlags(QtCore.Qt.ItemIsEnabled)
        file.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        label = QtWidgets.QTableWidgetItem(label)
        label.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        inx = self.T1_TableWidget_Files.rowCount()
        self.T1_TableWidget_Files.insertRow(inx)
        self.T1_TableWidget_Files.setCellWidget(inx, 0, cell_widget)
        self.T1_TableWidget_Files.setItem(inx, 1, label)
        self.T1_TableWidget_Files.setItem(inx, 2, file)
        #TODO connect signals on table change to do something...

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
        for f in filenames:
            # POPULATES HERE BUT NOT CORRECTLY
            # Add current filename to table with label (if possible)
            self.T1_fileTable_createRow(label="", file=f)

    def T1_openLabels(self):
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

            # Update total number of files and add new ones with labels (if possible) to table
            self.n_files += len(files)
            #TODO do something with selected labels
            print(files)

            for f in files:
                print(f)

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
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xlsx')
                     or f.endswith('.csv') or f.endswith('.xls')]
            print(files)

            # Update total number of files and add new ones with labels (if possible) to table
            self.n_files += len(files)
            for f in files:
                print(f)
                #TODO the following line is a test
                self.T1_fileTable_createRow(label="", file=f)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())