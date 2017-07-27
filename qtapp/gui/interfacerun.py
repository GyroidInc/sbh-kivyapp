# -*- coding: utf-8 -*-

# Standard imports
from __future__ import division
from io import StringIO
import csv

import matplotlib
matplotlib.use("Qt5Agg")
import os
from PyQt5 import QtCore, QtGui,  uic, QtWidgets, Qt
from PyQt5.QtWidgets import (QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox,
                             QWidget, QTableWidgetItem, QFileDialog)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import sys
import time
import traceback

# Imports from qtapp
from dynamicmplcanvas import DynamicMplCanvas
from qtapp.utils import constants, helper
from qtapp.utils.errorhandling import errorDialogOnException

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

        self.vbox = QtWidgets.QVBoxLayout()
        self.MplCanvas = DynamicMplCanvas()
        self.navi_toolbar = NavigationToolbar(self.MplCanvas, self)
        self.vbox.addWidget(self.navi_toolbar)
        self.vbox.addWidget(self.MplCanvas)
        self.vbox.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.T1_Frame_CanvasFrame.setLayout(self.vbox)

        self.columns = []
        self.freqs = []
        self.min_freq, self.max_freq = 0, 100

        # Clear table for files and labels (add rows to table dynamically)
        self.n_files = 0
        self.T1_TableWidget_Files.setRowCount(self.n_files)
        #self.T1_TableWidget_Files.setFocusPolicy(QtCore.Qt.NoFocus)

        # Ensure each item in the ListWidget has an unchecked box next to it
        for index in range(self.T2_ListWidget_Models.count()):
            self.T2_ListWidget_Models.item(index).setCheckState(QtCore.Qt.Unchecked)

        # Clear table for features/columns (add features/columns to list dynamically)
        self.T1_ListWidget_Features.clear()

        # Disable 'Load Labels File...' button until user selects Label Files By CSV File
        self.T1_Button_LoadLabelFiles.clicked.connect(self.T1_setLabelsByName)

        self.T1_ComboBox_LabelFilesBy.currentIndexChanged.connect(self.T1_selectLabelLoadMode)


        # TODO graph doesnt update on slider button press/change
        # connecting graph refresh
        self.T1_Button_RefreshPlot.clicked.connect(self.T1_Regraph)
        # Connect frequency sliders
        self.T1_HorizontalSlider_MaxFrequency.valueChanged.connect(self.T1_checkMinSlider)
        self.T1_HorizontalSlider_MinFrequency.valueChanged.connect(self.T1_checkMaxSlider)
        self.T1_HorizontalSlider_MaxFrequency.valueChanged.connect(self.T1_updateCounter)
        self.T1_HorizontalSlider_MinFrequency.valueChanged.connect(self.T1_updateCounter)
        # Connect 'Load Files...' and 'Load Directory...' buttons
        self.T1_Button_LoadFiles.clicked.connect(self.T1_openFiles)
        self.T1_Button_LoadDirectory.clicked.connect(self.T1_openDirectory)

        # Connect 'Ingest Files' button
        self.T1_Button_IngestFiles.clicked.connect(self.T1_ingestFiles)

    def T1_selectLabelLoadMode(self):
        """switches between two label calling methodologies

        Parameters
        ----------

        Returns
        -------
        """
        if self.T1_ComboBox_LabelFilesBy.currentText() == "CSV File":
            self.T1_Button_LoadLabelFiles.setText("Load Labels")
            self.T1_Button_LoadLabelFiles.disconnect()
            self.T1_Button_LoadLabelFiles.clicked.connect(self.T1_openLabels)
        else:
            self.T1_Button_LoadLabelFiles.setText("Label Data")
            self.T1_Button_LoadLabelFiles.disconnect()
            self.T1_Button_LoadLabelFiles.clicked.connect(self.T1_setLabelsByName)

    def T1_Regraph(self):
        """attempts to regraph the selected range after selection

        Parameters
        ----------

        Returns
        -------
        """
        if self.T1_ListWidget_Features.currentItem() != None:
            feat = self.T1_ListWidget_Features.currentItem().text()
            if len(self.freqs) > 1 and len(self.columns) > 0:
                #begin constructing list of dicts for graphing
                graphlist=[]
                freqlist = sorted(i for i in self.freqs if i >= self.min_freq and i <= self.max_freq)

                for baseName in self.data.keys():
                    df = self.data[baseName]["features"]
                    vals = df.loc[df["Freq"].isin(freqlist), feat].tolist()
                    graphlist.append({"label": baseName, "values" : vals})

                self.MplCanvas.update_figure(xindex=freqlist, plotlist=graphlist, ylabel=feat)







    def T1_setLabelsByFile(self, filename):
        """attempts to label each row by a dict constructed from a csv file

        Parameters
        ----------

        Returns
        -------
        """
        reader = csv.reader(open(filename, 'r'))
        toDict = {}
        for row in reader:
            k, v = row
            toDict[k] = v
        for Basename in self.data:
            self.data[Basename]["Label"] =  toDict[Basename]
        self.T1_UpdateFileList(toDict)

    def T1_setLabelsByName(self):
        """attempts to label each row by a the file name

        Parameters
        ----------

        Returns
        -------
        """
        toDict = dict(zip(self.data.keys(), helper.get_labels_from_filenames(self.data.keys())))
        for Basename in self.data:
            self.data[Basename]["label"] = toDict[Basename]
        self.T1_UpdateFileList(toDict)

    def T1_UpdateFileList(self, labelDict):
        """updates all of the labels by matching them to a dictionary

        Parameters
        ----------
        Dict{baseName, Label}

        Returns
        -------
        """

        for i in range(self.T1_TableWidget_Files.rowCount()):
            # Grab label and basename from table
            if self.T1_TableWidget_Files.item(i, 2).text() in labelDict:
                self.T1_TableWidget_Files.item(i, 1).setText(str(labelDict[self.T1_TableWidget_Files.item(i, 2).text()]))



    def T1_UpdateFigureTest(self):
        """Just a test

        Parameters
        ----------

        Returns
        -------
        """
        self.MplCanvas.update_figure([0, 1, 2, 3], [{"values": [0, 1, 2, 3], "label": "Test.xlsx"},
                                          {"values": [4, 0, 2, 3], "label": "Test2.xlsx"}])

    def T1_fileTable_createRow(self, label, file):
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
        chk_bx.setObjectName("checkbox")
        lay_out = QtWidgets.QHBoxLayout(cell_widget)
        lay_out.addWidget(chk_bx)
        lay_out.setAlignment(QtCore.Qt.AlignCenter)
        lay_out.setContentsMargins(0, 0, 0, 0)
        cell_widget.setLayout(lay_out)
        #cell_widget.findChild(QtWidgets.QCheckBox, "checkbox")
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

    def T1_updateCounter(self):
        """sets the lcd to the value of the slider freq

        Parameters
        ----------

        Returns
        -------
        """
        if len(self.freqs) > 1:
            toSet = self.T1_LCD_MinFrequency
            toSet.display(self.freqs[self.T1_HorizontalSlider_MinFrequency.value()])
            self.min_freq = self.freqs[self.T1_HorizontalSlider_MinFrequency.value()]

            toSet = self.T1_LCD_MaxFrequency
            toSet.display(self.freqs[self.T1_HorizontalSlider_MaxFrequency.value()])
            self.max_freq = self.freqs[self.T1_HorizontalSlider_MaxFrequency.value()]


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

            if toSet >= self.T1_HorizontalSlider_MaxFrequency.maximum():
                toSet = self.T1_HorizontalSlider_MaxFrequency.maximum()
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
                                                "*.csv files (*.csv)",
                                                options=options)
        if file:
            try:
                self.setLabelsByFile(file)
            except Exception as e:
                self.warningPopupMessage(message="Bad File Format", informativeText="The CSV Label file was mangled resulting in"
                                         "an exception: {}".format(e), windowTitle="Error")



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
                                                "*.xlsx files (*.xlsx);;"
                                                " *.xls files (*xls);; All files (*)",
                                                options=options)
        if files:
            # Add labels and files to table
            for f in files:
                basename = helper.get_base_filename(f)
                if basename not in self.data:
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
                    if basename not in self.data:
                        self.T1_fileTable_createRow(label=helper.parse_label(f), file=basename)
                        self.data[basename] = {'absolute_path': f, 'features': None, 'label': None}

    #@errorDialogOnException(exceptions=Exception)
    def T1_ingestFiles(self):
        """Does the major data ingestion based on prestaged setting

        Parameters
        ----------

        Returns
        -------
        """
        n_files_selected = 0  # Keep track of this for warning message

        labelsOk=True
        for i in range(self.T1_TableWidget_Files.rowCount()):
            if not self.T1_TableWidget_Files.item(i, 1).text():
                if self.T1_TableWidget_Files.cellWidget(i, 0).findChild(QtWidgets.QCheckBox, "checkbox").checkState()\
                    == QtCore.Qt.Checked:
                        self.warningPopupMessage(message="Not all Labels filled in for selected files",
                                         informativeText="Check selected files and try again",
                                         windowTitle="Label Warning")
                        labelsOk=False
                        break



        if(labelsOk):
            for i in range(self.T1_TableWidget_Files.rowCount()):

                # If checked, load file into memory
                if self.T1_TableWidget_Files.cellWidget(i, 0).findChild(QtWidgets.QCheckBox, "checkbox").checkState()\
                        == QtCore.Qt.Checked:

                    n_files_selected += 1
                    # Grab label and basename from table
                    label, basename = self.T1_TableWidget_Files.item(i, 1).text(), self.T1_TableWidget_Files.item(i, 2).text()

                    # Load data set and label
                    self.data[basename]['features'] = helper.load(file=self.data[basename]['absolute_path'])
                    self.data[basename]['label'] = label

                else:
                    continue

            # Check for intersection of columns and frequencies
            if n_files_selected > 0:
                self.columns = helper.find_unique_cols(self.data)
                self.freqs = helper.find_unique_freqs(self.data)

                # Remove columns that are usually constant
                for c in constants.COLUMNS_TO_DROP:
                    self.columns.pop(self.columns.index(c))

                # Sanity check (delete if working correctly)
                print(self.columns)
                print(self.freqs)

                if len(self.freqs) > 1:
                    self.min_freq, self.max_freq = min(self.freqs), max(self.freqs)


                    #set the increments to the frequency range selection
                    self.T1_HorizontalSlider_MinFrequency.setMaximum(len(self.freqs) -1)
                    self.T1_HorizontalSlider_MaxFrequency.setMaximum(len(self.freqs) -1)

                    # set list to columns selection
                    self.T1_ListWidget_Features.addItems(self.columns)
                    #Make all elements checkable
                    for i in range(self.T1_ListWidget_Features.count()):
                        item = self.T1_ListWidget_Features.item(i)
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        self.T1_ListWidget_Features.item(i).setCheckState(QtCore.Qt.Checked)

                    self.freqs.sort()


                else:
                    self.warningPopupMessage(message="Only one common frequency found across %d selected files" % n_files_selected,
                                             informativeText="Check selected files and try again",
                                             windowTitle="Frequency Warning")






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
    window.resize(1280, 960)
    sys.exit(app.exec_())