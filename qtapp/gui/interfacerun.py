# -*- coding: utf-8 -*-

# Standard imports
from __future__ import division

import csv
from io import StringIO
import json
import copy
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import os
import pandas as pd
import platform
from PyQt5 import QtCore, QtGui,  uic, QtWidgets, Qt
from PyQt5.QtWidgets import (QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox,
                             QWidget, QTableWidgetItem, QFileDialog)
import sys
from threading import Thread
import time
import traceback

# Append main module path and then import qtapp module
MAIN_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(MAIN_PATH)
import qtapp

# Imports from qtapp
try:
    from qtapp.gui.about_ui import AboutUI
    from qtapp.gui.hyperparameters_ui import HyperparametersUI
    from qtapp.gui.dynamicmplcanvas import DynamicMplCanvas
    from qtapp.model import pipeline_specifications as ps
    from qtapp.utils import constants, helper
    from qtapp.utils.errorhandling import errorDialogOnException
    from qtapp.utils.nonguiwrapper import nongui
    from qtapp.utils.filechecker import split_files
except:
    from about_ui import AboutUI
    from hyperparameters_ui import HyperparametersUI
    from filechecker_ui import FileCheckerUI
    from dynamicmplcanvas import DynamicMplCanvas
    from model import pipeline_specifications as ps
    from utils import constants, helper
    from utils.errorhandling import errorDialogOnException
    from utils.nonguiwrapper import nongui
    from utils.filechecker import split_files


def excepthook(excType, excValue, tracebackobj):
    """Global function to catch unhandled exceptions.

    Parameters
    ----------
    excType : str
        exception type

    excValue : str
        exception value

    tracebackobj : str
        traceback object

    Returns
    -------
    None
    """
    separator = '-' * 80
    logFile = "_errmsg_" + ".log"
    notice = \
        """An unhandled exception occurred. Please report the problem\n""" \
        """ via email to <%s>.\n""" \
        """A log has been written to "%s".\n\nError information:\n""" % \
        ("contact@gyriod.io", "_errmsg_.log")
    versionInfo = constants.VERSION
    timeString = time.strftime("%Y-%m-%d, %H:%M:%S")

    tbinfofile = StringIO()
    traceback.print_tb(tracebackobj, None, tbinfofile)
    tbinfofile.seek(0)
    tbinfo = tbinfofile.read()
    errmsg = '%s: \n%s' % (str(excType), str(excValue))
    sections = [separator, timeString, separator, errmsg, separator, tbinfo]
    msg = '\n'.join(sections)
    try:
        f = open(os.path.join(MAIN_PATH, logFile), "w")
        f.write(msg)
        f.write(versionInfo)
        f.close()
    except IOError:
        pass
    errorbox = QtWidgets.QMessageBox()
    errorbox.setText(str(notice) + str(msg) + str(versionInfo))
    errorbox.exec_()

# Define global except hook
sys.excepthook = excepthook


class Ui(QtWidgets.QMainWindow):
    """Main UI for application"""
    def __init__(self):
        super(Ui, self).__init__()

        # Dynamically load .ui file
        ui_path = os.path.join(os.path.join(os.path.join(MAIN_PATH, 'qtapp'), 'gui'), 'interface.ui')
        uic.loadUi(ui_path, self)
        self.statusBar().showMessage('Load Files or Configuration File to Begin...')

        # Load icon file
        icon_path = os.path.join(os.path.join(os.path.join(MAIN_PATH, 'qtapp'), 'gui', 'gyroid.png'))
        self.setWindowIcon(QtGui.QIcon(icon_path))

        # Force tab widget to open on Tab 1
        self.TabWidget.setCurrentIndex(0)

        # Initial parameters
        self.data = {}
        self.T1_stable = True
        self.test_data = {}
        self.config = helper.create_blank_config()
        self.learner_input = None
        self.columns = []
        self.freqs = []
        self.min_freq, self.max_freq = 0, 100
        self.n_files_selected = 0
        self.dataset_created = False

        # Set experiment names on Tab 2 and Tab 3
        self.T2_Label_ExperimentName.setText("No Experiment Name Saved in Configuration File")
        self.T3_Label_ExperimentName.setText("No Experiment Name Saved in Configuration File")

        # Create matplotlib widget
        self.vbox = QtWidgets.QVBoxLayout()
        self.MplCanvas = DynamicMplCanvas()
        self.navi_toolbar = NavigationToolbar(self.MplCanvas, self)
        self.vbox.addWidget(self.navi_toolbar)
        self.vbox.addWidget(self.MplCanvas)
        self.vbox.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.T1_Frame_CanvasFrame.setLayout(self.vbox)

        # Ensure each item in the ListWidget on Tab 2 has an unchecked box next to it
        for index in range(self.T2_ListWidget_Models.count()):
            self.T2_ListWidget_Models.item(index).setCheckState(QtCore.Qt.Unchecked)

        ###############################
        ## CONNECT MENU ITEM BUTTONS ##
        ###############################

        # Connect the menu item 'Save Configuration File'
        self.FileItem_SaveConfigurationFile.triggered.connect(self.saveConfigurationFile)

        # Connect the menu item 'Load Configuration File'
        self.FileItem_LoadConfigurationFile.triggered.connect(self.loadConfigurationFile)

        # Connect the menu item 'Exit'
        self.FileItem_Exit.triggered.connect(self.exitApplication)

        # Connect the menu item 'About'
        self.HelpItem_About.triggered.connect(AboutUI)

        # Connect the menu item 'Documentation'
        self.HelpItem_Documentation.triggered.connect(self.viewDocumentation)

        # Connect the menu item 'File Checker'
        self.HelpItem_fileChecker.triggered.connect(self.checkFiles)

        ###########################
        ## CONNECT TAB 1 BUTTONS ##
        ###########################

        # Disable 'Load Labels File...' button until user selects Label Files By CSV File
        self.T1_Button_LoadLabelFiles.clicked.connect(self.T1_setLabelsByName)
        self.T1_ComboBox_LabelFilesBy.currentIndexChanged.connect(self.T1_selectLabelLoadMode)

        # Connect button 'Refresh Plot'
        self.T1_Button_RefreshPlot.clicked.connect(self.T1_Regraph)

        # Connect minimum and maximum frequency sliders
        self.T1_HorizontalSlider_MinFrequency.valueChanged.connect(self.T1_checkMaxSlider)
        self.T1_HorizontalSlider_MinFrequency.valueChanged.connect(self.T1_updateCounter)

        self.T1_HorizontalSlider_MaxFrequency.valueChanged.connect(self.T1_checkMinSlider)
        self.T1_HorizontalSlider_MaxFrequency.valueChanged.connect(self.T1_updateCounter)

        # Connect 'Load Files...' and 'Load Directory...' buttons
        self.T1_Button_LoadFiles.clicked.connect(self.T1_openFiles)
        self.T1_Button_LoadDirectory.clicked.connect(self.T1_openDirectory)

        # Connect 'Ingest Files' button
        self.T1_Button_IngestFiles.clicked.connect(self.T1_LoadingPopupIngest)

        # Connect 'Create Dataset' button
        self.T1_Button_CreateDataset.clicked.connect(self.T1_createDataset)

        # Connect 'Save Configuration File' button
        self.T1_Button_SaveConfigurationFile.clicked.connect(self.T1_saveConfigurationFile)

        ###########################
        ## CONNECT TAB 2 BUTTONS ##
        ###########################

        # Connect 'Set Parameters' button
        self.T2_Button_SetParameters.clicked.connect(self.T2_setParameters)

        # Connect 'Begin Training' button
        self.T2_Button_BeginTraining.clicked.connect(self.T2_beginTraining)

        # Connect the 'Save Configuration File' button
        self.T2_Button_SaveConfigurationFile.clicked.connect(self.T2_saveConfigurationFile)

        # Connect the 'Clear Log' button
        self.T2_Button_ClearLog.clicked.connect(self.T2_TextBrowser_AnalysisLog.clear)

        ###########################
        ## CONNECT TAB 3 BUTTONS ##
        ###########################

        # Connect 'Load Files...' and 'Load Directory...' buttons
        self.T3_Button_LoadFiles.clicked.connect(self.T3_openFiles)
        self.T3_Button_LoadDirectory.clicked.connect(self.T3_openDirectory)

        # Connect the 'Clear Log' button
        self.T3_Button_ClearLog.clicked.connect(self.T3_TextBrowser_AnalysisLog.clear)

        # Connect the 'Load Trained Models...' button
        self.T3_Button_LoadTrainedModels.clicked.connect(self.T3_loadTrainedModels)

        # Connect the 'Begin Testing' button
        self.T3_Button_BeginTesting.clicked.connect(self.T3_beginTesting)

        # Connect the 'Generate Report' button
        self.T3_Button_GenerateReport.clicked.connect(self.T3_generateReport)


    ###########################################
    ############# TAB 1 FUNCTIONS #############
    ###########################################

    def T1_selectLabelLoadMode(self):
        """Switches two labeling methodologies on Tab 1"""
        if self.T1_ComboBox_LabelFilesBy.currentText() == "CSV File":
            self.T1_Button_LoadLabelFiles.setText("Load Labels")
            self.T1_Button_LoadLabelFiles.disconnect()
            self.T1_Button_LoadLabelFiles.clicked.connect(self.T1_openLabels)
        else:
            self.T1_Button_LoadLabelFiles.setText("Label Data")
            self.T1_Button_LoadLabelFiles.disconnect()
            self.T1_Button_LoadLabelFiles.clicked.connect(self.T1_setLabelsByName)


    def T1_Regraph(self):
        """Attempts to regraph the selected range after selection on Tab 1"""
        if not self.dataset_created:
            helper.messagePopUp(message="Dataset not created",
                              informativeText="Please create dataset",
                              windowTitle="Error: Missing Information",
                              type="error")
            self.statusBar().showMessage("Error: Missing Information")
            return

        if self.T1_ListWidget_Features.currentItem() != None:
            feat = self.T1_ListWidget_Features.currentItem().text()
            if len(self.freqs) > 1 and len(self.columns) > 0:
                # Begin constructing list of dicts for graphing
                graphlist=[]
                freqlist = sorted(i for i in self.freqs if i >= self.min_freq and i <= self.max_freq)

                for baseName in self.data.keys():
                    if self.data[baseName]["features"] is not None:
                        df = self.data[baseName]["features"]
                        vals = df.loc[df["Freq"].isin(freqlist), feat].tolist()
                        graphlist.append({"label": baseName, "values" : vals})

                self.MplCanvas.update_figure(xindex=freqlist, plotlist=graphlist, ylabel=feat)


    def T1_setLabelsByFile(self, filename):
        """Attempts to label each row by a dict constructed from a csv file on Tab 1"""
        reader = csv.reader(open(filename, 'r'))
        toDict = {}
        for row in reader:

            # If first row is headers, then skip
            if reader.line_num == 1:
                if set(row) == {'label', 'filename'}:
                    # Check if filenames are first column
                    if row[0] == 'filename':
                        filename_first = True
                    else:
                        filename_first = False
                    continue
                else:
                    # Try and figure out which column contains the filename and label
                    try:
                        float(row[1]) # Attempts to convert entry in second column to float --> label should convert
                        filename_first = True
                        pass
                    except ValueError:
                        filename_first = False
                        pass

            # Grab element in first column and second column
            k, v = row

            # Filename should be the key and label is the value
            if filename_first:
                try:
                    if float(v) == int(v):
                        toDict[k] = int(v)
                    else:
                        toDict[k] = float(v)
                except:
                    toDict[k] = float(v)
            else:
                try:
                    if float(k) == int(k):
                        toDict[v] = int(k)
                    else:
                        toDict[v] = float(k)
                except:
                    toDict[v] = float(k)

        # Add labels to dictionary and update file list
        for basename in self.data:
            self.data[basename]["Label"] = toDict[basename]
        self.T1_UpdateFileList(toDict)


    def T1_setLabelsByName(self):
        """Attempts to label each row by the filename on Tab 1"""
        if self.T1_TableWidget_Files.rowCount() == 0:
            helper.messagePopUp(message="No files loaded",
                                informativeText="Please load files and try again",
                                windowTitle="Error: No Files Selected",
                                type="error")
            self.statusBar().showMessage("Error: No Files Selected")
            return

        toDict = dict(zip(self.data.keys(), helper.get_labels_from_filenames(self.data.keys())))
        for Basename in self.data:
            self.data[Basename]["label"] = toDict[Basename]
        self.T1_UpdateFileList(toDict)


    def T1_UpdateFileList(self, labelDict):
        """updates all of the labels by matching them to a dictionary

        Parameters
        ----------
        labelDict: dict
            Dictionary with key = file basename and value = file label

        Returns
        -------
        None
        """
        for i in range(self.T1_TableWidget_Files.rowCount()):
            # Grab label and basename from table
            if self.T1_TableWidget_Files.item(i, 2).text() in labelDict:
                self.T1_TableWidget_Files.item(i, 1).setText(str(labelDict[self.T1_TableWidget_Files.item(i, 2).text()]))


    def T1_fileTable_createRow(self, label, file):
        """Adds new row to the file table on Tab 1

        Parameters
        ----------
        label : str
            Label for file

        file : str
            Basename for file

        Returns
        -------
        None
        """
        cell_widget = QWidget()
        chk_bx = QtWidgets.QCheckBox()
        chk_bx.setCheckState(QtCore.Qt.Checked)
        chk_bx.setObjectName("checkbox")

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


    def T1_updateCounter(self):
        """Sets the lcd to the value of the slider freq on Tab 1"""
        if len(self.freqs) > 1:
            toSet = self.T1_LCD_MinFrequency
            toSet.display(self.freqs[self.T1_HorizontalSlider_MinFrequency.value()])
            self.min_freq = self.freqs[self.T1_HorizontalSlider_MinFrequency.value()]

            toSet = self.T1_LCD_MaxFrequency
            toSet.display(self.freqs[self.T1_HorizontalSlider_MaxFrequency.value()])
            self.max_freq = self.freqs[self.T1_HorizontalSlider_MaxFrequency.value()]

            if self.dataset_created:
                self.statusBar().showMessage("""Click 'Create Dataset' to update frequencies""")


    def T1_checkMaxSlider(self):
        """Checks maximum value of slider"""
        if self.T1_HorizontalSlider_MaxFrequency.value() <= self.T1_HorizontalSlider_MinFrequency.value():
            toSet= self.T1_HorizontalSlider_MaxFrequency.value() - 1
            if toSet < 0:
                toSet = 0
                self.T1_HorizontalSlider_MaxFrequency.setValue(1)

            self.T1_HorizontalSlider_MinFrequency.setValue(toSet)


    def T1_checkMinSlider(self):
        """Checks minimum value of slider on Tab 1"""
        if self.T1_HorizontalSlider_MaxFrequency.value() <= self.T1_HorizontalSlider_MinFrequency.value():
            toSet = self.T1_HorizontalSlider_MinFrequency.value() + 1

            if toSet >= self.T1_HorizontalSlider_MaxFrequency.maximum():
                toSet = self.T1_HorizontalSlider_MaxFrequency.maximum()
                self.T1_HorizontalSlider_MinFrequency.setValue(toSet - 1)

            self.T1_HorizontalSlider_MaxFrequency.setValue(toSet)


    def T1_openLabels(self):
        """Opens .csv file for loading labels by file on Tab 1"""
        if self.T1_TableWidget_Files.rowCount() == 0:
            helper.messagePopUp(message="No files loaded",
                                informativeText="Please load files and try again",
                                windowTitle="Error: No Files Selected",
                                type="error")
            self.statusBar().showMessage("Error: No Files Selected")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "Load: SansEC labels file", "",
                                                "*.csv files (*.csv)",
                                                options=options)
        if file:
            try:
                self.T1_setLabelsByFile(file)
            except Exception as e:
                helper.messagePopUp(message="Error loading label file because",
                                  informativeText=str(e),
                                  windowTitle="Error: Loading Label File",
                                  type="error")
                self.statusBar().showMessage("Error: Loading Label File")
                return


    def T1_openFiles(self):
        """Opens single or multiple files for training on Tab 1"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Load : SansEC experiment files", "",
                                                "*.xlsx files (*.xlsx);;"
                                                " *.xls files (*xls);; All files (*)",
                                                options=options)
        if files:
            # Add labels and files to table
            for f in files:
                basename = helper.get_base_filename(f)
                if basename not in self.data:
                    self.T1_fileTable_createRow(label="", file=basename)
                    self.data[basename] = {'absolute_path': f, 'features': None, 'label': None, 'selected': True}


    def T1_openDirectory(self):
        """Opens directory of files for training on Tab 1"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,
                    "Load : SansEC directory with experiment files", os.path.expanduser("~"), options=options)

        if directory:
            # Grab files that end with .xlsx, .csv, and .xls
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xlsx')
                     or f.endswith('.xls')]

            if files:
                # Add labels and files to table
                for f in files:
                    basename = helper.get_base_filename(f)
                    if basename not in self.data:
                        self.T1_fileTable_createRow(label="", file=basename)
                        self.data[basename] = {'absolute_path': f, 'features': None, 'label': None, 'selected': True}


    def T1_LoadingPopupIngest(self):
        """Sets up ingestion with a loading bar on Tab 1"""
        # Define initial variables
        progress = QtWidgets.QProgressDialog(parent=self)
        progress.setCancelButton(None)
        progress.setLabelText('Ingesting files...')
        progress.setWindowTitle("Loading")
        progress.setMinimum(0)
        progress.setMaximum(0)

        deepcpy = copy.deepcopy(self.data)
        progress.forceShow()
        (message, informativeText, windowTitle, type) = self.T1_ingestFiles()
        if message is not None:
            helper.messagePopUp(message, informativeText, windowTitle, type)
        progress.cancel()
        if self.T1_stable is False:
            self.data = deepcpy
            self.T1_stable = True


    @nongui
    def T1_ingestFiles(self):
        """Does the major data ingestion based on prestaged setting on Tab 1"""
        # Define initial variables
        self.T1_stable = False
        self.n_files_selected, labelsOk, allOk, checkcnt = 0, True, True, 0

        # COMMENT HERE
        for i in range(self.T1_TableWidget_Files.rowCount()):
            if self.T1_TableWidget_Files.cellWidget(i, 0).findChild(QtWidgets.QCheckBox, "checkbox").checkState() \
                    == QtCore.Qt.Checked:
                checkcnt += 1

                # COMMENT HERE
                if not self.T1_TableWidget_Files.item(i, 1).text() and allOk == True:
                    allOk = False
                    message="Not all labels filled in for selected files"
                    informativeText="Check selected files and try again"
                    windowTitle="Error: Missing Labels"
                    type="error"
                    self.statusBar().showMessage("Error: Missing Labels")
                    return message, informativeText, windowTitle, type

        # COMMENT HERE
        if checkcnt < 2:
            allOk = False
            message="Error: %d file(s) selected" % checkcnt
            informativeText="Please select more files. Must be at least 2 files"
            windowTitle="Error: Missing Files"
            type="error"
            self.statusBar().showMessage("Error: Missing Files")
            return message, informativeText, windowTitle, type

        # COMMENT HERE
        if allOk:
            for i in range(self.T1_TableWidget_Files.rowCount()):
                # Grab label and basename from table
                label, basename = self.T1_TableWidget_Files.item(i, 1).text(), self.T1_TableWidget_Files.item(i, 2).text()

                # If checked, load file into memory
                if self.T1_TableWidget_Files.cellWidget(i, 0).findChild(QtWidgets.QCheckBox, "checkbox").checkState() \
                        == QtCore.Qt.Checked:

                    self.statusBar().showMessage('Ingesting files...')
                    self.n_files_selected += 1

                    # Grab label and basename from table
                    label, basename = self.T1_TableWidget_Files.item(i, 1).text(), self.T1_TableWidget_Files.item(i, 2).text()

                    # Load data set and label
                    self.data[basename]['features'] = helper.load(file=self.data[basename]['absolute_path'])
                    self.data[basename]['label'] = label
                    self.data[basename]['selected'] = True

                    # Update configuration file if file not already added
                    if self.data[basename]['absolute_path'] not in self.config["TrainFiles"]:
                        self.config["TrainFiles"].append(self.data[basename]['absolute_path'])

                else:
                    self.data[basename]['features'] = None
                    self.data[basename]['selected'] = False

            # Check for intersection of columns and frequencies
            if self.n_files_selected == 0:
                self.statusBar().showMessage('')
                message="No files to ingest"
                informativeText="Add files and try again",
                windowTitle="Error: No Files Selected",
                type="error"
                self.statusBar().showMessage("Error: No Files Selected")
                return message, informativeText, windowTitle, type

            # Find intersection of rows and columns
            self.freqs = helper.find_unique_freqs(self.data)
            self.columns = helper.find_unique_cols(self.data)

            # Check if at lseast one frequency and column are selected
            if len(self.freqs) < 2:
                self.statusBar().showMessage('Tip: Exclude files with different frequencies and try again')
                message="Less than 2 common frequencies found across {} selected files".format(self.n_files_selected)
                informativeText="Check selected files and try again"
                windowTitle="Error: Few Common Frequencies Across Files"
                type="error"
                self.statusBar().showMessage("Error: Less Than 2 Common Frequencies Across Files")
                return message, informativeText, windowTitle, type

            if len(self.columns) == 0:
                self.statusBar().showMessage('Tip: Exclude files with different features/columns and try again')
                message="No common features/columns found across {} selected files".format(self.n_files_selected)
                informativeText="Check selected files and try again"
                windowTitle="Error: No Common Features/Columns Across Files"
                type="error"
                self.statusBar().showMessage("Error: No Common Features/Columns Across Files")
                return message, informativeText, windowTitle, type

            # Remove columns that are usually constant
            for c in constants.COLUMNS_TO_DROP:
                self.columns.pop(self.columns.index(c))

            # Grab min and max frequency across all data sets
            self.min_freq, self.max_freq = min(self.freqs), max(self.freqs)

            # Set the increments to the frequency range selection
            self.T1_HorizontalSlider_MinFrequency.setMaximum(len(self.freqs) - 1)
            self.T1_HorizontalSlider_MaxFrequency.setMaximum(len(self.freqs) - 1)

            # Set slider positions
            self.T1_HorizontalSlider_MinFrequency.setSliderPosition(0)
            self.T1_HorizontalSlider_MaxFrequency.setSliderPosition(len(self.freqs))

            # Set display for LCDs - Sort freqs first!
            self.freqs.sort()
            self.T1_updateCounter()

            # Remove old column selections from list
            self.T1_ListWidget_Features.clear()

            # Set list to columns selection
            self.T1_ListWidget_Features.addItems(self.columns)

            # Make all elements checkable
            for i in range(self.T1_ListWidget_Features.count()):
                item = self.T1_ListWidget_Features.item(i)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                self.T1_ListWidget_Features.item(i).setCheckState(QtCore.Qt.Checked)

            self.statusBar().showMessage('Successfully ingested %d files' % self.n_files_selected)
            self.T1_stable = True
            return None, None, None, None


    def T1_createDataset(self):
        """Creates training data set on Tab 1"""
        # Make sure at least one file selected
        if self.T1_stable is False:
            helper.messagePopUp(message="Something happened during ingest.",
                                informativeText="Ingested files did not stabilize. Rerun Ingest.",
                                windowTitle="Error: No Files Ingested",
                                type="error")
            self.statusBar().showMessage("Error: Rerun \"Ingest files\"")
            return

        if self.n_files_selected == 0:
            helper.messagePopUp(message="No files ingested to create data set",
                              informativeText="Ingest files and try again",
                              windowTitle="Error: No Files Ingested",
                              type="error")
            self.statusBar().showMessage("Error: No Files Ingested")
            return

        self.statusBar().showMessage("Creating dataset for model training...")

        # Check for columns to use among those that are checked
        cols_to_use = []
        for index in range(self.T1_ListWidget_Features.count()):
            if self.T1_ListWidget_Features.item(index).checkState() == QtCore.Qt.Checked:
                cols_to_use.append(self.T1_ListWidget_Features.item(index).text())

        # Make sure at least one column selected
        if len(cols_to_use) == 0:
            helper.messagePopUp(message="No features/columns selected",
                              informativeText="Select one or more features/columns and try again",
                              windowTitle="Error: No Features/Columns Selected",
                              type="error")
            self.statusBar().showMessage("Error: No Features/Columns Selected")
            return

        # Get indices for closest frequencies
        min_idx, max_idx = helper.index_for_freq(self.freqs, self.min_freq), helper.index_for_freq(self.freqs, self.max_freq)

        # Create data set
        self.learner_input = helper.tranpose_and_append_columns(data=self.data,
                                                                freqs=self.freqs,
                                                                columns=cols_to_use,
                                                                idx_freq_ranges=(min_idx, max_idx))
        self.statusBar().showMessage("Dataset created with %d samples and %d features" % \
                                     (self.learner_input.shape[0], self.learner_input.shape[1]-1))
        self.dataset_created = True # Flag that dataset was created

        # Update configuration file
        self.config['TrainSamples'] = self.learner_input.shape[0]
        self.config['TrainFeatures'] = self.learner_input.shape[1] - 1  # Last column is for label
        self.config['Freqs'], self.config['Columns'] = self.freqs[min_idx:max_idx + 1], cols_to_use
        self.config['LearningTask'] = "Regressor" if self.T1_RadioButton_ContinuousLabels.isChecked() else "Classifier"


    def T1_saveConfigurationFile(self):
        """Saves configuration file on Tab 1"""
        if len(self.T1_Label_ExperimentName.text()) == 0:
            helper.messagePopUp(message="Experiment name not specified",
                              informativeText="Please enter experiment name",
                              windowTitle="Error: Missing Information",
                              type="error")
            self.statusBar().showMessage("Error: Missing Information")
            return

        if not self.dataset_created:
            helper.messagePopUp(message="Dataset not created",
                              informativeText="Please create dataset",
                              windowTitle="Error: Missing Information",
                              type="error")
            self.statusBar().showMessage("Error: Missing Information")
            return

        # Set experiment name
        try:
            self.config['ExperimentName'] = self.T1_Label_ExperimentName.text().replace(" ", "_")
        except Exception as e:
            helper.messagePopUp(message="Error setting experiment name because",
                              informativeText=str(e),
                              windowTitle="Error: Unable to Set Experiment Name",
                              type="error")
            self.statusBar().showMessage("Error: Unable to Set Experiment Name")
            return

        # Set learning task based on label type (categorical = Classifier, continuous = Regressor)
        self.config['LearningTask'] = "Regressor" if self.T1_RadioButton_ContinuousLabels.isChecked() else "Classifier"

        # Sets save directory to be directory ABOVE where qtapp resides
        self.config['SaveDirectory'] = os.path.join(os.path.dirname(MAIN_PATH),
                                                    self.config['ExperimentName'])

        # Create directory structure for current experiment
        if os.path.isdir(self.config['SaveDirectory']):
            overwriteFiles = helper.messagePopUp(message="Directory already exists for experiment: \n%s" % \
                                                       self.config['SaveDirectory'],
                                               informativeText="Do you want to overwrite files?",
                                               windowTitle="Warning: Directory Already Exists",
                                               type="warning",
                                               question=True)
            overwriteStatus = False if overwriteFiles == QMessageBox.No else True
        else:
            overwriteStatus = False
        helper.create_directory_structure(save_directory=self.config['SaveDirectory'],
                                          overwrite=overwriteStatus,
                                          configuration_file=self.config)

        # Save configuration file
        self.saveConfigurationFile()

        # Save learner_input data set
        try:
            self.learner_input.to_csv(os.path.join(self.config["SaveDirectory"], "training_data.csv"), index=False)
        except Exception as e:
            helper.messagePopUp(message="Warning: Error saving training data set because",
                              informativeText=str(e) + "\nThis may cause errors when generating summary report",
                              windowTitle="Warning: Error Saving Training Data",
                              type="warning")
            self.statusBar().showMessage("Warning: Error Saving Training Data")
            pass

        # Force tab widget to open on Tab 2
        self.TabWidget.setCurrentIndex(1)

    ###########################################
    ############# TAB 2 FUNCTIONS #############
    ###########################################

    def T2_setParameters(self):
        """Sets hyperparameters for model using UI on Tab 2"""
        if not self.T2_ListWidget_Models.currentItem():
            helper.messagePopUp(message="No model selected",
                              informativeText="Select model and try again",
                              windowTitle="Error: No Model Selected",
                              type="error")
            self.statusBar().showMessage("Error: No Model Selected")
            return

        # Grab current model name
        selectedModel = self.T2_ListWidget_Models.currentItem().text()
        if selectedModel == "K-Nearest Neighbors":
            model = "KNearestNeighbors"
        else:
            model = selectedModel.replace(" ", "")

        # Run UI for user to select hyperparameters
        HyperparametersUI(model=model, configuration_file=self.config)

        try:
            if self.config['Models'][model]['hyperparameters']:
                self.statusBar().showMessage('Hyperparameters set for %s' % selectedModel)
        except Exception as e:
            helper.messagePopUp(message="Error setting hyperparameters for %s because" % model,
                              informativeText=str(e),
                              windowTitle="Error: Setting Hyperparameters for %s" % model,
                              type="error")
            self.statusBar().showMessage("Error: Setting Hyperparameters for %s" % model)
            return


    def T2_beginTraining(self):
        """Trains selected models on Tab 2"""
        if self.learner_input is None:
            helper.messagePopUp(message="Training data not created",
                                informativeText="Create training dataset and try again",
                                windowTitle="Error: No Training Data",
                                type="error")
            self.statusBar().showMessage("Error: No Training Data")
            return

        # Check and make sure number of samples is sufficient for training
        if self.config["LearningTask"] == "Classifier":
            n_classes_lt3 = helper.check_categorical_labels(labels=self.learner_input.iloc[:, -1])
            if n_classes_lt3:
                helper.messagePopUp(message="Need at least 3 samples per class for classifier models",
                                    informativeText="%d classes with less than three samples" % n_classes_lt3,
                                    windowTitle="Error: Not Enough Samples Per Class",
                                    type = "error")
                self.statusBar().showMessage("Error: Not Enough Samples Per Class")
                return
        else:
            if self.learner_input.shape[0] < 9:
                helper.messagePopUp(message="Need at least 9 samples for regression models",
                                    informativeText="%d samples found" % self.learner_input.shape[0],
                                    windowTitle="Error: Not Enough Samples",
                                    type = "error")
                self.statusBar().showMessage("Error: Not Enough Samples")
                return

        # Keep track of number of models selected and define models to train
        self.n_models_selected, self.models_to_train, models_without_hypers = 0, {}, 0

        # See if any models selected
        for i in range(self.T2_ListWidget_Models.count()):
            if self.T2_ListWidget_Models.item(i).checkState() == QtCore.Qt.Checked:

                # Grab model name
                selectedModel = self.T2_ListWidget_Models.item(i).text()
                if selectedModel == "K-Nearest Neighbors":
                    model = "KNearestNeighbors"
                else:
                    model = selectedModel.replace(" ", "")

                # Update configuration file to flag that current model should be trained
                self.config['Models'][model]['selected'] = True

                # If not automatically tuning, check if hyperparameters specified; if not set to default sklearn params
                if not self.config['AutomaticallyTune']:
                    if len(self.config['Models'][model]['hyperparameters']) == 0:
                        self.config['Models'][model]['hyperparameters'] = {}
                        models_without_hypers += 1

                self.n_models_selected += 1
            else:
                # Grab model name
                selectedModel = self.T2_ListWidget_Models.item(i).text()
                if selectedModel == "K-Nearest Neighbors":
                    model = "KNearestNeighbors"
                else:
                    model = selectedModel.replace(" ", "")

                # Model not selected to set to False
                self.config['Models'][model]['selected'] = False

        if self.n_models_selected == 0:
            helper.messagePopUp(message="No models selected for training",
                              informativeText="Select models and try again",
                              windowTitle="Error: No Model Selected",
                              type="error")
            self.statusBar().showMessage("Error: No Model Selected")
            return

        self.T2_Button_BeginTraining.setEnabled(False)
        self.T2_ProgressBar_Training.setRange(0, 0)

        # Set configuration file to parameters from pipeline specifications

        # Standardize features
        if self.T2_ComboBox_StandardizeFeatures.currentText() == "Yes":
            self.config['StandardizeFeatures'] = True
        else:
            self.config['StandardizeFeatures'] = False

        # Feature reduction (convert to lower-case if not None)
        if self.T2_ComboBox_FeatureReduction.currentText() == "None":
            self.config['FeatureReduction'] = None
        else:
            self.config['FeatureReduction'] = self.T2_ComboBox_FeatureReduction.currentText().lower()

        # Training method (convert to lower-case)
        self.config['TrainingMethod'] = self.T2_ComboBox_TrainingMethod.currentText().lower()

        # Automatically tune
        if self.T2_ComboBox_AutomaticallyTune.currentText() == "Yes":
            self.config['AutomaticallyTune'] = True
        else:
            self.config['AutomaticallyTune'] = False

        # If not automatically tuning, inform user which model(s) specified with default hyperparameters
        if not self.config['AutomaticallyTune'] and models_without_hypers > 0:
            helper.messagePopUp(message="%d models selected and hyperparameters for %d of these models were not specified" % \
                                      (self.n_models_selected, models_without_hypers),
                              informativeText="Hyperparameters set to default values",
                              windowTitle="Setting Default Hyperparameters",
                              type="information")

        # PRINT SUMMARY INFORMATION HERE ABOUT ANALYSIS BEFORE IT STARTS
        if self.n_models_selected == 1:
            self.T2_TextBrowser_AnalysisLog.append("Training %d machine learning model..." % self.n_models_selected)
            self.statusBar().showMessage("Training %d machine learning model, see log for details" % self.n_models_selected)
        else:
            self.T2_TextBrowser_AnalysisLog.append("Training %d machine learning models..." % self.n_models_selected)
            self.statusBar().showMessage("Training %d machine learning models, see log for details" % self.n_models_selected)

        # Grab information from configuration file
        learner_type = self.config['LearningTask']
        standardize = self.config['StandardizeFeatures']
        feature_reduction_method = self.config['FeatureReduction']
        training_method = self.config['TrainingMethod']
        automatically_tune = self.config['AutomaticallyTune']
        X, y = self.learner_input.iloc[:, :-1], self.learner_input.iloc[:, -1]

        e = None
        # If automatically tune
        if automatically_tune:
            # Start in separate thread
           e = ps.automatically_tune(X, y, learner_type, standardize, feature_reduction_method,
                                 training_method, self.T2_TextBrowser_AnalysisLog, self.config)

        # Otherwise train using holdout or cross-validation
        else:
            if training_method == "holdout":
                e = ps.holdout(X, y, learner_type, standardize, feature_reduction_method,
                                self.T2_TextBrowser_AnalysisLog, self.config)
            else:
                e = ps.cv(X, y, learner_type, standardize, feature_reduction_method,
                          self.T2_TextBrowser_AnalysisLog, self.config)

        self.T2_Button_BeginTraining.setEnabled(True)
        self.T2_ProgressBar_Training.setRange(0, 1)
        if e is not None:
            raise e


    def T2_saveConfigurationFile(self):
        """Saves configuration file on Tab 2"""
        if len(self.config["ExperimentName"]) == 0:
            helper.messagePopUp(message="Experiment name not specified",
                                informativeText="Please enter experiment name and save configuration file",
                                windowTitle="Error: Missing Information",
                                type="error")
            self.statusBar().showMessage("Error: Missing Information")
            return

        try:
            # Save configuration file
            json.dump(self.config, open(os.path.join(self.config['SaveDirectory'], 'configuration.json'), 'w'))

            # Update status bar
            self.statusBar().showMessage("Successfully saved configuration file for experiment %s" % \
                                         self.config['ExperimentName'])

            # Update experiment name on Tab 2 and Tab 3
            self.T2_Label_ExperimentName.setText(self.config['ExperimentName'])
            self.T3_Label_ExperimentName.setText(self.config['ExperimentName'])
        except Exception as e:
            helper.messagePopUp(message="Error saving configuration file because",
                                informativeText=str(e),
                                windowTitle="Error: Saving Configuration File",
                                type="error")
            self.statusBar().showMessage("Error: Saving Configuration File")
            return
        self.TabWidget.setCurrentIndex(2)


    ###########################################
    ############# TAB 3 FUNCTIONS #############
    ###########################################

    def T3_fileTable_createRow(self, label, file):
        """Adds new row to the file table on Tab 3"""
        cell_widget = QWidget()
        chk_bx = QtWidgets.QCheckBox()
        chk_bx.setCheckState(QtCore.Qt.Checked)
        chk_bx.setObjectName("checkbox")

        lay_out = QtWidgets.QHBoxLayout(cell_widget)
        lay_out.addWidget(chk_bx)
        lay_out.setAlignment(QtCore.Qt.AlignCenter)
        lay_out.setContentsMargins(0, 0, 0, 0)
        cell_widget.setLayout(lay_out)

        file = QtWidgets.QTableWidgetItem(file)
        file.setFlags(QtCore.Qt.ItemIsEnabled)
        file.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        label = QtWidgets.QTableWidgetItem(str(label)) # Make sure to convert numeric to str
        label.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        inx = self.T3_TableWidget_TestFiles.rowCount()
        self.T3_TableWidget_TestFiles.insertRow(inx)
        self.T3_TableWidget_TestFiles.setCellWidget(inx, 0, cell_widget)
        self.T3_TableWidget_TestFiles.setItem(inx, 1, label)
        self.T3_TableWidget_TestFiles.setItem(inx, 2, file)


    def T3_openFiles(self):
        """Opens single or multiple files for testing data on Tab 3"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Load : SansEC experiment files", "",
                                                "*.xlsx files (*.xlsx);;"
                                                " *.xls files (*xls);; All files (*)",
                                                options=options)
        if files:
            for f in files:
                basename = helper.get_base_filename(f)
                label = helper.parse_label(basename)
                self.test_data[basename] = {'absolute_path': f,
                                            'selected': True,
                                            'features': '',
                                            'label': label}
                self.T3_fileTable_createRow(label=label, file=basename)


    def T3_openDirectory(self):
        """Opens directory of files for testing data on Tab 3"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,
                                                     "Load : SansEC directory with experiment files",
                                                     os.path.expanduser("~"),
                                                     options=options)

        if directory:
            # Grab files that end with .xlsx, .csv, and .xls
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xlsx')
                     or f.endswith('.xls')]

            if files:
                for f in files:
                    basename = helper.get_base_filename(f)
                    label = helper.parse_label(basename)
                    self.test_data[basename] = {'absolute_path': f,
                                                'selected': True,
                                                'features': '',
                                                'label': label}
                    self.T3_fileTable_createRow(label=label, file=basename)


    def T3_loadTrainedModels(self):
        """Loads directory for trained models on Tab 3"""
        if len(self.T3_Label_ExperimentName.text()) == 0:
            helper.messagePopUp(message="Experiment name not specified",
                              informativeText="Please enter experiment name",
                              windowTitle="Error: Missing Information",
                              type="error")
            self.statusBar().showMessage("Error: Missing Information")
            return

        # Saved models for current configuration file
        n_models_saved = 0
        for model_name, model_information in self.config["Models"].items():
            if model_information['path_trained_learner']:
                n_models_saved += 1
            else:
                continue

        if n_models_saved == 0:
            helper.messagePopUp(message="No models trained for experiment %s" % self.config["ExperimentName"],
                                informativeText="Please train at least one model and try again",
                                windowTitle="Error: No Models Trained",
                                type="error")
            self.statusBar().showMessage("Error: No Models Trained")
            return

        # Load directory for files
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,
                                                     "Load : Trained machine learning models directory",
                                                     os.path.join(self.config["SaveDirectory"], "Models"),
                                                     options=options)
        if directory:
            # Grab files that end with .pkl
            model_directory = os.path.join(self.config["SaveDirectory"], "Models")
            models = [m.split('.pkl')[0] for m in os.listdir(model_directory) if m.endswith('.pkl')]

            if models:
                # Remove old column selections from list and update with new models
                self.T3_ListWidget_Models.clear()

                # Make all elements checkable
                self.T3_ListWidget_Models.addItems(models)
                for i in range(self.T3_ListWidget_Models.count()):
                    item = self.T3_ListWidget_Models.item(i)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    self.T3_ListWidget_Models.item(i).setCheckState(QtCore.Qt.Checked)

                # Flag all test_models value in the configuration file to True by default
                for m in models:
                    self.config["Models"][m]["test_model"] = True


    def T3_updateLabels(self):
        """updates all of the labels by matching them to a dictionary

        Parameters
        ----------
        labelDict: dict
            Dictionary with key = file basename and value = file label

        Returns
        -------
        None
        """
        for i in range(self.T3_TableWidget_TestFiles.rowCount()):
            if self.T3_TableWidget_TestFiles.cellWidget(i, 0).findChild(QtWidgets.QCheckBox,
                                                                        "checkbox").checkState() \
                    == QtCore.Qt.Checked:
                basename = self.T3_TableWidget_TestFiles.item(i, 2).text()
                label = self.T3_TableWidget_TestFiles.item(i, 1).text()
                try:
                    if int(label) == float(label):
                        label = int(label)
                    else:
                        label = float(label)
                except:
                    label = float(label)

                self.test_data[basename]['label'] = label


    def T3_beginTesting(self):
        """Does model loading, file ingestion, testing data creation, and model testing on Tab 3"""
        self.T3_ProgressBar_Testing.setRange(0, 0)
        self.T3_Button_BeginTesting.setEnabled(False)

        # Check that at least one model loaded
        if self.T3_ListWidget_Models.count() == 0:
            helper.messagePopUp(message="No models loaded",
                                informativeText="Please load trained models and try again",
                                windowTitle="Error: No Models Loaded",
                                type="error")
            self.statusBar().showMessage("Error: No Models Loaded")
            self.T3_ProgressBar_Testing.setRange(0, 1)
            self.T3_Button_BeginTesting.setEnabled(True)
            return

        # Check that at least one file is loaded
        if self.T3_TableWidget_TestFiles.rowCount() == 0:
            helper.messagePopUp(message="No files loaded",
                                informativeText="Please load files and try again",
                                windowTitle="Error: No Files Selected",
                                type="error")
            self.statusBar().showMessage("Error: No Files Selected")
            self.T3_ProgressBar_Testing.setRange(0, 1)
            self.T3_Button_BeginTesting.setEnabled(True)
            return

        # Check that all files are labeled
        missing_labels, selected_files, selected_models = 0, 0, 0
        for i in range(self.T3_TableWidget_TestFiles.rowCount()):
            if self.T3_TableWidget_TestFiles.cellWidget(i, 0).findChild(QtWidgets.QCheckBox,
                                                                        "checkbox").checkState() \
                    == QtCore.Qt.Checked:
                selected_files += 1

                # Check if label exists
                if not self.T3_TableWidget_TestFiles.item(i, 1).text():
                    missing_labels += 1

        # Create error if missing labels found
        if missing_labels > 0:
            helper.messagePopUp(message="Not all labels filled in for selected files",
                                informativeText="Check selected files and try again",
                                windowTitle="Error: Missing Labels",
                                type="error")
            self.statusBar().showMessage("Error: Missing Labels")
            self.T3_ProgressBar_Testing.setRange(0, 1)
            self.T3_Button_BeginTesting.setEnabled(True)
            self.statusBar().showMessage("Error: Missing Labels")
            self.T3_ProgressBar_Testing.setRange(0, 1)
            self.T3_Button_BeginTesting.setEnabled(True)
            return
        else:
            try:
                self.T3_updateLabels()
            except Exception as e:
                helper.messagePopUp(message="Error labeling file(s) because",
                                    informativeText=str(e),
                                    windowTitle="Error: Labeling File(s)",
                                    type="error")
                self.statusBar().showMessage("Error: Labeling File(s)")
                self.T3_ProgressBar_Testing.setRange(0, 1)
                self.T3_Button_BeginTesting.setEnabled(True)
                return

        for index in range(self.T3_ListWidget_Models.count()):
            if self.T3_ListWidget_Models.item(index).checkState() == QtCore.Qt.Checked:
                selected_models += 1

        if selected_files < 1 or selected_models < 1:
            helper.messagePopUp(message="Less than one model or file selected",
                                informativeText="Check selected files and models and try again",
                                windowTitle="Error: Missing Labels",
                                type="error")
            self.statusBar().showMessage("Error: No selected model/file")
            self.T3_ProgressBar_Testing.setRange(0, 1)
            self.T3_Button_BeginTesting.setEnabled(True)
            return

        # Try and load each selected trained model
        models_to_test, models_failed = {}, 0
        for index in range(self.T3_ListWidget_Models.count()):
            if self.T3_ListWidget_Models.item(index).checkState() == QtCore.Qt.Checked:
                try:
                    model_name = self.T3_ListWidget_Models.item(index).text()
                    models_to_test[model_name] = \
                        helper.load_trained_model(self.config["Models"][model_name]["path_trained_learner"])
                except:
                    models_failed += 1
                    continue

        if models_failed > 0 and models_failed < self.T3_ListWidget_Models.count():
            helper.messagePopUp(message="Warning: %d/%d models failed loading" % (models_failed,
                                                                                  self.T3_ListWidget_Models.count()),
                                informativeText="Press OK to continue",
                                windowTitle="Warning: Some Models Failed Loading",
                                type="warning")
            self.statusBar().showMessage("Warning: Some Models Failed Loading")
            pass

        if models_failed == self.T3_ListWidget_Models.count():
            helper.messagePopUp(message="Error: All %d models failed loading" % self.T3_ListWidget_Models.count(),
                                informativeText="Check saved models and try again",
                                windowTitle="Error: No Models Loaded",
                                type="error")
            self.statusBar().showMessage("Error: No Models Loaded")
            self.T3_Button_BeginTesting.setEnabled(True)
            self.T3_ProgressBar_Testing.setRange(0, 1)
            return

        # Try and load each selected test file and check for same frequencies and columns as training data
        #first, reset data dictionary:
        for basename in self.test_data:
            self.test_data[basename]['features'] = ''

        self.config["TestFiles"] = []

        files_failed = 0
        for i in range(self.T3_TableWidget_TestFiles.rowCount()):
            if self.T3_TableWidget_TestFiles.cellWidget(i, 0).findChild(QtWidgets.QCheckBox,
                                                                        "checkbox").checkState() \
                    == QtCore.Qt.Checked:
                basename = self.T3_TableWidget_TestFiles.item(i, 2).text()
                try:
                    self.test_data[basename]['features'] = helper.load(self.test_data[basename]["absolute_path"])
                except:
                    files_failed += 1
                    continue

                # Update configuration file if file not already added
                if self.test_data[basename]['absolute_path'] not in self.config["TestFiles"]:
                    self.config["TestFiles"].append(self.test_data[basename]['absolute_path'])

        if files_failed > 0 and files_failed < self.T3_TableWidget_TestFiles.rowCount():
            helper.messagePopUp(message="Warning: %d/%d files failed loading" % (files_failed,
                                                                                 self.T3_TableWidget_TestFiles.rowCount()),
                                informativeText="Press OK to continue",
                                windowTitle="Warning: Some Files Failed Loading",
                                type="warning")
            self.statusBar().showMessage("Warning: Some Files Failed Loading")
            pass

        if files_failed == self.T3_TableWidget_TestFiles.rowCount():
            helper.messagePopUp(message="Error: All %d files failed loading" % self.T3_TableWidget_TestFiles.rowCount(),
                                informativeText="Check loaded files and try again",
                                windowTitle="Error: No Files Loaded",
                                type="error")
            self.statusBar().showMessage("Error: No Files Loaded")
            self.T3_Button_BeginTesting.setEnabled(True)
            self.T3_ProgressBar_Testing.setRange(0, 1)
            return

        # Check that all test files have the same frequencies and features as training data
        valid_test_data, invalid_files = {}, []
        for key, value in self.test_data.items():
            if value['features'] is not '':

                # Get frequencies and features for current file
                testing_freqs = value['features']['Freq'].tolist()
                testing_feats = value['features'].columns.tolist()

                # Remove columns that are usually constant
                for c in constants.COLUMNS_TO_DROP:
                    testing_feats.pop(testing_feats.index(c))

                try:
                    if helper.check_testing_freqs_and_features(testing_freqs=testing_freqs,
                                                               testing_feats=testing_feats,
                                                               training_freqs=self.config['Freqs'],
                                                               training_feats=self.config['Columns']):
                        valid_test_data[key] = value
                    else:
                        invalid_files.append(key)
                except:
                    invalid_files.append(key)
                    continue

        # Create warning and error messages if some files failed comparison to training data
        files_failed = len(invalid_files)
        if files_failed > 0 and files_failed < self.T3_TableWidget_TestFiles.rowCount():
            helper.messagePopUp(message="Warning: %d/%d files were dissimilar to training data" % (files_failed,
                                                                                 self.T3_TableWidget_TestFiles.rowCount()),
                                informativeText="Press OK to continue",
                                windowTitle="Warning: Some Files Dissimilar to Training Data",
                                type="warning")
            self.statusBar().showMessage("Warning: Some Files Dissimilar to Training Data")
            pass

        if files_failed == self.T3_TableWidget_TestFiles.rowCount():
            helper.messagePopUp(message="Error: All %d files dissimilar to training data" % self.T3_TableWidget_TestFiles.rowCount(),
                                informativeText="Check loaded files and try again",
                                windowTitle="Error: All Files Dissimilar to Training Data",
                                type="error")
            self.statusBar().showMessage("Error: All Files Dissimilar to Training Data")
            self.T3_Button_BeginTesting.setEnabled(True)
            self.T3_ProgressBar_Testing.setRange(0, 1)
            return

        # Create input for machine learning models and test for similar frequencies and columns as training data
        self.T3_TextBrowser_AnalysisLog.append("Creating testing data set...")
        try:
            min_freq, max_freq = min(self.config['Freqs']), max(self.config['Freqs'])
            min_idx = helper.index_for_freq(self.config['Freqs'], min_freq)
            max_idx = helper.index_for_freq(self.config['Freqs'], max_freq)
            self.test_input = helper.tranpose_and_append_columns(data=valid_test_data,
                                                            freqs=self.config['Freqs'],
                                                            columns=self.config['Columns'],
                                                            idx_freq_ranges=(min_idx, max_idx))
            X, y = self.test_input.iloc[:, :-1], self.test_input.iloc[:, -1]

        except Exception as e:
            helper.messagePopUp(message="Error: Creating testing dataset because",
                                informativeText=str(e),
                                windowTitle="Error: Creating Testing Dataset",
                                type="error")
            self.statusBar().showMessage("Error: Creating Testing Dataset")
            self.T3_Button_BeginTesting.setEnabled(True)
            self.T3_ProgressBar_Testing.setRange(0, 1)
            return

        # Deploy selected trained models on test data set
        e = ps.deploy_models(X, y, models_to_test, self.T3_TextBrowser_AnalysisLog,
                                              self.config)
        self.T3_Button_BeginTesting.setEnabled(True)
        self.T3_ProgressBar_Testing.setRange(0, 1)

        if e is not None:
            raise e


    def T3_generateReport(self):
        """Generates summary report for analysis on Tab 3"""
        # Check if any models have testing metrics
        testing_metric_found = False
        for model_information in self.config['Models'].values():
            if model_information['test_score'] is not None:
                testing_metric_found = True
            else:
                continue
        if not testing_metric_found:
            helper.messagePopUp(message="Error: No models tested",
                                informativeText="Deploy trained models and try again",
                                windowTitle="Error: No Models Tested",
                                type="error")
            self.statusBar().showMessage("Error: No Models Tested")
            return

        # Generate report
        self.T3_TextBrowser_AnalysisLog.append("\nGenerating summary report for analysis (%s), please wait...\n" % \
                                               self.config["ExperimentName"])
        e = self.generateReport()

        if e is not None:
            raise e


    @nongui
    def generateReport(self):
        """Helper function that generates report in separate thread"""
        # Generate feature importance analysis results
        try:
            importances_generated, predictions_generated = False, False
            if self.learner_input is not None:
                self.T3_TextBrowser_AnalysisLog.append("Running feature importance analysis...")
                try:
                    var_names, importances = ps.feature_importance_analysis(X=self.learner_input.iloc[:, :-1],
                                                                            y=self.learner_input.iloc[:, -1],
                                                                            configuration_file=self.config)
                    importances_generated = True
                    self.T3_TextBrowser_AnalysisLog.append("\tFeature analysis finished")
                except Exception as e:
                    self.T3_TextBrowser_AnalysisLog.append("\tFeature analysis failed because %s\n" % str(e))
                    pass

            try:
                self.T3_TextBrowser_AnalysisLog.append("\nGenerating predictions file...")
                ps.summary_model_predictions(y_test=self.test_input.iloc[:, -1], configuration_file=self.config)
                predictions_generated = True
                self.T3_TextBrowser_AnalysisLog.append("\tPredictions file finished")
            except Exception as e:
                return e

            helper.generate_summary_report(configuration_file=self.config, var_names=var_names, importances=importances)
            self.T3_TextBrowser_AnalysisLog.append("\nSummary report finished and saved as %s" % \
                                                   os.path.join(os.path.join(self.config["SaveDirectory"], "Summary"),
                                                   "analysis_summary.txt"))
            if importances_generated:
                self.T3_TextBrowser_AnalysisLog.append("\nFeature analysis report saved as %s" % \
                                                       os.path.join(os.path.join(self.config["SaveDirectory"], "Summary"),
                                                       "feature_importance_analysis.txt"))

            if predictions_generated:
                self.T3_TextBrowser_AnalysisLog.append("\nModel predictions report saved as %s" % \
                                                       os.path.join(os.path.join(self.config["SaveDirectory"], "Summary"),
                                                                    "summary_model_predictions.txt"))

            self.T3_TextBrowser_AnalysisLog.append("\n --- Analysis Finished ---\n")
        except Exception as e:
            return e
        return None


    ###########################################
    ########### MENU ITEM FUNCTIONS ###########
    ###########################################

    def saveConfigurationFile(self):
        """Saves configuration file at any given moment in app"""
        if len(self.config["ExperimentName"]) == 0:
            helper.messagePopUp(message="Experiment name not specified",
                                informativeText="Please enter experiment name and save configuration file",
                                windowTitle="Error: Missing Information",
                                type="error")
            self.statusBar().showMessage("Error: Missing Information")
            return

        try:
            # Save configuration file
            json.dump(self.config, open(os.path.join(self.config['SaveDirectory'], 'configuration.json'), 'w'))

            # Update status bar
            self.statusBar().showMessage("Successfully saved configuration file for experiment %s" % \
                                         self.config['ExperimentName'])

            # Update experiment name on Tab 2 and Tab 3
            self.T2_Label_ExperimentName.setText(self.config['ExperimentName'])
            self.T3_Label_ExperimentName.setText(self.config['ExperimentName'])
        except Exception as e:
            helper.messagePopUp(message="Error saving configuration file because",
                                informativeText=str(e),
                                windowTitle="Error: Saving Configuration File",
                                type="error")
            self.statusBar().showMessage("Error: Saving Configuration File")
            return


    def loadConfigurationFile(self):
        """Load configuration file at any given moment in app"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file = QFileDialog.getOpenFileName(self, "Load : Configuration file", "",
                                            "Configuration files (*.json)",
                                            options=options)
        if file[0]:
            try:
                self.config = json.load(open(file[0], 'r'))
                self.statusBar().showMessage("Configuration file loaded for %s" % self.config["ExperimentName"])
                self.T1_Label_ExperimentName.setText(self.config["ExperimentName"])
                self.T2_Label_ExperimentName.setText(self.config['ExperimentName'])
                self.T3_Label_ExperimentName.setText(self.config['ExperimentName'])

                # Try and load training data set
                try:
                    self.learner_input = pd.read_csv(os.path.join(self.config["SaveDirectory"], "training_data.csv"))
                except:
                    self.learner_input = None

            except Exception as e:
                helper.messagePopUp(message="Error loading configuration file %s because" % file[0],
                                    informativeText=str(e),
                                    windowTitle="Error: Loading Configuration File",
                                    type="error")
                self.statusBar().showMessage("Error: Loading Configuration File")
                return


    def viewDocumentation(self):
        """Loads user's manual documentation for viewing"""
        operating_system = platform.system()
        try:
            if operating_system == "Windows":
                status = os.system("""start "" /max "sbh-qtapp\qtapp\docs\MLM_manual.pdf" """)
            else:
                status = os.system("open sbh-qtapp/qtapp/docs/MLM_manual.pdf")

            if status != 0:
                helper.messagePopUp(message="Error loading documentation on detected OS %s" % operating_system,
                                    informativeText="Please check docs folder for copy of .pdf file",
                                    windowTitle="Error: Loading Documentation",
                                    type="error")
                self.statusBar().showMessage("Error: Loading Documentation")
                return

        except Exception as e:
            helper.messagePopUp(message="Error loading documentation on detected OS %s because" % operating_system,
                                informativeText=str(e),
                                windowTitle="Error: Loading Documentation - Check docs folder for copy of .pdf file",
                                type="error")
            self.statusBar().showMessage("Error: Loading Documentation - Check docs folder for copy of .pdf file")
            return


    def checkFiles(self):
        """Runs the file checker script on user selected directory"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,
                    "Select : SansEC directory with experiment files", os.path.expanduser("~"), options=options)

        if directory:
            self.statusBar().showMessage("Checking files in directory %s, please wait..." % directory)
            status = split_files(directory)
            if not status[0]:
                helper.messagePopUp(message="Error Checking Files",
                                    informativeText="Unable to check files in %s because %s" % (directory, status[1]),
                                    windowTitle="Error: Checking Files",
                                    type="error")
                self.statusBar().showMessage("Error checking files in %s" % directory)
                return
            else:
                self.statusBar().showMessage("Successfully split %d files into %d unique groupings" % (status[-1], status[-2]))


    def exitApplication(self):
        """Exits application"""
        answer = helper.messagePopUp(message="Are you sure you want to exit?",
                                     informativeText="Please make sure to save configuration file",
                                     windowTitle="Exit Application",
                                     type="warning",
                                     question=True)
        if answer == QMessageBox.Yes:
            sys.exit(0)
        else:
            pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui().showMaximized()
    sys.exit(app.exec_())