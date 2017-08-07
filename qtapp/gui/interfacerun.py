# -*- coding: utf-8 -*-

# Standard imports
from __future__ import division

import csv
from io import StringIO
import json
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import os
from PyQt5 import QtCore, QtGui,  uic, QtWidgets, Qt
from PyQt5.QtWidgets import (QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox,
                             QWidget, QTableWidgetItem, QFileDialog)
import sys
from threading import Thread
import time
import traceback

# Imports from qtapp

try:
    from qtapp.gui.hyperparameters_ui import HyperparametersUI
    from qtapp.gui.dynamicmplcanvas import DynamicMplCanvas
    from qtapp.model import pipeline_specifications as ps
    from qtapp.utils import constants, helper
    from qtapp.utils.errorhandling import errorDialogOnException
    from qtapp.utils.nonguiwrapper import nongui
except:
    from dynamicmplcanvas import DynamicMplCanvas
    from hyperparameters_ui import HyperparametersUI
    from model import pipeline_specifications as ps
    from utils import constants, helper
    from utils.errorhandling import errorDialogOnException
    from utils.nonguiwrapper import nongui


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

        # Force tab widget to open on Tab 1
        self.TabWidget.setCurrentIndex(0)

        # Create data structure to hold information about files and configuration file
        self.data = {}
        self.config = helper.create_blank_config()
        self.learner_input = None
        self.statusBar().showMessage('Load Files or Configuration File to Begin...')

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

        self.columns = []
        self.freqs = []
        self.min_freq, self.max_freq = 0, 100
        self.n_files_selected = 0
        self.dataset_created = False

        #self.T1_TableWidget_Files.setFocusPolicy(QtCore.Qt.NoFocus)

        # Ensure each item in the ListWidget has an unchecked box next to it
        for index in range(self.T2_ListWidget_Models.count()):
            self.T2_ListWidget_Models.item(index).setCheckState(QtCore.Qt.Unchecked)

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

        ## MENU ITEM BUTTONS ##

        # Connect the menu item 'Save Configuration File'
        self.FileItem_SaveConfigurationFile.triggered.connect(self.saveConfigurationFile)

        # Connect the menu item 'Load Configuration File'
        self.FileItem_LoadConfigurationFile.triggered.connect(self.loadConfigurationFile)

        # Connect the menu item 'Exit'
        self.FileItem_Exit.triggered.connect(self.exitApplication)

        ## TAB 1 BUTTONS ##

        # Connect 'Load Files...' and 'Load Directory...' buttons
        self.T1_Button_LoadFiles.clicked.connect(self.T1_openFiles)
        self.T1_Button_LoadDirectory.clicked.connect(self.T1_openDirectory)

        # Connect 'Ingest Files' button
        self.T1_Button_IngestFiles.clicked.connect(self.T1_LoadingPopupIngest)

        # Connect 'Create Dataset' button
        self.T1_Button_CreateDataset.clicked.connect(self.T1_createDataset)

        # Connect 'Save Configuration File' button
        self.T1_Button_SaveConfigurationFile.clicked.connect(self.T1_saveConfigurationFile)

        ## TAB 2 BUTTONS ##

        # Connect 'Set Parameters' button
        self.T2_Button_SetParameters.clicked.connect(self.T2_setParameters)

        # Connect 'Begin Training' button
        self.T2_Button_BeginTraining.clicked.connect(self.T2_beginTraining)

        # Connect the 'Save Configuration File' button
        self.T2_Button_SaveConfigurationFile.clicked.connect(self.saveConfigurationFile)

        # Connect the 'Clear Log' button
        self.T2_Button_ClearLog.clicked.connect(self.T2_TextBrowser_AnalysisLog.clear)

        ## TAB 3 BUTTONS ##

        # Connect the 'Clear Log' button
        self.T2_Button_ClearLog.clicked.connect(self.T3_TextBrowser_AnalysisLog.clear)


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
                    if self.data[baseName]["features"] is not None:
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
            self.data[Basename]["Label"] = toDict[Basename]
        self.T1_UpdateFileList(toDict)


    def T1_setLabelsByName(self):
        """attempts to label each row by a the file name

        Parameters
        ----------

        Returns
        -------
        """
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
            self.statusBar().showMessage("""Click 'Create Dataset' to update frequencies""")


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
        if self.T1_TableWidget_Files.rowCount() == 0:
            helper.messagePopUp(message="No files loaded",
                                informativeText="Please load files and try again",
                                windowTitle="Error: No Files Selected",
                                type="error")
            self.statusBar().showMessage("Error: No Files Selected")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileNames(self, "Load: SansEC experiment files", "",
                                                "*.csv files (*.csv)",
                                                options=options)
        if file:
            try:
                self.setLabelsByFile(file)
            except Exception as e:
                helper.messagePopUp(message="Error loading label file because",
                                  informativeText=str(e),
                                  windowTitle="Error: Loading Label File",
                                  type="error")
                self.statusBar().showMessage("Error: Loading Label File")
                return

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
                    self.T1_fileTable_createRow(label="", file=basename)
                    self.data[basename] = {'absolute_path': f, 'features': None, 'label': None, 'selected': True}


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
                        self.T1_fileTable_createRow(label="", file=basename)
                        self.data[basename] = {'absolute_path': f, 'features': None, 'label': None, 'selected': True}


    def T1_LoadingPopupIngest(self):

        """Sets up ingestion with a loading bar

        Parameters
        ----------

        Returns
        -------
        """
        # Define initial variables
        progress = QtWidgets.QProgressDialog(parent=self)
        progress.setCancelButton(None)
        progress.setLabelText('Ingesting files...')
        progress.setWindowTitle("Loading")
        progress.setMinimum(0)
        progress.setMaximum(0)
        progress.forceShow()
        message, informativeText, windowTitle, type = self.T1_ingestFiles()
        if message is not None:
            helper.messagePopUp(message, informativeText, windowTitle, type)
        progress.cancel()


    @nongui
    def T1_ingestFiles(self):
        """Does the major data ingestion based on prestaged setting

        Parameters
        ----------

        Returns
        -------
        """
        # Define initial variables
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
                    return message, informativeText, windowTitle, type

        # COMMENT HERE
        if checkcnt < 2:
            allOk = False
            message="%d file(s) selected" % checkcnt
            informativeText="select more files. Must be at least 2."
            windowTitle="Error: Missing Labels"
            type="error"
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

            # Check if at least one frequency and column are selected
            if len(self.freqs) == 0:
                self.statusBar().showMessage('Tip: Exclude files with different frequencies and try again')
                message="No common frequencies found across %d selected files" % self.n_files_selected,
                informativeText="Check selected files and try again",
                windowTitle="Error: No Common Frequencies Across Files",
                type="error"
                self.statusBar().showMessage("Error: No Common Frequencies Across Files")
                return message, informativeText, windowTitle, type

            if len(self.columns) == 0:
                self.statusBar().showMessage('Tip: Exclude files with different features/columns and try again')
                message="No common features/columns found across %d selected files" % self.n_files_selected
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
            return None, None, None, None


    def T1_createDataset(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        # Make sure at least one file selected
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
        self.dataset_created = True

        # Update configuration file
        self.config['TrainSamples'], self.config['TrainFeatures'] = self.learner_input.shape
        self.config['Freqs'], self.config['Columns'] = self.freqs[min_idx:max_idx + 1], cols_to_use
        self.config['LearningTask'] = "Regressor" if self.T1_RadioButton_ContinuousLabels.isChecked() else "Classifier"


    def T1_saveConfigurationFile(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
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

        # Set learning task based on label type (categorical = Classifier, continuous = Regressor)
        self.config['LearningTask'] = "Regressor" if self.T1_RadioButton_ContinuousLabels.isChecked() else "Classifier"

        # 'Safely' go up two directories and save all files related to current experiment run in 'sbh-qtapp' +
        # experiment name directory
        self.config['SaveDirectory'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
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

        # Force tab widget to open on Tab 2
        self.TabWidget.setCurrentIndex(1)


    def T2_setParameters(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
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
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
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
                helper.messagePopUp(message="Need at least three samples per class for classifier models",
                                    informativeText="%d classes with less than three samples" % n_classes_lt3,
                                    windowTitle="Error: Not Enough Samples Per Class",
                                    type = "error")
                self.statusBar().showMessage("Error: Not Enough Samples Per Class")
                return
        else:
            if self.learner_input.shape[0] < 3:
                helper.messagePopUp(message="Need at least three samples for regression models",
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

        # If automatically tune
        if automatically_tune:
            # Start in separate thread
            Thread(target=ps.automatically_tune, args=(X, y, learner_type, standardize, feature_reduction_method,
                                                       training_method, self.T2_TextBrowser_AnalysisLog,
                                                       self.config)).start()

        # Otherwise train using holdout or cross-validation
        else:
            if training_method == "holdout":
                Thread(target=ps.holdout, args=(X, y, learner_type, standardize, feature_reduction_method,
                                                self.T2_TextBrowser_AnalysisLog, self.config)).start()
            else:
                Thread(target=ps.cv, args=(X, y, learner_type, standardize,
                                           feature_reduction_method, self.T2_TextBrowser_AnalysisLog,
                                           self.config)).start()


    def T3_ingestFile(self, filepath):

        """Does the major data ingestion based on prestaged setting

        Parameters
        ----------

        Returns
        -------
        """
        features = helper.load(file=filepath)
        freq = set(features["Freq"])
        feat = set(features.keys())
        if self.freqs < freq and self.columns < feat:
            pass



    def T3_fileTable_createRow(self, label, file):
        """Adds new row to the file table

        Parameters
        ----------

        Returns
        -------
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
        self.T3_TableWidget_TestFiles.insertRow(inx)
        self.T3_TableWidget_TestFiles.setCellWidget(inx, 0, cell_widget)
        self.T3_TableWidget_TestFiles.setItem(inx, 1, label)
        self.T3_TableWidget_TestFiles.setItem(inx, 2, file)

    def T3_openFiles(self):
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
                    self.T3_fileTable_createRow(label="", file=basename)
                    self.data[basename] = {'absolute_path': f, 'features': None, 'label': None, 'selected': True}

    def T3_openDirectory(self):
        """Clicked action for 'Load Directory...' button

        Parameters
        ----------

        Returns
        -------
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,
                                                     "Load : SansEC directory with experiment files",
                                                     os.path.expanduser("~"), options=options)

        if directory:
            # Grab files that end with .xlsx, .csv, and .xls
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xlsx')
                     or f.endswith('.csv') or f.endswith('.xls')]

            if files:
                # Add labels and files to table
                for f in files:
                    basename = helper.get_base_filename(f)
                    if basename not in self.data:
                        self.T3_fileTable_createRow(label="", file=basename)
                        self.data[basename] = {'absolute_path': f, 'features': None, 'label': None, 'selected': True}

    def saveConfigurationFile(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
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
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file = QFileDialog.getOpenFileName(self, "Load: Configuration file", "",
                                            "Configuration files (*.json)",
                                            options=options)
        if file[0]:
            try:
                self.config = json.load(open(file[0], 'r'))
                self.statusBar().showMessage("Configuration file loaded for %s" % self.config["ExperimentName"])
                self.T1_Label_ExperimentName.setText(self.config["ExperimentName"])
                self.T2_Label_ExperimentName.setText(self.config['ExperimentName'])
                self.T3_Label_ExperimentName.setText(self.config['ExperimentName'])
            except Exception as e:
                helper.messagePopUp(message="Error loading configuration file %s because" % file[0],
                                    informativeText=str(e),
                                    windowTitle="Error: Loading Configuration File",
                                    type="error")
                self.statusBar().showMessage("Error: Loading Configuration File")
                return



    def exitApplication(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
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