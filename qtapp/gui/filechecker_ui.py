# -*- coding: utf-8 -*-

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QGridLayout, QSizePolicy, QWidget, QDialog, QPushButton
import sys


try:
    from qtapp.utils.filechecker import split_files
    from qtapp.utils import helper
except:
    from utils.filechecker import split_files
    from utils import helper


class FileCheckerUI(object):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, statusBar):
        # Define QDialog as modal application window
        self.dialog = QDialog()
        self.dialog.resize(450, 100)
        self.dialog.setObjectName("Dialog")
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialog.setModal(True)
        self.statusBar = statusBar

        # Set window title
        self.dialog.setWindowTitle("File Checker")

        # Define grid for layout
        self.gridLayout = QGridLayout(self.dialog)
        self.gridLayout.setObjectName("gridLayout")

        # Set size policies
        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHorizontalStretch(1)

        # Set font
        font = QtGui.QFont().setPointSize(12)

        # 1. Directory

        ## button
        self.button_directory = QPushButton(self.dialog)
        self.button_directory.setObjectName("button_directory")
        self.button_directory.setText("Select Directory for File Checking")
        self.gridLayout.addWidget(self.button_directory, 0, 0, 2, 2)
        self.button_directory.clicked.connect(self.check)

        # Show QDialog
        self.dialog.exec_()


    def check(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """

        self.dialog.close()
        options = QFileDialog.Options()
        options |= QFileDialog.DontResolveSymlinks | QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self.dialog,
                    "Load : SansEC directory with experiment files", os.path.expanduser("~"), options=options)

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
                self.statusBar().showMessage("Successfully split %d files into %d unique groupings" % (status[-2], status[-11]))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FileCheckerUI('')
    sys.exit(app.exec_())
