from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QComboBox, QLineEdit, QGridLayout, QSizePolicy, QMessageBox, QWidget, QDialog, QPushButton

class AboutUI(object):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self):
        # Define QDialog as modal application window
        self.dialog = QDialog()
        self.dialog.setObjectName("Dialog")
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialog.setModal(True)

        # Set window title
        self.dialog.setWindowTitle("About")

        # Define grid for layout
        self.gridLayout = QGridLayout(self.dialog)
        self.gridLayout.setObjectName("gridLayout")

        # Set size policies
        #self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        #self.sizePolicy.setHorizontalStretch(1)

        ## label
        self.label_author = QLabel(self.dialog)
        self.label_author.setAlignment(QtCore.Qt.AlignRight)
        self.label_author.setText("Author")
        self.gridLayout.addWidget(self.label_author, 0, 0, 1, 1)

        ## input
        self.input_author = QLabel(self.dialog)
        self.input_author.setAlignment(QtCore.Qt.AlignCenter)
        self.input_author.setText("Gyroid, Inc.")
        self.gridLayout.addWidget(self.input_author, 0, 1, 1, 1)