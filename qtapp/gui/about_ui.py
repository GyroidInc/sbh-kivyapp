from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, QSizePolicy, QWidget, QDialog, QPushButton
import sys

try:
    from qtapp.utils import constants as c
except:
    from utils import constants as c

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

        ## label
        self.label_version = QLabel(self.dialog)
        self.label_version.setAlignment(QtCore.Qt.AlignRight)
        self.label_version.setText("Version:")
        self.gridLayout.addWidget(self.label_version, 0, 0, 1, 1)

        ## input
        self.input_version = QLabel(self.dialog)
        self.input_version.setAlignment(QtCore.Qt.AlignLeft)
        self.input_version.setText(c.VERSION)
        self.gridLayout.addWidget(self.input_version, 0, 1, 1, 1)

        ## label
        self.label_author = QLabel(self.dialog)
        self.label_author.setAlignment(QtCore.Qt.AlignRight)
        self.label_author.setText("Author:")
        self.gridLayout.addWidget(self.label_author, 1, 0, 1, 1)

        ## input
        self.input_author = QLabel(self.dialog)
        self.input_author.setAlignment(QtCore.Qt.AlignLeft)
        self.input_author.setText(c.AUTHOR)
        self.gridLayout.addWidget(self.input_author, 1, 1, 1, 1)

        ## label
        self.label_builtwith = QLabel(self.dialog)
        self.label_builtwith.setAlignment(QtCore.Qt.AlignRight)
        self.label_builtwith.setText("Built With:")
        self.gridLayout.addWidget(self.label_builtwith, 2, 0, 1, 1)

        ## input
        self.input_builtwith = QLabel(self.dialog)
        self.input_builtwith.setAlignment(QtCore.Qt.AlignLeft)
        self.input_builtwith.setText(c.BUILT_WITH)
        self.gridLayout.addWidget(self.input_builtwith, 2, 1, 1, 1)

        # Spacer for adding space between hyperparameters and save button
        self.spacer = QtWidgets.QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.okButton = QPushButton(self.dialog)
        self.okButton.setText("Ok")

        # Add spacer and ok button
        self.gridLayout.addItem(self.spacer, 3, 1, 1, 1)
        self.gridLayout.addWidget(self.okButton, 4, 0, 1, 3, QtCore.Qt.AlignHCenter)
        self.okButton.clicked.connect(self.dialog.close)

        # Show QDialog
        self.dialog.exec_()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AboutUI()
    sys.exit(app.exec_())