from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QLabel, QComboBox, QLineEdit, QGridLayout, QSizePolicy, QMessageBox, QWidget, QDialog, QPushButton
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, Matern, RationalQuadratic, RBF
import sys


class HyperparametersUI(object):
    """ADD

    Parameters
    ----------

    Returns
    -------
    """
    def __init__(self, model=None, configuration_file=None):
        # Define QDialog as modal application window
        self.dialog = QDialog()
        self.dialog.setObjectName("Dialog")
        self.dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        self.dialog.setModal(True)
        self.model = model
        self.type = configuration_file['LearningTask']
        self.configuration_file = configuration_file

        # Set window title
        self.dialog.setWindowTitle(self.model + " : " + self.type)

        # Define grid for layout
        self.gridLayout = QGridLayout(self.dialog)
        self.gridLayout.setObjectName("gridLayout")

        # Set size policies
        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHorizontalStretch(1)

        # Set font (ADD THIS TO EACH WIDGET?)
        font = QtGui.QFont().setPointSize(12)

        # Validators
        self.IntValidator = QtGui.QIntValidator()
        self.DoubleValidator = QtGui.QDoubleValidator()

        # Spacer for adding space between hyperparameters and save button
        self.spacer = QtWidgets.QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Add 'Save' or 'Close' button
        if model == "LinearModel":
            self.closeButton = QPushButton(self.dialog)
            self.closeButton.setObjectName("closeButton")
            self.closeButton.setText("Close")
        else:
            self.saveButton = QPushButton(self.dialog)
            self.saveButton.setObjectName("saveButton")
            self.saveButton.setText("Save")

        # Load based on model
        if model == "ExtraTrees" or model == "RandomForest":
            # 1. n_estimators

            ## label
            self.label_n_estimators = QLabel(self.dialog)
            self.label_n_estimators.setAlignment(QtCore.Qt.AlignRight)
            self.label_n_estimators.setObjectName("label_n_estimators")
            self.label_n_estimators.setText("Number of Trees:")
            self.gridLayout.addWidget(self.label_n_estimators, 0, 0, 1, 1)

            ## input
            self.input_n_estimators = QLineEdit(self.dialog)
            self.input_n_estimators.setValidator(self.IntValidator)
            self.input_n_estimators.setAlignment(QtCore.Qt.AlignCenter)
            self.input_n_estimators.setObjectName("input_n_estimators")
            self.input_n_estimators.setText('10')
            self.gridLayout.addWidget(self.input_n_estimators, 0, 1, 1, 1)

            # 2. max_features

            ## label
            self.label_max_features = QLabel(self.dialog)
            self.label_max_features.setAlignment(QtCore.Qt.AlignRight)
            self.label_max_features.setObjectName("label_max_features")
            self.label_max_features.setText("Maximum Features (p):")
            self.gridLayout.addWidget(self.label_max_features, 1, 0, 1, 1)

            ## input
            self.input_max_features = QComboBox()
            self.input_max_features.setObjectName("input_max_features")
            self.input_max_features.addItems(["Sqrt(p)", "Log2(p)", "All"])
            self.input_max_features.setEditable(True)
            self.input_max_features.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
            self.input_max_features.lineEdit().setReadOnly(True)
            self.gridLayout.addWidget(self.input_max_features, 1, 1, 1, 1)

            # 3. criterion

            ## label
            self.label_criterion = QLabel(self.dialog)
            self.label_criterion.setAlignment(QtCore.Qt.AlignRight)
            self.label_criterion.setObjectName("label_criterion")
            self.label_criterion.setText("Criterion:")
            self.gridLayout.addWidget(self.label_criterion, 2, 0, 1, 1)

            ## input
            self.input_criterion = QComboBox()
            self.input_criterion.setObjectName("input_criterion")
            if self.type == "Regressor":
                self.input_criterion.addItems(["Mean Squared Error", "Mean Absolute Error"])
            else:
                self.input_criterion.addItems(["Gini Index", "Entropy"])
            self.input_criterion.setEditable(True)
            self.input_criterion.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
            self.input_criterion.lineEdit().setReadOnly(True)
            self.gridLayout.addWidget(self.input_criterion, 2, 1, 1, 1)

            # Add spacer and save button
            self.gridLayout.addItem(self.spacer, 3, 1, 1, 1)
            self.gridLayout.addWidget(self.saveButton, 4, 0, 1, 3, QtCore.Qt.AlignHCenter)
            self.saveButton.clicked.connect(self.action)


        elif model == "GaussianProcess":
            # 1. kernel

            ## label
            self.label_kernel = QLabel(self.dialog)
            self.label_kernel.setAlignment(QtCore.Qt.AlignRight)
            self.label_kernel.setObjectName("label_kernel")
            self.label_kernel.setText("Kernel:")
            self.gridLayout.addWidget(self.label_kernel, 0, 0, 1, 1)

            ## input
            self.input_kernel = QComboBox(self.dialog)
            self.input_kernel.setObjectName("input_kernel")
            self.input_kernel.addItems(["Radial Basis Function",
                                        "Dot Product",
                                        "Matern",
                                        "Rational Quadratic",
                                        "Exp-Sine Squared",
                                        "Constant"])
            self.input_kernel.setEditable(True)
            self.input_kernel.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
            self.input_kernel.lineEdit().setReadOnly(True)
            self.gridLayout.addWidget(self.input_kernel, 0, 1, 1, 1)

            # Add spacer and save button
            self.gridLayout.addItem(self.spacer, 2, 1, 1, 1)
            self.gridLayout.addWidget(self.saveButton, 3, 0, 1, 3, QtCore.Qt.AlignHCenter)
            self.saveButton.clicked.connect(self.action)


        elif model == "GradientBoostedTrees":
            # 1. n_estimators

            ## label
            self.label_n_estimators = QLabel(self.dialog)
            self.label_n_estimators.setAlignment(QtCore.Qt.AlignRight)
            self.label_n_estimators.setObjectName("label_n_estimators")
            self.label_n_estimators.setText("Number of Trees:")
            self.gridLayout.addWidget(self.label_n_estimators, 0, 0, 1, 1)

            ## input
            self.input_n_estimators = QLineEdit(self.dialog)
            self.input_n_estimators.setValidator(self.IntValidator)
            self.input_n_estimators.setAlignment(QtCore.Qt.AlignCenter)
            self.input_n_estimators.setObjectName("input_n_estimators")
            self.input_n_estimators.setText('100')
            self.gridLayout.addWidget(self.input_n_estimators, 0, 1, 1, 1)

            # 2. learning_rate

            ## label
            self.label_learning_rate = QLabel(self.dialog)
            self.label_learning_rate.setAlignment(QtCore.Qt.AlignRight)
            self.label_learning_rate.setObjectName("label_learning_rate")
            self.label_learning_rate.setText("Learning Rate:")
            self.gridLayout.addWidget(self.label_learning_rate, 1, 0, 1, 1)

            ## input
            self.input_learning_rate = QLineEdit(self.dialog)
            self.input_learning_rate.setValidator(self.DoubleValidator)
            self.input_learning_rate.setAlignment(QtCore.Qt.AlignCenter)
            self.input_learning_rate.setObjectName("input_learning_rate")
            self.input_learning_rate.setText('.01')
            self.gridLayout.addWidget(self.input_learning_rate, 1, 1, 1, 1)

            # 3. subsample

            ## label
            self.label_subsample = QLabel(self.dialog)
            self.label_subsample.setAlignment(QtCore.Qt.AlignRight)
            self.label_subsample.setObjectName("label_subsample")
            self.label_subsample.setText("Row Subsampling:")
            self.gridLayout.addWidget(self.label_subsample, 2, 0, 1, 1)

            ## input
            self.input_subsample = QLineEdit(self.dialog)
            self.input_subsample.setValidator(self.DoubleValidator)
            self.input_subsample.setAlignment(QtCore.Qt.AlignCenter)
            self.input_subsample.setObjectName("input_subsample")
            self.input_subsample.setText('1')
            self.gridLayout.addWidget(self.input_subsample, 2, 1, 1, 1)

            # 4. max_depth

            ## label
            self.label_max_depth = QLabel(self.dialog)
            self.label_max_depth.setAlignment(QtCore.Qt.AlignRight)
            self.label_max_depth.setObjectName("label_max_depth")
            self.label_max_depth.setText("Maximum Tree Depth:")
            self.gridLayout.addWidget(self.label_max_depth, 3, 0, 1, 1)

            ## input
            self.input_max_depth = QLineEdit(self.dialog)
            self.input_max_depth.setValidator(self.IntValidator)
            self.input_max_depth.setAlignment(QtCore.Qt.AlignCenter)
            self.input_max_depth.setObjectName("input_max_depth")
            self.input_max_depth.setText('3')
            self.gridLayout.addWidget(self.input_max_depth, 3, 1, 1, 1)

            # 5. loss

            ## label
            self.label_loss = QLabel(self.dialog)
            self.label_loss.setAlignment(QtCore.Qt.AlignRight)
            self.label_loss.setObjectName("label_loss")
            self.label_loss.setText("Loss Function:")
            self.gridLayout.addWidget(self.label_loss, 4, 0, 1, 1)

            ## input
            self.input_loss = QComboBox()
            self.input_loss.setObjectName("input_loss")
            if self.type == "Regressor":
                self.input_loss.addItems(["Least Squares",
                                          "Huber"])
            else:
                self.input_loss.addItems(["Deviance",
                                          "Exponential"])
            self.input_loss.setEditable(True)
            self.input_loss.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
            self.input_loss.lineEdit().setReadOnly(True)
            self.gridLayout.addWidget(self.input_loss, 4, 1, 1, 1)

            # Add spacer and save button
            self.gridLayout.addItem(self.spacer, 5, 1, 1, 1)
            self.gridLayout.addWidget(self.saveButton, 6, 0, 1, 3, QtCore.Qt.AlignHCenter)
            self.saveButton.clicked.connect(self.action)


        elif model == "KNearestNeighbors":
            # 1. n_neighbors

            ## label
            self.label_n_neighbors = QLabel(self.dialog)
            self.label_n_neighbors.setAlignment(QtCore.Qt.AlignRight)
            self.label_n_neighbors.setObjectName("label_n_neighbors")
            self.label_n_neighbors.setText("Number of Neighbors:")
            self.gridLayout.addWidget(self.label_n_neighbors, 0, 0, 1, 1)

            ## input
            self.input_n_neighbors = QLineEdit(self.dialog)
            self.input_n_neighbors.setValidator(self.IntValidator)
            self.input_n_neighbors.setAlignment(QtCore.Qt.AlignCenter)
            self.input_n_neighbors.setObjectName("input_n_neighbors")
            self.input_n_neighbors.setText('3')
            self.gridLayout.addWidget(self.input_n_neighbors, 0, 1, 1, 1)

            # 2. p

            ## label
            self.label_p = QLabel(self.dialog)
            self.label_p.setAlignment(QtCore.Qt.AlignRight)
            self.label_p.setObjectName("label_p")
            self.label_p.setText("Power Parameter:")
            self.gridLayout.addWidget(self.label_p, 1, 0, 1, 1)

            ## input
            self.input_p = QLineEdit(self.dialog)
            self.input_p.setValidator(self.IntValidator)
            self.input_p.setAlignment(QtCore.Qt.AlignCenter)
            self.input_p.setObjectName("input_p")
            self.input_p.setText('2')
            self.gridLayout.addWidget(self.input_p, 1, 1, 1, 1)

            # Add spacer and save button
            self.gridLayout.addItem(self.spacer, 2, 1, 1, 1)
            self.gridLayout.addWidget(self.saveButton, 3, 0, 1, 3, QtCore.Qt.AlignHCenter)
            self.saveButton.clicked.connect(self.action)


        elif model == "NeuralNetwork":
            # 1. hidden_layer_sizes

            ## label
            self.label_hidden_layer_sizes = QLabel(self.dialog)
            self.label_hidden_layer_sizes.setAlignment(QtCore.Qt.AlignRight)
            self.label_hidden_layer_sizes.setObjectName("label_hidden_layer_sizes")
            self.label_hidden_layer_sizes.setText("Number of Neurons:")
            self.gridLayout.addWidget(self.label_hidden_layer_sizes, 0, 0, 1, 1)

            ## input
            self.input_hidden_layer_sizes = QLineEdit(self.dialog)
            self.input_hidden_layer_sizes.setValidator(self.IntValidator)
            self.input_hidden_layer_sizes.setAlignment(QtCore.Qt.AlignCenter)
            self.input_hidden_layer_sizes.setObjectName("input_hidden_layer_sizes")
            self.input_hidden_layer_sizes.setText('100')
            self.gridLayout.addWidget(self.input_hidden_layer_sizes, 0, 1, 1, 1)

            # 2. learning_rate

            ## label
            self.label_learning_rate = QLabel(self.dialog)
            self.label_learning_rate.setAlignment(QtCore.Qt.AlignRight)
            self.label_learning_rate.setObjectName("label_learning_rate")
            self.label_learning_rate.setText("Learning Rate:")
            self.gridLayout.addWidget(self.label_learning_rate, 1, 0, 1, 1)

            ## input
            self.input_learning_rate = QComboBox(self.dialog)
            self.input_learning_rate.setObjectName("input_learning_rate")
            self.input_learning_rate.addItems(["Constant",
                                               "Adaptive",
                                               "Inverse Scaling"])
            self.input_learning_rate.setEditable(True)
            self.input_learning_rate.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
            self.input_learning_rate.lineEdit().setReadOnly(True)
            self.gridLayout.addWidget(self.input_learning_rate, 1, 1, 1, 1)

            # 3. solver

            ## label
            self.label_solver = QLabel(self.dialog)
            self.label_solver.setAlignment(QtCore.Qt.AlignRight)
            self.label_solver.setObjectName("label_solver")
            self.label_solver.setText("Weight Optimizer:")
            self.gridLayout.addWidget(self.label_solver, 2, 0, 1, 1)

            ## input
            self.input_solver = QComboBox(self.dialog)
            self.input_solver.setObjectName("input_solver")
            self.input_solver.addItems(["Adam",
                                        "LBFGS",
                                        "SGD"])
            self.input_solver.setEditable(True)
            self.input_solver.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
            self.input_solver.lineEdit().setReadOnly(True)
            self.gridLayout.addWidget(self.input_solver, 2, 1, 1, 1)

            # Add spacer and save button
            self.gridLayout.addItem(self.spacer, 3, 1, 1, 1)
            self.gridLayout.addWidget(self.saveButton, 4, 0, 1, 3, QtCore.Qt.AlignHCenter)
            self.saveButton.clicked.connect(self.action)


        elif model == "SupportVectorMachine":
            # 1. kernel

            ## label
            self.label_kernel = QLabel(self.dialog)
            self.label_kernel.setAlignment(QtCore.Qt.AlignRight)
            self.label_kernel.setObjectName("label_kernel")
            self.label_kernel.setText("Kernel:")
            self.gridLayout.addWidget(self.label_kernel, 0, 0, 1, 1)

            ## input
            self.input_kernel = QComboBox(self.dialog)
            self.input_kernel.setObjectName("input_kernel")
            self.input_kernel.addItems(["Radial Basis Function",
                                        "Polynomial"])
            self.input_kernel.setEditable(True)
            self.input_kernel.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
            self.input_kernel.lineEdit().setReadOnly(True)
            self.gridLayout.addWidget(self.input_kernel, 0, 1, 1, 1)

            # 2. degree

            ## label
            self.label_degree = QLabel(self.dialog)
            self.label_degree.setAlignment(QtCore.Qt.AlignRight)
            self.label_degree.setObjectName("label_degree")
            self.label_degree.setText("Polynomial Degree:")
            self.gridLayout.addWidget(self.label_degree, 1, 0, 1, 1)

            ## input
            self.input_degree = QLineEdit(self.dialog)
            self.input_degree.setValidator(self.IntValidator)
            self.input_degree.setAlignment(QtCore.Qt.AlignCenter)
            self.input_degree.setObjectName("input_degree")
            self.input_degree.setText('2')
            self.gridLayout.addWidget(self.input_degree, 1, 1, 1, 1)

            # 3. C

            ## label
            self.label_C = QLabel(self.dialog)
            self.label_C.setAlignment(QtCore.Qt.AlignRight)
            self.label_C.setObjectName("label_C")
            self.label_C.setText("Regularization:")
            self.gridLayout.addWidget(self.label_C, 2, 0, 1, 1)

            ## input
            self.input_C = QLineEdit(self.dialog)
            self.input_C.setValidator(self.DoubleValidator)
            self.input_C.setAlignment(QtCore.Qt.AlignCenter)
            self.input_C.setObjectName("input_C")
            self.input_C.setText('1')
            self.gridLayout.addWidget(self.input_C, 2, 1, 1, 1)

            # 4. gamma

            ## label
            self.label_gamma = QLabel(self.dialog)
            self.label_gamma.setAlignment(QtCore.Qt.AlignRight)
            self.label_gamma.setObjectName("label_gamma")
            self.label_gamma.setText("Gamma:")
            self.gridLayout.addWidget(self.label_gamma, 3, 0, 1, 1)

            ## input
            self.input_gamma = QLineEdit(self.dialog)
            self.input_gamma.setValidator(self.DoubleValidator)
            self.input_gamma.setAlignment(QtCore.Qt.AlignCenter)
            self.input_gamma.setObjectName("input_gamma")
            self.input_gamma.setText('.001')
            self.gridLayout.addWidget(self.input_gamma, 3, 1, 1, 1)

            # Add spacer and save button
            self.gridLayout.addItem(self.spacer, 4, 1, 1, 1)
            self.gridLayout.addWidget(self.saveButton, 5, 0, 1, 3, QtCore.Qt.AlignHCenter)
            self.saveButton.clicked.connect(self.action)


        elif model == "LinearModel":
            self.label_noparams = QLabel(self.dialog)
            self.label_noparams.setAlignment(QtCore.Qt.AlignHCenter)
            self.label_noparams.setObjectName("label_noparams")
            self.label_noparams.setText("No hyperparameters to set")
            self.gridLayout.addWidget(self.label_noparams, 0, 0, 1, 2)

            # Add spacer and close button
            self.gridLayout.addItem(self.spacer, 1, 1, 1, 1)
            self.gridLayout.addWidget(self.closeButton, 2, 0, 1, 3, QtCore.Qt.AlignHCenter)
            self.closeButton.clicked.connect(self.action)


        else:
            raise ValueError("%s not a recognized model" % model)

        # Show QDialog
        self.dialog.exec_()


    def action(self):
        """ADD

        Parameters
        ----------

        Returns
        -------
        """
        if self.model == "ExtraTrees" or self.model == "RandomForest":
            if self.type == "Regressor":
                if self.input_criterion.currentText() == "Mean Squared Error":
                    criterion = "mse"
                else:
                    criterion = "mae"
            else:
                criterion = self.input_criterion.currentText().lower()

            if self.input_max_features.currentText() == "Sqrt(p)":
                max_features = "sqrt"
            elif self.input_max_features.currentText() == "Log2(p)":
                max_features = "log2"
            else:
                max_features = None

            params =  {
                "n_estimators": int(self.input_n_estimators.text()),
                "max_features": max_features,
                "criterion": criterion
            }

        elif self.model == "GradientBoostedTrees":
            if self.type == "Regressor":
                if self.input_loss.currentText() == "Least Squares":
                    loss = "ls"
                else:
                    loss = "huber"
            else:
                loss = self.input_loss.currentText().lower()

            params = {
                "n_estimators": int(self.input_n_estimators.text()),
                "learning_rate": float(self.input_learning_rate.text()),
                "subsample": float(self.input_subsample.text()),
                "max_depth": int(self.input_max_depth.text()),
                "loss": loss
            }

        elif self.model == "SupportVectorMachine":
            params = {
                "kernel": self.input_kernel.currentText().lower(),
                "C": float(self.input_C.text()),
                "gamma": float(self.input_gamma.text()),
                "degree": int(self.input_degree.text()),
                "max_iter": 10000
            }

        elif self.model == "GaussianProcess":
            if self.input_kernel.currentText() == "Radial Basis Function":
                kernel = RBF()
            elif self.input_kernel.currentText() == "Dot Product":
                kernel = DotProduct()
            elif self.input_kernel.currentText() == "Matern":
                kernel = Matern()
            elif self.input_kernel.currentText() == "Rational Quadratic":
                kernel = RationalQuadratic()
            elif self.input_kernel.currentText() == "Constant":
                kernel = ConstantKernel()
            else:
                kernel = ExpSineSquared()

            params = {"kernel": kernel}

        elif self.model == "KNearestNeighbors":
            params = {
                "n_neighbors": int(self.input_n_neighbors.text()),
                "p": int(self.input_p.text())
            }

        elif self.model == "NeuralNetwork":
            if self.input_learning_rate.currentText() == "Inverse Scaling":
                learning_rate = "invscaling"
            else:
                learning_rate = self.input_learning_rate.currentText().lower()

            params = {
                "hidden_layer_sizes": int(self.input_hidden_layer_sizes.text()),
                "solver": self.input_solver.currentText().lower(),
                "learning_rate": learning_rate
            }

        else:
            params = {}

        # Close dialog and return params
        self.dialog.close()
        self.configuration_file['Models'][self.model]['hyperparameters'] = params


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = HyperparametersUI(model="LinearModel", type="Regressor")
    sys.exit(app.exec_())