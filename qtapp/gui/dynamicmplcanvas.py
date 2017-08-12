# -*- coding: utf-8 -*-

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from PyQt5 import QtCore, QtGui,  uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QFileDialog

plt.style.use('seaborn-whitegrid')

class MplCanvas(FigureCanvas):
    """Base MPL widget for plotting

    Parameters
    ----------
    parent : controller
        Main controller for widget to display within

    width : int
        Width of plotting canvas

    height : int
        Height of plotting canvas

    dpi : int
        Resolution of plotting canavas

    Returns
    -------
    None
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class DynamicMplCanvas(MplCanvas):
    """A canvas that updates itself on call with a new plot."""
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.compute_initial_figure()

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 1, 1, 1])

    def update_figure(self, xindex, plotlist, ylabel=""):
        """Updates figure based on new parameters

        Parameters
        ----------
        xindex : 1d array-like
            Array of indices for x-axis

        plotlist : list
            List of files to plot

        ylabel : str
            Label of feature to plot

        Returns
        -------
        None
        """
        # Iterate over files and add to same plot
        self.axes.clear()
        for entry in plotlist:
            self.axes.plot(xindex, entry["values"], label=entry["label"])
            self.axes.set_xlabel("Freq (Hz)")
            self.axes.set_ylabel(ylabel)

        # Add legend and then show on UI
        self.axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        self.draw()
