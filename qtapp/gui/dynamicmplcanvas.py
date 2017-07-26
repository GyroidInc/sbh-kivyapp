from PyQt5 import QtCore, QtGui,  uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QFileDialog
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
#TODO add toolbar to widget


plt.style.use('seaborn-whitegrid')

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):

        fig = plt.figure( dpi=dpi)

        self.axes = fig.add_subplot(111)

        #self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)



class DynamicMplCanvas(MplCanvas):
    """A canvas that updates itself on call with a new plot."""
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.compute_initial_figure()

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4])
        self.test1()
        #self.update_figure([0, 1, 2, 3], [{"values": [0, 1, 2, 3], "label": "Test.xlsx"},{"values" : [4,0,2,3], "label" : "Test2.xlsx" }])

    def test1(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        l = np.random.uniform(0, 10, size=4)
        self.axes.clear()
        self.axes.plot([0, 1, 2, 3], l, label="test1.xlsx")
        l = np.random.uniform(0, 10, size=4)
        self.axes.plot([0, 1, 2, 3], l, label="test2.xlsx")
        l = np.random.uniform(0, 10, size=4)
        self.axes.plot([0, 1, 2, 3], l, label="test3.xlsx")
        l = np.random.uniform(0, 10, size=4)
        self.axes.plot([0, 1, 2, 3], l, label="test4.xlsx")
        l = np.random.uniform(0, 10, size=4)
        self.axes.plot([0, 1, 2, 3], l, label="test5.xlsx")
        l = np.random.uniform(0, 10, size=4)
        self.axes.plot([0, 1, 2, 3], l, label="test6.xlsx")
        self.axes.set_xlabel("Freq (Hz)")
        self.axes.set_ylabel("Time taken (seconds)")
        self.axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
       #self.draw()

    def update_figure(self, xindex, plotlist, ylabel=""):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        self.axes.clear()
        for entry in plotlist:
            self.axes.plot(xindex, entry["values"], label=entry["label"])
            self.axes.set_xlabel("Freq (Hz)")
            self.axes.set_ylabel(ylabel)
        self.axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        self.draw()


"""class DynamicMplCanvas(MplCanvas):
    #A canvas that updates itself on call with a new plot.
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.compute_initial_figure()

    def compute_initial_figure(self):
        self.axes.plot([0,1,2,3], [4,0,2,3])
        self.draw()
        #self.update_figure2([0,1,2,3], [{"values" : [0,1,2,3], "label" : "Test.xlsx" },
        #                               {"values" : [4,0,2,3], "label" : "Test2.xlsx" }])

    def update_figure(self):
        pass

    def update_figure2(self, xindex, plotlist):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        self.axes.clear()
        for entry in plotlist:
            self.axes.plot(xindex, entry["values"], label=entry["label"])
        self.draw()
"""

