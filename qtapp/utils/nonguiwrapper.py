from multiprocessing.pool import ThreadPool
from PyQt5.QtWidgets import QApplication


def nongui(fun):
    """Decorator running the function in non-gui thread while
    processing the gui events."""
    def wrap(*args, **kwargs):
        pool = ThreadPool(processes=1)
        thisasync = pool.apply_async(fun, args, kwargs)
        while not thisasync.ready():
            thisasync.wait(0.01)
            QApplication.processEvents()
        return thisasync.get()

    return wrap