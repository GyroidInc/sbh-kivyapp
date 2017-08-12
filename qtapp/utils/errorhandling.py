# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QErrorMessage


class errorDialogOnException(object):
    """
    An error popup decorator for functions. Call as:
    @errorDialogOnException(ExceptionType1, ExceptionType2)
    def someFunct()
    ...


    Parameters
    ----------
    exceptions: Exceptions to be caught

    Returns
    -------
    """

    func = None
    def __init__(self, exceptions):
        self.exceptions = exceptions

    def __call__(self, *args, **kwargs):
        if self.func is None:
            self.func = args[0]
            return self
        try:
            return self.func(*args, **kwargs)
        except self.exceptions as e:
            error_dialog = QErrorMessage()
            error_dialog.showMessage(e)