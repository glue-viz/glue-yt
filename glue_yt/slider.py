import os
from qtpy import QtWidgets
from glue.utils.qt import load_ui
from glue.config import menubar_plugin
from glue.core.message import NumericalDataChangedMessage


class YTWidget(QtWidgets.QWidget):
    """
    The main widget that appears in the toolbar.
    """

    def __init__(self, data_collection=None, parent=None):

        super(YTWidget, self).__init__(parent=parent)

        self.data_collection = data_collection

        self.ui = load_ui('yt.ui', self,
                          directory=os.path.dirname(__file__))

        self.ui.value_step.setMinimum(0)
        self.ui.value_step.setMaximum(len(data_collection[0].ds_all) - 1)

        # Set up connections for UI elements
        self.ui.value_step.valueChanged.connect(self.set_step)
        self.ui.text_step.setText('')

        self.data = None
        self.n_steps = None
        self.filename = None

    def set_step(self, step_id):
        self.data_collection[0].current_step = step_id
        self.ui.text_step.setText("{0:6d}".format(step_id))
        msg = NumericalDataChangedMessage(self.data_collection[0])
        self.data_collection.hub.broadcast(msg)


@menubar_plugin("Browse yt time series")
def yt_plugin(session, data_collection):
    YT_widget = YTWidget(data_collection=data_collection)
    toolbar = QtWidgets.QToolBar()
    toolbar.addWidget(YT_widget)
    session.application.addToolBar(toolbar)
