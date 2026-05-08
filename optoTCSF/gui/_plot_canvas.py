"""Shared matplotlib canvas widget for embedding plots in PyQt5."""

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class PlotCanvas(QWidget):
    """A matplotlib Figure embedded in a QWidget with toolbar."""

    def __init__(self, parent=None, nrows=1, ncols=1, figsize=(6, 4), projection=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        if projection:
            self.ax = self.fig.add_subplot(111, projection=projection)
        elif nrows == 1 and ncols == 1:
            self.ax = self.fig.add_subplot(111)
        else:
            self.axes = self.fig.subplots(nrows, ncols)

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()

    def redraw(self):
        self.fig.tight_layout()
        self.canvas.draw()
