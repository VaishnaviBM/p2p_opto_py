"""Main application window with 4-tab interface."""

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QStatusBar,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .opsin_tab import OpsinSimulatorTab
from .experiment_tab import ExperimentTab
from .tcsf_tab import TCSFEstimatorTab
from .opto_tcsf_tab import OptoTCSFTab
from .video_simulation_tab import VideoSimulationTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("optoTCSF – Optogenetic TCSF Framework")
        self.setMinimumSize(1100, 750)

        font = QFont("Arial", 10)
        self.setFont(font)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        # Header
        header = QLabel(
            "optoTCSF  |  Optogenetic Temporal Contrast Sensitivity Framework"
        )
        header.setAlignment(Qt.AlignCenter)
        hfont = QFont("Arial", 12, QFont.Bold)
        header.setFont(hfont)
        header.setStyleSheet("color: #2c5f8a; padding: 4px 0;")
        layout.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        layout.addWidget(self.tabs)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # Instantiate tabs
        self.opsin_tab = OpsinSimulatorTab(self.status)
        self.expt_tab = ExperimentTab(self.status)
        self.tcsf_tab = TCSFEstimatorTab(self.status)
        self.opto_tcsf_tab = OptoTCSFTab(self.status)
        self.video_sim_tab = VideoSimulationTab(self.status)

        self.tabs.addTab(self.opsin_tab, "1 · Opsin Simulator")
        self.tabs.addTab(self.expt_tab, "2 · Psychophysics Experiment")
        self.tabs.addTab(self.tcsf_tab, "3 · TCSF Estimator")
        self.tabs.addTab(self.opto_tcsf_tab, "4 · Opto TCSF Prediction")
        self.tabs.addTab(self.video_sim_tab, "5 · Video Simulation")

        self.tabs.setStyleSheet("""
            QTabBar::tab { min-width: 200px; padding: 6px 12px; }
            QTabBar::tab:selected { background: #2c5f8a; color: white; font-weight: bold; }
        """)
