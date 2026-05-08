"""
Module 2 – Psychophysics Experiment tab.

Launches the qCSF-based 2AFC CSF experiment using PsychoPy.
Conditions: baseline_v1 (fixed TF), baseline_v2 (fixed SF),
            opto_v1 (fixed TF + opsin), opto_v2 (fixed SF + opsin).
"""

import json
import subprocess
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QPushButton, QFileDialog, QLineEdit,
    QFormLayout, QSplitter, QMessageBox, QTextEdit, QCheckBox, QSpinBox,
    QRadioButton, QButtonGroup, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess

from ..core.opsin_model import load_all_opsins


class ExperimentTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status = status_bar
        self._process = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        left.setMaximumWidth(420)
        lv = QVBoxLayout(left)
        lv.setAlignment(Qt.AlignTop)
        lv.setSpacing(8)

        # --- Subject ---
        grp_sub = QGroupBox("Subject")
        f1 = QFormLayout(grp_sub)
        self._sID_edit = QLineEdit()
        self._sID_edit.setPlaceholderText("e.g. NS_JD")
        f1.addRow("Subject ID:", self._sID_edit)
        lv.addWidget(grp_sub)

        # --- Condition ---
        grp_cond = QGroupBox("Experiment Condition")
        f2 = QFormLayout(grp_cond)

        self._cond_combo = QComboBox()
        self._cond_combo.addItems([
            "baseline_v1  (fixed TF, vary SF)",
            "baseline_v2  (fixed SF, vary TF)",
            "opto_v1  (opto, fixed TF)",
            "opto_v2  (opto, fixed SF)",
        ])
        self._cond_combo.currentIndexChanged.connect(self._on_cond_changed)
        f2.addRow("Condition:", self._cond_combo)

        self._opsin_label = QLabel("Opsin:")
        self._opsin_combo = QComboBox()
        self._refresh_opsins()
        f2.addRow(self._opsin_label, self._opsin_combo)

        self._fixed_freq_spin = QDoubleSpinBox()
        self._fixed_freq_spin.setRange(0.25, 60.0)
        self._fixed_freq_spin.setValue(5.0)
        f2.addRow("Fixed Freq (Hz or cpd):", self._fixed_freq_spin)

        lv.addWidget(grp_cond)

        # --- Display ---
        grp_disp = QGroupBox("Display")
        f3 = QFormLayout(grp_disp)

        self._frate_spin = QSpinBox()
        self._frate_spin.setRange(60, 240)
        self._frate_spin.setValue(120)
        self._frate_spin.setSuffix(" Hz")
        f3.addRow("Frame Rate:", self._frate_spin)

        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(10, 500)
        self._dist_spin.setValue(57.0)
        self._dist_spin.setSuffix(" cm")
        f3.addRow("Viewing Distance:", self._dist_spin)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(10, 200)
        self._width_spin.setValue(70.0)
        self._width_spin.setSuffix(" cm")
        f3.addRow("Screen Width:", self._width_spin)

        lv.addWidget(grp_disp)

        # --- Trial settings ---
        grp_trial = QGroupBox("Trial Settings")
        f4 = QFormLayout(grp_trial)

        self._n_trials_spin = QSpinBox()
        self._n_trials_spin.setRange(5, 500)
        self._n_trials_spin.setValue(50)
        f4.addRow("Trials per Freq:", self._n_trials_spin)

        self._stim_dur_spin = QDoubleSpinBox()
        self._stim_dur_spin.setRange(100, 5000)
        self._stim_dur_spin.setValue(800.0)
        self._stim_dur_spin.setSuffix(" ms")
        f4.addRow("Stimulus Duration:", self._stim_dur_spin)

        self._debug_check = QCheckBox("Debug mode (no screen)")
        self._debug_check.setChecked(False)
        f4.addRow(self._debug_check)

        lv.addWidget(grp_trial)

        # --- Output ---
        grp_out = QGroupBox("Output")
        f5 = QFormLayout(grp_out)
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setText(str(Path.home() / "optoTCSF_data"))
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_output)
        row = QHBoxLayout()
        row.addWidget(self._out_dir_edit)
        row.addWidget(btn_browse)
        f5.addRow("Output Directory:", row)
        lv.addWidget(grp_out)

        # --- Launch button ---
        self._launch_btn = QPushButton("Launch Experiment")
        self._launch_btn.setMinimumHeight(40)
        self._launch_btn.setStyleSheet("font-weight:bold; background:#2c5f8a; color:white;")
        self._launch_btn.clicked.connect(self._launch_experiment)
        lv.addWidget(self._launch_btn)

        self._stop_btn = QPushButton("Stop Experiment")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_experiment)
        lv.addWidget(self._stop_btn)

        # Right panel – log output
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Experiment Log:"))
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setStyleSheet("font-family: monospace; font-size: 10px;")
        rv.addWidget(self._log_edit)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._on_cond_changed(0)

    def _on_cond_changed(self, idx: int):
        is_opto = idx in (2, 3)
        self._opsin_label.setVisible(is_opto)
        self._opsin_combo.setVisible(is_opto)

    def _refresh_opsins(self):
        opsins = load_all_opsins()
        self._opsin_combo.clear()
        self._opsin_combo.addItems(sorted(opsins.keys()))

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._out_dir_edit.setText(path)

    def _launch_experiment(self):
        sID = self._sID_edit.text().strip().upper()
        if not sID:
            QMessageBox.warning(self, "Subject ID", "Please enter a subject ID.")
            return

        cond_map = [
            "baseline_v1", "baseline_v2", "opto_v1", "opto_v2"
        ]
        cond = cond_map[self._cond_combo.currentIndex()]
        opsin = self._opsin_combo.currentText() if "opto" in cond else ""
        out_dir = self._out_dir_edit.text().strip()
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        config = {
            "sID": sID,
            "cond": cond,
            "opsin": opsin,
            "frame_rate": self._frate_spin.value(),
            "view_dist": self._dist_spin.value(),
            "screen_width": self._width_spin.value(),
            "n_trials": self._n_trials_spin.value(),
            "stim_dur": self._stim_dur_spin.value(),
            "debug": self._debug_check.isChecked(),
            "out_dir": out_dir,
            "fixed_freq": self._fixed_freq_spin.value(),
        }

        # Write config to temp file and launch runner
        cfg_path = Path(out_dir) / f"_expt_config_{sID}.json"
        cfg_path.write_text(json.dumps(config, indent=2))

        runner_path = Path(__file__).parent.parent / "experiment" / "psychopy_runner.py"

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)

        # Disable Python output buffering so lines appear in real time
        env = self._process.processEnvironment()
        from PyQt5.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)

        self._process.readyReadStandardOutput.connect(self._on_process_output)
        self._process.finished.connect(self._on_process_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        # -u flag also disables buffering as a belt-and-suspenders measure
        self._process.start(sys.executable, ["-u", str(runner_path), str(cfg_path)])

        if not self._process.waitForStarted(3000):
            self._log_edit.append("[ERROR] Process failed to start.")
            self._launch_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            return

        self._launch_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.status.showMessage("Experiment running…")
        self._log_edit.append(f"--- Launched: {cond} | Subject: {sID} ---")

    def _stop_experiment(self):
        if self._process and self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._log_edit.append("--- Experiment stopped by user ---")
        self._launch_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _on_process_output(self):
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if line.strip():
                self._log_edit.append(line)

    def _on_process_error(self, error):
        from PyQt5.QtCore import QProcess as QP
        msgs = {
            QP.FailedToStart: "Failed to start — check Python path",
            QP.Crashed: "Process crashed",
            QP.Timedout: "Timed out",
            QP.WriteError: "Write error",
            QP.ReadError: "Read error",
        }
        msg = msgs.get(error, f"Unknown error ({error})")
        self._log_edit.append(f"[ERROR] {msg}")
        self._launch_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.status.showMessage(f"Experiment error: {msg}")

    def _on_process_finished(self, exit_code, exit_status):
        self._launch_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log_edit.append(f"--- Experiment finished (exit code {exit_code}) ---")
        self.status.showMessage("Experiment done.")
