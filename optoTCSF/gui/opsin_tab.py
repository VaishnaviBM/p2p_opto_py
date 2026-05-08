"""
Module 1 – Opsin Simulator tab.

Functionality:
  - Select a built-in opsin and display simulated photocurrents
    (for sinusoidal input given irradiance, TF, wavelength).
  - Load a CSV of patch-clamp data and fit the 4-state model.
  - Save/load user-defined opsin parameters.
"""

import copy
import json
import math
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QPushButton, QFileDialog, QLineEdit,
    QFormLayout, QSplitter, QSizePolicy, QProgressBar, QMessageBox,
    QTabWidget, QCheckBox, QSpinBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ._plot_canvas import PlotCanvas
from ..core.opsin_model import (
    load_all_opsins, BUILTIN_OPSINS, OpsinParams,
    simulate_sinusoidal, simulate_step, fit_opsin_from_csv, save_user_opsin,
    delete_user_opsin, predict_step_from_params,
)


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class SimWorker(QThread):
    finished = pyqtSignal(object, object, object, str)  # t, stim, I, title
    error = pyqtSignal(str)

    def __init__(self, params, irr, lam, V, mode, **kwargs):
        super().__init__()
        self.params = params
        self.irr = irr
        self.lam = lam
        self.V = V
        self.mode = mode
        self.kwargs = kwargs

    def run(self):
        try:
            if self.mode == "step":
                on_ms = self.kwargs["on_ms"]
                off_ms = self.kwargs["off_ms"]
                pad_ms = self.kwargs["pad_ms"]
                t, stim, I = simulate_step(self.params, self.irr, self.lam, self.V,
                                           stim_on_ms=on_ms, stim_off_ms=off_ms,
                                           pad_ms=pad_ms)
                title = (f"λ={self.lam:.0f} nm, Step ON={on_ms:.0f} ms,"
                         f" I₀={self.irr:.4f} W/mm²")
            else:
                freq = self.kwargs["freq"]
                pad_ms = self.kwargs["pad_ms"]
                t, stim, I = simulate_sinusoidal(self.params, self.irr, freq,
                                                 self.lam, self.V,
                                                 pad_dur_ms=pad_ms)
                title = (f"λ={self.lam:.0f} nm, TF={freq:.1f} Hz,"
                         f" I₀={self.irr:.4f} W/mm²")
            self.finished.emit(t, stim, I, title)
        except Exception as ex:
            self.error.emit(str(ex))


class FitWorker(QThread):
    finished = pyqtSignal(object, object, object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, csv_path, stim_on, stim_off, irr, lam, V, E):
        super().__init__()
        self.csv_path = csv_path
        self.stim_on = stim_on
        self.stim_off = stim_off
        self.irr = irr
        self.lam = lam
        self.V = V
        self.E = E

    def run(self):
        try:
            params, t_data, I_data = fit_opsin_from_csv(
                self.csv_path, self.stim_on, self.stim_off,
                self.irr, self.lam, self.V, self.E,
                progress_callback=lambda n: self.progress.emit(n),
            )
            self.finished.emit(params, t_data, I_data)
        except Exception as ex:
            self.error.emit(str(ex))


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class OpsinSimulatorTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status = status_bar
        self._all_opsins = load_all_opsins()
        self._current_params: OpsinParams = None
        self._fit_worker = None
        self._sim_worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel
        left = QWidget()
        left.setMaximumWidth(380)
        left_layout = QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(8)

        inner_tabs = QTabWidget()
        inner_tabs.addTab(self._build_simulate_panel(), "Simulate")
        inner_tabs.addTab(self._build_fit_panel(), "Fit from CSV")
        inner_tabs.addTab(self._build_params_panel(), "Parameters")

        left_layout.addWidget(inner_tabs)

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self._canvas = PlotCanvas(figsize=(7, 5))
        right_layout.addWidget(self._canvas)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Populate opsin combo and wire signal AFTER all child widgets are built
        self._opsin_combo.addItems(sorted(self._all_opsins.keys()))
        self._opsin_combo.currentTextChanged.connect(self._on_opsin_changed)
        self._on_opsin_changed(self._opsin_combo.currentText())

    # ---------- Simulate panel ----------

    def _build_simulate_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignTop)

        # Opsin selection
        grp = QGroupBox("Opsin Selection")
        form = QFormLayout(grp)
        self._opsin_combo = QComboBox()
        form.addRow("Opsin:", self._opsin_combo)
        self._delete_btn = QPushButton("Delete from Library")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_opsin)
        form.addRow(self._delete_btn)
        layout.addWidget(grp)

        # Stimulus parameters
        grp2 = QGroupBox("Stimulus Parameters")
        form2 = QFormLayout(grp2)

        self._stim_mode_combo = QComboBox()
        self._stim_mode_combo.addItems(["Sinusoidal", "Step (constant)"])
        self._stim_mode_combo.currentTextChanged.connect(self._on_stim_mode_changed)
        form2.addRow("Stimulus type:", self._stim_mode_combo)

        self._irr_spin = QDoubleSpinBox()
        self._irr_spin.setRange(1e-6, 1.0)
        self._irr_spin.setDecimals(6)
        self._irr_spin.setSingleStep(0.0001)
        self._irr_spin.setValue(0.001)
        form2.addRow("Irradiance (W/mm²):", self._irr_spin)

        self._lambda_spin = QDoubleSpinBox()
        self._lambda_spin.setRange(380, 780)
        self._lambda_spin.setValue(590.0)
        self._lambda_spin.setSuffix(" nm")
        form2.addRow("Wavelength:", self._lambda_spin)

        self._V_spin = QDoubleSpinBox()
        self._V_spin.setRange(-100, 0)
        self._V_spin.setValue(-60.0)
        self._V_spin.setSuffix(" mV")
        form2.addRow("Holding Potential:", self._V_spin)

        # Sinusoidal-only rows
        self._freq_label = QLabel("Temporal Freq:")
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(0.5, 60.0)
        self._freq_spin.setValue(5.0)
        self._freq_spin.setSuffix(" Hz")
        form2.addRow(self._freq_label, self._freq_spin)

        self._sin_pad_label = QLabel("Padding buffer:")
        self._sin_pad_spin = QDoubleSpinBox()
        self._sin_pad_spin.setRange(0.0, 30.0)
        self._sin_pad_spin.setDecimals(2)
        self._sin_pad_spin.setSingleStep(0.5)
        self._sin_pad_spin.setValue(2.0)
        self._sin_pad_spin.setSuffix(" s")
        self._sin_pad_spin.setToolTip(
            "Pre-stimulus pad driven at mean irradiance (Irr/2)"
        )
        form2.addRow(self._sin_pad_label, self._sin_pad_spin)

        # Step-only rows (hidden initially)
        self._stim_on_dur_label = QLabel("Stim ON duration:")
        self._stim_on_dur_spin = QDoubleSpinBox()
        self._stim_on_dur_spin.setRange(1, 10000)
        self._stim_on_dur_spin.setValue(500.0)
        self._stim_on_dur_spin.setSuffix(" ms")
        form2.addRow(self._stim_on_dur_label, self._stim_on_dur_spin)

        self._stim_off_dur_label = QLabel("Stim OFF duration:")
        self._stim_off_dur_spin = QDoubleSpinBox()
        self._stim_off_dur_spin.setRange(1, 10000)
        self._stim_off_dur_spin.setValue(500.0)
        self._stim_off_dur_spin.setSuffix(" ms")
        form2.addRow(self._stim_off_dur_label, self._stim_off_dur_spin)

        self._pad_dur_label = QLabel("Pre-stim pad:")
        self._pad_dur_spin = QDoubleSpinBox()
        self._pad_dur_spin.setRange(0, 5000)
        self._pad_dur_spin.setValue(100.0)
        self._pad_dur_spin.setSuffix(" ms")
        form2.addRow(self._pad_dur_label, self._pad_dur_spin)

        # Hide step controls at start
        for w_ in (self._stim_on_dur_label, self._stim_on_dur_spin,
                   self._stim_off_dur_label, self._stim_off_dur_spin,
                   self._pad_dur_label, self._pad_dur_spin):
            w_.setVisible(False)

        layout.addWidget(grp2)

        btn = QPushButton("Simulate Photocurrent")
        btn.clicked.connect(self._run_simulation)
        layout.addWidget(btn)

        # Multi-TF sweep (sinusoidal only)
        self._sweep_grp = QGroupBox("TF Sweep")
        form3 = QFormLayout(self._sweep_grp)
        self._tf_min_spin = QDoubleSpinBox()
        self._tf_min_spin.setRange(0.5, 30); self._tf_min_spin.setValue(1.5)
        form3.addRow("Min TF (Hz):", self._tf_min_spin)
        self._tf_max_spin = QDoubleSpinBox()
        self._tf_max_spin.setRange(1, 60); self._tf_max_spin.setValue(20.0)
        form3.addRow("Max TF (Hz):", self._tf_max_spin)
        self._tf_n_spin = QSpinBox()
        self._tf_n_spin.setRange(2, 20); self._tf_n_spin.setValue(8)
        form3.addRow("N TF points:", self._tf_n_spin)
        btn_sweep = QPushButton("Run TF Sweep")
        btn_sweep.clicked.connect(self._run_tf_sweep)
        form3.addRow(btn_sweep)
        layout.addWidget(self._sweep_grp)

        return w

    def _on_stim_mode_changed(self, mode: str):
        is_step = (mode == "Step (constant)")
        for w_ in (self._freq_label, self._freq_spin,
                   self._sin_pad_label, self._sin_pad_spin):
            w_.setVisible(not is_step)
        for w_ in (self._stim_on_dur_label, self._stim_on_dur_spin,
                   self._stim_off_dur_label, self._stim_off_dur_spin,
                   self._pad_dur_label, self._pad_dur_spin):
            w_.setVisible(is_step)
        self._sweep_grp.setVisible(not is_step)

    # ---------- Fit panel ----------

    def _build_fit_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignTop)

        grp = QGroupBox("Load Patch-Clamp CSV")
        form = QFormLayout(grp)

        self._csv_path_edit = QLineEdit()
        self._csv_path_edit.setPlaceholderText("time(s), current(nA)")
        btn_csv = QPushButton("Browse…")
        btn_csv.clicked.connect(self._browse_csv)
        row = QHBoxLayout()
        row.addWidget(self._csv_path_edit)
        row.addWidget(btn_csv)
        form.addRow("CSV file:", row)

        self._stim_on_spin = QDoubleSpinBox()
        self._stim_on_spin.setRange(0, 10); self._stim_on_spin.setDecimals(4)
        self._stim_on_spin.setValue(0.19)
        form.addRow("Stim ON (s):", self._stim_on_spin)

        self._stim_off_spin = QDoubleSpinBox()
        self._stim_off_spin.setRange(0, 10); self._stim_off_spin.setDecimals(4)
        self._stim_off_spin.setValue(0.29)
        form.addRow("Stim OFF (s):", self._stim_off_spin)

        self._fit_irr_spin = QDoubleSpinBox()
        self._fit_irr_spin.setRange(1e-7, 1.0); self._fit_irr_spin.setDecimals(6)
        self._fit_irr_spin.setValue(0.02e-3)
        form.addRow("Irradiance (W/mm²):", self._fit_irr_spin)

        self._fit_lambda_spin = QDoubleSpinBox()
        self._fit_lambda_spin.setRange(380, 780)
        self._fit_lambda_spin.setValue(650.0)
        self._fit_lambda_spin.setSuffix(" nm")
        form.addRow("Wavelength:", self._fit_lambda_spin)

        self._fit_V_spin = QDoubleSpinBox()
        self._fit_V_spin.setRange(-100, 0); self._fit_V_spin.setValue(-60.0)
        form.addRow("Holding Pot. (mV):", self._fit_V_spin)

        self._fit_E_spin = QDoubleSpinBox()
        self._fit_E_spin.setRange(-50, 50); self._fit_E_spin.setValue(0.0)
        form.addRow("Reversal Pot. (mV):", self._fit_E_spin)

        layout.addWidget(grp)

        self._fit_progress = QProgressBar()
        self._fit_progress.setVisible(False)
        layout.addWidget(self._fit_progress)

        btn_fit = QPushButton("Fit Opsin Parameters")
        btn_fit.clicked.connect(self._run_fit)
        layout.addWidget(btn_fit)

        # Save fitted opsin
        grp2 = QGroupBox("Save Fitted Opsin")
        form2 = QFormLayout(grp2)
        self._save_name_edit = QLineEdit()
        self._save_name_edit.setPlaceholderText("e.g. MyOpsin_650nm")
        form2.addRow("Name:", self._save_name_edit)
        btn_save = QPushButton("Save to Library")
        btn_save.clicked.connect(self._save_fitted_opsin)
        form2.addRow(btn_save)
        layout.addWidget(grp2)

        return w

    # ---------- Parameters panel ----------

    def _build_params_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignTop)

        self._param_fields = {}
        grp = QGroupBox("Current Opsin Parameters")
        form = QFormLayout(grp)

        param_names = [
            ("Gd1", "Gd1 (ms⁻¹)"), ("Gd2", "Gd2 (ms⁻¹)"), ("Gr", "Gr (ms⁻¹)"),
            ("g0_ph", "g0_ph (nS)"), ("phim", "ɸm (ph/mm²/s)"),
            ("k1", "k1 (ms⁻¹)"), ("k2", "k2 (ms⁻¹)"),
            ("Gf0", "Gf0 (ms⁻¹)"), ("Gb0", "Gb0 (ms⁻¹)"),
            ("kf", "kf (ms⁻¹)"), ("kb", "kb (ms⁻¹)"),
            ("gamma", "γ"), ("p", "p"), ("q", "q"), ("E", "E (mV)"),
        ]
        for key, label in param_names:
            spin = QDoubleSpinBox()
            spin.setRange(-1e10, 1e10)
            spin.setDecimals(8)
            spin.setSingleStep(0.001)
            spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._param_fields[key] = spin
            form.addRow(label, spin)

        layout.addWidget(grp)

        btn_update = QPushButton("Use These Parameters for Simulation")
        btn_update.clicked.connect(self._apply_manual_params)
        layout.addWidget(btn_update)

        btn_reset = QPushButton("Reset to Default Parameters")
        btn_reset.clicked.connect(self._reset_params)
        layout.addWidget(btn_reset)

        return w

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _refresh_opsin_combo(self, select: str = None):
        self._all_opsins = load_all_opsins()
        self._opsin_combo.blockSignals(True)
        current = select if select else self._opsin_combo.currentText()
        self._opsin_combo.clear()
        self._opsin_combo.addItems(sorted(self._all_opsins.keys()))
        idx = self._opsin_combo.findText(current)
        if idx >= 0:
            self._opsin_combo.setCurrentIndex(idx)
        self._opsin_combo.blockSignals(False)
        self._on_opsin_changed(self._opsin_combo.currentText())

    def _on_opsin_changed(self, name: str):
        if name in self._all_opsins:
            self._current_params = self._all_opsins[name]
            self._lambda_spin.setValue(self._current_params.peak_lambda)
            self._populate_param_fields(self._current_params)
            self._delete_btn.setEnabled(name not in BUILTIN_OPSINS)
            self.status.showMessage(f"Loaded {name}")

    def _delete_opsin(self):
        name = self._opsin_combo.currentText()
        if name in BUILTIN_OPSINS:
            return
        reply = QMessageBox.question(
            self, "Delete Opsin",
            f"Permanently delete '{name}' from the library?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if delete_user_opsin(name):
            self.status.showMessage(f"Deleted '{name}' from library.")
        self._refresh_opsin_combo()

    def _populate_param_fields(self, params: OpsinParams):
        for key, spin in self._param_fields.items():
            spin.blockSignals(True)
            spin.setValue(float(getattr(params, key, 0.0)))
            spin.blockSignals(False)

    def _apply_manual_params(self):
        if self._current_params is None:
            self._current_params = OpsinParams()
        # Work on a copy so the original stored in _all_opsins stays clean for reset
        edited = copy.copy(self._current_params)
        for key, spin in self._param_fields.items():
            setattr(edited, key, spin.value())
        self._current_params = edited
        self.status.showMessage("Using manually edited parameters.")

    def _reset_params(self):
        name = self._opsin_combo.currentText()
        if name in self._all_opsins:
            self._current_params = self._all_opsins[name]
            self._populate_param_fields(self._current_params)
            self.status.showMessage(f"Reset to default parameters for {name}.")

    def _run_simulation(self):
        if self._current_params is None:
            QMessageBox.warning(self, "No Opsin", "Please select an opsin first.")
            return
        if self._sim_worker is not None and self._sim_worker.isRunning():
            return
        irr = self._irr_spin.value()
        lam = self._lambda_spin.value()
        V = self._V_spin.value()

        if self._stim_mode_combo.currentText() == "Step (constant)":
            self._sim_worker = SimWorker(
                self._current_params, irr, lam, V, "step",
                on_ms=self._stim_on_dur_spin.value(),
                off_ms=self._stim_off_dur_spin.value(),
                pad_ms=self._pad_dur_spin.value(),
            )
        else:
            self._sim_worker = SimWorker(
                self._current_params, irr, lam, V, "sin",
                freq=self._freq_spin.value(),
                pad_ms=self._sin_pad_spin.value() * 1000.0,
            )
        self._sim_worker.finished.connect(self._on_sim_done)
        self._sim_worker.error.connect(self._on_sim_error)
        self._sim_worker.start()
        self.status.showMessage("Simulating…")

    def _on_sim_done(self, t, stim, I, title):
        self._plot_simulation(t, stim, I, title)
        self.status.showMessage(f"Done. Peak current: {np.abs(I).max():.2f} pA")

    def _on_sim_error(self, msg: str):
        QMessageBox.critical(self, "Simulation Error", msg)
        self.status.showMessage("Simulation failed.")

    def _run_tf_sweep(self):
        if self._current_params is None:
            QMessageBox.warning(self, "No Opsin", "Please select an opsin first.")
            return
        irr = self._irr_spin.value()
        lam = self._lambda_spin.value()
        V = self._V_spin.value()
        tf_min = self._tf_min_spin.value()
        tf_max = self._tf_max_spin.value()
        n_tf = self._tf_n_spin.value()
        tfs = np.logspace(np.log10(tf_min), np.log10(tf_max), n_tf)

        self.status.showMessage("Running TF sweep…")
        self._canvas.fig.clear()

        n_cols = 2
        n_rows = math.ceil(n_tf / n_cols)
        axes_grid = self._canvas.fig.subplots(n_rows, n_cols, sharex=False)
        # Normalise to a flat list regardless of shape
        if n_rows == 1 and n_cols == 1:
            axes_flat = [axes_grid]
        elif n_rows == 1:
            axes_flat = list(axes_grid)
        else:
            axes_flat = [ax for row in axes_grid for ax in row]

        for pi, tf in enumerate(tfs):
            ax = axes_flat[pi]
            t, stim, I = simulate_sinusoidal(self._current_params, irr, tf, lam, V)
            # Centre both signals so they oscillate around 0, then scale to [-1, 1]
            stim_c = stim - stim.mean()
            I_c = I - I.mean()
            stim_norm = stim_c / (np.abs(stim_c).max() + 1e-12)
            I_norm = I_c / (np.abs(I_c).max() + 1e-12)
            ax.plot(t, stim_norm, "r--", lw=0.8, label="Stimulus (norm)")
            ax.plot(t, I_norm, "b", lw=1.2, label="Photocurrent (norm)")
            ax.set_title(f"TF = {tf:.1f} Hz", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_xlabel("Time (ms)")
            ax.grid(True, alpha=0.3)

        # Hide any unused axes in the last row
        for pi in range(n_tf, len(axes_flat)):
            axes_flat[pi].set_visible(False)

        opsin_name = self._opsin_combo.currentText()
        self._canvas.fig.suptitle(f"{opsin_name} – TF Sweep", fontsize=10)
        self._canvas.fig.tight_layout()
        self._canvas.redraw()
        self.status.showMessage("TF sweep complete.")

    def _plot_simulation(self, t, stim, I, title: str):
        self._canvas.fig.clear()
        ax1 = self._canvas.fig.add_subplot(211)
        ax2 = self._canvas.fig.add_subplot(212)

        ax1.plot(t, stim * 1e3, "r-", lw=1.2, label="Irradiance (mW/mm²)")
        ax1.set_ylabel("Irradiance (mW/mm²)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title, fontsize=9)

        ax2.plot(t, I, "b-", lw=1.5, label="Photocurrent (pA)")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Photocurrent (pA)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        opsin_name = self._opsin_combo.currentText()
        self._canvas.fig.suptitle(f"Opsin: {opsin_name}", fontsize=10)
        self._canvas.redraw()

    # ---------- Fitting ----------

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if path:
            self._csv_path_edit.setText(path)

    def _run_fit(self):
        csv_path = self._csv_path_edit.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "No file", "Please select a CSV file.")
            return
        if not Path(csv_path).exists():
            QMessageBox.warning(self, "File not found", f"Cannot find: {csv_path}")
            return

        self._fit_progress.setVisible(True)
        self._fit_progress.setValue(0)
        self._fit_progress.setMaximum(3000)

        stim_on = self._stim_on_spin.value()
        stim_off = self._stim_off_spin.value()
        irr = self._fit_irr_spin.value()
        lam = self._fit_lambda_spin.value()
        V = self._fit_V_spin.value()
        E = self._fit_E_spin.value()

        self._fit_worker = FitWorker(csv_path, stim_on, stim_off, irr, lam, V, E)
        self._fit_worker.finished.connect(self._on_fit_done)
        self._fit_worker.progress.connect(self._fit_progress.setValue)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.start()
        self.status.showMessage("Fitting…")

    def _on_fit_done(self, params: OpsinParams, t_data, I_data):
        self._fit_progress.setVisible(False)
        self._current_params = params
        self._populate_param_fields(params)

        try:
            I_pred = predict_step_from_params(
                params, t_data,
                self._stim_on_spin.value(), self._stim_off_spin.value(),
                self._fit_irr_spin.value(), self._fit_V_spin.value(),
            )
        except Exception as ex:
            QMessageBox.critical(self, "Prediction Error", str(ex))
            self.status.showMessage("Fit complete (prediction plot failed).")
            return

        self._canvas.fig.clear()
        ax = self._canvas.fig.add_subplot(111)
        ax.plot(t_data, I_data, "b-", lw=1.5, label="Data (pA)")
        ax.plot(t_data[: len(I_pred)], I_pred, "r--", lw=1.5, label="Fit (pA)")
        ax.axvline(self._stim_on_spin.value() * 1000, color="gray", ls=":", lw=0.8)
        ax.axvline(self._stim_off_spin.value() * 1000, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Photocurrent (pA)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("Fitted Opsin Model vs Data")
        self._canvas.redraw()
        self.status.showMessage("Fit complete.")

    def _on_fit_error(self, msg: str):
        self._fit_progress.setVisible(False)
        QMessageBox.critical(self, "Fitting Error", msg)
        self.status.showMessage("Fit failed.")

    def _save_fitted_opsin(self):
        if self._current_params is None:
            QMessageBox.warning(self, "No parameters", "Run a fit first.")
            return
        name = self._save_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required", "Enter a name for the opsin.")
            return
        self._current_params.name = name
        self._current_params.peak_lambda = self._fit_lambda_spin.value()
        save_user_opsin(self._current_params)
        self._refresh_opsin_combo(select=name)
        self.status.showMessage(f"Saved '{name}' to library.")
