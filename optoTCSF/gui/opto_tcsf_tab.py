"""
Module 4 – Opto TCSF Prediction tab.

Predicts optogenetic TCSF by combining:
  1. A neurotypical baseline TCSF (loaded from .mat, CSV, or JSON).
  2. Opsin photocurrent traces (from Module 1 or stored parameters).
  3. A linking hypothesis model (decision rule, Probability Summation by default).
"""

import io as _io
import json
import re
import numpy as np
import scipy.io as sio
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QPushButton, QFileDialog, QLineEdit,
    QFormLayout, QSplitter, QMessageBox, QTabWidget, QSpinBox,
    QRadioButton, QButtonGroup, QCheckBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QStackedWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ._plot_canvas import PlotCanvas
from ..core.opsin_model import load_all_opsins, simulate_sinusoidal, get_scale_factor
from ..core.attenuation import DecisionParams, decision_rule, AVAILABLE_RULES
from ..core.qcsf_algo import estimate_sensitivity_grid


import colorsys as _colorsys
import hashlib as _hashlib

# Fixed colors for known opsins — always the same regardless of list order or size.
_OPSIN_COLORS = {
    "baseline":   (0.49, 0.68, 0.00),   # olive green
    "ChRmine":    (0.00, 0.75, 0.77),
    "ChR2":       (0.78, 0.49, 1.00),
    "ReaChR":     (0.97, 0.46, 0.43),
    "ChrimsonR":  (0.90, 0.20, 0.20),
    "CsChrimson": (0.95, 0.60, 0.10),
    "bReaChES":   (0.10, 0.55, 0.90),
    "mco_450":    (0.60, 0.20, 0.80),
}


def _color_for_opsin(label: str, fallback_idx: int = 0):
    """Return a unique RGB colour for *label*.

    Known opsins get a fixed colour from _OPSIN_COLORS.
    Any other label (user-added opsins) gets a deterministic colour derived
    from the MD5 hash of its name, spread evenly across the HSV hue wheel with
    good saturation and brightness so it is always visually distinct.
    """
    for key, col in _OPSIN_COLORS.items():
        if key.lower() in label.lower():
            return col
    digest = int(_hashlib.md5(label.encode()).hexdigest()[:8], 16)
    hue = (digest / 0xFFFFFFFF) % 1.0
    return _colorsys.hsv_to_rgb(hue, 0.75, 0.88)

# ---------------------------------------------------------------------------
# .mat parsing helpers (same logic as tcsf_tab.py)
# ---------------------------------------------------------------------------

def _parse_mat_filename(fname: str) -> dict:
    m = re.search(r'(baseline|opto)_(v1|v2)', fname)
    if m is None:
        raise ValueError(
            f"Cannot infer condition from filename: '{fname}'\n"
            "Expected: <sID>_qCSF_<baseline|opto>_<v1|v2>[_<opsin>]_<timestamp>"
        )
    cond_type = m.group(1)
    cond_ver  = m.group(2)
    fix_temporal = (cond_ver == 'v1')
    sid_m = re.match(r'^(.+?)_qCSF_', fname)
    sID = sid_m.group(1) if sid_m else fname[:8]
    opsin = ''
    if cond_type == 'opto':
        after = fname[m.end():]
        op_m = re.match(r'_([A-Za-z][A-Za-z0-9]+?)_\d{2}-', after)
        if op_m:
            opsin = op_m.group(1)
    return {'sID': sID, 'cond_type': cond_type,
            'fix_temporal': fix_temporal, 'opsin': opsin}


def _parse_freq_field(field: str, fix_temporal: bool) -> float:
    prefix = 'TF_' if fix_temporal else 'SF_'
    if not field.startswith(prefix):
        raise ValueError(f"Unexpected field '{field}'")
    parts = field[len(prefix):].split('_')
    if len(parts) == 1:
        return float(parts[0])
    try:
        return float(f"{parts[0]}.{''.join(parts[1:])}")
    except ValueError:
        return float(parts[0])


def _parse_dict_key(key: str, fix_temporal: bool) -> float:
    prefix = 'TF_' if fix_temporal else 'SF_'
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected key '{key}'")
    val_str = key[len(prefix):]
    try:
        return float(val_str)
    except ValueError:
        parts = val_str.split('_')
        if len(parts) >= 2:
            return float(f"{parts[0]}.{''.join(parts[1:])}")
        raise


_matlab_engine = None


def _get_matlab_engine():
    global _matlab_engine
    if _matlab_engine is None:
        try:
            import matlab.engine
        except ImportError:
            return None
        _matlab_engine = matlab.engine.start_matlab("-nojvm -nosplash -nodesktop")
    return _matlab_engine


def _history_block_to_rows(history: np.ndarray,
                            fixed_val: float,
                            fix_temporal: bool) -> np.ndarray:
    vf           = history[:, 1].astype(float)
    contrast_lin = history[:, 2].astype(float)
    response     = history[:, 3].astype(float)
    contrast_lvl = -np.log10(np.clip(contrast_lin, 1e-12, 1.0))
    if fix_temporal:
        tf = np.full(len(vf), fixed_val)
        sf = vf
    else:
        tf = vf
        sf = np.full(len(vf), fixed_val)
    return np.column_stack([tf, sf, contrast_lvl, response])


def _load_mat_to_array(path: str) -> np.ndarray:
    """Load a single .mat file and return a [TF, SF, contrast_level, response] array."""
    fname = Path(path).stem
    info = _parse_mat_filename(fname)
    fix_temporal = info['fix_temporal']

    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    data_key = next((k for k in mat if not k.startswith('__')), None)
    if data_key is None:
        raise ValueError("No data variable found in file.")

    data_obj = mat[data_key]
    all_rows = []

    if hasattr(data_obj, '_fieldnames'):
        for field in data_obj._fieldnames:
            try:
                fixed_val = _parse_freq_field(field, fix_temporal)
            except ValueError:
                continue
            try:
                block = getattr(data_obj, field)
                history = block.qcsf.data.history
                if history.ndim == 1:
                    history = history.reshape(1, -1)
                if history.shape[1] < 4:
                    continue
                all_rows.append(_history_block_to_rows(history, fixed_val, fix_temporal))
            except AttributeError:
                continue
    else:
        eng = _get_matlab_engine()
        if eng is None:
            raise RuntimeError(
                "MCOS/dictionary .mat format requires the MATLAB Python engine.\n"
                "Install: cd /usr/local/MATLAB/R2022b/extern/engines/python && python setup.py install"
            )
        eng.eval(
            f"clear all; load('{path}'); "
            f"tmpv = who(); "
            f"eval(['tmpdict = ' tmpv{{1}} ';']); "
            f"matkeys = keys(tmpdict);",
            nargout=0, stdout=_io.StringIO())
        n_keys = int(eng.eval("numel(matkeys)", nargout=1))
        for i in range(1, n_keys + 1):
            field = eng.eval(f"matkeys{{{i}}}", nargout=1)
            try:
                fixed_val = _parse_dict_key(field, fix_temporal)
            except ValueError:
                continue
            try:
                history = np.array(
                    eng.eval(f"tmpdict(matkeys{{{i}}}).qcsf.data.history", nargout=1))
                if history.ndim == 1:
                    history = history.reshape(1, -1)
                if history.shape[1] < 4:
                    continue
                all_rows.append(_history_block_to_rows(history, fixed_val, fix_temporal))
            except Exception:
                continue

    if not all_rows:
        raise ValueError("No valid trial blocks found in file.")
    return np.vstack(all_rows)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class PredictionWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, opsin_names, tf_list, sf_list, ntf, nsf,
                 baseline_S, irr, V, dec_params, win_size):
        super().__init__()
        self.opsin_names = opsin_names
        self.tf_list = tf_list
        self.sf_list = sf_list
        self.ntf = ntf
        self.nsf = nsf
        self.baseline_S = baseline_S
        self.irr = irr
        self.V = V
        self.dec_params = dec_params
        self.win_size = win_size

    def run(self):
        try:
            opsins = load_all_opsins()
            results = {"baseline": (self.tf_list, self.sf_list, self.baseline_S)}

            for oname in self.opsin_names:
                if oname not in opsins:
                    continue
                params = opsins[oname]
                lam = params.peak_lambda
                resp = np.zeros(len(self.tf_list))

                for i, tf in enumerate(self.tf_list):
                    t, stim, I = simulate_sinusoidal(
                        params, self.irr, tf, lam, self.V,
                        stim_dur_ms=800.0, pad_dur_ms=2000.0, dt=0.1,
                    )
                    dt = t[1] - t[0] if len(t) > 1 else 0.1
                    bins_per_ms = int(1.0 / dt)
                    pad_pts = int(2000.0 * bins_per_ms) if dt > 0 else 0

                    if i == 0:
                        offset, scale_fac = get_scale_factor(I, dt, 0)

                    y_norm = (I - offset) * scale_fac
                    resp[i] = decision_rule(y_norm, dt, self.dec_params)

                resp_ref = resp[0] if resp[0] != 0 else 1.0
                attenuation = resp / resp_ref

                S_opto = np.zeros_like(self.baseline_S)
                for j in range(self.nsf):
                    for i in range(self.ntf):
                        S_opto[i, j] = self.baseline_S[i, j] + np.log10(attenuation[i] + 1e-12)

                results[f"opto_{oname}"] = (self.tf_list, self.sf_list, S_opto)

            self.finished.emit(results)
        except Exception as ex:
            self.error.emit(str(ex))


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class OptoTCSFTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status = status_bar
        self._baseline_S = None
        self._baseline_tf = None
        self._baseline_sf = None
        self._results = {}
        self._mat_trial_data = {}   # {label: np.ndarray} merged by sID+cond (v1+v2)
        self._dec_params = DecisionParams()
        self._worker = None
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

        # --- Baseline TCSF ---
        grp_base = QGroupBox("Neurotypical Baseline TCSF")
        fb = QVBoxLayout(grp_base)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._base_src_combo = QComboBox()
        self._base_src_combo.addItems([
            "Load from .mat file",
            "Load from CSV file",
            "Load from JSON file",
        ])
        self._base_src_combo.currentIndexChanged.connect(self._on_src_changed)
        src_row.addWidget(self._base_src_combo)
        fb.addLayout(src_row)

        # --- Stacked: .mat list  vs  single-file path edit ---
        self._src_stack = QStackedWidget()

        # Page 0: .mat list widget
        mat_page = QWidget()
        mat_lv = QVBoxLayout(mat_page)
        mat_lv.setContentsMargins(0, 0, 0, 0)
        btn_mat_row = QHBoxLayout()
        btn_add_mat = QPushButton("Add .mat file(s)")
        btn_add_mat.clicked.connect(self._add_mat_files)
        btn_rem_mat = QPushButton("Remove selected")
        btn_rem_mat.clicked.connect(self._remove_mat_selected)
        btn_mat_row.addWidget(btn_add_mat)
        btn_mat_row.addWidget(btn_rem_mat)
        mat_lv.addLayout(btn_mat_row)
        self._mat_list = QListWidget()
        self._mat_list.setMaximumHeight(100)
        self._mat_list.setSelectionMode(QAbstractItemView.MultiSelection)
        mat_lv.addWidget(self._mat_list)
        self._src_stack.addWidget(mat_page)

        # Page 1: single file path + browse
        file_page = QWidget()
        file_lv = QHBoxLayout(file_page)
        file_lv.setContentsMargins(0, 0, 0, 0)
        self._base_path_edit = QLineEdit()
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_baseline)
        file_lv.addWidget(self._base_path_edit)
        file_lv.addWidget(btn_browse)
        self._src_stack.addWidget(file_page)

        fb.addWidget(self._src_stack)

        btn_load_base = QPushButton("Load Baseline")
        btn_load_base.clicked.connect(self._load_baseline)
        fb.addWidget(btn_load_base)

        self._base_status_lbl = QLabel("Not loaded")
        self._base_status_lbl.setStyleSheet("color: gray;")
        fb.addWidget(self._base_status_lbl)

        lv.addWidget(grp_base)

        # --- Opsin selection ---
        grp_opsin = QGroupBox("Opsins to Predict")
        fo = QVBoxLayout(grp_opsin)
        self._opsin_checks = {}
        for name in sorted(load_all_opsins().keys()):
            cb = QCheckBox(name)
            fo.addWidget(cb)
            self._opsin_checks[name] = cb
        lv.addWidget(grp_opsin)

        # --- Stimulus params ---
        grp_stim = QGroupBox("Stimulus Parameters")
        fs = QFormLayout(grp_stim)

        self._irr_spin = QDoubleSpinBox()
        self._irr_spin.setRange(1e-6, 1.0)
        self._irr_spin.setDecimals(6)
        self._irr_spin.setValue(0.001)
        fs.addRow("Irradiance (W/mm²):", self._irr_spin)

        self._V_spin = QDoubleSpinBox()
        self._V_spin.setRange(-100, 0)
        self._V_spin.setValue(-60.0)
        self._V_spin.setSuffix(" mV")
        fs.addRow("Holding Potential:", self._V_spin)

        lv.addWidget(grp_stim)

        # --- Decision rule ---
        grp_link = QGroupBox("Linking Hypothesis (Decision Rule)")
        fl = QFormLayout(grp_link)

        self._rule_combo = QComboBox()
        self._rule_combo.addItems(AVAILABLE_RULES)
        self._rule_combo.currentTextChanged.connect(self._on_rule_changed)
        fl.addRow("Rule:", self._rule_combo)

        self._beta0_spin = QDoubleSpinBox()
        self._beta0_spin.setRange(-10, 10); self._beta0_spin.setValue(1.0)
        fl.addRow("β₀ (positive phase):", self._beta0_spin)

        self._beta1_spin = QDoubleSpinBox()
        self._beta1_spin.setRange(-10, 10); self._beta1_spin.setValue(-0.5)
        fl.addRow("β₁ (negative phase):", self._beta1_spin)

        self._tau_spin = QDoubleSpinBox()
        self._tau_spin.setRange(0.1, 1000); self._tau_spin.setValue(10.0)
        fl.addRow("τ (filter, ms):", self._tau_spin)

        self._fc_spin = QDoubleSpinBox()
        self._fc_spin.setRange(0.1, 100); self._fc_spin.setValue(8.0)
        fl.addRow("fc (Butterworth, Hz):", self._fc_spin)

        btn_fit_rule = QPushButton("Fit Rule to Psychophysics Data")
        btn_fit_rule.clicked.connect(self._fit_linking_model)
        fl.addRow(btn_fit_rule)

        lv.addWidget(grp_link)

        # --- TF/SF grid ---
        grp_grid = QGroupBox("TF / SF Grid")
        fg = QFormLayout(grp_grid)

        self._ntf_spin = QSpinBox()
        self._ntf_spin.setRange(3, 20); self._ntf_spin.setValue(12)
        fg.addRow("N TF:", self._ntf_spin)

        self._nsf_spin = QSpinBox()
        self._nsf_spin.setRange(3, 20); self._nsf_spin.setValue(12)
        fg.addRow("N SF:", self._nsf_spin)

        lv.addWidget(grp_grid)

        # --- Predict ---
        btn_predict = QPushButton("Predict Opto TCSF")
        btn_predict.setMinimumHeight(40)
        btn_predict.setStyleSheet("font-weight:bold; background:#2c5f8a; color:white;")
        btn_predict.clicked.connect(self._run_prediction)
        lv.addWidget(btn_predict)

        btn_export = QPushButton("Export Results CSV")
        btn_export.clicked.connect(self._export_results)
        lv.addWidget(btn_export)

        # --- Right panel ---
        right = QWidget()
        rv = QVBoxLayout(right)
        self._inner_tabs = QTabWidget()
        rv.addWidget(self._inner_tabs)

        self._canvas_3d = PlotCanvas(figsize=(7, 5))
        self._canvas_tf = PlotCanvas(figsize=(7, 4))
        self._inner_tabs.addTab(self._canvas_3d, "3-D TCSF")
        self._inner_tabs.addTab(self._canvas_tf, "Attenuation vs TF")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._on_src_changed(0)

    # ------------------------------------------------------------------
    # Source / UI helpers
    # ------------------------------------------------------------------

    def _on_src_changed(self, idx: int):
        # 0 = .mat, 1 = CSV, 2 = JSON
        self._src_stack.setCurrentIndex(0 if idx == 0 else 1)

    def _on_rule_changed(self, rule: str):
        self._beta0_spin.setEnabled(rule in ("Probability Summation", "1OrderFilter", "2OrderFilter"))
        self._beta1_spin.setEnabled(rule in ("Probability Summation", "1OrderFilter", "2OrderFilter"))
        self._tau_spin.setEnabled(rule == "1OrderFilter")
        self._fc_spin.setEnabled(rule == "2OrderFilter")

    # ------------------------------------------------------------------
    # .mat list management
    # ------------------------------------------------------------------

    def _add_mat_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select baseline .mat files", "", "MATLAB files (*.mat)"
        )
        for path in paths:
            self._load_single_mat(path)

    def _remove_mat_selected(self):
        rows = sorted(
            (self._mat_list.row(item) for item in self._mat_list.selectedItems()),
            reverse=True,
        )
        for row in rows:
            item = self._mat_list.item(row)
            if item is None:
                continue
            label = item.data(Qt.UserRole)
            self._mat_trial_data.pop(label, None)
            self._mat_list.takeItem(row)
        self._base_status_lbl.setText("Not loaded")
        self._base_status_lbl.setStyleSheet("color: gray;")
        self._baseline_S = None

    def _load_single_mat(self, path: str):
        try:
            fname = Path(path).stem
            info = _parse_mat_filename(fname)
        except ValueError as ex:
            QMessageBox.critical(self, "Filename error", str(ex))
            return

        # Label = sID + cond_type (merges v1+v2 from the same subject+condition)
        label = f"{info['sID']}_{info['cond_type']}"
        if info['opsin']:
            label += f"_{info['opsin']}"

        try:
            arr = _load_mat_to_array(path)
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))
            return

        # Merge v1+v2: find existing entry with same sID+cond_type
        merge_key = None
        for existing in list(self._mat_trial_data.keys()):
            if existing == label:
                merge_key = existing
                break

        if merge_key is not None:
            self._mat_trial_data[merge_key] = np.vstack(
                [self._mat_trial_data[merge_key], arr]
            )
            total = len(self._mat_trial_data[merge_key])
            for i in range(self._mat_list.count()):
                item = self._mat_list.item(i)
                if item.data(Qt.UserRole) == merge_key:
                    item.setText(f"{merge_key}  ({total} trials)")
                    break
            self.status.showMessage(f"Merged into: {merge_key}  ({total} trials)")
        else:
            self._mat_trial_data[label] = arr
            item = QListWidgetItem(f"{label}  ({len(arr)} trials)")
            item.setData(Qt.UserRole, label)
            self._mat_list.addItem(item)
            self.status.showMessage(f"Loaded: {label}")

    # ------------------------------------------------------------------
    # Single-file browse (CSV / JSON)
    # ------------------------------------------------------------------

    def _browse_baseline(self):
        idx = self._base_src_combo.currentIndex()
        filt = "CSV files (*.csv)" if idx == 1 else "JSON files (*.json)"
        path, _ = QFileDialog.getOpenFileName(self, "Open baseline file", "", filt)
        if path:
            self._base_path_edit.setText(path)

    # ------------------------------------------------------------------
    # Load Baseline
    # ------------------------------------------------------------------

    def _load_baseline(self):
        src = self._base_src_combo.currentIndex()
        ntf = self._ntf_spin.value()
        nsf = self._nsf_spin.value()

        try:
            if src == 0:  # .mat
                if not self._mat_trial_data:
                    QMessageBox.warning(self, "No files", "Add at least one .mat file first.")
                    return
                # Average all loaded baseline datasets
                all_arrays = list(self._mat_trial_data.values())
                arr = np.vstack(all_arrays)
                tf, sf, S = estimate_sensitivity_grid(arr, ntf=ntf, nsf=nsf)

            elif src == 1:  # CSV
                path = self._base_path_edit.text().strip()
                if not path or not Path(path).exists():
                    QMessageBox.warning(self, "File not found", "Select a valid CSV file.")
                    return
                import pandas as pd
                df = pd.read_csv(path)
                if "Condition" in df.columns:
                    df = df[df["Condition"].str.lower().str.contains("baseline")]
                arr = df[["TF", "SF", "Sensitivity"]].values
                tf = np.unique(arr[:, 0])
                sf = np.unique(arr[:, 1])
                S = arr[:, 2].reshape(len(tf), len(sf))

            else:  # JSON
                path = self._base_path_edit.text().strip()
                if not path or not Path(path).exists():
                    QMessageBox.warning(self, "File not found", "Select a valid JSON file.")
                    return
                with open(path) as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    arr = np.array([[r["tf"], r["sf"], r["contrast_level"], r["response"]]
                                    for r in raw])
                else:
                    cond_data = next(iter(raw.values()))
                    arr = np.array([[r["tf"], r["sf"], r["contrast_level"], r["response"]]
                                    for r in cond_data])
                tf, sf, S = estimate_sensitivity_grid(arr, ntf=ntf, nsf=nsf)

            self._baseline_tf = tf
            self._baseline_sf = sf
            self._baseline_S = S
            self._base_status_lbl.setText(f"Loaded ({S.shape[0]}×{S.shape[1]} grid)")
            self._base_status_lbl.setStyleSheet("color: green;")

        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _get_dec_params(self) -> DecisionParams:
        rule = self._rule_combo.currentText()
        return DecisionParams(
            rule=rule,
            beta=[self._beta0_spin.value(), self._beta1_spin.value()],
            tau=self._tau_spin.value(),
            fc=self._fc_spin.value(),
        )

    def _run_prediction(self):
        if self._baseline_S is None:
            QMessageBox.warning(self, "No baseline", "Load a baseline TCSF first.")
            return

        selected_opsins = [n for n, cb in self._opsin_checks.items() if cb.isChecked()]
        if not selected_opsins:
            QMessageBox.warning(self, "No opsins", "Select at least one opsin.")
            return

        ntf = self._ntf_spin.value()
        nsf = self._nsf_spin.value()
        tf_list = np.exp(np.linspace(np.log(1.5), np.log(20), ntf))
        sf_list = self._baseline_sf if self._baseline_sf is not None else \
            np.exp(np.linspace(np.log(0.25), np.log(16), nsf))

        baseline_S = self._baseline_S
        if baseline_S.shape != (ntf, nsf):
            from scipy.interpolate import RegularGridInterpolator
            try:
                interp = RegularGridInterpolator(
                    (self._baseline_tf, self._baseline_sf), baseline_S,
                    method="linear", bounds_error=False,
                    fill_value=np.nanmean(baseline_S)
                )
                TF_new, SF_new = np.meshgrid(tf_list, sf_list, indexing="ij")
                baseline_S = interp((TF_new, SF_new))
            except Exception:
                baseline_S = np.full((ntf, nsf), np.nanmean(baseline_S))

        self.status.showMessage("Predicting…")
        self._worker = PredictionWorker(
            opsin_names=selected_opsins,
            tf_list=tf_list,
            sf_list=sf_list,
            ntf=ntf,
            nsf=nsf,
            baseline_S=baseline_S,
            irr=self._irr_spin.value(),
            V=self._V_spin.value(),
            dec_params=self._get_dec_params(),
            win_size=1.0,
        )
        self._worker.finished.connect(self._on_prediction_done)
        self._worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
        self._worker.start()

    def _on_prediction_done(self, results: dict):
        self._results = results
        self.status.showMessage("Prediction complete.")
        self._plot_3d()
        self._plot_attenuation()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_3d(self):
        self._canvas_3d.fig.clear()
        ax = self._canvas_3d.fig.add_subplot(111, projection="3d")

        from matplotlib.patches import Patch
        legend_elems = []
        for ci, (label, (tf, sf, S)) in enumerate(self._results.items()):
            color = _color_for_opsin(label, ci)
            LSF, LTF = np.meshgrid(np.log(sf), np.log(tf))
            ax.plot_surface(LSF, LTF, S, color=color, alpha=0.5, edgecolor="none")
            legend_elems.append(Patch(facecolor=color, alpha=0.6, label=label))

        tf_ticks = [1.5, 5, 18]
        sf_ticks = [1.5, 5, 16]
        ax.set_xticks(np.log(sf_ticks))
        ax.set_xticklabels([str(t) for t in sf_ticks], fontsize=7)
        ax.set_yticks(np.log(tf_ticks))
        ax.set_yticklabels([str(t) for t in tf_ticks], fontsize=7)
        ax.set_xlabel("SF (cpd)", fontsize=8, labelpad=8)
        ax.set_ylabel("TF (Hz)", fontsize=8, labelpad=8)
        ax.set_zlabel("Log sensitivity", fontsize=8)
        ax.view_init(elev=15, azim=105)
        ax.set_title("Predicted Opto TCSF", fontsize=10)
        ax.legend(handles=legend_elems, fontsize=7, loc="upper right")
        self._canvas_3d.redraw()

    def _plot_attenuation(self):
        self._canvas_tf.fig.clear()
        ax = self._canvas_tf.fig.add_subplot(111)

        for ci, (label, (tf, sf, S)) in enumerate(self._results.items()):
            color = _color_for_opsin(label, ci)
            ax.plot(np.log(tf), np.nanmean(S, axis=1),
                    color=color, lw=1.5, marker="o", ms=4, label=label)

        tf_ticks = [1.5, 5, 18]
        from matplotlib.ticker import FixedLocator
        ax.xaxis.set_major_locator(FixedLocator(np.log(tf_ticks)))
        ax.set_xticklabels([str(t) for t in tf_ticks])
        ax.set_xlabel("TF (Hz)")
        ax.set_ylabel("Mean log sensitivity")
        ax.set_title("Mean TCSF vs Temporal Frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._canvas_tf.redraw()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _fit_linking_model(self):
        QMessageBox.information(
            self, "Fit Linking Model",
            "Load psychophysics data in the TCSF Estimator tab first, then return here.\n\n"
            "The fitting optimises the decision rule parameters to minimise MSE "
            "between the predicted and measured opsin TCSF curves.",
        )

    def _export_results(self):
        if not self._results:
            QMessageBox.warning(self, "No results", "Run prediction first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not path:
            return
        rows = []
        for label, (tf, sf, S) in self._results.items():
            for i, t in enumerate(tf):
                for j, s in enumerate(sf):
                    rows.append(f"{t:.6f},{s:.6f},{label},{S[i,j]:.6f}")
        with open(path, "w") as f:
            f.write("TF,SF,Condition,PredictedSensitivity\n")
            f.write("\n".join(rows))
        self.status.showMessage(f"Exported to {path}")
