"""
Module 3 – TCSF Estimator tab.

Loads psychophysics data (JSON / CSV / MATLAB .mat files saved by the experiment
module), fits Weibull functions per TF-SF grid cell, and displays TCSF surfaces.

.mat naming convention (from the MATLAB experiment):
  <sID>_qCSF_baseline_v1_<timestamp>.mat   – fixed TF blocks
  <sID>_qCSF_baseline_v2_<timestamp>.mat   – fixed SF blocks
  <sID>_qCSF_opto_v1_<opsin>_<timestamp>.mat
  <sID>_qCSF_opto_v2_<opsin>_<timestamp>.mat
"""

import json
import re
import numpy as np
from pathlib import Path

import scipy.io as sio

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QFormLayout, QSplitter, QMessageBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QAbstractItemView, QTabWidget, QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import Qt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ._plot_canvas import PlotCanvas
from ..core.qcsf_algo import estimate_sensitivity_grid


# ---------------------------------------------------------------------------
# .mat filename / field-name helpers
# ---------------------------------------------------------------------------

def _parse_mat_filename(fname: str) -> dict:
    """Extract sID, condition, opsin, fix_temporal from a .mat stem.

    Expected patterns:
      <sID>_qCSF_baseline_v1_<ts>
      <sID>_qCSF_opto_v1_<opsin>_<ts>   (opsin may be absent for older files)
    """
    m = re.search(r'(baseline|opto)_(v1|v2)', fname)
    if m is None:
        raise ValueError(f"Cannot infer condition from filename: '{fname}'\n"
                         "Expected pattern: <sID>_qCSF_<baseline|opto>_<v1|v2>[_<opsin>]_<timestamp>")

    cond_type = m.group(1)     # 'baseline' or 'opto'
    cond_ver  = m.group(2)     # 'v1' or 'v2'
    fix_temporal = (cond_ver == 'v1')  # v1 = fixed TF, vary SF; v2 = fixed SF, vary TF

    sid_m = re.match(r'^(.+?)_qCSF_', fname)
    sID = sid_m.group(1) if sid_m else fname[:8]

    opsin = ''
    if cond_type == 'opto':
        after = fname[m.end():]  # text after "opto_v1" or "opto_v2"
        op_m = re.match(r'_([A-Za-z][A-Za-z0-9]+?)_\d{2}-', after)
        if op_m:
            opsin = op_m.group(1)

    # v1+v2 are two sweeps of the same condition; merge under one label
    label = f"{sID}_{cond_type}"
    cond_key = cond_type
    if opsin:
        label += f"_{opsin}"
        cond_key = f"{cond_type}_{opsin}"

    return {'sID': sID, 'cond': f"{cond_type}_{cond_ver}", 'cond_key': cond_key,
            'opsin': opsin, 'fix_temporal': fix_temporal, 'label': label}


def _parse_freq_field(field: str, fix_temporal: bool) -> float:
    """Convert a struct field name like 'TF_5' or 'SF_1_5' to a float.

    MATLAB replaces dots in struct field names with underscores, so
    TF_1.5 becomes TF_1_5. We reverse that heuristically.
    """
    prefix = 'TF_' if fix_temporal else 'SF_'
    if not field.startswith(prefix):
        raise ValueError(f"Unexpected field name '{field}' for prefix '{prefix}'")
    val_str = field[len(prefix):]
    parts = val_str.split('_')
    if len(parts) == 1:
        return float(parts[0])
    try:
        return float(f"{parts[0]}.{''.join(parts[1:])}")
    except ValueError:
        return float(parts[0])


def _parse_dict_key(key: str, fix_temporal: bool) -> float:
    """Convert a dictionary key like 'TF_5' or 'SF_1.5' to a float."""
    prefix = 'TF_' if fix_temporal else 'SF_'
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected key '{key}' for prefix '{prefix}'")
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


_SCALE_FACTORS_CACHE = None


def _load_scale_factors() -> dict:
    """Load physiological scale factors from scaling.mat files.

    Mirrors FitPsycho2DataSpotlight.m lines 71-81:
        sc.ChRmine = 1
        sc.OpsinX  = (chrmine_sc(1)*OpsinX_sc(2)) / (chrmine_sc(2)*OpsinX_sc(1))

    Returns {opsin_name: scale_factor} where ChRmine is the reference (=1.0).
    Returns an empty dict if the files are not found.
    """
    global _SCALE_FACTORS_CACHE
    if _SCALE_FACTORS_CACHE is not None:
        return _SCALE_FACTORS_CACHE

    loadmats_dir = Path(__file__).parent.parent.parent / 'Data' / 'LoadMats'
    if not loadmats_dir.is_dir():
        _SCALE_FACTORS_CACHE = {}
        return _SCALE_FACTORS_CACHE

    try:
        chrmine_sc = sio.loadmat(
            str(loadmats_dir / 'ChRmine' / 'scaling.mat'), squeeze_me=True
        )['scaling']
    except Exception:
        _SCALE_FACTORS_CACHE = {}
        return _SCALE_FACTORS_CACHE

    sc = {'ChRmine': 1.0}
    for opsin in ['ChR2', 'ReaChR']:
        try:
            opsin_sc = sio.loadmat(
                str(loadmats_dir / opsin / 'scaling.mat'), squeeze_me=True
            )['scaling']
            sc[opsin] = float(
                (chrmine_sc[0] * opsin_sc[1]) / (chrmine_sc[1] * opsin_sc[0])
            )
        except Exception:
            pass

    _SCALE_FACTORS_CACHE = sc
    return sc


_COLORS = {
    "chrmine": (0.00, 0.75, 0.77),
    "chr2":    (0.78, 0.49, 1.00),
    "reachr":  (0.97, 0.46, 0.43),
    "baseline": (0.49, 0.68, 0.00),
}

# Extra palette for labels that don't match any key above
_EXTRA_COLORS = [
    (0.20, 0.60, 0.86),
    (0.95, 0.75, 0.10),
    (0.80, 0.40, 0.10),
    (0.60, 0.20, 0.60),
]


def _color_for(label: str, extra_idx: int = 0):
    low = label.lower()
    for k, v in _COLORS.items():
        if k in low:
            return v
    return _EXTRA_COLORS[extra_idx % len(_EXTRA_COLORS)]


class TCSFEstimatorTab(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status = status_bar
        self._datasets = {}             # {label: np.ndarray [TF, SF, log_contrast, response]}
        self._surfaces = {}             # {label: (tf, sf, S)} — what gets plotted
        self._surface_errors = {}       # {label: (tf, sf, SEM)} — only in averaged mode
        self._per_subject_surfaces = {} # {label: (tf, sf, S)} — always one per dataset
        self._dataset_meta = {}         # {label: {'sID': str, 'cond_key': str}}
        self._subject_canvases = []     # dynamically-added per-subject plot tabs
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel
        left = QWidget()
        left.setMaximumWidth(380)
        lv = QVBoxLayout(left)
        lv.setAlignment(Qt.AlignTop)
        lv.setSpacing(8)

        # File loading
        grp_load = QGroupBox("Load Data")
        fl = QVBoxLayout(grp_load)

        btn_row = QHBoxLayout()
        btn_json = QPushButton("Add JSON file(s)")
        btn_json.clicked.connect(self._load_json)
        btn_csv = QPushButton("Add CSV file(s)")
        btn_csv.clicked.connect(self._load_csv)
        btn_mat = QPushButton("Add .mat file(s)")
        btn_mat.clicked.connect(self._load_mat)
        btn_mat.setToolTip(
            "Load MATLAB qCSF .mat files.\n"
            "Naming convention:\n"
            "  <sID>_qCSF_baseline_v1_<ts>.mat  (fixed TF)\n"
            "  <sID>_qCSF_baseline_v2_<ts>.mat  (fixed SF)\n"
            "  <sID>_qCSF_opto_v1_<opsin>_<ts>.mat\n"
            "  <sID>_qCSF_opto_v2_<opsin>_<ts>.mat\n\n"
            "Supports both struct and dictionary (MCOS) formats.\n"
            "MCOS files require MATLAB R2022b with the Python engine installed."
        )
        btn_row.addWidget(btn_json)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_mat)
        fl.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(130)
        self._file_list.setSelectionMode(QAbstractItemView.MultiSelection)
        fl.addWidget(QLabel("Loaded datasets:"))
        fl.addWidget(self._file_list)

        btn_remove = QPushButton("Remove selected")
        btn_remove.clicked.connect(self._remove_selected)
        fl.addWidget(btn_remove)
        lv.addWidget(grp_load)

        # Grid parameters
        grp_grid = QGroupBox("Grid Parameters")
        fg = QFormLayout(grp_grid)

        self._ntf_spin = QSpinBox()
        self._ntf_spin.setRange(2, 20); self._ntf_spin.setValue(12)
        fg.addRow("N TF bins:", self._ntf_spin)

        self._nsf_spin = QSpinBox()
        self._nsf_spin.setRange(2, 20); self._nsf_spin.setValue(12)
        fg.addRow("N SF bins:", self._nsf_spin)

        self._win_spin = QDoubleSpinBox()
        self._win_spin.setRange(0.1, 5.0); self._win_spin.setValue(1.0)
        fg.addRow("Window size (log units):", self._win_spin)

        lv.addWidget(grp_grid)

        # Scaling options
        grp_scale = QGroupBox("Sensitivity Scaling")
        fs = QFormLayout(grp_scale)

        self._unscale_check = QCheckBox("Preserve relative sensitivity across opsins")
        self._unscale_check.setToolTip(
            "Align each opto surface to the average baseline surface\n"
            "so that sensitivity differences reflect opsin gain, not\n"
            "overall scaling differences between conditions."
        )
        self._unscale_check.setChecked(True)
        fs.addRow(self._unscale_check)

        lv.addWidget(grp_scale)

        # Multi-subject plot mode
        grp_mode = QGroupBox("Multi-subject Plot Mode")
        fm = QVBoxLayout(grp_mode)
        self._radio_avg   = QRadioButton("Average across subjects")
        self._radio_indiv = QRadioButton("Individual subplots per subject")
        self._radio_avg.setChecked(True)
        _bg = QButtonGroup(self)
        _bg.addButton(self._radio_avg)
        _bg.addButton(self._radio_indiv)
        fm.addWidget(self._radio_avg)
        fm.addWidget(self._radio_indiv)
        lv.addWidget(grp_mode)

        # Estimate button
        btn_est = QPushButton("Estimate TCSF")
        btn_est.setMinimumHeight(36)
        btn_est.setStyleSheet("font-weight:bold;")
        btn_est.clicked.connect(self._estimate)
        lv.addWidget(btn_est)

        # Export
        btn_export = QPushButton("Export to CSV")
        btn_export.clicked.connect(self._export_csv)
        lv.addWidget(btn_export)

        # Right panel
        right = QWidget()
        rv = QVBoxLayout(right)
        self._inner_tabs = QTabWidget()
        rv.addWidget(self._inner_tabs)

        self._canvas_3d = PlotCanvas(figsize=(7, 5))
        self._canvas_2d = PlotCanvas(figsize=(7, 4))
        self._inner_tabs.addTab(self._canvas_3d, "3-D TCSF Surface")
        self._inner_tabs.addTab(self._canvas_2d, "TF slices")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------

    def _load_json(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open JSON experiment files", "", "JSON files (*.json)"
        )
        for p in paths:
            self._load_json_file(p)

    def _load_json_file(self, path: str):
        try:
            with open(path) as f:
                raw = json.load(f)
            # Expected format: list of dicts with keys tf, sf, contrast_level, response
            # OR {condition: {tf: .., sf: .., contrast_level: .., response: ..}}
            label = Path(path).stem
            if isinstance(raw, list):
                arr = np.array([[r["tf"], r["sf"], r["contrast_level"], r["response"]]
                                for r in raw])
                self._add_dataset(label, arr)
            elif isinstance(raw, dict):
                # Multi-condition dict
                for cond, trials in raw.items():
                    if isinstance(trials, list):
                        arr = np.array([[r["tf"], r["sf"], r["contrast_level"], r["response"]]
                                        for r in trials])
                        self._add_dataset(f"{label}/{cond}", arr)
        except Exception as ex:
            QMessageBox.critical(self, "Load error", str(ex))

    def _load_csv(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open CSV data files", "", "CSV files (*.csv)"
        )
        for p in paths:
            try:
                arr = np.loadtxt(p, delimiter=",", skiprows=1)
                label = Path(p).stem
                self._add_dataset(label, arr)
            except Exception as ex:
                QMessageBox.critical(self, "Load error", str(ex))

    def _load_mat(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open MATLAB qCSF data files", "", "MATLAB files (*.mat)"
        )
        for p in paths:
            self._load_mat_file(p)

    def _load_mat_file(self, path: str):
        """Load a MATLAB .mat qCSF file and extract trial data.

        The file must contain a struct variable whose fields correspond to
        frequency blocks (e.g. TF_5, SF_1_5).  Each block must have a
        nested qcsf.data.history array with columns:
            [trial, VF, contrast_linear, response]
        where VF is the varying frequency (SF for v1, TF for v2).

        If the file uses MATLAB's dictionary type (MCOS), a clear error is
        shown with instructions for fixing the MATLAB experiment code.
        """
        try:
            fname = Path(path).stem
            cond_info = _parse_mat_filename(fname)
        except ValueError as ex:
            QMessageBox.critical(self, "Filename error", str(ex))
            return

        try:
            mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
        except Exception as ex:
            QMessageBox.critical(self, "Load error", f"scipy.io failed:\n{ex}")
            return

        # Find the main data variable (skip private __ keys)
        data_key = next((k for k in mat if not k.startswith('__')), None)
        if data_key is None:
            QMessageBox.critical(self, "Load error", "No data variable found in file.")
            return

        data_obj = mat[data_key]

        fix_temporal = cond_info['fix_temporal']
        label = cond_info['label']
        all_rows = []
        bad_fields = []

        if hasattr(data_obj, '_fieldnames'):
            # Struct format — read directly
            for field in data_obj._fieldnames:
                try:
                    fixed_val = _parse_freq_field(field, fix_temporal)
                except ValueError:
                    bad_fields.append(field)
                    continue
                try:
                    block = getattr(data_obj, field)
                    history = block.qcsf.data.history
                    if history.ndim == 1:
                        history = history.reshape(1, -1)
                    if history.shape[1] < 4:
                        bad_fields.append(field)
                        continue
                    block_rows = self._history_block_to_rows(
                        history, fixed_val, fix_temporal)
                    all_rows.append(block_rows)
                except AttributeError as ex:
                    bad_fields.append(f"{field} ({ex})")
        else:
            # MCOS dictionary format — try MATLAB engine
            eng = _get_matlab_engine()
            if eng is None:
                QMessageBox.critical(
                    self, "Unsupported .mat format",
                    "This file uses MATLAB's dictionary type (R2022b+).\n"
                    "The MATLAB engine for Python is not installed.\n\n"
                    "Install it with:\n"
                    "  cd /usr/local/MATLAB/R2022b/extern/engines/python\n"
                    "  python setup.py install"
                )
                return
            import io as _io
            try:
                self.status.showMessage("Loading via MATLAB engine…")
                # scipy.io returns the key as the Python string 'None' for
                # MCOS files (it cannot decode the variable name).  We must
                # not pass that string into MATLAB.  Instead: clear the
                # workspace entirely, load the file, ask MATLAB for the
                # actual variable name via who(), and alias it to tmpdict so
                # all subsequent accesses use a stable name.
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
                        bad_fields.append(field)
                        continue
                    try:
                        history = np.array(
                            eng.eval(
                                f"tmpdict(matkeys{{{i}}}).qcsf.data.history",
                                nargout=1))
                        if history.ndim == 1:
                            history = history.reshape(1, -1)
                        if history.shape[1] < 4:
                            bad_fields.append(field)
                            continue
                        block_rows = self._history_block_to_rows(
                            history, fixed_val, fix_temporal)
                        all_rows.append(block_rows)
                    except Exception as ex:
                        bad_fields.append(f"{field} ({ex})")
            except Exception as ex:
                QMessageBox.critical(self, "MATLAB engine error", str(ex))
                return

        if bad_fields:
            self.status.showMessage(
                f"Skipped {len(bad_fields)} block(s) with unexpected structure."
            )

        if not all_rows:
            QMessageBox.critical(
                self, "Load error",
                "No valid trial blocks were found.\n"
                f"Check that the fields follow the pattern "
                f"{'TF_<hz>' if fix_temporal else 'SF_<cpd>'} and that "
                "qcsf.data.history exists in each block."
            )
            return

        arr = np.vstack(all_rows)
        self._add_dataset(label, arr,
                          sID=cond_info['sID'],
                          cond_key=cond_info['cond_key'])

    @staticmethod
    def _history_block_to_rows(history: np.ndarray, fixed_val: float,
                               fix_temporal: bool) -> np.ndarray:
        """Convert a history matrix (N×4) to a [TF, SF, contrast_level, response] array."""
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

    def _add_dataset(self, label: str, arr: np.ndarray,
                     sID: str = '', cond_key: str = ''):
        if arr.shape[1] < 4:
            QMessageBox.warning(self, "Format error",
                                f"'{label}': expected ≥4 columns [TF, SF, contrast_level, response].")
            return

        # Merge v1+v2 sweeps from the SAME subject+condition into one dataset.
        # Match on sID+cond_key stored in metadata so that two different subjects
        # whose filenames happen to produce the same label string never silently
        # collapse into each other.
        merge_label = None
        if sID:
            for existing_label, meta in self._dataset_meta.items():
                if meta.get('sID') == sID and meta.get('cond_key') == cond_key:
                    merge_label = existing_label
                    break

        if merge_label is not None:
            self._datasets[merge_label] = np.vstack([self._datasets[merge_label], arr])
            total = len(self._datasets[merge_label])
            for i in range(self._file_list.count()):
                item = self._file_list.item(i)
                if item.data(Qt.UserRole) == merge_label:
                    item.setText(f"{merge_label}  ({total} trials)")
                    break
            self.status.showMessage(f"Merged into: {merge_label}  ({total} trials total)")
        else:
            # Guarantee the label key is unique even if two subjects share a label string
            unique_label = label
            suffix = 1
            while unique_label in self._datasets:
                unique_label = f"{label}_{suffix}"
                suffix += 1
            self._datasets[unique_label] = arr
            self._dataset_meta[unique_label] = {
                'sID':      sID      or unique_label,
                'cond_key': cond_key or unique_label,
            }
            item = QListWidgetItem(f"{unique_label}  ({len(arr)} trials)")
            item.setData(Qt.UserRole, unique_label)
            self._file_list.addItem(item)
            self.status.showMessage(f"Loaded: {unique_label}")

    def _remove_selected(self):
        # Collect rows in descending order so removing one doesn't shift others
        rows = sorted(
            (self._file_list.row(item) for item in self._file_list.selectedItems()),
            reverse=True,
        )
        for row in rows:
            item = self._file_list.item(row)
            if item is None:
                continue
            label = item.data(Qt.UserRole)
            self._datasets.pop(label, None)
            self._dataset_meta.pop(label, None)
            self._file_list.takeItem(row)
        self.status.showMessage("Removed selected datasets.")

    def _estimate(self):
        if not self._datasets:
            QMessageBox.warning(self, "No data", "Load at least one data file first.")
            return

        ntf = self._ntf_spin.value()
        nsf = self._nsf_spin.value()
        win = self._win_spin.value()
        unscale = self._unscale_check.isChecked()

        # Estimate one surface per dataset (always per-subject)
        per_subject = {}
        for label, data in self._datasets.items():
            tf, sf, S = estimate_sensitivity_grid(data, ntf=ntf, nsf=nsf, win_size=win)
            per_subject[label] = (tf, sf, S)

        # Sensitivity correction for opto surfaces.
        #
        # CHECKED ("Preserve relative sensitivity"):
        #   Align each opsin independently to the averaged baseline surface at the
        #   SF index where the baseline peaks in the lowest TF row.  This preserves
        #   the shape of each surface while matching absolute level to baseline.
        #
        # UNCHECKED:
        #   Apply physiological scale factors from scaling.mat
        #   (addFac = -log10(sc[opsin]), mirroring FitPsycho2DataSpotlight.m lines 133-142).
        #   ChRmine is the reference (sc=1, addFac=0); ChR2 and ReaChR are scaled relative to it.

        baseline_surfaces = [
            S for label, (_, __, S) in per_subject.items()
            if self._dataset_meta.get(label, {}).get('cond_key', '') == 'baseline'
        ]
        baseline_S = np.nanmean(baseline_surfaces, axis=0) if baseline_surfaces else None

        if unscale:
            # CHECKED: align each opto surface to baseline at the peak-SF index
            if baseline_S is not None:
                idx = int(np.nanargmax(baseline_S[0, :]))
                maxSens_ctl = float(baseline_S[0, idx])
                for label in list(per_subject):
                    if self._dataset_meta.get(label, {}).get('cond_key', '') != 'baseline':
                        tf_, sf_, S = per_subject[label]
                        addSens = maxSens_ctl - float(S[0, idx])
                        if np.isfinite(addSens):
                            per_subject[label] = (tf_, sf_, S + addSens)
        else:
            # UNCHECKED: apply physiological scale factors from scaling.mat
            # (preserves relative sensitivity differences between opsins), then
            # translate all opto surfaces together so that ChRmine aligns with
            # baseline at the SF where baseline peaks in the lowest TF row.
            scale_facs = _load_scale_factors()
            if not scale_facs:
                self.status.showMessage(
                    "Warning: scaling.mat files not found — scale factors not applied."
                )

            # Step 1: apply per-opsin scale factors
            for label in list(per_subject):
                cond_key = self._dataset_meta.get(label, {}).get('cond_key', '')
                if cond_key.startswith('opto_'):
                    opsin_name = cond_key[len('opto_'):]
                    if opsin_name in scale_facs:
                        addFac = float(-np.log10(scale_facs[opsin_name]))
                        tf_, sf_, S = per_subject[label]
                        per_subject[label] = (tf_, sf_, S + addFac)

            # Step 2: translate all opto surfaces together so that at the
            # lowest TF (row 0) and lowest SF (col 0), ChRmine aligns with
            # baseline — which should bring all opsins to the same value there
            # once the scale factors have normalised their relative sensitivity.
            if baseline_S is not None:
                chrmine_Ss = [
                    S for label, (_, __, S) in per_subject.items()
                    if self._dataset_meta.get(label, {}).get('cond_key', '') == 'opto_ChRmine'
                ]
                if chrmine_Ss:
                    mean_chrmine_S = np.nanmean(chrmine_Ss, axis=0)
                    global_shift = float(baseline_S[0, 0]) - float(mean_chrmine_S[0, 0])
                    for label in list(per_subject):
                        if self._dataset_meta.get(label, {}).get('cond_key', '') != 'baseline':
                            tf_, sf_, S = per_subject[label]
                            per_subject[label] = (tf_, sf_, S + global_shift)

        if self._radio_avg.isChecked():
            # Average surfaces across subjects within each condition group
            from collections import defaultdict
            groups = defaultdict(list)
            for label, (tf, sf, S) in per_subject.items():
                ck = self._dataset_meta.get(label, {}).get('cond_key', label)
                groups[ck].append(S)
            ref_tf, ref_sf = next(iter(per_subject.values()))[:2]
            self._surfaces = {
                ck: (ref_tf, ref_sf, np.nanmean(Ss, axis=0))
                for ck, Ss in groups.items()
            }
            # SEM across subjects (zero / omitted when only one subject)
            self._surface_errors = {
                ck: (ref_tf, ref_sf,
                     np.nanstd(Ss, axis=0, ddof=1) / np.sqrt(len(Ss))
                     if len(Ss) > 1 else np.zeros_like(Ss[0]))
                for ck, Ss in groups.items()
            }
        else:
            self._surfaces = per_subject
            self._surface_errors = {}

        self._per_subject_surfaces = per_subject  # kept for individual subplot mode

        self._plot_3d()
        self._plot_2d_slices()
        self.status.showMessage("TCSF estimation complete.")

    def _draw_3d_ax(self, ax, surfaces: dict, title: str = "TCSF Surfaces"):
        """Populate a 3D axes with TCSF surfaces from *surfaces* {label: (tf, sf, S)}."""
        from matplotlib.patches import Patch
        tf_ticks = [1.5, 5, 18]
        sf_ticks = [1.5, 5, 16]
        legend_elems = []
        for ei, (label, (tf, sf, S)) in enumerate(surfaces.items()):
            color = _color_for(label, ei)
            LSF, LTF = np.meshgrid(np.log(sf), np.log(tf))
            ax.plot_surface(LSF, LTF, np.where(np.isnan(S), np.nan, S),
                            color=color, alpha=0.5, edgecolor="none")
            legend_elems.append(Patch(facecolor=color, alpha=0.7, label=label))
        ax.set_xticks(np.log(sf_ticks))
        ax.set_xticklabels([str(t) for t in sf_ticks], fontsize=7)
        ax.set_yticks(np.log(tf_ticks))
        ax.set_yticklabels([str(t) for t in tf_ticks], fontsize=7)
        ax.set_xlabel("SF (cpd)", fontsize=8, labelpad=8)
        ax.set_ylabel("TF (Hz)", fontsize=8, labelpad=8)
        ax.set_zlabel("Log sensitivity", fontsize=8)
        ax.view_init(elev=15, azim=105)
        ax.set_title(title, fontsize=10)
        ax.legend(handles=legend_elems, fontsize=7, loc="upper right")

    def _plot_3d(self):
        self._canvas_3d.fig.clear()
        if self._radio_indiv.isChecked():
            # Individual mode: each subject gets its own 3D in a per-subject tab
            self._canvas_3d.redraw()
            return
        ax = self._canvas_3d.fig.add_subplot(111, projection="3d")
        self._draw_3d_ax(ax, self._surfaces)
        self._canvas_3d.redraw()

    def _clear_subject_tabs(self):
        """Remove all dynamically-added per-subject plot tabs."""
        for canvas in self._subject_canvases:
            idx = self._inner_tabs.indexOf(canvas)
            if idx >= 0:
                self._inner_tabs.removeTab(idx)
            canvas.deleteLater()
        self._subject_canvases.clear()

    def _plot_2d_slices(self):
        self._clear_subject_tabs()
        self._canvas_2d.fig.clear()

        if not self._surfaces:
            self._canvas_2d.redraw()
            return

        if self._radio_indiv.isChecked():
            self._plot_2d_individual()
        else:
            self._plot_2d_overlaid(self._surfaces)
            self._canvas_2d.redraw()

    def _get_fixed_sf_values(self) -> list:
        """Return sorted unique SF values that came from v2 (fixed-SF) blocks.

        In v2 blocks the SF column is np.full(n, fixed_val), so every value in
        that group is bitwise identical (std ≈ 0).  Adaptive SF values from v1
        blocks vary, so they won't satisfy this condition for any meaningful
        cluster size.
        """
        all_fixed_sfs = set()
        for data in self._datasets.values():
            sf_col = data[:, 1]
            for sf_val in np.unique(np.round(sf_col, 4)):
                mask = np.abs(sf_col - sf_val) < 1e-9
                if mask.sum() >= 5 and np.std(sf_col[mask]) < 1e-9:
                    all_fixed_sfs.add(round(float(sf_val), 4))
        return sorted(all_fixed_sfs)

    def _pick_sf_indices(self, sf_grid: np.ndarray) -> np.ndarray:
        """Return 3 SF grid indices: nearest to first, middle, and last fixed-SF value.

        Falls back to evenly-spaced linspace indices when no fixed-SF data is found.
        """
        fixed_sfs = self._get_fixed_sf_values()
        n_panels = min(3, len(sf_grid))
        if len(fixed_sfs) >= 2:
            n = len(fixed_sfs)
            picks = [fixed_sfs[0], fixed_sfs[n // 2], fixed_sfs[-1]]
            return np.array([int(np.argmin(np.abs(sf_grid - v))) for v in picks])
        return np.linspace(0, len(sf_grid) - 1, n_panels, dtype=int)

    @staticmethod
    def _style_2d_ax(ax, si, first_sf, tf_ticks, pi, n_panels,
                     ylim=None, xlim=None, show_ylabel=True):
        """Apply consistent axis styling to one TF-slice panel."""
        from matplotlib.ticker import FixedLocator, MaxNLocator
        ax.xaxis.set_major_locator(FixedLocator(np.log(tf_ticks)))
        ax.set_xticklabels([str(t) for t in tf_ticks], fontsize=7)
        ax.set_xlabel("TF (Hz)", fontsize=8)
        ax.set_title(f"SF ≈ {first_sf[si]:.1f} cpd", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if pi == 0 and show_ylabel:
            ax.set_ylabel("Log sensitivity", fontsize=8)
        else:
            ax.tick_params(labelleft=False)

    def _plot_2d_overlaid(self, surfaces: dict):
        """One row of TF-slice panels, all conditions overlaid in colour."""
        first_tf, first_sf = next(iter(surfaces.values()))[:2]
        sf_idx = self._pick_sf_indices(first_sf)
        n_panels = len(sf_idx)

        all_S = [S for _, __, S in surfaces.values()]
        ylim = (np.nanmin(all_S) - 0.1, np.nanmax(all_S) + 0.1)
        xlim = (float(np.log(first_tf[0])), float(np.log(first_tf[-1])))

        axes = self._canvas_2d.fig.subplots(1, n_panels)
        if n_panels == 1:
            axes = [axes]

        tf_ticks = [1.5, 5, 18]
        for pi, si in enumerate(sf_idx):
            ax = axes[pi]
            for ei, (label, (tf, sf, S)) in enumerate(surfaces.items()):
                color = _color_for(label, ei)
                x = np.log(tf)
                ax.plot(x, S[:, si], color=color, lw=1.5, marker="o", ms=3, label=label)
                if label in self._surface_errors:
                    sem = self._surface_errors[label][2][:, si]
                    ax.fill_between(x, S[:, si] - sem, S[:, si] + sem,
                                    color=color, alpha=0.2, linewidth=0)
            self._style_2d_ax(ax, si, first_sf, tf_ticks, pi, n_panels,
                               ylim=ylim, xlim=xlim)
            if pi == 0:
                ax.legend(fontsize=6)

    def _plot_2d_individual(self):
        """One tab per subject containing a 3-D surface (top) and TF slices (bottom)."""
        per_subj = self._per_subject_surfaces

        from collections import OrderedDict
        subj_groups = OrderedDict()
        for label in per_subj:
            sid = self._dataset_meta.get(label, {}).get('sID', label)
            subj_groups.setdefault(sid, []).append(label)

        first_sf = next(iter(per_subj.values()))[1]
        sf_idx = self._pick_sf_indices(first_sf)
        n_panels = len(sf_idx)
        tf_ticks = [1.5, 5, 18]

        all_S = [S for _, __, S in per_subj.values()]
        vmin = np.nanmin(all_S)
        vmax = np.nanmax(all_S)

        for sid, labels in subj_groups.items():
            subj_surfaces = {
                self._dataset_meta.get(lb, {}).get('cond_key', lb): per_subj[lb]
                for lb in labels
            }

            # Container: vertical splitter with 3D on top, 2D slices on bottom
            container = QWidget()
            splitter = QSplitter(Qt.Vertical, container)
            QVBoxLayout(container).addWidget(splitter)
            container.layout().setContentsMargins(0, 0, 0, 0)

            # ---- 3D surface ----
            canvas_3d = PlotCanvas(figsize=(7, 4))
            canvas_3d.fig.clear()
            ax3d = canvas_3d.fig.add_subplot(111, projection="3d")
            self._draw_3d_ax(ax3d, subj_surfaces, title=sid)
            canvas_3d.redraw()
            splitter.addWidget(canvas_3d)

            # ---- 2D TF slices ----
            canvas_2d = PlotCanvas(figsize=(7, 3))
            axes_2d = canvas_2d.fig.subplots(1, n_panels, squeeze=False)[0]
            ref_tf = next(iter(subj_surfaces.values()))[0]
            ylim = (vmin - 0.1, vmax + 0.1)
            xlim = (float(np.log(ref_tf[0])), float(np.log(ref_tf[-1])))
            for pi, si in enumerate(sf_idx):
                ax = axes_2d[pi]
                for ei, (ck, (tf, sf, S)) in enumerate(subj_surfaces.items()):
                    ax.plot(np.log(tf), S[:, si], color=_color_for(ck, ei),
                            lw=1.5, marker="o", ms=3, label=ck)
                self._style_2d_ax(ax, si, first_sf, tf_ticks, pi, n_panels,
                                   ylim=ylim, xlim=xlim)
                if pi == n_panels - 1:
                    ax.legend(fontsize=6, loc="upper right")
            canvas_2d.redraw()
            splitter.addWidget(canvas_2d)

            splitter.setSizes([400, 250])

            self._inner_tabs.addTab(container, sid)
            self._subject_canvases.append(container)

        if self._subject_canvases:
            self._inner_tabs.setCurrentWidget(self._subject_canvases[0])

    def _export_csv(self):
        if not self._surfaces:
            QMessageBox.warning(self, "No data", "Run estimation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if not path:
            return

        rows = []
        for label, (tf, sf, S) in self._surfaces.items():
            for i, t in enumerate(tf):
                for j, s in enumerate(sf):
                    rows.append(f"{t:.6f},{s:.6f},{label},{S[i,j]:.6f}")

        with open(path, "w") as f:
            f.write("TF,SF,Condition,Sensitivity\n")
            f.write("\n".join(rows))
        self.status.showMessage(f"Exported to {path}")
