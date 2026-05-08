"""
Video Simulation tab – applies the 4-state opsin photocurrent model to every
pixel of an input video and writes a temporally-processed output video.

Pipeline (mirrors opto_simulation.m):
  1. Read video → grayscale → resize by scale_factor → normalise [0, 1]
  2. Upsample each pixel's luminance trace from video fps to upsample_fps
  3. Prepend background-luminance padding (default 0.5 s)
  4. Run 4-state opsin Euler integration (all pixels simultaneously)
  5. Downsample back to video fps, normalise to [0, 1]
  6. Reshape to uint8 [T, H, W] frames for preview and saving
  7. Write output video at input_fps * fps_multiplier
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from skimage.transform import resize as sk_resize
import imageio

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QFileDialog,
    QLineEdit, QFormLayout, QSplitter, QMessageBox, QProgressBar, QSlider,
    QTabWidget, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from ..core.opsin_model import load_all_opsins, OpsinParams

# ---------------------------------------------------------------------------
# Physical constants (same as opsin_model.py)
# ---------------------------------------------------------------------------
H_PLANCK = 6.62607015e-34
C_LIGHT   = 2.99792458e8
_EPS      = 1e-30   # avoid division-by-zero in rate functions


# ---------------------------------------------------------------------------
# GPU / accelerator backend detection
# ---------------------------------------------------------------------------

def _detect_backend():
    """Return (name, module) for the best available numeric backend."""
    try:
        import cupy as cp
        cp.zeros(1)                  # triggers device initialisation
        return "cupy", cp
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "torch", torch
    except Exception:
        pass
    return "numpy", np


_BACKEND, _XP = _detect_backend()


def _backend_label() -> str:
    """Human-readable description of the active backend."""
    if _BACKEND == "cupy":
        try:
            import cupy as cp
            dev = cp.cuda.Device(0)
            name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"]
            if isinstance(name, bytes):
                name = name.decode()
            return f"CuPy  –  {name}"
        except Exception:
            return "CuPy (GPU)"
    if _BACKEND == "torch":
        try:
            import torch
            return f"PyTorch CUDA  –  {torch.cuda.get_device_name(0)}"
        except Exception:
            return "PyTorch CUDA"
    return "NumPy (CPU)"


# ---------------------------------------------------------------------------
# Vectorised 4-state Euler integrator
# ---------------------------------------------------------------------------

def _euler_vectorized(
    params: OpsinParams,
    V: float,
    lambda_nm: float,
    Irr_batch: np.ndarray,   # shape [N_pixels, T], float32
    dt: float,               # time-step in ms
) -> np.ndarray:
    """
    Run the 4-state opsin photocurrent model for all pixels simultaneously.

    Parameters
    ----------
    params     : OpsinParams instance.
    V          : Holding potential (mV).
    lambda_nm  : Stimulus wavelength (nm).
    Irr_batch  : Irradiance traces, shape [N_pixels, T] (W/mm²), float32.
    dt         : Time-step in ms.

    Returns
    -------
    I_out : numpy array [N_pixels, T], photocurrent (pA).
    """
    # ---- choose array namespace ----
    if _BACKEND == "cupy":
        xp = _XP
        data = xp.array(Irr_batch, dtype=xp.float32)
    elif _BACKEND == "torch":
        xp = None          # use torch directly
        import torch
        data = torch.as_tensor(Irr_batch, dtype=torch.float32, device="cuda")
    else:
        xp = np
        data = np.array(Irr_batch, dtype=np.float32)

    N, T = Irr_batch.shape

    # ---- pre-compute constants ----
    phi_scale = (lambda_nm * 1e-9) / (H_PLANCK * C_LIGHT)  # photon flux per W/mm²
    phim_p = params.phim ** params.p
    phim_q = params.phim ** params.q
    p = params.p
    q = params.q

    # ---- helpers to create zero / ones arrays in correct namespace ----
    def _zeros(shape):
        if _BACKEND == "torch":
            return torch.zeros(shape, dtype=torch.float32, device="cuda")
        return xp.zeros(shape, dtype=xp.float32)

    def _ones(shape):
        if _BACKEND == "torch":
            return torch.ones(shape, dtype=torch.float32, device="cuda")
        return xp.ones(shape, dtype=xp.float32)

    def _maximum(a, b):
        if _BACKEND == "torch":
            return torch.clamp(a, min=b)
        return xp.maximum(a, b)

    # ---- initial state: C1=1, O1=O2=C2=0 ----
    C1 = _ones(N)
    O1 = _zeros(N)
    O2 = _zeros(N)
    C2 = _zeros(N)

    # ---- output array ----
    if _BACKEND == "torch":
        I_out = torch.zeros((N, T), dtype=torch.float32, device="cuda")
    elif _BACKEND == "cupy":
        I_out = xp.zeros((N, T), dtype=xp.float32)
    else:
        I_out = np.zeros((N, T), dtype=np.float32)

    # ---- Euler loop over time ----
    for t in range(T - 1):
        if _BACKEND == "torch":
            irr_t = data[:, t]
        else:
            irr_t = data[:, t]

        # photon flux  [N]
        phi = _maximum(irr_t, 0.0) * phi_scale

        # rate functions (vectorised)
        if _BACKEND == "torch":
            pp  = phi ** p
            pq  = phi ** q
            Ga1 = params.k1 * pp / (pp + phim_p + _EPS)
            Ga2 = params.k2 * pp / (pp + phim_p + _EPS)
            Gf  = params.Gf0 + params.kf * pq / (pq + phim_q + _EPS)
            Gb  = params.Gb0 + params.kb * pq / (pq + phim_q + _EPS)
        else:
            pp  = phi ** p
            pq  = phi ** q
            Ga1 = params.k1 * pp / (pp + phim_p + _EPS)
            Ga2 = params.k2 * pp / (pp + phim_p + _EPS)
            Gf  = params.Gf0 + params.kf * pq / (pq + phim_q + _EPS)
            Gb  = params.Gb0 + params.kb * pq / (pq + phim_q + _EPS)

        dC1 = (params.Gd1 * O1 + params.Gr  * C2 - Ga1 * C1) * dt
        dO1 = (Ga1 * C1 + Gb  * O2 - (params.Gd1 + Gf) * O1) * dt
        dO2 = (Ga2 * C2 + Gf  * O1 - (params.Gd2 + Gb) * O2) * dt
        dC2 = (params.Gd2 * O2 - (params.Gr + Ga2) * C2) * dt

        C1 = _maximum(C1 + dC1, 0.0)
        O1 = _maximum(O1 + dO1, 0.0)
        O2 = _maximum(O2 + dO2, 0.0)
        C2 = _maximum(C2 + dC2, 0.0)

        # renormalise: C1 + O1 + O2 + C2 = 1
        total = C1 + O1 + O2 + C2 + _EPS
        C1 = C1 / total
        O1 = O1 / total
        O2 = O2 / total
        C2 = C2 / total

        f_phi = O1 + params.gamma * O2
        I_out[:, t] = params.g0_ph * f_phi * (V - params.E)

    # ---- return numpy array ----
    if _BACKEND == "torch":
        return I_out.cpu().numpy()
    if _BACKEND == "cupy":
        return _XP.asnumpy(I_out)
    return I_out


# ---------------------------------------------------------------------------
# Simulation worker thread
# ---------------------------------------------------------------------------

class SimulationWorker(QThread):
    """Runs the full video-to-opsin-current pipeline off the GUI thread."""

    progress    = pyqtSignal(int, str)             # (percent 0-100, message)
    input_ready = pyqtSignal(object, float)        # (frames_uint8 [T,H,W], input_fps)
    finished    = pyqtSignal(object, float, float) # (frames_uint8 [T,H,W], input_fps, fps_mult)
    error       = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        params: OpsinParams,
        V: float,
        lambda_nm: float,
        irradiance: float,
        scale_factor: float,
        upsample_fps: int,
        pad_duration: float,
        chunk_size: int,
        fps_multiplier: float,
        input_fps_override: float = 0.0,
    ):
        super().__init__()
        self.video_path         = video_path
        self.params             = params
        self.V                  = V
        self.lambda_nm          = lambda_nm
        self.irradiance         = irradiance
        self.scale_factor       = scale_factor
        self.upsample_fps       = upsample_fps
        self.pad_duration       = pad_duration
        self.chunk_size         = chunk_size
        self.fps_multiplier     = fps_multiplier
        self.input_fps_override = input_fps_override
        self._cancelled         = False

    def cancel(self):
        self._cancelled = True

    # ------------------------------------------------------------------
    def run(self):
        try:
            self._run_pipeline()
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    def _run_pipeline(self):
        # ---- 1. Read video ----
        self.progress.emit(1, "Opening video …")
        reader = imageio.get_reader(self.video_path, format="ffmpeg")
        meta   = reader.get_meta_data()
        meta_fps  = float(meta.get("fps", 25.0))
        input_fps = self.input_fps_override if self.input_fps_override > 0 else meta_fps
        _nf = meta.get("nframes", 0)
        n_frames = int(_nf) if _nf and np.isfinite(_nf) else 0

        frames_raw = []
        for idx, frame in enumerate(reader):
            if self._cancelled:
                reader.close()
                return
            # convert to grayscale
            if frame.ndim == 3:
                gray = (
                    0.2989 * frame[:, :, 0].astype(np.float32)
                    + 0.5870 * frame[:, :, 1].astype(np.float32)
                    + 0.1140 * frame[:, :, 2].astype(np.float32)
                )
            else:
                gray = frame.astype(np.float32)
            frames_raw.append(gray)
            if idx == 0:
                self.progress.emit(2, f"Reading frames  (fps={input_fps:.2f}) …")
        reader.close()

        if not frames_raw:
            self.error.emit("No frames could be read from the video.")
            return

        T_orig = len(frames_raw)
        H_orig, W_orig = frames_raw[0].shape

        # ---- 2. Resize ----
        self.progress.emit(5, "Resizing frames …")
        H_small = max(1, int(round(H_orig * self.scale_factor)))
        W_small = max(1, int(round(W_orig * self.scale_factor)))

        frames_small = np.empty((T_orig, H_small, W_small), dtype=np.float32)
        for i, f in enumerate(frames_raw):
            if self._cancelled:
                return
            resized = sk_resize(f, (H_small, W_small), anti_aliasing=True,
                                preserve_range=True)
            frames_small[i] = resized

        # Normalise [0, 1]
        fmin = frames_small.min()
        fmax = frames_small.max()
        if fmax > fmin:
            frames_small = (frames_small - fmin) / (fmax - fmin)
        else:
            frames_small = np.zeros_like(frames_small)

        # ---- Emit input preview ----
        input_preview = (frames_small * 255).astype(np.uint8)  # [T, H, W]
        self.input_ready.emit(input_preview, input_fps)

        # ---- 3. Build pixel matrix [N_pixels, T_orig] ----
        N_pixels = H_small * W_small
        # scale luminance by irradiance so model sees correct physical units
        pix_matrix = frames_small.reshape(T_orig, N_pixels).T  # [N, T]
        pix_matrix = pix_matrix * self.irradiance               # W/mm²

        # ---- 4. Upsample parameters ----
        dt_us   = 1.0 / self.upsample_fps * 1e3            # ms
        scale   = int(round(self.upsample_fps / input_fps))
        t_orig  = np.arange(T_orig) / input_fps            # seconds
        t_up    = np.arange(0, T_orig / input_fps, dt_us * 1e-3)  # seconds
        T_up    = len(t_up)
        pad_smp = int(round(self.pad_duration * self.upsample_fps))

        # ---- 5. Process in chunks ----
        I_out_ds = np.zeros((N_pixels, T_orig), dtype=np.float32)
        n_chunks  = (N_pixels + self.chunk_size - 1) // self.chunk_size

        self.progress.emit(8, f"Running model on {N_pixels:,} pixels in "
                              f"{n_chunks} chunks  ({_BACKEND}) …")

        for c_idx in range(n_chunks):
            if self._cancelled:
                return
            pct = 8 + int(90 * c_idx / n_chunks)
            self.progress.emit(pct, f"Chunk {c_idx + 1}/{n_chunks} …")

            start = c_idx * self.chunk_size
            end   = min(start + self.chunk_size, N_pixels)
            chunk = pix_matrix[start:end]          # [n_chunk, T_orig]
            n_ch  = end - start

            # ---- upsample each pixel with linear interpolation ----
            f_interp = interp1d(t_orig, chunk, kind="linear", axis=1,
                                fill_value="extrapolate")
            chunk_up  = f_interp(t_up).astype(np.float32)  # [n_ch, T_up]

            # ---- prepend background pad (background luminance = 0.5 * irr) ----
            bg_val    = 0.5 * self.irradiance
            pad_block = np.full((n_ch, pad_smp), bg_val, dtype=np.float32)
            chunk_pad = np.concatenate([pad_block, chunk_up], axis=1)  # [n_ch, T_pad]

            # ---- run vectorised Euler ----
            I_chunk = _euler_vectorized(
                self.params, self.V, self.lambda_nm, chunk_pad, dt_us
            )                                                # [n_ch, T_pad]

            # ---- downsample: skip pad, then take every scale-th sample ----
            I_ds = I_chunk[:, pad_smp::scale][:, :T_orig]  # [n_ch, T_orig]

            # pad to T_orig if rounding shortened it
            if I_ds.shape[1] < T_orig:
                I_ds = np.pad(I_ds, ((0, 0), (0, T_orig - I_ds.shape[1])),
                              mode="edge")

            I_out_ds[start:end] = I_ds.astype(np.float32)

        # ---- 6. Normalise output to [0, 1] → uint8 ----
        self.progress.emit(98, "Normalising output …")
        I_min = I_out_ds.min()
        I_max = I_out_ds.max()
        if I_max > I_min:
            I_norm = (I_out_ds - I_min) / (I_max - I_min)
        else:
            I_norm = np.zeros_like(I_out_ds)

        # reshape to [T, H, W]
        frames_out = (I_norm.T.reshape(T_orig, H_small, W_small) * 255).astype(np.uint8)

        self.progress.emit(100, "Done.")
        self.finished.emit(frames_out, input_fps, self.fps_multiplier)


# ---------------------------------------------------------------------------
# Playback widget (shared between Input and Output tabs)
# ---------------------------------------------------------------------------

class _PreviewPanel(QWidget):
    """A video-label + playback controls panel."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.video_label = QLabel(title)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "background: #3a3a3a; color: #aaaaaa; border: 1px solid #555;"
        )
        layout.addWidget(self.video_label)

        # Playback controls
        ctrl = QHBoxLayout()
        self.btn_play  = QPushButton("Play")
        self.btn_stop  = QPushButton("Stop")
        self.slider    = QSlider(Qt.Horizontal)
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setFixedWidth(80)
        self.lbl_frame.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for b in (self.btn_play, self.btn_stop):
            b.setFixedWidth(60)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.lbl_frame)
        layout.addLayout(ctrl)

        self.frames= None   # [T, H, W] uint8
        self._cur_idx = 0
        self._playing = False

        self.timer = QTimer()
        self.timer.timeout.connect(self._advance)

        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop.clicked.connect(self._stop)
        self.slider.valueChanged.connect(self._on_slider)

        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    def load_frames(self, frames: np.ndarray, display_fps: float):
        """Load [T, H, W] uint8 frames and set playback fps."""
        self.frames    = frames
        self._cur_idx  = 0
        self._playing  = False
        self.btn_play.setText("Play")
        interval_ms = max(1, int(1000.0 / display_fps))
        self.timer.setInterval(interval_ms)
        T = frames.shape[0]
        self.slider.setRange(0, max(0, T - 1))
        self.slider.setValue(0)
        self.lbl_frame.setText(f"0 / {T}")
        self._show_frame(0)
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        self.btn_play.setEnabled(enabled)
        self.btn_stop.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def _toggle_play(self):
        if self.frames is None:
            return
        self._playing = not self._playing
        if self._playing:
            self.btn_play.setText("Pause")
            self.timer.start()
        else:
            self.btn_play.setText("Play")
            self.timer.stop()

    def _stop(self):
        self._playing = False
        self.btn_play.setText("Play")
        self.timer.stop()
        if self.frames is not None:
            self._cur_idx = 0
            self.slider.setValue(0)
            self._show_frame(0)

    def _advance(self):
        if self.frames is None:
            return
        T = self.frames.shape[0]
        self._cur_idx = (self._cur_idx + 1) % T
        self.slider.blockSignals(True)
        self.slider.setValue(self._cur_idx)
        self.slider.blockSignals(False)
        self._show_frame(self._cur_idx)

    def _on_slider(self, value: int):
        self._cur_idx = value
        self._show_frame(value)

    def _show_frame(self, idx: int):
        if self.frames is None:
            return
        T = self.frames.shape[0]
        self.lbl_frame.setText(f"{idx} / {T}")
        frame = self.frames[idx]          # [H, W] uint8
        H, W  = frame.shape
        # Ensure contiguous C-order buffer
        frame_c = np.ascontiguousarray(frame)
        qimg = QImage(frame_c.data, W, H, W, QImage.Format_Grayscale8)
        pix  = QPixmap.fromImage(qimg)
        pix  = pix.scaled(self.video_label.size(),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class VideoSimulationTab(QWidget):
    """Tab 5 – Video opsin simulation."""

    def __init__(self, status_bar):
        super().__init__()
        self._status_bar    = status_bar
        self._opsins: dict  = {}
        self._worker= None
        self._input_frames= None
        self._output_frames= None
        self._input_fps: float  = 25.0
        self._fps_mult: float   = 4.0
        self._video_w= None
        self._video_h= None

        self._build_ui()
        self._load_opsins()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---- Left control panel ----
        left = QWidget()
        left.setMaximumWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        left_layout.addWidget(self._build_video_group())
        left_layout.addWidget(self._build_opsin_group())
        left_layout.addWidget(self._build_stimulus_group())
        left_layout.addWidget(self._build_processing_group())
        left_layout.addWidget(self._build_output_group())
        left_layout.addWidget(self._build_run_group())
        left_layout.addStretch(1)

        splitter.addWidget(left)

        # ---- Right preview panel ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._preview_tabs = QTabWidget()
        self._in_panel  = _PreviewPanel("Input Preview")
        self._out_panel = _PreviewPanel("Output Preview")
        self._preview_tabs.addTab(self._in_panel,  "Input Preview")
        self._preview_tabs.addTab(self._out_panel, "Output Preview")
        right_layout.addWidget(self._preview_tabs)

        # Save button
        self._btn_save = QPushButton("Save Output Video …")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        right_layout.addWidget(self._btn_save)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ---------- group builders ----------

    def _build_video_group(self) -> QGroupBox:
        gb = QGroupBox("Video Input")
        lay = QVBoxLayout(gb)

        row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("No file selected …")
        self._path_edit.setReadOnly(True)
        self._btn_browse = QPushButton("Browse …")
        self._btn_browse.clicked.connect(self._on_browse)
        row.addWidget(self._path_edit, 1)
        row.addWidget(self._btn_browse)
        lay.addLayout(row)

        self._lbl_info = QLabel("—")
        self._lbl_info.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self._lbl_info)

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Input frame rate:"))
        self._spn_input_fps = QDoubleSpinBox()
        self._spn_input_fps.setRange(1.0, 960.0)
        self._spn_input_fps.setDecimals(3)
        self._spn_input_fps.setValue(30.0)
        self._spn_input_fps.setSuffix("  fps")
        self._spn_input_fps.setToolTip(
            "Auto-filled from video metadata on browse. "
            "Override manually if the detected value is wrong."
        )
        self._spn_input_fps.valueChanged.connect(self._sync_sampling_fps)
        fps_row.addWidget(self._spn_input_fps)
        lay.addLayout(fps_row)

        return gb

    def _build_opsin_group(self) -> QGroupBox:
        gb  = QGroupBox("Opsin")
        lay = QVBoxLayout(gb)
        self._cmb_opsin = QComboBox()
        self._cmb_opsin.currentIndexChanged.connect(self._on_opsin_changed)
        lay.addWidget(self._cmb_opsin)
        return gb

    def _build_stimulus_group(self) -> QGroupBox:
        gb  = QGroupBox("Stimulus")
        form = QFormLayout(gb)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self._spn_irr = QDoubleSpinBox()
        self._spn_irr.setRange(1e-6, 1.0)
        self._spn_irr.setDecimals(6)
        self._spn_irr.setValue(0.001)
        self._spn_irr.setSuffix("  W/mm²")
        form.addRow("Irradiance", self._spn_irr)

        self._spn_V = QDoubleSpinBox()
        self._spn_V.setRange(-100.0, 0.0)
        self._spn_V.setDecimals(1)
        self._spn_V.setValue(-60.0)
        self._spn_V.setSuffix("  mV")
        form.addRow("Holding potential", self._spn_V)

        self._spn_lambda = QSpinBox()
        self._spn_lambda.setRange(300, 800)
        self._spn_lambda.setValue(590)
        self._spn_lambda.setSuffix("  nm")
        form.addRow("Wavelength", self._spn_lambda)

        return gb

    def _build_processing_group(self) -> QGroupBox:
        gb   = QGroupBox("Processing")
        form = QFormLayout(gb)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self._edit_resize = QLineEdit("1/8")
        self._edit_resize.setPlaceholderText("e.g. 1/8  or  3/4")
        self._edit_resize.setToolTip(
            "Enter a fraction (e.g. 1/8, 3/4) or decimal (e.g. 0.125).\n"
            "Must produce integer pixel dimensions — validated after loading a video."
        )
        self._edit_resize.textChanged.connect(self._validate_resize)
        form.addRow("Resize factor", self._edit_resize)

        self._lbl_resize_dims = QLabel("— (load a video first)")
        self._lbl_resize_dims.setStyleSheet("color: gray; font-size: 10px;")
        self._lbl_resize_dims.setWordWrap(True)
        form.addRow("", self._lbl_resize_dims)

        self._spn_ups_fps = QSpinBox()
        self._spn_ups_fps.setRange(1000, 10000)
        self._spn_ups_fps.setSingleStep(500)
        self._spn_ups_fps.setValue(5000)
        self._spn_ups_fps.setSuffix("  fps")
        form.addRow("Upsample FPS", self._spn_ups_fps)

        self._spn_pad = QDoubleSpinBox()
        self._spn_pad.setRange(0.0, 10.0)
        self._spn_pad.setDecimals(2)
        self._spn_pad.setValue(0.5)
        self._spn_pad.setSuffix("  s")
        form.addRow("Pad duration", self._spn_pad)

        self._spn_chunk = QSpinBox()
        self._spn_chunk.setRange(256, 16384)
        self._spn_chunk.setSingleStep(256)
        self._spn_chunk.setValue(4096)
        form.addRow("Chunk size", self._spn_chunk)

        lbl_backend = QLineEdit(_backend_label())
        lbl_backend.setReadOnly(True)
        lbl_backend.setStyleSheet("background: #f0f0f0; color: #333;")
        form.addRow("GPU backend", lbl_backend)

        return gb

    def _build_output_group(self) -> QGroupBox:
        gb   = QGroupBox("Frame Rates")
        form = QFormLayout(gb)

        self._lbl_sampling_fps = QLineEdit("—")
        self._lbl_sampling_fps.setReadOnly(True)
        self._lbl_sampling_fps.setStyleSheet("background: #f0f0f0; color: #333;")
        self._lbl_sampling_fps.setToolTip(
            "Sampling frame rate — matches the input video frame rate.\n"
            "Adjust via the Input frame rate field in the Video Input group."
        )
        form.addRow("Sampling FR (fps)", self._lbl_sampling_fps)

        self._spn_playback_mult = QDoubleSpinBox()
        self._spn_playback_mult.setRange(0.25, 20.0)
        self._spn_playback_mult.setDecimals(2)
        self._spn_playback_mult.setSingleStep(0.25)
        self._spn_playback_mult.setValue(4.0)
        self._spn_playback_mult.setSuffix("×")
        self._spn_playback_mult.setToolTip(
            "Multiplier applied to both input and output frame rates for\n"
            "preview playback and video saving.\n"
            "Effective playback fps = Sampling FR × this value."
        )
        form.addRow("Playback FR (mult)", self._spn_playback_mult)

        return gb

    def _build_run_group(self) -> QGroupBox:
        gb  = QGroupBox("Control")
        lay = QVBoxLayout(gb)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("Run Simulation")
        self._btn_run.setStyleSheet(
            "background-color: #2c5f8a; color: white; font-weight: bold; padding: 6px;"
        )
        self._btn_run.clicked.connect(self._on_run)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)

        btn_row.addWidget(self._btn_run, 2)
        btn_row.addWidget(self._btn_cancel, 1)
        lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        lay.addWidget(self._progress)

        self._lbl_status = QLabel("Ready.")
        self._lbl_status.setStyleSheet("color: #444; font-size: 10px;")
        lay.addWidget(self._lbl_status)

        return gb

    # ------------------------------------------------------------------
    # Opsin loading
    # ------------------------------------------------------------------

    def _load_opsins(self):
        try:
            self._opsins = load_all_opsins()
        except Exception as exc:
            self._opsins = {}
            self._lbl_status.setText(f"Could not load opsins: {exc}")
            return

        self._cmb_opsin.blockSignals(True)
        self._cmb_opsin.clear()
        for name in sorted(self._opsins.keys()):
            self._cmb_opsin.addItem(name)
        self._cmb_opsin.blockSignals(False)

        if self._cmb_opsin.count() > 0:
            self._on_opsin_changed(0)

    def _on_opsin_changed(self, _index: int):
        name = self._cmb_opsin.currentText()
        if name in self._opsins:
            opsin = self._opsins[name]
            self._spn_lambda.blockSignals(True)
            self._spn_lambda.setValue(int(round(opsin.peak_lambda)))
            self._spn_lambda.blockSignals(False)

    def _current_opsin(self) -> OpsinParams:
        name = self._cmb_opsin.currentText()
        if name in self._opsins:
            return self._opsins[name]
        return OpsinParams()

    def _sync_sampling_fps(self, value: float):
        self._lbl_sampling_fps.setText(f"{value:.3f}")

    # ------------------------------------------------------------------
    # Resize factor validation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fraction(text: str):
        """Parse '1/8', '3/4', or '0.125' into a float. Returns None on error."""
        text = text.strip()
        try:
            if '/' in text:
                num, den = text.split('/', 1)
                den_val = float(den)
                if den_val == 0:
                    return None
                return float(num) / den_val
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _as_fraction_str(f: float) -> str:
        """Express a float as a neat fraction string where possible."""
        from fractions import Fraction
        frac = Fraction(f).limit_denominator(200)
        if abs(frac - f) < 1e-9:
            return str(frac)          # e.g. '1/8'
        return f"{f:.4g}"

    def _find_valid_resize_factors(self, current: float) -> list:
        """Return (factor, fraction_str) pairs near *current* that yield integer dims."""
        if self._video_w is None or self._video_h is None:
            return []
        W, H = self._video_w, self._video_h
        valid = {}
        for n in range(1, 201):
            for k in range(1, n + 1):
                f = k / n
                if 0 < f <= 1.0:
                    if abs(W * f - round(W * f)) < 1e-6 and abs(H * f - round(H * f)) < 1e-6:
                        if round(W * f) >= 1 and round(H * f) >= 1:
                            key = round(f, 9)
                            if key not in valid:
                                valid[key] = self._as_fraction_str(f)
        return sorted(valid.items(), key=lambda kv: abs(kv[0] - current))

    def _validate_resize(self) -> bool:
        """Parse the resize field, validate dimensions, update label. Returns True if valid."""
        factor = self._parse_fraction(self._edit_resize.text())

        if factor is None or not (0 < factor <= 1.0):
            self._lbl_resize_dims.setText("Invalid — enter a fraction like 1/8 or 3/4")
            self._lbl_resize_dims.setStyleSheet("color: red; font-size: 10px;")
            self._btn_run.setEnabled(False)
            return False

        if self._video_w is None or self._video_h is None:
            self._lbl_resize_dims.setText("— (load a video first)")
            self._lbl_resize_dims.setStyleSheet("color: gray; font-size: 10px;")
            return True

        new_w = self._video_w * factor
        new_h = self._video_h * factor
        if abs(new_w - round(new_w)) < 1e-6 and abs(new_h - round(new_h)) < 1e-6:
            self._lbl_resize_dims.setText(
                f"→ {int(round(new_w))} × {int(round(new_h))} px"
            )
            self._lbl_resize_dims.setStyleSheet("color: green; font-size: 10px;")
            self._btn_run.setEnabled(True)
            return True
        else:
            nearby = self._find_valid_resize_factors(factor)[:4]
            sugg = "  ".join(s for _, s in nearby)
            self._lbl_resize_dims.setText(
                f"→ {new_w:.2f} × {new_h:.2f} px — non-integer!\n"
                f"Valid nearby: {sugg}"
            )
            self._lbl_resize_dims.setStyleSheet("color: red; font-size: 10px;")
            self._btn_run.setEnabled(False)
            return False

    # ------------------------------------------------------------------
    # Browse / video info
    # ------------------------------------------------------------------

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All files (*)",
        )
        if not path:
            return
        self._path_edit.setText(path)
        self._read_video_info(path)

    def _read_video_info(self, path: str):
        try:
            reader = imageio.get_reader(path, format="ffmpeg")
            meta   = reader.get_meta_data()
            reader.close()
            fps    = meta.get("fps", "?")
            size   = meta.get("size", ("?", "?"))
            dur    = meta.get("duration", None)
            nfr    = meta.get("nframes", None)
            nfr    = nfr if (nfr and np.isfinite(nfr)) else None
            dur_s  = f"{dur:.1f} s" if dur else (f"{nfr/detected_fps:.1f} s" if nfr else "? s")
            detected_fps = float(fps) if fps != "?" else 25.0
            self._video_w = int(size[0]) if size[0] != "?" else None
            self._video_h = int(size[1]) if size[1] != "?" else None
            self._lbl_info.setText(
                f"{detected_fps:.3f} fps  |  {size[0]}×{size[1]} px  |  {dur_s}"
            )
            self._spn_input_fps.setValue(detected_fps)   # triggers _sync_sampling_fps
            self._input_fps = detected_fps
            self._validate_resize()
        except Exception as exc:
            self._lbl_info.setText(f"Error reading meta: {exc}")

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------

    def _on_run(self):
        path = self._path_edit.text().strip()
        if not path or not Path(path).is_file():
            QMessageBox.warning(self, "No video", "Please select a valid video file first.")
            return

        if not self._validate_resize():
            QMessageBox.warning(self, "Invalid resize factor",
                                "The resize factor produces non-integer frame dimensions.\n"
                                "Check the suggested values shown below the resize field.")
            return

        opsin     = self._current_opsin()
        scale_fac = self._parse_fraction(self._edit_resize.text())
        fps_mult  = self._spn_playback_mult.value()

        self._worker = SimulationWorker(
            video_path          = path,
            params              = opsin,
            V                   = self._spn_V.value(),
            lambda_nm           = float(self._spn_lambda.value()),
            irradiance          = self._spn_irr.value(),
            scale_factor        = scale_fac,
            upsample_fps        = self._spn_ups_fps.value(),
            pad_duration        = self._spn_pad.value(),
            chunk_size          = self._spn_chunk.value(),
            fps_multiplier      = fps_mult,
            input_fps_override  = self._spn_input_fps.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.input_ready.connect(self._on_input_ready)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_save.setEnabled(False)
        self._progress.setValue(0)
        self._lbl_status.setText("Starting …")
        self._worker.start()

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._reset_run_buttons()
        self._lbl_status.setText("Cancelled.")
        self._progress.setValue(0)

    def _reset_run_buttons(self):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker signals
    # ------------------------------------------------------------------

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._lbl_status.setText(msg)
        if self._status_bar:
            self._status_bar.showMessage(msg)

    def _on_input_ready(self, frames: np.ndarray, input_fps: float):
        self._input_frames = frames
        playback_fps = input_fps * self._spn_playback_mult.value()
        self._in_panel.load_frames(frames, playback_fps)

    def _on_finished(self, frames: np.ndarray, input_fps: float, fps_mult: float):
        self._output_frames = frames
        self._input_fps     = input_fps
        self._fps_mult      = fps_mult

        display_fps = input_fps * fps_mult
        self._out_panel.load_frames(frames, display_fps)
        self._preview_tabs.setCurrentIndex(1)   # switch to Output Preview
        self._btn_save.setEnabled(True)

        self._reset_run_buttons()
        self._lbl_status.setText(
            f"Done. Output: {frames.shape[0]} frames @ {display_fps:.1f} fps"
        )
        if self._status_bar:
            self._status_bar.showMessage("Simulation complete.")

    def _on_error(self, msg: str):
        self._reset_run_buttons()
        self._lbl_status.setText("Error – see dialog.")
        QMessageBox.critical(self, "Simulation Error", msg)
        if self._status_bar:
            self._status_bar.showMessage("Simulation error.")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------

    def _on_save(self):
        if self._output_frames is None:
            return

        out_fps = self._input_fps * self._fps_mult
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video", "opsin_output.mp4",
            "MP4 video (*.mp4);;AVI video (*.avi);;All files (*)",
        )
        if not path:
            return

        try:
            writer = imageio.get_writer(path, format="ffmpeg", fps=out_fps,
                                        codec="libx264", quality=8)
            T = self._output_frames.shape[0]
            for i in range(T):
                frame_gray = self._output_frames[i]          # [H, W] uint8
                frame_rgb  = np.stack([frame_gray] * 3, axis=-1)  # [H, W, 3]
                writer.append_data(frame_rgb)
            writer.close()
            QMessageBox.information(self, "Saved",
                                    f"Output video saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
