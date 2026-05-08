"""
mat2csv.py  –  Convert qCSF .mat data files to CSV.

Usage
-----
  python mat2csv.py <file1.mat> [file2.mat ...]   # explicit files
  python mat2csv.py output/*.mat                  # glob (shell expands it)
  python mat2csv.py output/ --recursive           # convert every .mat in a folder

The .mat file must have been saved as a MATLAB struct (not a dictionary).
See the comment in CSFExpt.m for the required change.

Output
------
One CSV per .mat file, written next to the original:
  NS_qCSF_baseline_v1_26-05-04_14-30.mat
  → NS_qCSF_baseline_v1_26-05-04_14-30.csv

CSV columns:
  sID, condition, opsin, block, fixed_freq, TF, SF,
  trial, contrast, contrast_level, response

  block        – the MATLAB struct field name (e.g. TF_5, SF_1_5)
  fixed_freq   – the frequency value that was held constant in that block
  TF / SF      – temporal / spatial frequency for each trial (Hz / cpd)
  contrast     – linear contrast used on that trial
  contrast_level – –log10(contrast)  (higher = lower contrast)
  response     – 1 = correct, 0 = incorrect
"""

import sys
import re
import csv
import argparse
import numpy as np
from pathlib import Path

try:
    import scipy.io as sio
except ImportError:
    sys.exit("scipy is required:  pip install scipy")

# MATLAB engine is optional; used as fallback for MCOS/dictionary .mat files
_matlab_engine = None

def _get_matlab_engine():
    global _matlab_engine
    if _matlab_engine is None:
        try:
            import matlab.engine
        except ImportError:
            return None
        print("Starting MATLAB engine (first MCOS file)…", flush=True)
        _matlab_engine = matlab.engine.start_matlab("-nojvm -nosplash -nodesktop")
    return _matlab_engine


# ---------------------------------------------------------------------------
# Filename parsing  (same logic as tcsf_tab.py)
# ---------------------------------------------------------------------------

def parse_filename(stem: str) -> dict:
    """Return {'sID', 'cond', 'opsin', 'fix_temporal'} from a .mat stem."""
    m = re.search(r'(baseline|opto)_(v1|v2)', stem)
    if m is None:
        raise ValueError(
            f"Cannot infer condition from '{stem}'.\n"
            "Expected pattern: <sID>_qCSF_<baseline|opto>_<v1|v2>[_<opsin>]_<timestamp>"
        )

    cond_type = m.group(1)        # 'baseline' or 'opto'
    cond_ver  = m.group(2)        # 'v1' or 'v2'
    fix_temporal = (cond_ver == 'v1')   # v1 = fix TF, vary SF; v2 = fix SF, vary TF

    sid_m = re.match(r'^(.+?)_qCSF_', stem)
    sID   = sid_m.group(1) if sid_m else stem[:8]

    opsin = ''
    if cond_type == 'opto':
        after  = stem[m.end():]
        op_m   = re.match(r'_([A-Za-z][A-Za-z0-9]+?)_\d{2}-', after)
        if op_m:
            opsin = op_m.group(1)

    return {
        'sID':          sID,
        'cond':         f"{cond_type}_{cond_ver}",
        'opsin':        opsin,
        'fix_temporal': fix_temporal,
    }


def parse_freq_field(field: str, fix_temporal: bool) -> float:
    """'TF_5'→5.0, 'TF_1_5'→1.5 (struct field names use _ for decimal point)."""
    prefix = 'TF_' if fix_temporal else 'SF_'
    if not field.startswith(prefix):
        raise ValueError(f"Field '{field}' does not start with '{prefix}'")
    parts = field[len(prefix):].split('_')
    if len(parts) == 1:
        return float(parts[0])
    try:
        return float(f"{parts[0]}.{''.join(parts[1:])}")
    except ValueError:
        return float(parts[0])


def parse_dict_key(key: str, fix_temporal: bool) -> float:
    """'TF_5'→5.0, 'SF_1.5'→1.5 (dictionary keys may use . for decimal point)."""
    prefix = 'TF_' if fix_temporal else 'SF_'
    if not key.startswith(prefix):
        raise ValueError(f"Key '{key}' does not start with '{prefix}'")
    val_str = key[len(prefix):]
    try:
        return float(val_str)
    except ValueError:
        # Also handle underscore-decimal form just in case
        parts = val_str.split('_')
        if len(parts) >= 2:
            return float(f"{parts[0]}.{''.join(parts[1:])}")
        raise


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

def _rows_from_struct(data_obj, info: dict) -> list:
    """Extract trial rows from a scipy struct object."""
    fix_temporal = info['fix_temporal']
    rows = []
    for field in data_obj._fieldnames:
        try:
            fixed_val = parse_freq_field(field, fix_temporal)
        except ValueError:
            print(f"  [skip] unexpected field name '{field}'")
            continue
        try:
            block   = getattr(data_obj, field)
            history = block.qcsf.data.history
            if history.ndim == 1:
                history = history.reshape(1, -1)
            if history.shape[1] < 4:
                print(f"  [skip] '{field}': history has fewer than 4 columns")
                continue
        except AttributeError as ex:
            print(f"  [skip] '{field}': {ex}")
            continue
        rows.extend(_history_to_rows(history, field, fixed_val, info))
    return rows


def _rows_from_matlab_engine(mat_path: Path, info: dict) -> list:
    """Extract trial rows from an MCOS dictionary .mat via MATLAB engine."""
    import io
    eng = _get_matlab_engine()
    if eng is None:
        return None  # engine unavailable

    fix_temporal = info['fix_temporal']
    rows = []

    try:
        # clear all → load file → detect actual variable name via who() →
        # alias to tmpdict.  scipy.io returns 'None' as the key for MCOS
        # files, so we cannot rely on the Python-side variable name.
        eng.eval(
            f"clear all; load('{mat_path}'); "
            f"tmpv = who(); "
            f"eval(['tmpdict = ' tmpv{{1}} ';']); "
            f"matkeys = keys(tmpdict);",
            nargout=0, stdout=io.StringIO())
        n_keys = int(eng.eval("numel(matkeys)", nargout=1))

        for i in range(1, n_keys + 1):
            field = eng.eval(f"matkeys{{{i}}}", nargout=1)
            try:
                fixed_val = parse_dict_key(field, fix_temporal)
            except ValueError:
                print(f"  [skip] unexpected key '{field}'")
                continue
            try:
                history = np.array(eng.eval(f"tmpdict(matkeys{{{i}}}).qcsf.data.history", nargout=1))
                if history.ndim == 1:
                    history = history.reshape(1, -1)
                if history.shape[1] < 4:
                    print(f"  [skip] '{field}': history has fewer than 4 columns")
                    continue
            except Exception as ex:
                print(f"  [skip] '{field}': {ex}")
                continue
            rows.extend(_history_to_rows(history, field, fixed_val, info))
    except Exception as ex:
        print(f"  [MATLAB engine error] {ex}")
        return None

    return rows


def _history_to_rows(history: np.ndarray, field: str, fixed_val: float, info: dict) -> list:
    """Convert a history matrix (N×4) to a list of row dicts."""
    fix_temporal = info['fix_temporal']
    trial_nums   = history[:, 0].astype(int)
    vf           = history[:, 1].astype(float)
    contrast_lin = history[:, 2].astype(float)
    response     = history[:, 3].astype(int)
    contrast_lvl = -np.log10(np.clip(contrast_lin, 1e-12, 1.0))
    rows = []
    for k in range(len(trial_nums)):
        tf, sf = (fixed_val, vf[k]) if fix_temporal else (vf[k], fixed_val)
        rows.append({
            'sID':            info['sID'],
            'condition':      info['cond'],
            'opsin':          info['opsin'],
            'block':          field,
            'fixed_freq':     fixed_val,
            'TF':             tf,
            'SF':             sf,
            'trial':          trial_nums[k],
            'contrast':       contrast_lin[k],
            'contrast_level': contrast_lvl[k],
            'response':       response[k],
        })
    return rows


def convert(mat_path: Path, out_path: Path, verbose: bool = True) -> int:
    """Convert one .mat file.  Returns number of rows written (0 on error)."""

    # --- Parse filename ---
    try:
        info = parse_filename(mat_path.stem)
    except ValueError as ex:
        print(f"[SKIP] {mat_path.name}: {ex}")
        return 0

    # --- Load via scipy ---
    try:
        mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    except Exception as ex:
        print(f"[ERROR] {mat_path.name}: scipy.io failed — {ex}")
        return 0

    data_key = next((k for k in mat if not k.startswith('__')), None)
    if data_key is None:
        print(f"[ERROR] {mat_path.name}: no data variable found.")
        return 0

    data_obj = mat[data_key]

    # --- Route to struct or MCOS path ---
    if hasattr(data_obj, '_fieldnames'):
        rows = _rows_from_struct(data_obj, info)
    else:
        # MCOS dictionary — try MATLAB engine
        rows = _rows_from_matlab_engine(mat_path, info)
        if rows is None:
            print(
                f"[ERROR] {mat_path.name}: MCOS/dictionary format and MATLAB engine unavailable.\n"
                "Install the engine:  cd /usr/local/MATLAB/R2022b/extern/engines/python && python setup.py install"
            )
            return 0

    if not rows:
        print(f"[ERROR] {mat_path.name}: no valid trial data found.")
        return 0

    # --- Write CSV ---
    fieldnames = [
        'sID', 'condition', 'opsin', 'block', 'fixed_freq',
        'TF', 'SF', 'trial', 'contrast', 'contrast_level', 'response',
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"[OK]  {mat_path.name}  →  {out_path.name}  ({len(rows)} trials)")
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_mat_files(paths: list[str], recursive: bool) -> list[Path]:
    """Expand directories and return a deduplicated list of .mat files."""
    result = []
    seen = set()
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            pattern = '**/*.mat' if recursive else '*.mat'
            candidates = sorted(pp.glob(pattern))
        else:
            candidates = [pp]
        for c in candidates:
            if c.suffix.lower() == '.mat' and c not in seen:
                result.append(c)
                seen.add(c)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert qCSF .mat experiment files to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('files', nargs='+', metavar='FILE_OR_DIR',
                        help='.mat file(s) or directories containing them')
    parser.add_argument('-o', '--outdir', metavar='DIR',
                        help='Write CSV files here instead of next to each .mat')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Recurse into subdirectories when a folder is given')
    args = parser.parse_args()

    mat_files = collect_mat_files(args.files, args.recursive)
    if not mat_files:
        sys.exit("No .mat files found.")

    out_root = Path(args.outdir) if args.outdir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_trials = 0
    for mat_path in mat_files:
        dest_dir = out_root if out_root else mat_path.parent
        csv_path = dest_dir / (mat_path.stem + '.csv')
        n = convert(mat_path, csv_path)
        if n:
            total_files += 1
            total_trials += n

    print(f"\nDone. Converted {total_files}/{len(mat_files)} file(s), {total_trials} total trials.")

    if _matlab_engine is not None:
        _matlab_engine.quit()


if __name__ == '__main__':
    main()
