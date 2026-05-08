"""Entry point: python -m optoTCSF  or  optoTCSF (console script)."""

import os
import sys


def _fix_qt_plugin_path():
    """Ensure Qt can find the xcb platform plugin on conda/pip environments."""
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        return  # already set by user

    import sysconfig
    from pathlib import Path

    # Candidates in order of preference
    candidates = []

    # 1. PyQt5-bundled plugins (pip wheel layout)
    try:
        import PyQt5
        pyqt5_dir = Path(PyQt5.__file__).parent
        candidates.append(pyqt5_dir / "Qt5" / "plugins" / "platforms")
        candidates.append(pyqt5_dir / "Qt" / "plugins" / "platforms")
    except Exception:
        pass

    # 2. Conda prefix plugins
    prefix = Path(sys.prefix)
    candidates.append(prefix / "plugins" / "platforms")
    candidates.append(prefix / "lib" / "qt5" / "plugins" / "platforms")

    # 3. System Qt
    for p in ["/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
              "/usr/lib/qt5/plugins/platforms",
              "/usr/lib64/qt5/plugins/platforms"]:
        candidates.append(Path(p))

    for path in candidates:
        if path.is_dir() and any(path.glob("libqxcb*")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(path)
            return


def main():
    _fix_qt_plugin_path()

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from .gui.main_window import MainWindow

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("optoTCSF")
    app.setOrganizationName("UW Vision & Cognition")

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
