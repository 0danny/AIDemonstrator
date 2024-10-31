# System Imports
import sys
sys.dont_write_bytecode = True

import platform

# Library Imports
from PyQt5.QtWidgets import QApplication

import pathlib
plt = platform.system()

if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Project Imports
from gui import SignViewer

def main():
    app = QApplication(sys.argv)
        
    app.setStyle('Fusion')

    # force light mode
    app.setPalette(app.style().standardPalette())
        
    viewer = SignViewer()
    viewer.show()
        
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()