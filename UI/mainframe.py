import sys
from UI.widgets import fileopen

from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Proteus")
        self.setWindowIcon(QtGui.QIcon("Proteus.png"))
        self.setGeometry(450, 130, 800, 600)

        self.open = fileopen.uiopen()
        self.layout().addWidget(self.open)

    def closeEvent(self, event):
        for window in QApplication.topLevelWidgets():
            window.close()
            exit()

def start():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()

    sys.exit(app.exec())

