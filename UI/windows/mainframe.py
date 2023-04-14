import sys
from pathlib import Path

from PyQt6 import QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QListWidget, QLabel, QGridLayout, QLineEdit, \
    QFileDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Proteus")
        self.setWindowIcon(QtGui.QIcon("Proteus.png"))
        self.setGeometry(450, 130, 800, 600)

        add = QPushButton("Add File",self)
        add.setGeometry(20, 20, 100, 40)
        add.setCheckable(True)
        add.clicked.connect(self.open_file)

        self.file_list = QLineEdit(self)
        self.file_list.setGeometry(140,25,300,30)


    def closeEvent(self, event):
        for window in QApplication.topLevelWidgets():
            window.close()
            exit()

    def open_file(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "Images (*.png *.jpg)"
        )
        if filename:
            path = Path(filename)
            self.file_list.setText(str(path))


app = QApplication(sys.argv)

w = MainWindow()
w.show()

sys.exit(app.exec())

