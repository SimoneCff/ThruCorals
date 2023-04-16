from pathlib import Path
from PyQt6.QtWidgets import QPushButton, QLineEdit, QFileDialog, QWidget


class uiopen(QWidget):
    def __init__(self):
        super().__init__()
        self.path = None
        self.have_it = False

        self.add_section()

    def add_section(self):
        # Button
        add = QPushButton("Add File", self)
        add.setGeometry(20, 20, 100, 40)
        add.clicked.connect(self.open_file)

        # Path
        self.file_list = QLineEdit(self)
        self.file_list.setGeometry(140, 25, 300, 30)

    def open_file(self):
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            "Images (*.png *.jpg)"
        )
        if filename:
            self.have_it = True
            self.path = Path(filename)
            self.file_list.setText(str(self.path))
