# Library Imports
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class DetectionBox(QFrame):
    def __init__(self, class_name, color):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.setMinimumWidth(120)
        self.setMaximumHeight(60)

        layout = QVBoxLayout(self)

        self.class_label = QLabel(str(class_name))
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet("font-weight: bold;")

        self.count_label = QLabel("0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("font-size: 18px;")

        layout.addWidget(self.class_label)
        layout.addWidget(self.count_label)

        self.setStyleSheet(f"""
            DetectionBox {{
                background-color: {color};
                border-radius: 5px;
                margin: 2px;
            }}
        """)

    def update_count(self, count):
        self.count_label.setText(str(count))