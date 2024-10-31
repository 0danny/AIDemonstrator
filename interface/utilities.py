# Library Imports
from PyQt5.QtGui import QColor

def generate_class_colors(class_names):
        colors = {}
        hue_step = 360 / len(class_names)

        for i, class_name in enumerate(class_names):
            hue = i * hue_step
            color = QColor.fromHsv(int(hue), 100, 240)
            colors[class_name] = color.name()

        return colors