# IronManTitles.py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

class IronManTitles(QWidget):
    def __init__(self, titles=None):
        super().__init__()
        self.initUI(titles)

    def initUI(self, titles):
        self.setWindowTitle('J.A.R.V.I.S. - Iron Man Titles')
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: #1F1F1F; color: #EFEFEF;")

        # Create a vertical layout
        layout = QVBoxLayout()

        if titles:
            # Create QLabel instances for each title and subtext
            for main_text, sub_text in titles:
                title_label = QLabel(f"<b>{main_text}</b><br/>{sub_text}")  # Use HTML formatting for bold main_text
                title_label.setStyleSheet("color: #EFEFEF;")  # Set text color
                layout.addWidget(title_label)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = IronManTitles()
    ex.show()
    sys.exit(app.exec_())
