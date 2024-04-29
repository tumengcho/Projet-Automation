# main.py
import sys
import threading
from PyQt5.QtWidgets import QApplication, QLabel, QDesktopWidget, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QMovie
from Jarvis_Ui import IronManTitles  # Corrected import statement
from main import JarvisAi, capture_audio, listen


class JarvisApp(QLabel):
    image_changed = pyqtSignal(str)
    add_layout = pyqtSignal(list)
    remove_layout = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.movie_path = "images/listening.gif"
        self.init_ui()

    def init_ui(self):
        self.current_layout = None
        self.set_movie(self.movie_path)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)  # Scale the movie to fit the label

    def set_movie(self, movie_path):
        self.movie_path = movie_path
        self.movie = QMovie(self.movie_path)
        self.setMovie(self.movie)
        self.movie.start()

    def update_movie(self, movie_path):
        self.set_movie(movie_path)

    def add_layout_to_label(self, titles):
        iron_man_titles = IronManTitles(titles)
        layout = QVBoxLayout()
        layout.addWidget(iron_man_titles)
        self.setLayout(layout)

    def remove_current_layout(self):
        # Clear any existing layout
        while self.layout().count():
            item = self.layout().takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

def main_loop(jarvis_app):
    mode = input("Speak or Write: ")
    if "write" in mode.lower():
        while True:
            jarvis_app.image_changed.emit("images/speaking.gif")
            query = input('You: ')
            if query.lower() == 'quit':
                break
            elif "test web" in query.lower():
                titles = [
                    ("OpenAI", "A platform for artificial intelligence research and development"),
                    ("GitHub", "A platform for hosting and collaborating on code projects"),
                    ("Stack Overflow", "A community-driven question and answer site for programmers"),
                    ("Medium", "A platform for publishing and reading articles and blog posts"),
                    ("YouTube", "A video-sharing platform")
                ]
                jarvis_app.add_layout.emit(titles)
            elif "close web" in query.lower():
                jarvis_app.remove_layout.emit()
            jarvis_app.image_changed.emit("images/listening.gif")
            # JarvisAi(query)
    elif "speak" in mode.lower():
        while True:
            jarvis_app.show()
            jarvis_app.image_changed.emit("images/speaking.gif")
            while True:
                audio = capture_audio()
                query = listen(audio)
                if query:
                    break

            jarvis_app.image_changed.emit("images/listening.gif")

            if "iron man" in query.lower():
                titles = [
                    ("OpenAI", "A platform for artificial intelligence research and development"),
                    ("GitHub", "A platform for hosting and collaborating on code projects"),
                    ("Stack Overflow", "A community-driven question and answer site for programmers"),
                    ("Medium", "A platform for publishing and reading articles and blog posts"),
                    ("YouTube", "A video-sharing platform")
                ]
                jarvis_app.add_layout.emit(titles)
            elif "hello" in query.lower():
                jarvis_app.remove_layout.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    jarvis_app = JarvisApp()
    jarvis_app.setWindowFlags(Qt.FramelessWindowHint)  # Remove window frame
    jarvis_app.setAttribute(Qt.WA_TranslucentBackground)  # Make window background transparent

    # Get the screen resolution
    desktop = QDesktopWidget()
    screen_rect = desktop.screenGeometry()
    jarvis_app.setGeometry(screen_rect)

    # Connect the image_changed signal to the update_movie slot
    jarvis_app.image_changed.connect(jarvis_app.update_movie)

    # Connect the add_layout signal to the add_layout_to_label slot
    jarvis_app.add_layout.connect(jarvis_app.add_layout_to_label)

    jarvis_app.remove_layout.connect(jarvis_app.remove_current_layout)
    # Start the Jarvis main loop in a separate thread
    thread = threading.Thread(target=main_loop, args=(jarvis_app,))
    thread.start()
    jarvis_app.show()

    sys.exit(app.exec_())
