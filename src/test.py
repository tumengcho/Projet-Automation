import sys
from PyQt5.QtWidgets import QApplication, QLabel, QDesktopWidget
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QMovie
import threading


class JarvisApp(QLabel):
    image_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.movie_path = "images/listening.gif"
        self.init_ui()

    def init_ui(self):
        self.set_movie(self.movie_path)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)  # Scale the movie to fit the label
        # Start a timer to change the movie every 2 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.change_movie)
        self.timer.start(2000)

    def set_movie(self, movie_path):
        self.movie_path = movie_path
        self.movie = QMovie(self.movie_path)
        self.setMovie(self.movie)
        self.movie.start()

    def change_movie(self):
        # Change the movie dynamically based on user input or other conditions
        pass

    def update_movie(self, movie_path):
        self.set_movie(movie_path)


def JarvisAi(query):
    # Example function to determine movie path based on user input
    if "hello" in query.lower():
        return "images/speaking.gif"
    else:
        return "images/listening.gif"


def main_loop(jarvis_app):
    while True:
        query = input('You: ')
        if query.lower() == 'quit':
            break
        movie_path = JarvisAi(query)
        # Emit the image_changed signal with the new movie path
        jarvis_app.image_changed.emit(movie_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    jarvis_app = JarvisApp()
    jarvis_app.setWindowFlags(Qt.FramelessWindowHint)  # Remove window frame
    jarvis_app.setAttribute(Qt.WA_TranslucentBackground)  # Make window background transparent

    # Get the screen resolution
    desktop = QDesktopWidget()
    screen_rect = desktop.screenGeometry()
    jarvis_app.setGeometry(screen_rect)

    jarvis_app.show()

    # Connect the image_changed signal to the update_movie slot
    jarvis_app.image_changed.connect(jarvis_app.update_movie)

    # Start the Jarvis main loop in a separate thread
    thread = threading.Thread(target=main_loop, args=(jarvis_app,))
    thread.start()

    sys.exit(app.exec_())
