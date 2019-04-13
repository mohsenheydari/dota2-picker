import win32gui
import win32con

from PySide2.QtCore import QUrl, Qt, Slot, Signal, QEvent, QTimer, QObject
from PySide2.QtWidgets import QDesktopWidget
from PySide2.QtWebEngineWidgets import QWebEngineView
from PySide2.QtGui import QColor
from PySide2.QtWebChannel import QWebChannel


class BrowserSharedObject(QObject):
    reset_clicked = Signal()

    @Slot()
    def reset_clicked_slot(self):
        self.reset_clicked.emit()


class PickerBrowserView(QWebEngineView):
    closing = Signal()
    minimized = Signal()
    reset_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        # self.setFocusPolicy(Qt.NoFocus)
        #self.setWindowFlag(Qt.WindowTransparentForInput, True)
        #self.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
        #self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_window)

        available_geometry = QDesktopWidget().availableGeometry()
        width = max(available_geometry.width() * .55, 900)
        height = max(available_geometry.height() * .55, 650)

        self.resize(width, height)
        self.load(QUrl("http://dotapicker.com/herocounter/"))
        self.setWindowTitle('AI Picker By Mohsen H')
        self.page().setBackgroundColor(QColor('black'))
        self.sharedobj = BrowserSharedObject()
        self.sharedobj.reset_clicked.connect(self.reset_clicked.emit)
        self.channel = QWebChannel()
        self.channel.registerObject('backend', self.sharedobj)
        self.page().setWebChannel(self.channel)
        self.loadFinished.connect(self.load_done)

    def update_window(self):
        if self.winId():
            win32gui.SetWindowPos(self.winId(), win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

    def load_done(self):
        self.inject_scripts()
        self.timer.start(30)

    def disable_contextmenu(self):
        script = """ document.addEventListener('contextmenu', event => event.preventDefault()); """
        self.page().runJavaScript(script)

    def select_hero(self, hero, left=True):
        button_selector = '.heroSelectContainerTopLeft' if left else '.heroSelectContainerTopRight'

        script = '''
        element = document.querySelector('.searchContainer > div:nth-child({child}) {element}');
        element && element.click();
        '''.format(child=str(hero+1), element=button_selector)

        self.page().runJavaScript(script)

    def select_team(self, team=False):
        script = '''
        team_buttons = document.querySelectorAll('a[ng-click~="show.teamRadiant"]');
        team_buttons[{index}] && team_buttons[{index}].click();
        '''.format(index=str(1 if team else 0))

        self.page().runJavaScript(script)

    def reset_selections(self):
        script = '''
        reset = document.querySelector('button[type="reset"]');
        reset && reset.click();
        '''

        self.page().runJavaScript(script)

    def inject_scripts(self):
        with open('./js/page.js', 'r') as myfile:
            script = myfile.read()

        self.page().runJavaScript(script)

    def closeEvent(self, event):
        self.closing.emit()
        self.timer.stop()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMinimized:
                self.minimized.emit()
