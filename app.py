''' Module contains application entry point and process handling code '''

import os
import sys
from time import sleep
from multiprocessing import Process, Pipe
import array
import numpy as np

from PIL import Image

from PySide2.QtCore import QRunnable, QThreadPool, Slot, Signal, QObject
from PySide2.QtWidgets import QApplication

from keras.preprocessing.image import img_to_array

from screenprocessing import get_player_images, get_player_image
from image import image_crop_to_square, get_avg_image_channel_diff
from win32 import capture_window, window_exist
from playerdata import PlayerPosition, PlayerDataOverTime

from ui.browserwidget import PickerBrowserView

from pickscreenclassifier import PickScreenClassifier
from heroclassifier import HeroClassifier

from controller_message import ControllerMessage

# Use only CPU when predicting ( in case GPU version installed )
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class ControllerSignals(QObject):
    '''
    Contains UI thread signlas
    '''
    add_heroes = Signal(array.array)
    select_team = Signal(bool)


class MessageProcessorThread(QRunnable):
    '''
    Thread class that handles controller messages and signals the UI
    '''

    def __init__(self, pipe, signals):
        super(MessageProcessorThread, self).__init__()
        self.signals = signals
        self.pipe = pipe
        self.isrunning = True

    def process_messages(self, message):
        ''' Process received messages '''

        print('Thread Received: ' + message.msgtype)

        if message.msgtype == 'add_heroes':
            self.signals.add_heroes.emit(message.data)

        if message.msgtype == 'select_team':
            self.signals.select_team.emit(message.data)

        if message.msgtype == 'kill':
            self.isrunning = False

    @Slot()
    def run(self):
        while self.isrunning:
            try:
                message = self.pipe.recv()
                self.process_messages(message)

            except EOFError:
                print("Exiting thread...")
                break


class ControllerProc(Process):
    '''
    Process class that process Dota screens and send messages to UI process
    '''

    def __init__(self, pipe):
        super(ControllerProc, self).__init__()

        self.daemon = True
        self.pipe = pipe
        self.window_name = "Dota 2"
        self.prev_player_pos = -1
        self.pos_detector = PlayerPosition()
        self.selections = PlayerDataOverTime()
        self.isrunning = True
        self.prev_selections = []
        self.reset()

        # Init pick screen classifier model
        self.pick_screen_classifier = PickScreenClassifier()
        self.pick_screen_classifier.init_model()
        self.pick_screen_classifier.load(
            './resources/models/screen_weights.h5')
        self.pick_screen_classifier.model._make_predict_function()

        # Init hero classifier model
        self.hero_classifier = HeroClassifier()
        self.hero_classifier.init_model()
        self.hero_classifier.load(
            './resources/models/hero_weights.h5')
        self.hero_classifier.model._make_predict_function()

    def reset(self):
        ''' resets internal state '''
        self.prev_selections = np.full((10), -1)
        self.prev_player_pos = -1
        self.pos_detector.reset()
        self.selections.reset()

    def is_pick_screen(self, window):
        ''' Checks if given image is picking screen '''

        window = image_crop_to_square(window)
        window = img_to_array(Image.fromarray(window).resize((224, 224)))
        window = window.reshape((1,) + window.shape)

        return self.pick_screen_classifier.classify(window)

    def print_hero_stat(self, window):
        values = []

        for i in range(10):
            hero_imgarr = get_player_image(window, i, True)
            values.append(get_avg_image_channel_diff(hero_imgarr))

        print(", ".join("%.2f" % f for f in values))

    def get_heroes(self, window):
        '''
        Returns array of 10 indexes of heros (sorted by name)
        If a hero is not selected -1 is returned
        '''

        images, selections = get_player_images(window)

        # reduce noise by accomulating data over several frames
        self.selections.update(selections)

        if self.selections.iterations < 5:
            return self.prev_selections

        selections = self.selections.get()
        self.selections.reset()

        predictions = np.full((10), -1)

        for i, img in enumerate(images):
            if selections[i] > .5:
                predictions[i] = self.hero_classifier.classify(img)

        return predictions

    def should_reset_heroes(self, heroes):

        for i, value in enumerate(self.prev_selections):
            if value != -1 and heroes[i] != value:
                return True

        return False

    def get_new_selections(self, heroes):
        retval = []

        for i, value in enumerate(heroes):
            if value != self.prev_selections[i]:
                retval.append((value, i > 4))

        return retval

    def process_messages(self, message):
        ''' Process received messages '''

        print('Proc Received: ' + message.msgtype)

        if message.msgtype == 'reset':
            self.reset()

        if message.msgtype == 'kill':
            self.isrunning = False

    def run(self):
        ''' Process main loop '''

        while self.isrunning:

            try:
                if self.pipe.poll():
                    message = self.pipe.recv()
                    self.process_messages(message)

            except EOFError:
                print("Exiting process...")
                break

            if not window_exist(self.window_name):
                print(self.window_name + " is not running")
                sleep(1)
                continue

            # remove alpha channel from captured image
            window = capture_window(self.window_name)[..., [0, 1, 2]]

            if not self.is_pick_screen(window):
                sleep(1)
                continue

            if self.pos_detector.iterations < 5:
                self.pos_detector.update(window)
                continue

            player_pos = self.pos_detector.get()

            if player_pos != self.prev_player_pos:
                self.pipe.send(ControllerMessage(
                    "select_team", False if player_pos < 5 else True))
                self.prev_player_pos = player_pos

            heroes = self.get_heroes(window)
            newheroes = self.get_new_selections(heroes)

            if newheroes:
                self.pipe.send(ControllerMessage("add_heroes", newheroes))

            self.prev_selections = heroes


if __name__ == '__main__':

    app = QApplication(sys.argv)

    mother_pipe, child_pipe = Pipe()
    controller = ControllerProc(child_pipe)

    browser = PickerBrowserView()

    browser.loadFinished.connect(controller.start)
    browser.show()

    def add_heroes(items):
        for item in items:
            browser.select_hero(item[0], not item[1])

    browser.reset_clicked.connect(
        lambda: mother_pipe.send(ControllerMessage("reset")))

    signals = ControllerSignals()

    signals.select_team.connect(browser.select_team)
    signals.add_heroes.connect(add_heroes)

    threadpool = QThreadPool()
    msgthread = MessageProcessorThread(mother_pipe, signals)
    threadpool.start(msgthread)

    def cleanup():
        mother_pipe.send(ControllerMessage("kill"))
        child_pipe.send(ControllerMessage("kill"))

    browser.closing.connect(cleanup)

    sys.exit(app.exec_())
