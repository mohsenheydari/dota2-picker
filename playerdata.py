''' Module contains helper class to accomulate and take average of players data over time '''

import numpy as np
from image import get_avg_image_value
from screenprocessing import get_player_name


class PlayerDataOverTime:
    ''' helper class responsible for accomulating and taking average of players data over time '''

    player_data = np.zeros(10)
    iterations = 0

    def __init__(self, filterfn=None):

        if filterfn is not None:
            self.filter = filterfn
        else:
            self.filter = self.passthrough

    def passthrough(self, i, data):
        return data[i]

    def update(self, data):

        for i in range(10):
            self.player_data[i] += self.filter(i, data)

        self.iterations += 1

    def reset(self):
        self.player_data = np.zeros(10)
        self.iterations = 0

    def get(self):
        ''' Returns average value of players data over time (iterations)'''

        if self.iterations == 0:
            return None

        return self.player_data/self.iterations


class PlayerPosition(PlayerDataOverTime):
    ''' Helper class to reduce player position detection errors '''

    def __init__(self):
        super().__init__(self.proc)

    def proc(self, i, window):
        playername = get_player_name(window, i)
        return get_avg_image_value(playername)

    def get(self):
        return np.argmax(super().get())
