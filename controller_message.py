''' Module contains Controller Message data class '''


class ControllerMessage:
    ''' Controller message data class '''
    msgtype = ""
    data = None

    def __init__(self, msgtype, data=None):
        self.msgtype = msgtype
        self.data = data
