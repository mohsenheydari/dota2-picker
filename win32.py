""" Module contains win32 helper functions """

import win32gui
import win32ui
import win32con
import win32api
import numpy as np


def get_window_rect(window_name):
    ''' Returns window rect by window name '''
    hwnd = win32gui.FindWindow(None, window_name)
    return win32gui.GetWindowRect(hwnd)


def get_client_rect(window_name):
    ''' Returns window client rect by window name '''
    hwnd = win32gui.FindWindow(None, window_name)
    return win32gui.GetClientRect(hwnd)


def window_exist(name):
    ''' Returns if window handle exist for given window name '''

    hwnd = win32gui.FindWindow(None, name)

    if hwnd < 1:
        return False

    return True


def click(x: int, y: int):
    ''' Emulate mouse click on specific absolute position '''

    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE |
                         win32con.MOUSEEVENTF_ABSOLUTE, x, y)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


def capture_window(name):
    ''' Return captured window in RGBA format (numpy array) '''

    hwnd = win32gui.FindWindow(None, name)

    cr = win32gui.GetClientRect(hwnd)
    #leftTop = win32gui.ClientToScreen(hwnd, (cr[0], cr[1]))

    width = cr[2]
    height = cr[3]

    hDC = win32gui.GetDC(hwnd)
    myDC = win32ui.CreateDCFromHandle(hDC)
    newDC = myDC.CreateCompatibleDC()

    mybitmap = win32ui.CreateBitmap()
    mybitmap.CreateCompatibleBitmap(myDC, width, height)

    newDC.SelectObject(mybitmap)

    # win32gui.SetForegroundWindow(hwnd)

    newDC.BitBlt((0, 0), (width, height), myDC, (0, 0), win32con.SRCCOPY)
    bminfo = mybitmap.GetInfo()

    signedintarray = mybitmap.GetBitmapBits(True)
    img = np.fromstring(signedintarray, dtype='uint8')
    img = img.reshape(bminfo['bmHeight'], bminfo['bmWidth'], 4)
    img = img[..., [2, 1, 0, 3]]  # convert BGRA to RGBA

    myDC.DeleteDC()
    newDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hDC)
    win32gui.DeleteObject(mybitmap.GetHandle())

    return img
