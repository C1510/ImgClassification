import cv2 as cv
import sys, math
import numpy as np


def threshold_image(img, thresh, mode = 'mg'):
    if mode == 'mg':
        thresh_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    elif mode == 'bi':
        thresh_method = cv.THRESH_BINARY
    ret, thresh = cv.threshold(img, thresh, 255, thresh_method)
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    return ret, thresh

def get_connected_components(img, min_side, max_side=None):
    b = cv.bitwise_not(img)
    n, labels, stats, centroids = cv.connectedComponentsWithStats(b)

    min = min_side**2
    if max_side is not None:
        max = max_side **2

    stats = np.array(stats)
    if max_side is not None:
        mask = np.array([True if (max > stats[i, 2] * stats[i, 3] > min and stats[i, 2]>1 and stats[i, 3]>1) else False for i in range(stats.shape[0])])
    else:
        mask = np.array([True if (stats[i, 2] * stats[i, 3] > min and stats[i, 2]>1 and stats[i, 3]>1) else False for i in range(stats.shape[0])])

    stats = stats[mask, :]

    mask = [True if (np.mean(img[col[1]:col[1]+col[3], col[0]:col[0]+col[2]])>10 and not math.isnan(np.mean(img[col[1]:col[1]+col[3], col[0]:col[0]+col[2]]))) else False for col in stats]
    stats = stats[mask, :]

    return stats

def plt_rectangles(img, stats):
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for col in stats:
        color = (255, 0, 0)
        cv.rectangle(img, (col[0], col[1]), (col[0] + col[2], col[1] + col[3]), color, 1)
    return img

def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch

