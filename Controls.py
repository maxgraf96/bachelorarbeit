import sys
import select
import tty
import termios

def start_listener():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            if isData():
                c = sys.stdin.read(1)
                if c is not None:
                    print(c)
                if c == '\x1b':  # x1b is ESC
                    break

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

