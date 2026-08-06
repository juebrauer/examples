import time

from pyfirmata2 import Arduino


PORT = "/dev/ttyACM0"


def main() -> None:
    board = Arduino(PORT)

    # Die eingebaute LED des Uno hängt an Pin 13.
    led = board.get_pin("d:13:o")

    try:
        for _ in range(5):
            led.write(True)
            time.sleep(0.5)

            led.write(False)
            time.sleep(0.5)

    finally:
        led.write(False)
        board.exit()


if __name__ == "__main__":
    main()
