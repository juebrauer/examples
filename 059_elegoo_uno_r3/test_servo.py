import time

from pyfirmata2 import Arduino


PORT = "/dev/ttyACM0"
SERVO_PIN = 9


def move_slowly(
    servo,
    start_angle: int,
    target_angle: int,
    delay_seconds: float = 0.1,
) -> int:
    """Bewegt den Servo in Ein-Grad-Schritten."""

    if start_angle == target_angle:
        return target_angle

    step = 1 if target_angle > start_angle else -1

    for angle in range(
        start_angle,
        target_angle + step,
        step,
    ):
        servo.write(angle)
        time.sleep(delay_seconds)

    return target_angle


def main() -> None:
    board = Arduino(PORT)

    # d = digital pin
    # 9 = pin 9
    # s = servo-mode
    servo = board.get_pin(f"d:{SERVO_PIN}:s")

    for i in range(5):
        # Zunächst in die ungefähre Mittelstellung.
        current_angle = 90
        servo.write(current_angle)
        time.sleep(1)

        # Absichtlich nur ein kleiner Testbereich.
        current_angle = move_slowly(
            servo,
            current_angle,
            120,
            delay_seconds=0.1,
        )

        time.sleep(0.5)

        current_angle = move_slowly(
            servo,
            current_angle,
            60,
            delay_seconds=0.1,
        )

        time.sleep(0.5)

        current_angle = move_slowly(
            servo,
            current_angle,
            90,
            delay_seconds=0.1,
        )

    board.exit()


if __name__ == "__main__":
    main()
