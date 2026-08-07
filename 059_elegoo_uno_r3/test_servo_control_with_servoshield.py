import time
import serial


PORT = "/dev/ttyACM0"
SERVO_CHANNEL = 7


def set_servo(ser, channel: int, angle: int) -> None:
    command = f"{channel} {angle}\n"
    ser.write(command.encode("ascii"))
    ser.flush()


def disable_servo(ser, channel: int) -> None:
    command = f"OFF {channel}\n"
    ser.write(command.encode("ascii"))
    ser.flush()


def move_slowly(
    ser,
    channel: int,
    start_angle: int,
    target_angle: int,
    delay_seconds: float = 0.1,
) -> int:

    if start_angle == target_angle:
        return target_angle

    step = 1 if target_angle > start_angle else -1

    for angle in range(
        start_angle,
        target_angle + step,
        step,
    ):
        set_servo(ser, channel, angle)
        time.sleep(delay_seconds)

    return target_angle


def main() -> None:

    print("Start to move servo ...")

    with serial.Serial(PORT, 115200, timeout=1) as ser:

        # Beim Öffnen der seriellen Schnittstelle
        # resettiert der Uno normalerweise.
        time.sleep(2)

        current_angle = 90
        set_servo(ser, SERVO_CHANNEL, current_angle)
        time.sleep(1)

        for run in range(3):

            print(f"{run=}")

            current_angle = move_slowly(
                ser,
                SERVO_CHANNEL,
                current_angle,
                120,
            )

            time.sleep(0.5)

            current_angle = move_slowly(
                ser,
                SERVO_CHANNEL,
                current_angle,
                60,
            )

            time.sleep(0.5)

            current_angle = move_slowly(
                ser,
                SERVO_CHANNEL,
                current_angle,
                90,
            )

        # PWM für diesen Kanal abschalten.
        disable_servo(ser, SERVO_CHANNEL)

    print("Finished servo demo!")


if __name__ == "__main__":
    main()