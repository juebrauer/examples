import time
import serial


PORT = "/dev/ttyACM0"

SERVO_1 = 7
SERVO_2 = 8


def set_servo(ser, channel: int, angle: int) -> None:
    ser.write(f"{channel} {angle}\n".encode("ascii"))
    ser.flush()


def disable_servo(ser, channel: int) -> None:
    ser.write(f"OFF {channel}\n".encode("ascii"))
    ser.flush()


def main() -> None:

    print("Start to move servo ...")

    with serial.Serial(PORT, 115200, timeout=1) as ser:

        # Uno resettiert beim Öffnen der Schnittstelle
        time.sleep(2)

        # Beide zunächst in Mittelstellung
        set_servo(ser, SERVO_1, 90)
        set_servo(ser, SERVO_2, 90)

        time.sleep(2)

        for run in range(3):

            print(f"{run=}")

            # Gegensinnige Bewegung
            set_servo(ser, SERVO_1, 120)
            set_servo(ser, SERVO_2, 60)

            time.sleep(2)

            set_servo(ser, SERVO_1, 60)
            set_servo(ser, SERVO_2, 120)

            time.sleep(2)

            set_servo(ser, SERVO_1, 90)
            set_servo(ser, SERVO_2, 90)

            time.sleep(2)

        # PWM abschalten
        disable_servo(ser, SERVO_1)
        disable_servo(ser, SERVO_2)

    print("Servo demo finished!")


if __name__ == "__main__":
    main()