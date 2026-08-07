import time
import serial

PORT = "/dev/ttyACM0"

SERVO_1 = 7
SERVO_2 = 8

current_angles = {}


def send_command(ser, command: str) -> str:
    """Sendet einen Befehl und liest die Antwort des Arduino."""
    ser.write((command + "\n").encode("ascii"))
    ser.flush()

    response = ser.readline().decode("ascii").strip()

    if not response:
        raise RuntimeError(
            f"Keine Antwort vom Arduino auf: {command}"
        )

    if response.startswith("ERROR"):
        raise RuntimeError(
            f"Arduino meldet Fehler: {response}"
        )

    return response


def write_servo(ser, channel: int, angle: int) -> None:
    send_command(ser, f"{channel} {angle}")


def disable_servo(ser, channel: int) -> None:
    send_command(ser, f"OFF {channel}")


def set_servo(
    ser,
    channel: int,
    target_angle: int,
    speed: float | None = None,
) -> None:
    """
    Bewegt einen Servo zur Zielposition.

    speed:
        Geschwindigkeit in Grad pro Sekunde.
        Bei None wird die Position sofort gesetzt.
    """

    target_angle = max(0, min(180, target_angle))

    # Position des Servos noch nicht bekannt
    if channel not in current_angles:
        write_servo(ser, channel, target_angle)
        current_angles[channel] = target_angle
        return

    current_angle = current_angles[channel]

    # Ohne Geschwindigkeitsbegrenzung direkt setzen
    if speed is None:
        write_servo(ser, channel, target_angle)
        current_angles[channel] = target_angle
        return

    if speed <= 0:
        raise ValueError("speed muss > 0 sein")

    if current_angle == target_angle:
        return

    step = 1 if target_angle > current_angle else -1

    # Gewünschte Zeit pro Grad
    period = 1.0 / speed

    next_time = time.monotonic()

    for angle in range(
        current_angle + step,
        target_angle + step,
        step,
    ):
        write_servo(ser, channel, angle)

        # Nächsten Schritt zeitlich möglichst genau planen.
        next_time += period

        remaining = next_time - time.monotonic()

        if remaining > 0:
            time.sleep(remaining)

    current_angles[channel] = target_angle


def main() -> None:

    print("Start to move servos ...")

    with serial.Serial(
        PORT,
        115200,
        timeout=1,
        write_timeout=1,
    ) as ser:

        # Uno resettiert beim Öffnen der Schnittstelle.
        time.sleep(2)

        # Der Arduino hat beim Start "READY" gesendet.
        # Diese Meldung brauchen wir hier nicht mehr.
        ser.reset_input_buffer()

        set_servo(ser, SERVO_1, 90)
        set_servo(ser, SERVO_2, 90)

        time.sleep(1)

        for run in range(3):

            print(f"{run=}")

            set_servo(
                ser,
                SERVO_1,
                120,
                speed=20,
            )

            set_servo(
                ser,
                SERVO_2,
                60,
                speed=60,
            )

            time.sleep(0.5)

            set_servo(
                ser,
                SERVO_1,
                60,
                speed=30,
            )

            set_servo(
                ser,
                SERVO_2,
                120,
                speed=30,
            )

            time.sleep(0.5)

            set_servo(
                ser,
                SERVO_1,
                90,
                speed=60,
            )

            set_servo(
                ser,
                SERVO_2,
                90,
                speed=60,
            )

            time.sleep(0.5)

        disable_servo(ser, SERVO_1)
        disable_servo(ser, SERVO_2)

    print("Servo demo finished!")


if __name__ == "__main__":
    main()