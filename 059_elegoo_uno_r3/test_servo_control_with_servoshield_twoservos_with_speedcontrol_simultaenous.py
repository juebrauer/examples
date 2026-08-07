import time
import serial


PORT = "/dev/ttyACM0"

SERVO_1 = 7
SERVO_2 = 8

# Zuletzt bekannte Position jedes Servos
current_angles: dict[int, int] = {}


def send_command(ser, command: str) -> str:
    """Sendet einen Befehl und wartet auf die Antwort des Arduino."""

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
    """Sendet eine Position direkt an den Arduino."""
    send_command(ser, f"{channel} {angle}")


def set_servo(ser, channel: int, angle: int) -> None:
    """
    Setzt einen Servo sofort auf eine Zielposition.

    Keine Geschwindigkeitskontrolle.
    """

    angle = max(0, min(180, angle))

    write_servo(ser, channel, angle)

    current_angles[channel] = angle


def set_servos(
    ser,
    targets: dict[int, int],
) -> None:
    """Setzt mehrere Servos direkt auf ihre Zielpositionen."""

    for channel, angle in targets.items():
        set_servo(ser, channel, angle)


def move_servo(
    ser,
    channel: int,
    target_angle: int,
    speed: float,
) -> None:
    """Bewegt einen einzelnen Servo mit kontrollierter Geschwindigkeit."""

    move_servos(
        ser,
        {channel: target_angle},
        speed=speed,
    )


def move_servos(
    ser,
    targets: dict[int, int],
    speed: float,
    update_hz: float = 50,
) -> None:
    """
    Bewegt mehrere Servos synchron zu ihren Zielpositionen.

    targets:
        Dictionary:
            Kanal -> Zielwinkel

        Beispiel:
            {
                7: 120,
                8: 60,
            }

    speed:
        Maximale Geschwindigkeit in Grad pro Sekunde.

    Alle Servos starten gleichzeitig und erreichen ihr
    jeweiliges Ziel gleichzeitig.
    """

    if speed <= 0:
        raise ValueError("speed muss > 0 sein")

    if update_hz <= 0:
        raise ValueError("update_hz muss > 0 sein")

    if not targets:
        return

    # Winkel auf gültigen Bereich begrenzen
    targets = {
        channel: max(0, min(180, angle))
        for channel, angle in targets.items()
    }

    # Für eine kontrollierte Bewegung müssen wir die
    # aktuelle Position jedes beteiligten Servos kennen.
    for channel in targets:
        if channel not in current_angles:
            raise ValueError(
                f"Aktuelle Position von Servo {channel} "
                "ist nicht bekannt."
            )

    start_angles = {
        channel: current_angles[channel]
        for channel in targets
    }

    distances = {
        channel: abs(
            targets[channel] - start_angles[channel]
        )
        for channel in targets
    }

    max_distance = max(distances.values())

    # Alle Servos stehen bereits richtig.
    if max_distance == 0:
        return

    # Der Servo mit dem längsten Weg bestimmt die Dauer.
    duration = max_distance / speed

    print(
        f"Move {targets}, "
        f"duration={duration:.2f}s"
    )

    start_time = time.monotonic()
    next_update = start_time

    # Damit wir nicht unnötig dieselbe Position mehrfach senden.
    last_sent = start_angles.copy()

    while True:

        now = time.monotonic()

        elapsed = now - start_time

        progress = min(
            elapsed / duration,
            1.0,
        )

        for channel, target_angle in targets.items():

            start_angle = start_angles[channel]

            angle = round(
                start_angle
                + (target_angle - start_angle) * progress
            )

            if angle != last_sent[channel]:
                write_servo(
                    ser,
                    channel,
                    angle,
                )

                last_sent[channel] = angle

        if progress >= 1.0:
            break

        next_update += 1.0 / update_hz

        remaining = next_update - time.monotonic()

        if remaining > 0:
            time.sleep(remaining)

    # Zur Sicherheit exakt das Ziel setzen.
    for channel, target_angle in targets.items():

        if last_sent[channel] != target_angle:
            write_servo(
                ser,
                channel,
                target_angle,
            )

        current_angles[channel] = target_angle


def disable_servo(ser, channel: int) -> None:
    send_command(
        ser,
        f"OFF {channel}",
    )


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

        # "READY" des Arduino verwerfen.
        ser.reset_input_buffer()

        # --------------------------------------------------
        # Ausgangsposition
        # --------------------------------------------------

        set_servos(
            ser,
            {
                SERVO_1: 90,
                SERVO_2: 90,
            },
        )

        time.sleep(1)

        # --------------------------------------------------
        # Demo
        # --------------------------------------------------

        for run in range(3):

            print(f"{run=}")

            move_servos(
                ser,
                {
                    SERVO_1: 120,
                    SERVO_2: 60,
                },
                speed=20,
            )

            time.sleep(0.5)

            move_servos(
                ser,
                {
                    SERVO_1: 60,
                    SERVO_2: 120,
                },
                speed=30,
            )

            time.sleep(0.5)

            move_servos(
                ser,
                {
                    SERVO_1: 90,
                    SERVO_2: 90,
                },
                speed=60,
            )

            time.sleep(0.5)

        # PWM abschalten
        disable_servo(ser, SERVO_1)
        disable_servo(ser, SERVO_2)

    print("Servo demo finished!")


if __name__ == "__main__":
    main()