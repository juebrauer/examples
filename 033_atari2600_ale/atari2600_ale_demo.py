import gymnasium as gym
import ale_py

import cv2

gym.register_envs(ale_py)

# Gymnasium/ALE's built-in `render_mode="human"` window doesn't expose a supported
# "window size" parameter. To control the window size, render frames as arrays and
# display them yourself.

WINDOW_SCALE = 3.5  # <1.0 smaller, 1.0 native Atari frame size, >1.0 bigger
WINDOW_NAME = "ALE"

STEPS_PER_GAME = 500
NUM_GAMES_TO_SHOW = 7

STEP_DELAY_MS = 10

# The 7 Atari 2600 games often reported in the original DQN results.
# If a ROM is missing, env creation will fail and the game will be skipped.
CANDIDATE_GAMES = [
    "ALE/BeamRider-v5",
    "ALE/Breakout-v5",
    "ALE/Enduro-v5",
    "ALE/Pong-v5",
    "ALE/Qbert-v5",
    "ALE/Seaquest-v5",
    "ALE/SpaceInvaders-v5",
]

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

shown_games = 0
should_quit = False
window_sized = False

for game_id in CANDIDATE_GAMES:
    if shown_games >= NUM_GAMES_TO_SHOW or should_quit:
        break

    try:
        env = gym.make(game_id, render_mode="rgb_array")
    except Exception as e:
        print(f"Skipping {game_id} (could not create env): {e}")
        continue

    # Show game title + number in the window title bar (not overlaid onto the image).
    try:
        game_nr = shown_games + 1
        cv2.setWindowTitle(WINDOW_NAME, f"{game_nr}/{NUM_GAMES_TO_SHOW}: {game_id}")
    except Exception:
        pass

    observation, info = env.reset(seed=42)
    for _ in range(STEPS_PER_GAME):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        frame = observation
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if WINDOW_SCALE != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (max(1, int(w * WINDOW_SCALE)), max(1, int(h * WINDOW_SCALE))),
                interpolation=cv2.INTER_NEAREST,
            )

        if not window_sized:
            h, w = frame.shape[:2]
            try:
                cv2.resizeWindow(WINDOW_NAME, w, h)
            except Exception:
                pass
            window_sized = True

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(STEP_DELAY_MS) & 0xFF
        if key in (27, ord("q")):
            should_quit = True
            break

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    shown_games += 1

cv2.destroyAllWindows()