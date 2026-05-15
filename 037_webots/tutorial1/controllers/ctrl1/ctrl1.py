from controller import Robot, Motor

print("Controller v7 - simple obstacle avoidance with state machine")

TIME_STEP = 64

MAX_SPEED = 6.28

# create the Robot instance.
robot = Robot()

# get a handler to the motors and set target position to infinity (speed control)
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# set up the motor speeds at 10% of the MAX_SPEED.
leftMotor.setVelocity(0 * MAX_SPEED)
rightMotor.setVelocity(0 * MAX_SPEED)

# get handlers to the 8 IR sensors and enable them
ir_sensors = []
ir_sensor_names = [
    'ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7'
]
for name in ir_sensor_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    ir_sensors.append(sensor)

state = "FORWARD"

while robot.step(TIME_STEP) != -1:
   # read values from the IR sensors
   ir_vec = []
   ir_values = ""
   for sensor in ir_sensors:
      value = sensor.getValue()
      ir_values += f"{value:.2f} "
      ir_vec.append(value)
   print(f"{state} : {ir_values}")

   # simple obstacle avoidance logic
   if state == "FORWARD":
      leftMotor.setVelocity(0.5 * MAX_SPEED)
      rightMotor.setVelocity(0.5 * MAX_SPEED)
      if ir_vec[0] > 80 or ir_vec[7] > 80 or ir_vec[1] > 80 or ir_vec[6] > 80:
         state = "TURN"
   elif state == "TURN":
      leftMotor.setVelocity(0.5 * MAX_SPEED)
      rightMotor.setVelocity(-0.5 * MAX_SPEED)
      if ir_vec[0] < 80 and ir_vec[7] < 80 and ir_vec[1] < 80 and ir_vec[6] < 80:
         state = "FORWARD"

   
   