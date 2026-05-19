import numpy as np

start_pos = 0 # [m], start position of the vehicle
start_speed = 13.8 # [m/s], 50 km/h = 50.000 m / 3600 s = 13.8 m/s

dt = 1.0 # [s] time step

max_simu_time = 60 # [s]

# x_k = [new_pos, new_speed] = true_F @ x_(k-1) = true_F @ [old_pos, old_speed]
#                            = [old_pos + dt*old_speed, old_speed]
true_F = np.array( [[1.0, dt],
                    [0.0, 1.0]] )

# Wir beschleunigen Auto von 50 km/h auf 100 km/h in einer Stunde:
# a [m/s^2] = delta_speed / delta_time = 50000m / (3600^2)s^2 = 0.003858025 m/s^2
u = np.array( [0.003858025] )

true_B = np.array( [[0.0],
                    [dt]] )

# true_B (Steuermatrix) ist aus R^(2,1)
# u ist aus R^(1,1)
# damit ist true_B @ u aus R^(2,1)

# setze Prozessrauschkovarianzmatrix
true_Q = np.array( [[1, 0.0],
                    [0.0, 0.01]])

true_H = np.array( [[1.0, 0.0],
                    [0.0, 1.0]] )

# setze Messrauschkovarianzmatrix
true_R = np.array( [[1000, 0.0],
                    [0.0, 1.0]])

measurement_each_nth_step = 5


