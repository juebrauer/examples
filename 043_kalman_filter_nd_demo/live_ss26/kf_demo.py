import numpy as np
import params as p
import matplotlib.pyplot as plt
import kalman_filter as kfi

true_x = np.array([p.start_pos, p.start_speed])
simu_time = 0.0

true_xs = []
zs = []
kf_est = []
state_uncertainties = []

initial_P = np.array([[10.0, 0.0],
                      [0.0, 10.0]])
kf = kfi.kalman_filter(initial_x = true_x,
                       initial_P = initial_P,
                       F=p.true_F,
                       B=p.true_B,
                       Q=p.true_Q,
                       H=p.true_H,
                       R=p.true_R)
                       

while simu_time < p.max_simu_time:

    # System evolution
    process_noise = np.random.multivariate_normal( np.array([0,0]), p.true_Q)
    true_x = p.true_F @ true_x + p.true_B @ p.u + process_noise

    # Simuluiere verrauschte Messung
    measurement_noise = np.random.multivariate_normal( np.array([0,0]), p.true_R)
    z = p.true_H @ true_x + measurement_noise

    # KF Updates
    kf.predict(p.u)
    if simu_time != 0 and simu_time % p.measurement_each_nth_step == 0:
        kf.correct(z)


    true_xs.append( true_x )
    zs.append( z )
    kf_est.append( kf.x )
    state_uncertainties.append( kf.get_scalar_measure_of_uncertainty() )

    simu_time += p.dt


plt.figure(figsize=(15,10))
plt.subplot(5,1, 1)
plt.title("Position")
plt.xlabel("time [s]")
plt.ylabel("position [m]")
plt.plot( [x[0] for x in true_xs], color="black", label="true" )
plt.plot( [z[0] for z in zs],      color="red",   label="measured" )
plt.plot( [x[0] for x in kf_est],  color="green", label="KF" )
plt.legend()

plt.subplot(5,1, 2)
plt.title("Speed")
plt.xlabel("time [s]")
plt.ylabel("speed [m/s]")
plt.plot( [x[1] for x in true_xs], color="black", label="true" )
plt.plot( [z[1] for z in zs],      color="red",   label="measured" )
plt.plot( [x[1] for x in kf_est],  color="green", label="KF" )
plt.legend()

plt.subplot(5,1, 3)
plt.title("Position errors per time step")
plt.xlabel("time [s]")
plt.ylabel("pos error [m]")
i = 0
errors_kf   = [est[i]-actual[i] for actual,est in zip(true_xs, kf_est)]
errors_meas = [z[i]-actual[i]   for actual,z in zip(true_xs, zs)]
errors_kf_mean = np.mean(np.abs(errors_kf))
errors_meas_mean = np.mean(np.abs(errors_meas))
plt.plot( errors_kf,   color="green", label=f"KF errors ({errors_kf_mean:.2f})" )
plt.plot( errors_meas, color="red",   label=f"Meas. errors ({errors_meas_mean:.2f}) " )
plt.legend()

plt.subplot(5,1, 4)
plt.title("Speed errors per time step")
plt.xlabel("time [s]")
plt.ylabel("speed error [m/s]")
i = 1
errors_kf   = [est[i]-actual[i] for actual,est in zip(true_xs, kf_est)]
errors_meas = [z[i]-actual[i]   for actual,z in zip(true_xs, zs)]
errors_kf_mean = np.mean(np.abs(errors_kf))
errors_meas_mean = np.mean(np.abs(errors_meas))
plt.plot( errors_kf,   color="green", label=f"KF errors ({errors_kf_mean:.2f})" )
plt.plot( errors_meas, color="red",   label=f"Meas. errors ({errors_meas_mean:.2f}) " )
plt.legend()

plt.subplot(5,1, 5)
plt.title("Uncertainty about KF estimated state")
plt.xlabel("time [s]")
plt.ylabel("det|P|")
plt.plot( state_uncertainties, color="black")



plt.subplots_adjust(hspace=1)

plt.show()