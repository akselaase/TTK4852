import numpy as np

# The data-values are currently just hardcoded, but must be extracted from the satel
channel_dt_s = np.array([0.507, 0.498]) # Time difference between channels 2-3 and 3-4
channel_resolution_m = 10 * np.ones(3, dtype=int) # Channel 2, 3 and 4 has an accuracy of 10 m

# Pixel values for channel 2, 3 and 4
measurements = np.array(
  [
    [5687, 5680, 5671], # x
    [1911, 1913, 1914]  # y
  ], 
  dtype=int) 

# Data assumed about the satellite
sentinel_lat_deg          = 50.042677   # [deg]
sentinel_inclination_deg  = 98.6        # [deg]
sentinel_abs_vel          = 7460        # [m/s]
sentinel_height_m         = 786000      # [m]

sentinel_angle_vel_x = np.sin(
  np.deg2rad(
    90 - sentinel_inclination_deg
  )
) 
sentinel_angle_vel_y = np.cos(
  np.deg2rad(
    90 - sentinel_inclination_deg
  )
) 

sentinel_vel_vec = sentinel_abs_vel * np.array(
  [
    [sentinel_angle_vel_x], 
    [sentinel_angle_vel_y]
  ]
)

# Assumption that the altitude of the aircraft is low enough, such that the resolution
# of each channel matches the ground. This is not really the case when the aircraft has
# an altitude greater than 0 
num_measurements = measurements.shape[1]
assert num_measurements >= 2, "Need at least two measurements to calculate the velocity"

# Calculating the change in distance between the channels
delta_pos = np.zeros((2, num_measurements - 1))
rel_vel_hat_arr = np.zeros((2, num_measurements - 1))

for idx in range(num_measurements - 1):
  current_measurement = measurements[:, idx]
  next_measurement = measurements[:, idx + 1]
  
  d_pos = next_measurement - current_measurement

  # Currently, only one resolution on the camera is used to calculate the real position
  # Unsure how to mix multiple resolutions, when converting into a change in meter

  # Converting into actual displacement in m
  resolution = channel_resolution_m[0]
  delta_pos[:, idx] = d_pos * resolution

  # Calculating the velocities
  dt = channel_dt_s[idx]
  rel_vel_hat_arr[:, idx] = delta_pos[:, idx] / dt


# TODO: Implement a least-square method to calculate the velocities
rel_vel_hat = np.average(a=rel_vel_hat_arr, axis=1)

# Calculate the different angles 
# phi:    Relative course-angle
# theta:  Aircraft's course TODO: Determine this somehow
# psi:    Satelitte's course  
phi = np.arctan2(rel_vel_hat[0], rel_vel_hat[1])
theta = np.deg2rad(-70) # TODO
psi = np.cos(np.deg2rad(sentinel_inclination_deg) / np.deg2rad(sentinel_lat_deg))

# Calculate the velocity and altitude of the aircraft
vel_aircraft_hat = np.sin(phi - psi) / np.sin(theta - psi) * np.linalg.norm(rel_vel_hat)
h_aircraft_hat = np.sin(phi - theta) / np.sin(theta - psi) * np.linalg.norm(rel_vel_hat) / sentinel_abs_vel * sentinel_height_m 

print(vel_aircraft_hat)
print(h_aircraft_hat)
