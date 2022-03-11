import numpy as np

# Data for channels
num_channels = 3

# The data-values are currently just hardcoded, but must be extracted from the satel
channel_dt_s_arr = np.array([0.507, 0.498]) # Time difference between channels 2-3 and 3-4
channel_resolution_m_arr = 10 * np.ones(3, dtype=int) # Channel 2, 3 and 4 has an accuracy of 10 m

# Pixel values for channel 2, 3 and 4
measurements = np.array(
  [
    [5687, 5680, 5671], # x (width)
    [1911, 1913, 1914]  # y (height)
  ], 
  dtype=int) 

# Data assumed about the satellite - haven't found better parameters
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
  resolution = channel_resolution_m_arr[0]
  delta_pos[:, idx] = d_pos * resolution

  # Calculating the velocities
  dt = channel_dt_s_arr[idx]
  rel_vel_hat_arr[:, idx] = delta_pos[:, idx] / dt


# TODO: Implement a least-square method to calculate the velocities
rel_vel_hat = np.average(a=rel_vel_hat_arr, axis=1)

def calculate_aircraft_heading(
    channel_intensities : np.ndarray, 
    indeces             : np.ndarray
  ) -> float:
  """
  Estimates the aircraft's heading based on the weighted covariance 
  of the pixel-values.

  This code is currently written in a form of a pseudo-code, and assumes 
  that all of the measurements are to be given simultaneously. It will
  therefore calculate the final estimate as an average of the current 
  estimates

  Input:
    channel_intensities : Measured intensities for each channel in an area around the aircraft
    indeces             : Pixel-indeces corresponding to the measured intensities

  Output:
    Estimated aircraft heading in rad
  """

  reflectance_arr = np.zeros(num_channels)
  # Iterate over all measured intensities in the channel_intensities, 
  # and calculate channel reflectance
  return 0.0
  for ch in range(num_channels):
    # Iterate over all indeces assumed relevant
    channel_resolution_m = channel_resolution_m_arr[ch]
    for idx in indeces:
      # Sum up the intensities at the channels 
      pass
    
  # Calculate weighted center, covariances and estimated heading for each channel
  weighted_coordinates = np.zeros((2, num_channels))
  covariances = np.zeros((3, num_channels)) # sigma_xx, sigma_xy, sigma_yy stacked vertically for each channel
  estimated_headings = np.zeros(num_channels)
  for ch in range(num_channels):
    channel_resolution_m = channel_resolution_m_arr[ch]
    channel_reflectance = reflectance_arr[ch]

    # Weighted center
    x_bar = 0
    y_bar = 0
    for idx in indeces:
      # Sum up the weighted intensities at the channels 
      # Unsure how these are set
      x_idx = idx[0]
      y_idx = idx[1]
      x_bar += x_idx * channel_intensities[ch][x_idx, y_idx] # Unsure how these are given as parameters
      y_bar += y_idx * channel_intensities[ch][x_idx, y_idx] # Unsure how these are given as parameters
    
    x_bar = x_bar * channel_resolution_m / channel_reflectance
    y_bar = y_bar * channel_resolution_m / channel_reflectance

    weighted_coordinates[:, ch] = np.array([x_bar, y_bar]).T

    # Covariances
    sigma_xx = 0
    sigma_xy = 0
    sigma_yy = 0
    for idx in indeces:
      # Sum up the weighted intensities at the channels 
      # Unsure how these are set
      x_idx = idx[0]
      y_idx = idx[1]
      sigma_xx += (x_idx**2) * channel_intensities[ch][x_idx, y_idx] # Unsure how these are given as parameters
      sigma_xy += (x_idx * y_idx) * channel_intensities[ch][x_idx, y_idx] # Unsure how these are given as parameters
      sigma_yy += (y_idx**2) * channel_intensities[ch][x_idx, y_idx] # Unsure how these are given as parameters
    
    sigma_xx = sigma_xx * ((channel_resolution_m / channel_reflectance)**2) - (x_bar**2)
    sigma_xy = sigma_xy * ((channel_resolution_m / channel_reflectance)**2) - (x_bar * y_bar)
    sigma_yy = sigma_yy * ((channel_resolution_m / channel_reflectance)**2) - (y_bar**2)

    covariances[:, ch] = np.array([sigma_xx, sigma_xy, sigma_yy]).T

    # Estimated headings
    if sigma_xx**2 - sigma_yy**2 == 0:
      # Not enough information from this channel
      continue
    heading_hat = 0.5 * np.arctan((2 * sigma_xy**2) / (sigma_xx**2 - sigma_yy**2))
    estimated_headings[ch] = heading_hat

  # Return averaged headings
  return np.mean(estimated_headings)

# Calculate the different angles 
# phi:    Relative course-angle
# theta:  Aircraft's course/heading TODO: Determine this somehow
# psi:    Satelitte's course  
phi = np.arctan2(rel_vel_hat[0], rel_vel_hat[1])
theta = np.deg2rad(-70) # TODO
psi = np.cos(np.deg2rad(sentinel_inclination_deg) / np.deg2rad(sentinel_lat_deg))

# Calculate the velocity and altitude of the aircraft
vel_aircraft_hat = np.sin(phi - psi) / np.sin(theta - psi) * np.linalg.norm(rel_vel_hat)
h_aircraft_hat = np.sin(phi - theta) / np.sin(theta - psi) * np.linalg.norm(rel_vel_hat) / sentinel_abs_vel * sentinel_height_m 

print(vel_aircraft_hat)
print(h_aircraft_hat)
calculate_aircraft_heading(channel_intensities=None, indeces=None)
