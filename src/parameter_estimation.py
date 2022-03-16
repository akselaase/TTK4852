from dataclasses import dataclass
import numpy as np

@dataclass
class SentinelData:
  channel_dt_s_arr =  np.array([0.507, 0.498])          # Time difference between channels 2-3 and 3-4
  channel_resolution_m_arr = 10 * np.ones(3, dtype=int) # Channel 2, 3 and 4 has an accuracy of 10 m

  # Data assumed about the satellite - haven't found better parameters
  sentinel_lat_deg          = 50.042677   # [deg]
  sentinel_inclination_deg  = 98.6        # [deg]
  sentinel_height_m         = 786000      # [m]
  sentinel_abs_vel          = 7460        # [m/s]

  __sentinel_angle_vel_x = np.sin(
    np.deg2rad(
      90 - sentinel_inclination_deg
    )
  ) 
  __sentinel_angle_vel_y = np.cos(
    np.deg2rad(
      90 - sentinel_inclination_deg
    )
  ) 

  sentinel_vel_vec = sentinel_abs_vel * np.array(
    [
      [__sentinel_angle_vel_x], 
      [__sentinel_angle_vel_y]
    ]
  )

class EstimateAircraftParameters:
  def __init__(self) -> None:
    self.__sentinel_data = SentinelData()
    
    self.__estimated_velocity = None
    self.__estimated_height = None
    self.__estimated_heading = None

    self.__num_channels = 0

    self.__min_channel_reflectance = 1e-4
    self.__invalid_heading = -1.0

  def display_estimates(self):
    print("Estimated velocity: {} [m/s]".format(self.__estimated_velocity))
    print("Estimated height: {} [m]".format(self.__estimated_height))
    print("Estimated heading: {} [deg]".format(self.__estimated_heading * 180 / np.pi))

  def estimate_parameters(self, cropped_image) -> tuple:
    """
    Tries to estimate an aircraft's velocity, height and heading

    Input: Cropped image containing an aircraft

    Output: Tuple containing (velocity, height, heading) 
    """
    intensities, indeces, coordinates = self.__extract_parameters(cropped_image=cropped_image) 

    if intensities is None or indeces is None:
      # Not enough information
      return self.__invalid()

    # Assumption that the altitude of the aircraft is low enough, such that the resolution
    # of each channel matches the ground. This is not really the case when the aircraft has
    # an altitude greater than 0 
    num_measurements = coordinates.shape[1]
    assert num_measurements >= 2, "Need at least two measurements to calculate the velocity"

    # Calculating the change in distance between the channels
    delta_pos = np.zeros((2, num_measurements - 1))
    rel_vel_hat_arr = np.zeros((2, num_measurements - 1))

    for idx in range(num_measurements - 1):
      current_measurement = coordinates[:, idx]
      next_measurement = coordinates[:, idx + 1]
      
      d_pos = next_measurement - current_measurement

      # Currently, only one resolution on the camera is used to calculate the real position
      # Unsure how to mix multiple resolutions, when converting into a change in meter

      # Converting into actual displacement in m
      resolution = self.__sentinel_data.channel_resolution_m_arr[0]
      delta_pos[:, idx] = d_pos * resolution

      # Calculating the velocities
      dt = self.__sentinel_data.channel_dt_s_arr[idx]
      rel_vel_hat_arr[:, idx] = delta_pos[:, idx] / dt

    # TODO: Implement a least-square method to calculate the velocities
    rel_vel_hat = np.average(a=rel_vel_hat_arr, axis=1)

    # Calculate the different angles 
    # phi:    Relative course-angle
    # theta:  Aircraft's course/heading TODO: Determine this somehow
    # psi:    Satelitte's course  
    phi = np.arctan2(rel_vel_hat[0], rel_vel_hat[1])
    theta = self.__calculate_aircraft_heading(channel_intensities=intensities, indeces=indeces)
    psi = np.cos(
      np.deg2rad(self.__sentinel_data.sentinel_inclination_deg) / np.deg2rad(self.__sentinel_data.sentinel_lat_deg)
    )

    # Validating if incorrect angle
    if theta == self.__invalid_heading:
      return self.__invalid()

    # Calculate the velocity and altitude of the aircraft
    vel_aircraft_hat = np.sin(phi - psi) / np.sin(theta - psi) * np.linalg.norm(rel_vel_hat)
    h_aircraft_hat = np.sin(phi - theta) / np.sin(theta - psi) * np.linalg.norm(rel_vel_hat) / self.__sentinel_data.sentinel_abs_vel * self.__sentinel_data.sentinel_height_m 

    self.__estimated_velocity = vel_aircraft_hat
    self.__estimated_height = h_aircraft_hat
    self.__estimated_heading = theta

    return self.__estimated_velocity, self.__estimated_height, self.__estimated_heading

  
  def __extract_parameters(self, cropped_image) -> tuple:
    """
    Tries to extract the intensities and indices for the different
    channel-intensities  
    """
    return None, None, None

  def __calculate_aircraft_heading(
      self,
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
    reflectance_arr = np.zeros(self.__num_channels)
    # Iterate over all measured intensities in the channel_intensities, 
    # and calculate channel reflectance
    # return -1.0 # Until the code is developed properly

    num_channels = self.__num_channels

    for ch in range(num_channels):
      # Iterate over all indeces assumed relevant
      channel_resolution_m = self.__sentinel_data.channel_resolution_m_arr[ch]
      for idx in indeces:
        # Sum up the intensities at the channels 
        pass
      
    # Calculate weighted center, covariances and estimated heading for each channel
    weighted_coordinates = np.zeros((2, num_channels))
    covariances = np.zeros((3, num_channels)) # sigma_xx, sigma_xy, sigma_yy stacked vertically for each channel
    estimated_headings = np.zeros(num_channels)
    for ch in range(num_channels):
      channel_resolution_m = self.__sentinel_data.channel_resolution_m_arr[ch]
      channel_reflectance = reflectance_arr[ch]

      if channel_reflectance <= self.__min_channel_reflectance:
        return self.__invalid_heading

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
        return self.__invalid_heading

      heading_hat = 0.5 * np.arctan((2 * sigma_xy**2) / (sigma_xx**2 - sigma_yy**2))
      estimated_headings[ch] = heading_hat

    # Return averaged headings
    return np.mean(estimated_headings) % (np.pi / 180)

  def __invalid(self) -> tuple:
    """
    Sets and returns values indicating that an error has occured
    """

    self.__estimated_velocity = None
    self.__estimated_heading = None 
    self.__estimated_height = None 

    return self.__estimated_velocity, self.__estimated_height, self.__estimated_heading

if __name__ == '__main__':
  est_aircraft_params = EstimateAircraftParameters()
  est_aircraft_params.estimate_parameters(cropped_image=None)
  est_aircraft_params.display_estimates()