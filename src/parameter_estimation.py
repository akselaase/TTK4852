from dataclasses import dataclass
from cv2 import CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR
import numpy as np
import warnings

from pathlib import Path
from process import ValidationResult

@dataclass
class SentinelData:
  channel_dt_s_arr          = np.array([0.507, 0.498])  # Time difference between channels 2-3 and 3-4
  channel_resolution_m_arr  = 10 * np.ones(3, dtype=int) # Channel 2, 3 and 4 has an accuracy of 10 m

  # Data assumed about the satellite - haven't found better parameters
  sentinel_lat_deg          = 50.042677   # [deg]
  sentinel_inclination_deg  = 98.6        # [deg]
  sentinel_height_m         = 786000      # [m]
  sentinel_abs_vel          = 7460        # [m/s]

  # Unsure if these are useful at all!
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
    if self.__estimated_heading is None:
      estimated_heading_deg = None
    else:
      estimated_heading_deg = self.__estimated_heading * 180 / np.pi

    print("Estimated velocity: {} [m/s]".format(self.__estimated_velocity))
    print("Estimated height: {} [m]".format(self.__estimated_height))
    print("Estimated velocity: {} [deg]".format(estimated_heading_deg))

  def estimate_parameters(
        self, 
        image       : np.ndarray,
        diffed      : np.ndarray,
        validation  : ValidationResult
      ) -> tuple:
    """
    Tries to estimate an aircraft's velocity, height and heading

    Input: Cropped image containing an aircraft

    Output: Tuple containing (velocity, height, heading) 
    """
    intensities, coordinates = self.__extract_parameters(
      image=image,
      diffed=diffed,
      validation=validation
    ) 

    if intensities is None:
      # Not enough information
      return self.__invalid()

    # Assumption that the altitude of the aircraft is low enough, such that the resolution
    # of each channel matches the ground. This is not really the case when the aircraft has
    # an altitude greater than 0 
    num_measurements = coordinates.shape[1]
    if num_measurements < 2:
      warnings.warn("Need at least two measurements to calculate the parameters")
      return self.__invalid()

    # Calculating the change in distance between the channels
    rel_vel_hat_arr = np.zeros((2, num_measurements - 1))

    for idx in range(num_measurements - 1):
      # Must iterate from the 'incorrect' order to counteract the effect from the 
      # parallax effect
      current_measurement = coordinates[:, -(coordinates.shape[1] - 1) + idx]
      next_measurement = coordinates[:, -(coordinates.shape[1] - 1) + idx + 1]

      # Currently, only one resolution on the camera is used to calculate the real position
      # Unsure how to mix multiple resolutions, when converting into a change in meter
      resolution = self.__sentinel_data.channel_resolution_m_arr[idx]
      d_pos = (next_measurement - current_measurement) * resolution

      # Calculating the velocities
      dt = self.__sentinel_data.channel_dt_s_arr[-idx]
      rel_vel_hat_arr[:, idx] = d_pos / dt

    # Might be better to use a least-square estimation instead of using the average
    rel_vel_hat = np.mean(a=rel_vel_hat_arr, axis=1)

    # Calculate the different angles 
    # phi:    Relative course-angle
    # theta:  Aircraft's heading
    # psi:    Satelitte's course  
    phi = np.arctan2(rel_vel_hat[0], rel_vel_hat[1])
    theta = self.__calculate_aircraft_heading(channel_intensities=intensities)
    psi = np.cos(
      np.deg2rad(self.__sentinel_data.sentinel_inclination_deg) / np.deg2rad(self.__sentinel_data.sentinel_lat_deg)
    )

    # Validating if incorrect angle
    if theta == self.__invalid_heading:
      warnings.warn("Invalid heading")
      return self.__invalid()

    if abs(np.sin(theta - psi)) < 1e-2:
      warnings.warn("Aircraft and satelitte almost parallell")
      return self.__invalid()

    # Calculate the velocity and altitude of the aircraft
    vel_aircraft_hat = np.sin(phi - psi) / np.sin(theta - psi) * np.linalg.norm(rel_vel_hat)
    h_aircraft_hat = \
        (np.sin(phi - theta) / np.sin(theta - psi)) \
      * (np.linalg.norm(rel_vel_hat) / self.__sentinel_data.sentinel_abs_vel) \
      * self.__sentinel_data.sentinel_height_m 

    self.__estimated_velocity = np.abs(vel_aircraft_hat)
    self.__estimated_height = np.abs(h_aircraft_hat)
    self.__estimated_heading = theta

    return self.__estimated_velocity, self.__estimated_height, self.__estimated_heading
  
  def __extract_parameters(
      self, 
      image       : np.ndarray,
      diffed      : np.ndarray,
      validation  : ValidationResult
    ) -> tuple:
    """
    Tries to extract the intensities for the different
    channel-intensities, as well as the channel-coordinates 
    in [row, col]^T
    
    images will be given as ndarray: [row, col, color]
    b, g, r = 0, 1, 2

    Unsure how to include the raw image, and if it is usable
    at all?
    """
    radius = validation.radius
    self.__num_channels = diffed.shape[2]

    # Set memory
    intensities = np.zeros((self.__num_channels, 2*radius + 1, 2*radius + 1))
    coordinates = np.zeros((2, self.__num_channels))

    # Extract data from the images
    green_center = validation.green_center
    blue_center = validation.blue_center
    red_center = validation.red_center

    # Images taken in order: bgr
    center_list = [blue_center, green_center, red_center]

    for (idx, center) in enumerate(center_list):
      row, col = center[0], center[1]
      coordinates[:, idx] = np.array([row, col]).T

      intensities[idx] = diffed[row - radius : row + radius + 1, col - radius : col + radius + 1, idx]

    return intensities, coordinates

  def __calculate_aircraft_heading(
      self,
      channel_intensities : np.ndarray
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
    num_channels = self.__num_channels
    
    # Calculate weighted center, covariances and estimated heading for each channel
    estimated_headings = np.zeros(num_channels)

    for ch in range(num_channels):
      # Iterate over all indeces assumed relevant
      channel_resolution_m = self.__sentinel_data.channel_resolution_m_arr[ch]
      channel_reflectance = np.sum(channel_intensities[ch].flatten(), axis=0)

      if channel_reflectance <= self.__min_channel_reflectance:
        warnings.warn("Invalid channel reflectance for channel {}".format(ch))
        return self.__invalid_heading

      # Weighted center
      x_bar = 0
      y_bar = 0
      (num_rows, num_cols) = channel_intensities.shape[1], channel_intensities.shape[2]
      for row in range(num_rows):
        for col in range(num_cols):
          x_bar += col * channel_intensities[ch, row, col] 
          y_bar += row * channel_intensities[ch, row, col] 

      # x_bar = np.mean(np.mean(channel_intensities[ch], axis=1)) # The difference between 0 and 1
      # y_bar = np.mean(np.mean(channel_intensities[ch], axis=0))

      x_bar = x_bar * channel_resolution_m / channel_reflectance
      y_bar = y_bar * channel_resolution_m / channel_reflectance

      # Covariances
      sigma_xx = 0
      sigma_xy = 0
      sigma_yy = 0
      for row in range(num_rows):
        for col in range(num_cols):
          sigma_xx += (col**2) * channel_intensities[ch, row, col] 
          sigma_xy += (col * row) * channel_intensities[ch, row, col] 
          sigma_yy += (row**2) * channel_intensities[ch, row, col] 

      # channel_intensity_cov = np.cov(channel_intensities[ch])
      
      # sigma_xx = channel_intensity_cov[0,0]
      # sigma_yy = channel_intensity_cov[1,1]
      # sigma_xy = channel_intensity_cov[0,1]

      sigma_xx = sigma_xx * ((channel_resolution_m**2) / channel_reflectance) - (x_bar**2)
      sigma_xy = sigma_xy * ((channel_resolution_m**2) / channel_reflectance) - (x_bar * y_bar)
      sigma_yy = sigma_yy * ((channel_resolution_m**2) / channel_reflectance) - (y_bar**2)

      # Estimated headings
      if sigma_xx**2 - sigma_yy**2 == 0:
        warnings.warn("Perhaps not enough information for channel {}. Measurements may be fucked!".format(ch))
      #   return self.__invalid_heading

      # heading_hat = 0.5 * np.arctan((2 * sigma_xy**2) / (sigma_xx**2 - sigma_yy**2))
      heading_hat = np.pi / 2 - 0.5 * np.arctan2((2 * sigma_xy**2), (sigma_xx**2 - sigma_yy**2)) # 90 deg - angle due to defining from north
      estimated_headings[ch] = heading_hat

    # print(estimated_headings * 180 / np.pi + 180)
    return np.mean(estimated_headings) % (2 * np.pi)

  def __invalid(self) -> tuple:
    """
    Sets and returns values indicating that an error has occured
    """

    self.__estimated_velocity = None
    self.__estimated_heading = None 
    self.__estimated_height = None 

    return self.__estimated_velocity, self.__estimated_height, self.__estimated_heading

  def save_parameters(
        self, 
        filename : str
      ) -> None:
    if self.__estimated_heading is None:
      estimated_heading = None
    else:
      estimated_heading = self.__estimated_heading * 180 / np.pi

    with open(filename, 'w') as file:
      file.write(
        "Parameters: \t Values: \n Velocity: \t {} [m/s] \n Height: \t {} [m] \n Heading: {} [deg]".format(
          self.__estimated_velocity,
          self.__estimated_height,
          estimated_heading
        )
      )

def do_parameter_est(
      image     : np.ndarray,
      diffed    : np.ndarray,
      validation: ValidationResult,
      filename  : str
    ) -> None:
  est_aircraft_params = EstimateAircraftParameters()
  est_aircraft_params.estimate_parameters(image=image, diffed=diffed, validation=validation)
  est_aircraft_params.save_parameters(filename=filename)

if __name__ == '__main__':
  est_aircraft_params = EstimateAircraftParameters()
  est_aircraft_params.estimate_parameters()
  est_aircraft_params.display_estimates()