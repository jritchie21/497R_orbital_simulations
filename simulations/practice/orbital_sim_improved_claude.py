import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
import time
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import warnings

# Optional imports for validation
try:
    from sgp4.api import Satrec, jday
    from sgp4.api import SGP4_ERRORS
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    warnings.warn("sgp4 not installed. Install with 'pip install sgp4' for TLE support and validation")

try:
    from skyfield.api import load, EarthSatellite
    from skyfield.api import utc
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False
    warnings.warn("skyfield not installed. Install with 'pip install skyfield' for additional validation")


@dataclass
class SatelliteState:
    """Complete state of a satellite"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    time: float = 0.0  # seconds since epoch


@dataclass
class SatelliteConfig:
    """Configuration for a satellite"""
    name: str
    color: str = 'red'
    size: float = 300.0  # km
    trail_length: int = 200
    show_velocity: bool = True
    show_attitude: bool = True
    orbital_elements: Dict = field(default_factory=dict)


class CoordinateFrames:
    """Handle coordinate frame transformations"""
    
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """Rotation matrix around X axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """Rotation matrix around Z axis"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    @staticmethod
    def eci_to_ecef(position_eci: np.ndarray, time: float, omega_earth: float = 7.2921159e-5) -> np.ndarray:
        """Convert from Earth-Centered Inertial to Earth-Centered Earth-Fixed frame"""
        theta = omega_earth * time  # Earth rotation angle
        R = CoordinateFrames.rotation_matrix_z(-theta)
        return R @ position_eci
    
    @staticmethod
    def ecef_to_eci(position_ecef: np.ndarray, time: float, omega_earth: float = 7.2921159e-5) -> np.ndarray:
        """Convert from Earth-Centered Earth-Fixed to Earth-Centered Inertial frame"""
        theta = omega_earth * time
        R = CoordinateFrames.rotation_matrix_z(theta)
        return R @ position_ecef
    
    @staticmethod
    def eci_to_lvlh(position_eci: np.ndarray, velocity_eci: np.ndarray) -> np.ndarray:
        """Get transformation matrix from ECI to Local-Vertical-Local-Horizontal (LVLH) frame"""
        # Z axis points toward Earth center (negative position)
        z_lvlh = -position_eci / np.linalg.norm(position_eci)
        
        # Y axis is negative orbit normal
        h = np.cross(position_eci, velocity_eci)
        y_lvlh = -h / np.linalg.norm(h)
        
        # X axis completes the right-handed system
        x_lvlh = np.cross(y_lvlh, z_lvlh)
        
        # Transformation matrix
        return np.array([x_lvlh, y_lvlh, z_lvlh])
    
    @staticmethod
    def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
        """Convert geodetic coordinates to ECEF"""
        # WGS84 parameters
        a = 6378137.0  # Semi-major axis (m)
        f = 1/298.257223563  # Flattening
        e2 = 2*f - f**2  # First eccentricity squared
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)
        
        return np.array([x, y, z])


class AttitudeDynamics:
    """Handle satellite attitude dynamics and control"""
    
    def __init__(self, inertia_tensor: Optional[np.ndarray] = None):
        """Initialize attitude dynamics
        
        Args:
            inertia_tensor: 3x3 inertia tensor (kg*m^2)
        """
        if inertia_tensor is None:
            # Default: small satellite with principal axes aligned
            self.inertia = np.diag([100, 100, 50])  # kg*m^2
        else:
            self.inertia = inertia_tensor
        self.inertia_inv = np.linalg.inv(self.inertia)
    
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quaternion_derivative(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Calculate quaternion time derivative given angular velocity"""
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        return q_dot
    
    def euler_equations(self, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """Calculate angular acceleration using Euler's equations"""
        # Euler's equations: I*omega_dot = torque - omega x (I*omega)
        L = self.inertia @ omega  # Angular momentum
        omega_dot = self.inertia_inv @ (torque - np.cross(omega, L))
        return omega_dot
    
    def gravity_gradient_torque(self, q: np.ndarray, position: np.ndarray, mu: float) -> np.ndarray:
        """Calculate gravity gradient torque"""
        r = np.linalg.norm(position)
        r_hat = position / r
        
        # Transform to body frame
        R = self.quaternion_to_rotation_matrix(q)
        r_body = R.T @ r_hat
        
        # Gravity gradient torque
        torque = (3 * mu / r**3) * np.cross(r_body, self.inertia @ r_body)
        return torque
    
    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        q = q / np.linalg.norm(q)  # Normalize
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
    
    def propagate_attitude(self, state: SatelliteState, torque: np.ndarray, dt: float) -> SatelliteState:
        """Propagate attitude dynamics one time step"""
        # Calculate angular acceleration
        omega_dot = self.euler_equations(state.angular_velocity, torque)
        
        # Update angular velocity
        new_omega = state.angular_velocity + omega_dot * dt
        
        # Update quaternion
        q_dot = self.quaternion_derivative(state.quaternion, state.angular_velocity)
        new_q = state.quaternion + q_dot * dt
        new_q = new_q / np.linalg.norm(new_q)  # Normalize
        
        state.quaternion = new_q
        state.angular_velocity = new_omega
        return state


class TLEHandler:
    """Handle Two-Line Element set operations"""
    
    @staticmethod
    def keplerian_to_tle(name: str, orbital_elements: Dict, 
                         epoch: datetime = None, 
                         bstar: float = 0.0,
                         sat_num: int = 99999) -> str:
        """Convert Keplerian elements to TLE format
        
        Args:
            name: Satellite name
            orbital_elements: Dictionary with Keplerian elements
            epoch: Epoch time (default: now)
            bstar: B* drag term
            sat_num: Satellite catalog number
        """
        if epoch is None:
            epoch = datetime.utcnow()
        
        # Extract elements
        a = orbital_elements['a'] / 1000  # Convert to km
        e = orbital_elements['e']
        i = orbital_elements['i']
        omega = orbital_elements['omega']
        w = orbital_elements['w']
        M = orbital_elements.get('M', 0)  # Mean anomaly
        
        # Calculate mean motion (revolutions per day)
        mu_earth = 398600.4418  # km^3/s^2
        n = 86400 / (2 * np.pi * np.sqrt(a**3 / mu_earth))  # rev/day
        
        # Format epoch
        year = epoch.year % 100
        day_of_year = epoch.timetuple().tm_yday
        fraction_of_day = (epoch.hour * 3600 + epoch.minute * 60 + epoch.second) / 86400
        epoch_str = f"{year:02d}{day_of_year:03d}.{int(fraction_of_day * 100000000):08d}"
        
        # Create TLE lines
        line0 = f"{name[:24]:24s}"
        line1 = f"1 {sat_num:05d}U {epoch.year:04d}001A   {epoch_str}  .00000000  00000-0 {bstar:8.4e} 0  9999"
        line2 = f"2 {sat_num:05d} {i:8.4f} {omega:8.4f} {e:.7f} {w:8.4f} {M:8.4f} {n:11.8f}    10"
        
        # Calculate checksums
        line1 = TLEHandler._add_checksum(line1)
        line2 = TLEHandler._add_checksum(line2)
        
        return f"{line0}\n{line1}\n{line2}"
    
    @staticmethod
    def _add_checksum(line: str) -> str:
        """Add checksum to TLE line"""
        checksum = 0
        for char in line[:-1]:
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        return line[:-1] + str(checksum % 10)
    
    @staticmethod
    def tle_to_keplerian(tle_lines: List[str]) -> Dict:
        """Parse TLE and extract Keplerian elements"""
        if len(tle_lines) < 2:
            raise ValueError("Need at least 2 TLE lines")
        
        line1 = tle_lines[-2] if len(tle_lines) == 3 else tle_lines[0]
        line2 = tle_lines[-1]
        
        # Parse line 2 (orbital elements)
        i = float(line2[8:16])  # Inclination (degrees)
        omega = float(line2[17:25])  # RAAN (degrees)
        e = float("0." + line2[26:33])  # Eccentricity
        w = float(line2[34:42])  # Argument of perigee (degrees)
        M = float(line2[43:51])  # Mean anomaly (degrees)
        n = float(line2[52:63])  # Mean motion (rev/day)
        
        # Convert mean motion to semi-major axis
        mu_earth = 398600.4418  # km^3/s^2
        a = (mu_earth / (n * 2 * np.pi / 86400)**2)**(1/3) * 1000  # Convert to meters
        
        return {
            'a': a,
            'e': e,
            'i': i,
            'omega': omega,
            'w': w,
            'M': M,
            'n': n
        }


class ValidationTools:
    """Tools for validating orbital calculations"""
    
    def __init__(self):
        self.sgp4_available = SGP4_AVAILABLE
        self.skyfield_available = SKYFIELD_AVAILABLE
    
    def validate_with_sgp4(self, tle_lines: List[str], epoch: datetime, 
                           minutes_after: float = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Validate position/velocity using SGP4
        
        Returns:
            Tuple of (position, velocity) in TEME frame, or None if not available
        """
        if not self.sgp4_available:
            print("SGP4 not available for validation")
            return None
        
        # Parse TLE
        if len(tle_lines) == 3:
            line1, line2 = tle_lines[1], tle_lines[2]
        else:
            line1, line2 = tle_lines[0], tle_lines[1]
        
        # Create satellite object
        satellite = Satrec.twoline2rv(line1, line2)
        
        # Calculate Julian date
        jd, fr = jday(epoch.year, epoch.month, epoch.day,
                     epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6)
        
        # Propagate
        error, position, velocity = satellite.sgp4(jd, fr + minutes_after/1440)
        
        if error == 0:
            # Convert from km to m
            return np.array(position) * 1000, np.array(velocity) * 1000
        else:
            print(f"SGP4 error: {SGP4_ERRORS[error]}")
            return None
    
    def validate_with_skyfield(self, tle_lines: List[str], epoch: datetime) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Validate using Skyfield library
        
        Returns:
            Tuple of (position, velocity) in GCRS frame, or None if not available
        """
        if not self.skyfield_available:
            print("Skyfield not available for validation")
            return None
        
        ts = load.timescale()
        
        # Create satellite from TLE
        if len(tle_lines) == 3:
            satellite = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
        else:
            satellite = EarthSatellite(tle_lines[0], tle_lines[1], "SATELLITE", ts)
        
        # Get position at epoch
        t = ts.utc(epoch.year, epoch.month, epoch.day, 
                  epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6)
        
        geocentric = satellite.at(t)
        position = geocentric.position.m  # meters
        velocity = geocentric.velocity.m_per_s  # m/s
        
        return position, velocity
    
    def compare_results(self, pos1: np.ndarray, vel1: np.ndarray,
                       pos2: np.ndarray, vel2: np.ndarray,
                       label1: str = "Method 1", label2: str = "Method 2") -> Dict:
        """Compare two sets of position/velocity vectors"""
        pos_diff = np.linalg.norm(pos1 - pos2)
        vel_diff = np.linalg.norm(vel1 - vel2)
        
        pos_error_pct = (pos_diff / np.linalg.norm(pos1)) * 100
        vel_error_pct = (vel_diff / np.linalg.norm(vel1)) * 100
        
        results = {
            'position_difference_m': pos_diff,
            'velocity_difference_m_s': vel_diff,
            'position_error_percent': pos_error_pct,
            'velocity_error_percent': vel_error_pct,
            'position_1': pos1,
            'velocity_1': vel1,
            'position_2': pos2,
            'velocity_2': vel2
        }
        
        print(f"\nValidation Comparison: {label1} vs {label2}")
        print(f"Position difference: {pos_diff:.2f} m ({pos_error_pct:.4f}%)")
        print(f"Velocity difference: {vel_diff:.4f} m/s ({vel_error_pct:.4f}%)")
        
        return results


class EnhancedOrbitSimulation:
    def __init__(self, test_mode: bool = False):
        # Earth constants
        self.mu_earth = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        self.earth_radius = 6.371e6  # Earth radius (m)
        self.omega_earth = 7.2921159e-5  # Earth rotation rate (rad/s)
        
        # Simulation parameters
        self.test_mode = test_mode
        self.animation_speed = 100  # Speed multiplier for animation
        self.current_time = 0.0
        self.epoch = datetime.utcnow()
        
        # Multiple satellites
        self.satellites: Dict[str, Dict[str, Any]] = {}
        
        # Coordinate frames
        self.coord_frames = CoordinateFrames()
        
        # Validation tools
        self.validator = ValidationTools()
        
        # Visualization
        self.fig = None
        self.ax = None
        self.gui_controls = {}
        self.animation = None
        self.is_paused = False
        self.show_earth = True
        self.show_grid = True
        self.frame_type = 'ECI'  # ECI, ECEF, or LVLH
        self.locked_axis_limits = None
        
        # Plot objects
        self.earth_surface = None
        self.earth_wireframe = None
        
    def add_satellite(self, config: SatelliteConfig) -> None:
        """Add a satellite to the simulation"""
        # Initialize satellite data
        self.satellites[config.name] = {
            'config': config,
            'state': SatelliteState(),
            'orbital_elements': config.orbital_elements,
            'trail_points': [],
            'attitude_dynamics': AttitudeDynamics(),
            'visual_elements': {
                'body': None,
                'trail': None,
                'velocity_vector': None,
                'attitude_axes': None,
                'full_orbit': None
            }
        }
        
        # Set initial state
        self._update_satellite_state(config.name, 0)
        
        # Precompute full orbit points for plotting and axis sizing
        try:
            self.satellites[config.name]['full_orbit_points'] = self._generate_full_orbit_points(
                config.orbital_elements
            )
        except Exception:
            self.satellites[config.name]['full_orbit_points'] = None
        
        # Recompute and lock axes after adding a satellite
        self._recompute_and_lock_axes()
        
        print(f"Added satellite: {config.name}")
    
    def remove_satellite(self, name: str) -> None:
        """Remove a satellite from the simulation"""
        if name in self.satellites:
            # Remove visual elements
            sat_data = self.satellites[name]
            for element in sat_data['visual_elements'].values():
                if element is not None:
                    try:
                        element.remove()
                    except:
                        pass
            
            del self.satellites[name]
            print(f"Removed satellite: {name}")
            # Recompute and lock axes after removal
            self._recompute_and_lock_axes()
    
    def save_configuration(self, filename: str) -> None:
        """Save current simulation configuration to JSON file"""
        config = {
            'epoch': self.epoch.isoformat(),
            'satellites': {}
        }
        
        for name, sat_data in self.satellites.items():
            config['satellites'][name] = {
                'name': sat_data['config'].name,
                'color': sat_data['config'].color,
                'size': sat_data['config'].size,
                'trail_length': sat_data['config'].trail_length,
                'orbital_elements': sat_data['orbital_elements']
            }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {filename}")
    
    def load_configuration(self, filename: str) -> None:
        """Load simulation configuration from JSON file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        # Set epoch
        self.epoch = datetime.fromisoformat(config['epoch'])
        
        # Clear existing satellites
        for name in list(self.satellites.keys()):
            self.remove_satellite(name)
        
        # Add satellites from config
        for sat_config in config['satellites'].values():
            self.add_satellite(SatelliteConfig(
                name=sat_config['name'],
                color=sat_config['color'],
                size=sat_config['size'],
                trail_length=sat_config['trail_length'],
                orbital_elements=sat_config['orbital_elements']
            ))
        
        print(f"Configuration loaded from {filename}")
    
    def export_to_tle(self, satellite_name: str, filename: str) -> None:
        """Export satellite orbital elements to TLE format"""
        if satellite_name not in self.satellites:
            raise ValueError(f"Satellite {satellite_name} not found")
        
        sat_data = self.satellites[satellite_name]
        tle = TLEHandler.keplerian_to_tle(
            satellite_name,
            sat_data['orbital_elements'],
            self.epoch
        )
        
        with open(filename, 'w') as f:
            f.write(tle)
        
        print(f"TLE exported to {filename}")
        return tle
    
    def import_from_tle(self, tle_lines: List[str], name: Optional[str] = None, 
                       color: str = 'blue') -> None:
        """Import satellite from TLE"""
        # Parse TLE
        elements = TLEHandler.tle_to_keplerian(tle_lines)
        
        # Get name
        if name is None:
            if len(tle_lines) == 3:
                name = tle_lines[0].strip()
            else:
                name = f"SAT_{len(self.satellites)+1}"
        
        # Create satellite config
        config = SatelliteConfig(
            name=name,
            color=color,
            orbital_elements=elements
        )
        
        self.add_satellite(config)
        print(f"Imported satellite {name} from TLE")
    
    def keplerian_to_cartesian(self, elements: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Keplerian elements to Cartesian state vectors"""
        a = elements['a']
        e = elements['e']
        i = np.radians(elements['i'])
        omega = np.radians(elements['omega'])
        w = np.radians(elements['w'])
        
        # Handle both true anomaly and mean anomaly
        if 'nu' in elements:
            nu = np.radians(elements['nu'])
        elif 'M' in elements:
            M = np.radians(elements['M'])
            # Solve Kepler's equation for eccentric anomaly
            E = self._solve_kepler_equation(M, e)
            # Convert to true anomaly
            nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        else:
            nu = 0
        
        # Calculate position and velocity in perifocal frame
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(nu))
        
        r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        
        h = np.sqrt(self.mu_earth * p)
        v_pf = np.array([
            -(self.mu_earth / h) * np.sin(nu),
            (self.mu_earth / h) * (e + np.cos(nu)),
            0
        ])
        
        # Transformation matrix to inertial frame
        cos_o, sin_o = np.cos(omega), np.sin(omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_w, sin_w = np.cos(w), np.sin(w)
        
        R = np.array([
            [cos_o*cos_w - sin_o*sin_w*cos_i, -cos_o*sin_w - sin_o*cos_w*cos_i, sin_o*sin_i],
            [sin_o*cos_w + cos_o*sin_w*cos_i, -sin_o*sin_w + cos_o*cos_w*cos_i, -cos_o*sin_i],
            [sin_w*sin_i, cos_w*sin_i, cos_i]
        ])
        
        return R @ r_pf, R @ v_pf
    
    def _solve_kepler_equation(self, M: float, e: float, tol: float = 1e-10) -> float:
        """Solve Kepler's equation M = E - e*sin(E) for E"""
        E = M if e < 0.8 else np.pi
        
        for _ in range(50):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E_new = E - f / f_prime
            
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        
        return E
    
    def _update_satellite_state(self, name: str, time: float) -> None:
        """Update satellite position, velocity, and attitude"""
        sat_data = self.satellites[name]
        elements = sat_data['orbital_elements']
        
        # Calculate orbital period
        a = elements['a']
        period = 2 * np.pi * np.sqrt(a**3 / self.mu_earth)
        
        # Update mean anomaly based on time
        n = 2 * np.pi / period  # Mean motion
        M0 = np.radians(elements.get('M', 0))
        M = M0 + n * time
        
        # Update elements with new mean anomaly
        current_elements = elements.copy()
        current_elements['M'] = np.degrees(M % (2 * np.pi))
        
        # Get Cartesian state
        position, velocity = self.keplerian_to_cartesian(current_elements)
        
        # Update state
        sat_data['state'].position = position
        sat_data['state'].velocity = velocity
        sat_data['state'].time = time
        
        # Update attitude dynamics
        if sat_data['config'].show_attitude:
            # Calculate torques (gravity gradient + control)
            gg_torque = sat_data['attitude_dynamics'].gravity_gradient_torque(
                sat_data['state'].quaternion,
                position,
                self.mu_earth
            )
            
            # Simple nadir-pointing control torque
            control_torque = np.array([0.001, 0.0005, 0.0002])  # Small control torque
            
            total_torque = gg_torque + control_torque
            
            # Propagate attitude
            dt = 1.0  # Time step for attitude propagation
            sat_data['state'] = sat_data['attitude_dynamics'].propagate_attitude(
                sat_data['state'],
                total_torque,
                dt
            )
        
        # Update trail
        sat_data['trail_points'].append(position.copy())
        if len(sat_data['trail_points']) > sat_data['config'].trail_length:
            sat_data['trail_points'].pop(0)
    
    def setup_plot(self) -> None:
        """Initialize the 3D plot with GUI controls"""
        if self.test_mode:
            return
        
        # Create figure with gridspec for layout
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(5, 3, width_ratios=[1, 3, 1], height_ratios=[1, 10, 1, 1, 1])
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax.set_xlabel('X (km)', fontsize=10)
        self.ax.set_ylabel('Y (km)', fontsize=10)
        self.ax.set_zlabel('Z (km)', fontsize=10)
        self.ax.set_title('Multi-Satellite Orbit Simulation', fontsize=14, fontweight='bold')
        self.ax.set_box_aspect([1, 1, 1])
        
        # Control panel
        self._setup_gui_controls(gs)
        
        plt.tight_layout()
    
    def _setup_gui_controls(self, gs) -> None:
        """Setup GUI control panels"""
        # Speed control slider
        ax_speed = self.fig.add_subplot(gs[2, 1])
        self.gui_controls['speed_slider'] = Slider(
            ax_speed, 'Speed', 0.1, 1000, valinit=self.animation_speed, valstep=10, color='lightblue'
        )
        self.gui_controls['speed_slider'].on_changed(self._on_speed_change)
        
        # Time display
        ax_time = self.fig.add_subplot(gs[3, 1])
        ax_time.axis('off')
        self.gui_controls['time_text'] = ax_time.text(
            0.5, 0.5, '', transform=ax_time.transAxes,
            ha='center', va='center', fontsize=11
        )
        
        # Control buttons
        ax_pause = self.fig.add_subplot(gs[4, 0])
        self.gui_controls['pause_button'] = Button(ax_pause, 'Pause/Resume', color='lightgreen')
        self.gui_controls['pause_button'].on_clicked(self._on_pause_click)
        
        ax_reset = self.fig.add_subplot(gs[4, 1])
        self.gui_controls['reset_button'] = Button(ax_reset, 'Reset', color='lightcoral')
        self.gui_controls['reset_button'].on_clicked(self._on_reset_click)
        
        ax_frame = self.fig.add_subplot(gs[4, 2])
        self.gui_controls['frame_button'] = Button(ax_frame, f'Frame: {self.frame_type}', color='lightyellow')
        self.gui_controls['frame_button'].on_clicked(self._on_frame_click)
        
        # Checkboxes for display options
        ax_checks = self.fig.add_subplot(gs[0:2, 0])
        ax_checks.axis('off')
        self.gui_controls['check_boxes'] = CheckButtons(
            ax_checks,
            ['Earth', 'Grid', 'Trails', 'Velocity', 'Attitude'],
            [True, True, True, True, False]
        )
        self.gui_controls['check_boxes'].on_clicked(self._on_check_click)
        
        # Satellite selector
        ax_sats = self.fig.add_subplot(gs[0:2, 2])
        ax_sats.axis('off')
        ax_sats.text(0.5, 0.95, 'Satellites:', transform=ax_sats.transAxes,
                    ha='center', fontsize=11, fontweight='bold')
        self.gui_controls['sat_list'] = ax_sats
    
    def _on_speed_change(self, val) -> None:
        """Handle speed slider change"""
        self.animation_speed = val
    
    def _on_pause_click(self, event) -> None:
        """Handle pause button click"""
        self.is_paused = not self.is_paused
        if self.animation:
            if self.is_paused:
                self.animation.pause()
            else:
                self.animation.resume()
    
    def _on_reset_click(self, event) -> None:
        """Handle reset button click"""
        self.current_time = 0
        for sat_data in self.satellites.values():
            sat_data['trail_points'] = []
            sat_data['state'].quaternion = np.array([1, 0, 0, 0])
            sat_data['state'].angular_velocity = np.zeros(3)
    
    def _on_frame_click(self, event) -> None:
        """Handle coordinate frame button click"""
        frames = ['ECI', 'ECEF', 'LVLH']
        idx = frames.index(self.frame_type)
        self.frame_type = frames[(idx + 1) % len(frames)]
        self.gui_controls['frame_button'].label.set_text(f'Frame: {self.frame_type}')
    
    def _on_check_click(self, label) -> None:
        """Handle checkbox clicks"""
        if label == 'Earth':
            self.show_earth = not self.show_earth
            if self.earth_surface:
                self.earth_surface.set_visible(self.show_earth)
        elif label == 'Grid':
            self.show_grid = not self.show_grid
            self.ax.grid(self.show_grid)
    
    def plot_earth(self) -> None:
        """Draw Earth in the plot"""
        if self.test_mode or self.ax is None:
            return
        
        # Create Earth sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_earth = self.earth_radius/1000 * np.outer(np.cos(u), np.sin(v))
        y_earth = self.earth_radius/1000 * np.outer(np.sin(u), np.sin(v))
        z_earth = self.earth_radius/1000 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.earth_surface = self.ax.plot_surface(
            x_earth, y_earth, z_earth,
            alpha=0.3, color='lightblue',
            linewidth=0, antialiased=True
        )
        
        # Add coordinate axes
        axis_length = self.earth_radius/1000 * 1.5
        self.ax.plot([0, 0], [0, 0], [0, axis_length], 'r-', linewidth=2, alpha=0.7, label='Z-axis')
        self.ax.plot([0, axis_length], [0, 0], [0, 0], 'g-', linewidth=2, alpha=0.7, label='X-axis')
        self.ax.plot([0, 0], [0, axis_length], [0, 0], 'b-', linewidth=2, alpha=0.7, label='Y-axis')
        
        # Add equator and meridians
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Equator
        x_eq = self.earth_radius/1000 * np.cos(theta)
        y_eq = self.earth_radius/1000 * np.sin(theta)
        z_eq = np.zeros_like(theta)
        self.ax.plot(x_eq, y_eq, z_eq, 'k-', linewidth=0.5, alpha=0.3)
        
        # Prime meridian
        phi = np.linspace(-np.pi/2, np.pi/2, 50)
        x_mer = self.earth_radius/1000 * np.cos(phi)
        y_mer = np.zeros_like(phi)
        z_mer = self.earth_radius/1000 * np.sin(phi)
        self.ax.plot(x_mer, y_mer, z_mer, 'k-', linewidth=0.5, alpha=0.3)
    
    def draw_satellite(self, name: str) -> None:
        """Draw a satellite and its associated elements"""
        if self.test_mode or self.ax is None or name not in self.satellites:
            return
        
        sat_data = self.satellites[name]
        config = sat_data['config']
        state = sat_data['state']
        visuals = sat_data['visual_elements']
        
        # Clear previous visual elements except the full-orbit curve (it's static)
        for key, element in visuals.items():
            if element is not None:
                try:
                    if key != 'full_orbit':
                        element.remove()
                except:
                    pass
        
        # Get position based on coordinate frame
        if self.frame_type == 'ECI':
            pos = state.position
        elif self.frame_type == 'ECEF':
            pos = self.coord_frames.eci_to_ecef(state.position, self.current_time)
        else:  # LVLH (show relative to first satellite)
            if len(self.satellites) > 1 and name != list(self.satellites.keys())[0]:
                ref_sat = list(self.satellites.values())[0]
                pos = state.position - ref_sat['state'].position
            else:
                pos = state.position
        
        pos_km = pos / 1000
        
        # Draw full orbit (ECI frame only)
        full_orbit_pts = self.satellites[name].get('full_orbit_points')
        if full_orbit_pts is not None:
            if self.frame_type == 'ECI':
                if visuals.get('full_orbit') is None or visuals['full_orbit'] not in self.ax.lines:
                    orbit_km = full_orbit_pts / 1000
                    visuals['full_orbit'] = self.ax.plot(
                        orbit_km[:, 0], orbit_km[:, 1], orbit_km[:, 2],
                        color=config.color, linestyle=':', linewidth=1.0, alpha=0.6
                    )[0]
                else:
                    visuals['full_orbit'].set_visible(True)
            else:
                if visuals.get('full_orbit') is not None:
                    try:
                        visuals['full_orbit'].set_visible(False)
                    except:
                        pass
        
        # Draw satellite body
        visuals['body'] = self.ax.scatter(
            pos_km[0], pos_km[1], pos_km[2],
            s=config.size, c=config.color, marker='o', alpha=0.9
        )
        
        # Draw velocity vector
        if config.show_velocity and self.gui_controls['check_boxes'].get_status()[3]:
            vel_scaled = state.velocity / 1000 * 0.3
            visuals['velocity_vector'] = self.ax.quiver(
                pos_km[0], pos_km[1], pos_km[2],
                vel_scaled[0], vel_scaled[1], vel_scaled[2],
                color='orange', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7
            )
        
        # Draw attitude axes
        if config.show_attitude and self.gui_controls['check_boxes'].get_status()[4]:
            R = sat_data['attitude_dynamics'].quaternion_to_rotation_matrix(state.quaternion)
            axis_length = 500  # km
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                axis_dir = R[:, i] * axis_length
                self.ax.quiver(
                    pos_km[0], pos_km[1], pos_km[2],
                    axis_dir[0], axis_dir[1], axis_dir[2],
                    color=color, arrow_length_ratio=0.1, linewidth=1, alpha=0.6
                )
        
        # Draw trail
        if len(sat_data['trail_points']) > 1 and self.gui_controls['check_boxes'].get_status()[2]:
            trail_array = np.array(sat_data['trail_points'])
            
            # Transform trail based on frame
            if self.frame_type == 'ECEF':
                trail_transformed = []
                for i, point in enumerate(trail_array):
                    t = self.current_time - (len(trail_array) - i - 1)
                    trail_transformed.append(self.coord_frames.eci_to_ecef(point, t))
                trail_array = np.array(trail_transformed)
            elif self.frame_type == 'LVLH' and len(self.satellites) > 1:
                if name != list(self.satellites.keys())[0]:
                    ref_trail = self.satellites[list(self.satellites.keys())[0]]['trail_points']
                    if len(ref_trail) >= len(trail_array):
                        trail_array = trail_array - np.array(ref_trail[-len(trail_array):])
            
            trail_km = trail_array / 1000
            visuals['trail'] = self.ax.plot(
                trail_km[:, 0], trail_km[:, 1], trail_km[:, 2],
                color=config.color, linewidth=1, alpha=0.5
            )[0]
    
    def update_satellite_list_display(self) -> None:
        """Update the satellite list in the GUI"""
        if 'sat_list' not in self.gui_controls:
            return
        
        ax = self.gui_controls['sat_list']
        ax.clear()
        ax.axis('off')
        ax.text(0.5, 0.95, 'Satellites:', transform=ax.transAxes,
               ha='center', fontsize=11, fontweight='bold')
        
        y_pos = 0.85
        for name, sat_data in self.satellites.items():
            color = sat_data['config'].color
            ax.text(0.1, y_pos, '●', transform=ax.transAxes,
                   color=color, fontsize=16)
            ax.text(0.3, y_pos, name[:12], transform=ax.transAxes,
                   fontsize=9, va='center')
            y_pos -= 0.1
    
    def animate_frame(self, frame: int) -> None:
        """Animation update function"""
        if self.is_paused:
            return
        
        # Update time
        dt = 10 * self.animation_speed / 100  # seconds
        self.current_time += dt
        
        # Update all satellites
        for name in self.satellites:
            self._update_satellite_state(name, self.current_time)
            self.draw_satellite(name)
        
        # Update time display
        elapsed_hours = self.current_time / 3600
        elapsed_days = elapsed_hours / 24
        current_date = self.epoch + timedelta(seconds=self.current_time)
        
        time_text = (f"Elapsed: {elapsed_hours:.2f} hours ({elapsed_days:.3f} days)\n"
                    f"Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                    f"Frame: {self.frame_type}")
        self.gui_controls['time_text'].set_text(time_text)
        
        # Update axis limits if needed
        self._update_axis_limits()
    
    def _update_axis_limits(self) -> None:
        """Update axis limits based on satellite positions"""
        if self.locked_axis_limits is None:
            return
        xlim, ylim, zlim = self.locked_axis_limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)

    def _generate_full_orbit_points(self, elements: Dict, num_points: int = 360) -> np.ndarray:
        """Generate ECI positions for a full orbit using true anomaly sampling."""
        nu_vals = np.linspace(0.0, 360.0, num_points, endpoint=False)
        positions = []
        base = elements.copy()
        if 'M' in base:
            base.pop('M')
        for nu in nu_vals:
            base['nu'] = nu
            pos, _ = self.keplerian_to_cartesian(base)
            positions.append(pos)
        return np.array(positions)

    def _recompute_and_lock_axes(self) -> None:
        """Recompute best symmetric axis limits from satellites' full orbits and lock them."""
        if self.ax is None:
            return
        max_extent_km = self.earth_radius / 1000 * 1.5
        for sat_data in self.satellites.values():
            pts = sat_data.get('full_orbit_points')
            if pts is not None and len(pts) > 0:
                ext = np.max(np.abs(pts)) / 1000.0
                if np.isnan(ext) or np.isinf(ext):
                    continue
                max_extent_km = max(max_extent_km, ext)
        max_range = max_extent_km * 1.3
        self.locked_axis_limits = ([-max_range, max_range], [-max_range, max_range], [-max_range, max_range])
        # Apply immediately if axes exist
        try:
            self.ax.set_xlim(self.locked_axis_limits[0])
            self.ax.set_ylim(self.locked_axis_limits[1])
            self.ax.set_zlim(self.locked_axis_limits[2])
        except Exception:
            pass
    
    def run_animation(self, duration_hours: float = 2.0) -> None:
        """Run the orbital animation"""
        if not self.satellites:
            raise ValueError("No satellites added. Use add_satellite() first.")
        
        # Setup plot
        self.setup_plot()
        self.plot_earth()
        self.update_satellite_list_display()
        
        # Calculate number of frames
        fps = 30  # frames per second
        interval = 1000 / fps  # milliseconds per frame
        total_frames = int(duration_hours * 3600 * fps / (self.animation_speed / 100))
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=total_frames,
            interval=interval, blit=False, repeat=True
        )
        
        plt.show()
    
    def validate_satellite(self, name: str, use_sgp4: bool = True, 
                          use_skyfield: bool = True) -> Dict:
        """Validate satellite propagation against standard libraries"""
        if name not in self.satellites:
            raise ValueError(f"Satellite {name} not found")
        
        sat_data = self.satellites[name]
        results = {'satellite': name, 'comparisons': []}
        
        # Get current state from our propagator
        our_pos = sat_data['state'].position
        our_vel = sat_data['state'].velocity
        
        print(f"\n{'='*60}")
        print(f"Validation for satellite: {name}")
        print(f"{'='*60}")
        print(f"Our propagator:")
        print(f"  Position: {our_pos/1000} km")
        print(f"  Velocity: {our_vel/1000} km/s")
        
        # Generate TLE for validation
        tle_str = self.export_to_tle(name, "temp_validation.tle")
        tle_lines = tle_str.strip().split('\n')
        
        # Validate with SGP4
        if use_sgp4 and self.validator.sgp4_available:
            sgp4_result = self.validator.validate_with_sgp4(
                tle_lines, self.epoch, self.current_time/60
            )
            if sgp4_result:
                sgp4_pos, sgp4_vel = sgp4_result
                print(f"\nSGP4 propagator:")
                print(f"  Position: {sgp4_pos/1000} km")
                print(f"  Velocity: {sgp4_vel/1000} km/s")
                
                comparison = self.validator.compare_results(
                    our_pos, our_vel, sgp4_pos, sgp4_vel,
                    "Our Propagator", "SGP4"
                )
                results['comparisons'].append(comparison)
        
        # Validate with Skyfield
        if use_skyfield and self.validator.skyfield_available:
            skyfield_result = self.validator.validate_with_skyfield(
                tle_lines, self.epoch + timedelta(seconds=self.current_time)
            )
            if skyfield_result:
                sky_pos, sky_vel = skyfield_result
                print(f"\nSkyfield propagator:")
                print(f"  Position: {sky_pos/1000} km")
                print(f"  Velocity: {sky_vel/1000} km/s")
                
                comparison = self.validator.compare_results(
                    our_pos, our_vel, sky_pos, sky_vel,
                    "Our Propagator", "Skyfield"
                )
                results['comparisons'].append(comparison)
        
        return results


def create_example_simulation():
    """Create an example simulation with multiple satellites"""
    print("Creating Enhanced Multi-Satellite Orbital Simulation")
    print("="*60)
    
    # Create simulation
    sim = EnhancedOrbitSimulation()
    
    # Add ISS-like satellite
    iss_config = SatelliteConfig(
        name="ISS",
        color='red',
        size=300,
        trail_length=200,
        orbital_elements={
            'a': 6.78e6,  # ~400 km altitude
            'e': 0.0001,
            'i': 51.6,
            'omega': 0,
            'w': 0,
            'M': 0
        }
    )
    sim.add_satellite(iss_config)
    
    # Add polar satellite
    polar_config = SatelliteConfig(
        name="POLAR-1",
        color='blue',
        size=250,
        trail_length=150,
        orbital_elements={
            'a': 7.2e6,  # ~800 km altitude
            'e': 0.001,
            'i': 90,  # Polar orbit
            'omega': 0,
            'w': 90,
            'M': 45
        }
    )
    sim.add_satellite(polar_config)
    
    # Add Molniya-type satellite
    molniya_config = SatelliteConfig(
        name="MOLNIYA-1",
        color='green',
        size=250,
        trail_length=300,
        show_attitude=True,
        orbital_elements={
            'a': 2.66e7,  # Molniya orbit
            'e': 0.74,
            'i': 63.4,
            'omega': 45,
            'w': 270,
            'M': 0
        }
    )
    sim.add_satellite(molniya_config)
    
    # Save configuration
    sim.save_configuration("multi_satellite_config.json")
    
    # Validate if libraries are available
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    for sat_name in sim.satellites:
        try:
            sim.validate_satellite(sat_name)
        except Exception as e:
            print(f"Validation failed for {sat_name}: {e}")
    
    # Run animation
    print("\n" + "="*60)
    print("Starting animation with GUI controls...")
    print("Controls:")
    print("  - Speed slider: Adjust animation speed")
    print("  - Pause/Resume: Pause or resume animation")
    print("  - Reset: Reset simulation to initial state")
    print("  - Frame button: Switch between ECI/ECEF/LVLH frames")
    print("  - Checkboxes: Toggle display elements")
    print("="*60)
    
    sim.run_animation(duration_hours=2.0)


if __name__ == "__main__":
    create_example_simulation()