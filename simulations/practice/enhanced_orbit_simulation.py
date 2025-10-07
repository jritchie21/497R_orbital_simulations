import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
import unittest
from basic_quat_practice import QuaternionArrowPlotter

class EnhancedOrbitSimulation:
    def __init__(self, test_mode=False):
        # Earth constants
        self.mu_earth = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        self.earth_radius = 6.371e6  # Earth radius (m)
        
        # Simulation parameters
        self.test_mode = test_mode
        self.satellite_quaternion = np.array([1, 0, 0, 0])  # Initial satellite orientation
        self.satellite_scale = 0.5  # Scale factor for satellite size (increased for better visibility)
        self.animation_speed = 1000  # Speed multiplier for animation
        
        # Plot objects
        self.fig = None
        self.ax = None
        self.orbit_line = None
        self.earth_surface = None
        self.satellite_plot = None
        self.velocity_quiver = None
        self.trail_points = []
        self.max_trail_length = 100
        
        # Animation state
        self.current_time = 0
        self.orbital_period = 0
        self.true_anomaly = 0
        self.satellite_position = np.array([0, 0, 0])
        self.satellite_velocity = np.array([0, 0, 0])
        
        # Orbital elements
        self.orbital_elements = None
        
    def keplerian_to_cartesian(self, a, e, i, omega, w, nu):
        """
        Convert Keplerian orbital elements to Cartesian coordinates
        
        Parameters:
        a: semi-major axis (m)
        e: eccentricity (0-1)
        i: inclination (degrees)
        omega: longitude of ascending node (degrees)
        w: argument of periapsis (degrees)
        nu: true anomaly (degrees)
        """
        # Convert angles to radians
        i = np.radians(i)
        omega = np.radians(omega)
        w = np.radians(w)
        nu = np.radians(nu)
        
        # Calculate orbital parameter
        p = a * (1 - e**2)
        
        # Calculate radius
        r = p / (1 + e * np.cos(nu))
        
        # Position in perifocal frame
        x_pf = r * np.cos(nu)
        y_pf = r * np.sin(nu)
        z_pf = 0
        
        # Velocity in perifocal frame
        h = np.sqrt(self.mu_earth * p)
        vx_pf = -(self.mu_earth / h) * np.sin(nu)
        vy_pf = (self.mu_earth / h) * (e + np.cos(nu))
        vz_pf = 0
        
        # Rotation matrices for Earth-centered inertial (ECI) frame
        R3_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1]
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        R3_w = np.array([
            [np.cos(w), -np.sin(w), 0],
            [np.sin(w), np.cos(w), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = R3_omega @ R1_i @ R3_w
        
        # Transform to inertial frame
        pos_pf = np.array([x_pf, y_pf, z_pf])
        vel_pf = np.array([vx_pf, vy_pf, vz_pf])
        
        pos_inertial = R @ pos_pf
        vel_inertial = R @ vel_pf
        
        return pos_inertial, vel_inertial
    
    def modified_equinoctial_to_keplerian(self, p, f, g, h, k, L):
        """
        Convert modified equinoctial elements to Keplerian elements
        
        Parameters:
        p: semi-latus rectum (m)
        f: e*cos(w+omega) 
        g: e*sin(w+omega)
        h: tan(i/2)*cos(omega)
        k: tan(i/2)*sin(omega)
        L: true longitude (degrees)
        """
        # Calculate eccentricity
        e = np.sqrt(f**2 + g**2)
        
        # Calculate semi-major axis
        a = p / (1 - e**2) if e < 1 else p / (e**2 - 1)
        
        # Calculate inclination
        i = 2 * np.arctan(np.sqrt(h**2 + k**2))
        
        # Calculate longitude of ascending node
        omega = np.arctan2(k, h)
        
        # Calculate argument of periapsis
        # Note: arctan2(g, f) gives us (ω + Ω), so we need to subtract Ω
        w_plus_omega = np.arctan2(g, f)
        w = w_plus_omega - omega
        
        # Calculate true anomaly: nu = L - omega - w
        # Use proper angle arithmetic to handle wrapping
        L_deg = np.degrees(L)
        omega_deg = np.degrees(omega)
        w_deg = np.degrees(w)
        
        # Calculate nu with proper angle wrapping
        # First normalize all angles to [0, 360)
        L_deg = L_deg % 360
        omega_deg = omega_deg % 360
        w_deg = w_deg % 360
        
        # Calculate nu = L - omega - w
        nu = L_deg - omega_deg - w_deg
        # Normalize to [0, 360) - but be careful about negative angles
        if nu < 0:
            nu = nu + 360
        nu = nu % 360
        
        # Normalize angles to [0, 360) for omega, w, nu and [0, 180] for i
        omega = np.degrees(omega) % 360
        w = np.degrees(w) % 360
        # nu is already normalized above, don't normalize again
        i = np.degrees(i) % 180
        
        return a, e, i, omega, w, nu
    
    def keplerian_to_modified_equinoctial(self, a, e, i, omega, w, nu):
        """
        Convert Keplerian elements to modified equinoctial elements
        
        Parameters:
        a: semi-major axis (m)
        e: eccentricity (0-1)
        i: inclination (degrees)
        omega: longitude of ascending node (degrees)
        w: argument of periapsis (degrees)
        nu: true anomaly (degrees)
        """
        # Convert to radians
        i_rad = np.radians(i)
        omega_rad = np.radians(omega)
        w_rad = np.radians(w)
        nu_rad = np.radians(nu)
        
        # Calculate semi-latus rectum
        p = a * (1 - e**2)
        
        # Calculate f and g
        f = e * np.cos(w_rad + omega_rad)
        g = e * np.sin(w_rad + omega_rad)
        
        # Calculate h and k
        h = np.tan(i_rad / 2) * np.cos(omega_rad)
        k = np.tan(i_rad / 2) * np.sin(omega_rad)
        
        # Calculate true longitude: L = omega + w + nu
        # This is the sum of the three angles
        L = omega_rad + w_rad + nu_rad
        
        return p, f, g, h, k, np.degrees(L)
    
    def set_orbital_elements(self, element_type, **kwargs):
        """
        Set orbital elements from either Keplerian or modified equinoctial elements
        
        Parameters:
        element_type: 'keplerian' or 'equinoctial'
        **kwargs: orbital element values
        """
        if element_type == 'keplerian':
            a = kwargs['a']
            e = kwargs['e']
            i = kwargs['i']
            omega = kwargs['omega']
            w = kwargs['w']
            nu = kwargs.get('nu', 0)  # Default to 0 if not provided
            
            # Validate Keplerian elements
            if a <= 0:
                raise ValueError(f"Invalid semi-major axis: {a}. Must be positive.")
            if e < 0 or e >= 1:
                raise ValueError(f"Invalid eccentricity: {e}. Must be 0 <= e < 1.")
            if i < 0 or i > 180:
                raise ValueError(f"Invalid inclination: {i}. Must be 0 <= i <= 180 degrees.")
            
            self.orbital_elements = {
                'type': 'keplerian',
                'a': a, 'e': e, 'i': i, 'omega': omega, 'w': w, 'nu': nu
            }
            
        elif element_type == 'equinoctial':
            p = kwargs['p']
            f = kwargs['f']
            g = kwargs['g']
            h = kwargs['h']
            k = kwargs['k']
            L = kwargs['L']
            
            # Validate equinoctial elements
            if p <= 0:
                raise ValueError(f"Invalid semi-latus rectum: {p}. Must be positive.")
            
            # Convert to Keplerian
            a, e, i, omega, w, nu = self.modified_equinoctial_to_keplerian(p, f, g, h, k, L)
            
            # Validate converted Keplerian elements
            if a <= 0:
                raise ValueError(f"Invalid semi-major axis from conversion: {a}. Check equinoctial elements.")
            if e < 0 or e >= 1:
                raise ValueError(f"Invalid eccentricity from conversion: {e}. Check f and g values.")
            
            self.orbital_elements = {
                'type': 'equinoctial',
                'p': p, 'f': f, 'g': g, 'h': h, 'k': k, 'L': L,
                'keplerian': {'a': a, 'e': e, 'i': i, 'omega': omega, 'w': w, 'nu': nu}
            }
        
        # Calculate orbital period
        if 'a' in self.orbital_elements:
            a = self.orbital_elements['a']
        else:
            a = self.orbital_elements['keplerian']['a']
        
        self.orbital_period = 2 * np.pi * np.sqrt(a**3 / self.mu_earth)
        
        print(f"Orbital elements set: {element_type}")
        print(f"Orbital period: {self.orbital_period/3600:.2f} hours")
    
    def generate_orbit_trail(self, num_points=1000):
        """Generate points along the complete orbit for trail visualization"""
        if self.orbital_elements is None:
            raise ValueError("Orbital elements not set. Call set_orbital_elements() first.")
        
        # Get Keplerian elements
        if self.orbital_elements['type'] == 'keplerian':
            a = self.orbital_elements['a']
            e = self.orbital_elements['e']
            i = self.orbital_elements['i']
            omega = self.orbital_elements['omega']
            w = self.orbital_elements['w']
        else:
            keplerian = self.orbital_elements['keplerian']
            a = keplerian['a']
            e = keplerian['e']
            i = keplerian['i']
            omega = keplerian['omega']
            w = keplerian['w']
        
        # Generate true anomaly array
        nu_array = np.linspace(0, 360, num_points)
        
        positions = []
        for nu in nu_array:
            pos, _ = self.keplerian_to_cartesian(a, e, i, omega, w, nu)
            positions.append(pos)
        
        return np.array(positions)
    
    def update_satellite_position(self, time_step):
        """Update satellite position based on current time"""
        if self.orbital_elements is None:
            return
        
        # Get Keplerian elements
        if self.orbital_elements['type'] == 'keplerian':
            a = self.orbital_elements['a']
            e = self.orbital_elements['e']
            i = self.orbital_elements['i']
            omega = self.orbital_elements['omega']
            w = self.orbital_elements['w']
        else:
            keplerian = self.orbital_elements['keplerian']
            a = keplerian['a']
            e = keplerian['e']
            i = keplerian['i']
            omega = keplerian['omega']
            w = keplerian['w']
        
        # Calculate mean anomaly from time
        n = 2 * np.pi / self.orbital_period  # Mean motion
        M = n * time_step  # Mean anomaly
        
        # Solve Kepler's equation for eccentric anomaly (simplified Newton-Raphson)
        E = M  # Initial guess
        for _ in range(10):  # Iterate to solve Kepler's equation
            E_new = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            if abs(E_new - E) < 1e-8:
                break
            E = E_new
        
        # Convert eccentric anomaly to true anomaly
        nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        nu = np.degrees(nu) % 360
        
        # Update satellite position and velocity
        self.satellite_position, self.satellite_velocity = self.keplerian_to_cartesian(
            a, e, i, omega, w, nu
        )
        
        # Add to trail
        self.trail_points.append(self.satellite_position.copy())
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
    
    def draw_satellite(self, position, quaternion):
        """Draw a 3D satellite at the given position with given orientation"""
        if self.test_mode or self.ax is None:
            return
        
        # Scale position to km
        pos_km = position / 1000
        
        # Create satellite body (larger, more visible box with solar panels)
        size = self.satellite_scale * 1000  # Scale in km
        
        # Define satellite corners in local frame (main body)
        main_body = np.array([
            [-size/2, -size/6, -size/12],  # Back-left-bottom
            [size/2, -size/6, -size/12],   # Front-left-bottom
            [size/2, size/6, -size/12],    # Front-right-bottom
            [-size/2, size/6, -size/12],   # Back-right-bottom
            [-size/2, -size/6, size/12],   # Back-left-top
            [size/2, -size/6, size/12],    # Front-left-top
            [size/2, size/6, size/12],     # Front-right-top
            [-size/2, size/6, size/12]     # Back-right-top
        ])
        
        # Define solar panels (wings)
        left_panel = np.array([
            [-size/2, -size/3, -size/24],  # Back-left-bottom
            [-size/2, -size/2, -size/24],  # Back-left-bottom
            [-size/2, -size/2, size/24],   # Back-left-top
            [-size/2, -size/3, size/24]    # Back-left-top
        ])
        
        right_panel = np.array([
            [-size/2, size/3, -size/24],   # Back-right-bottom
            [-size/2, size/2, -size/24],   # Back-right-bottom
            [-size/2, size/2, size/24],    # Back-right-top
            [-size/2, size/3, size/24]     # Back-right-top
        ])
        
        # Combine all parts
        corners = main_body
        
        # Convert quaternion to rotation matrix
        w, x, y, z = quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Rotate all satellite parts
        rotated_main = (R @ main_body.T).T
        rotated_left_panel = (R @ left_panel.T).T
        rotated_right_panel = (R @ right_panel.T).T
        
        # Translate to satellite position
        main_corners = rotated_main + pos_km
        left_panel_corners = rotated_left_panel + pos_km
        right_panel_corners = rotated_right_panel + pos_km
        
        # Draw main satellite body faces
        main_faces = [
            ([0, 1, 2, 3], 'blue'),      # bottom
            ([4, 5, 6, 7], 'lightblue'), # top
            ([1, 2, 6, 5], 'red'),       # front
            ([0, 3, 7, 4], 'darkblue'),  # back
            ([0, 1, 5, 4], 'green'),     # left
            ([2, 3, 7, 6], 'yellow')     # right
        ]
        
        for face_indices, color in main_faces:
            face_corners = main_corners[face_indices]
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection([face_corners], alpha=0.9, facecolor=color, edgecolor='black', linewidth=1)
            self.ax.add_collection3d(poly)
        
        # Draw solar panels
        left_panel_faces = [
            ([0, 1, 2, 3], 'darkgreen')  # solar panel face
        ]
        
        right_panel_faces = [
            ([0, 1, 2, 3], 'darkgreen')  # solar panel face
        ]
        
        for face_indices, color in left_panel_faces:
            face_corners = left_panel_corners[face_indices]
            poly = Poly3DCollection([face_corners], alpha=0.8, facecolor=color, edgecolor='black', linewidth=1)
            self.ax.add_collection3d(poly)
            
        for face_indices, color in right_panel_faces:
            face_corners = right_panel_corners[face_indices]
            poly = Poly3DCollection([face_corners], alpha=0.8, facecolor=color, edgecolor='black', linewidth=1)
            self.ax.add_collection3d(poly)
    
    def setup_plot(self):
        """Initialize the 3D plot"""
        if self.test_mode:
            return
            
        plt.ion()
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.grid(True)
        
        # Set reasonable plot limits based on orbital radius
        if self.orbital_elements:
            if self.orbital_elements['type'] == 'keplerian':
                a = self.orbital_elements['a']
            else:
                a = self.orbital_elements['keplerian']['a']
            
            # Set limits to show the orbit clearly
            max_range = a * 1.2 / 1000  # Convert to km and add margin
            self.ax.set_xlim([-max_range, max_range])
            self.ax.set_ylim([-max_range, max_range])
            self.ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        plt.show(block=False)
    
    def plot_earth(self):
        """Draw Earth in the plot"""
        if self.test_mode or self.ax is None:
            return
        
        # Create Earth sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_earth = self.earth_radius/1000 * np.outer(np.cos(u), np.sin(v))
        y_earth = self.earth_radius/1000 * np.outer(np.sin(u), np.sin(v))
        z_earth = self.earth_radius/1000 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.earth_surface = self.ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.3, color='lightblue')
        
        # Add coordinate axes
        axis_length = self.earth_radius/1000 * 1.5
        self.ax.plot([0, 0], [0, 0], [0, axis_length], 'r-', linewidth=4, label='North Pole')
        self.ax.plot([0, axis_length], [0, 0], [0, 0], 'g-', linewidth=2, alpha=0.7, label='Prime Meridian')
        self.ax.plot([0, 0], [0, axis_length], [0, 0], 'b-', linewidth=2, alpha=0.7, label='90°E Meridian')
    
    def plot_orbit_trail(self):
        """Plot the complete orbit trail"""
        if self.test_mode or self.ax is None:
            return
        
        # Generate orbit points
        positions = self.generate_orbit_trail()
        positions_km = positions / 1000
        
        # Plot orbit
        self.orbit_line = self.ax.plot(positions_km[:, 0], positions_km[:, 1], positions_km[:, 2], 
                                      'b-', linewidth=2, alpha=0.7, label='Orbit')[0]
    
    def plot_satellite_trail(self):
        """Plot the satellite's actual path"""
        if self.test_mode or self.ax is None or len(self.trail_points) < 2:
            return
        
        trail_array = np.array(self.trail_points)
        trail_km = trail_array / 1000
        
        # Plot trail
        self.ax.plot(trail_km[:, 0], trail_km[:, 1], trail_km[:, 2], 
                    'r-', linewidth=1, alpha=0.5, label='Satellite Trail')
    
    def update_plot(self):
        """Update the entire plot"""
        if self.test_mode or self.ax is None:
            return
        
        # Clear previous satellite
        if self.satellite_plot is not None:
            try:
                self.satellite_plot.remove()
            except:
                pass
        
        # Clear previous velocity vector
        if self.velocity_quiver is not None:
            try:
                self.velocity_quiver.remove()
            except:
                pass
        
        # Draw satellite
        self.draw_satellite(self.satellite_position, self.satellite_quaternion)
        
        # Draw velocity vector (properly scaled)
        pos_km = self.satellite_position / 1000
        vel_km = self.satellite_velocity / 1000
        
        # Scale velocity vector to be visible but not too large
        if self.orbital_elements:
            if self.orbital_elements['type'] == 'keplerian':
                a = self.orbital_elements['a']
            else:
                a = self.orbital_elements['keplerian']['a']
            
            # Scale velocity to be about 5% of orbital radius
            scale_factor = (a / 1000) * 0.05 / np.linalg.norm(vel_km) if np.linalg.norm(vel_km) > 0 else 1
            vel_scaled = vel_km * scale_factor
            
            self.velocity_quiver = self.ax.quiver(pos_km[0], pos_km[1], pos_km[2],
                                                vel_scaled[0], vel_scaled[1], vel_scaled[2],
                                                color='orange', alpha=0.8, linewidth=2)
        
        # Update plot
        plt.draw()
        plt.pause(0.01)
    
    def run_simulation(self, duration_hours=1.0, time_step_seconds=1.0):
        """Run the orbital simulation"""
        if self.orbital_elements is None:
            raise ValueError("Orbital elements not set. Call set_orbital_elements() first.")
        
        print(f"Starting simulation for {duration_hours} hours...")
        print(f"Time step: {time_step_seconds} seconds")
        
        # Setup plot
        self.setup_plot()
        self.plot_earth()
        self.plot_orbit_trail()
        
        # Simulation loop
        start_time = time.time()
        simulation_time = 0
        max_time = duration_hours * 3600  # Convert to seconds
        
        while simulation_time < max_time:
            # Update satellite position
            self.update_satellite_position(simulation_time)
            
            # Update satellite orientation (simple rotation for demo)
            # In a real simulation, this would be based on attitude control
            rotation_rate = 0.1  # rad/s
            angle = rotation_rate * simulation_time
            self.satellite_quaternion = np.array([
                np.cos(angle/2), 0, 0, np.sin(angle/2)
            ])
            
            # Update plot
            self.update_plot()
            self.plot_satellite_trail()
            
            # Update time
            simulation_time += time_step_seconds * self.animation_speed
            
            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < simulation_time / self.animation_speed:
                time.sleep((simulation_time / self.animation_speed) - elapsed)
        
        print("Simulation complete!")
    
    def print_orbital_info(self):
        """Print detailed orbital information"""
        if self.orbital_elements is None:
            print("No orbital elements set.")
            return
        
        print("\n" + "="*60)
        print("ORBITAL ELEMENTS")
        print("="*60)
        
        if self.orbital_elements['type'] == 'keplerian':
            elements = self.orbital_elements
            print("Type: Keplerian Elements")
            print(f"Semi-major axis (a): {elements['a']/1000:.1f} km")
            print(f"Eccentricity (e): {elements['e']:.6f}")
            print(f"Inclination (i): {elements['i']:.2f}°")
            print(f"Longitude of ascending node (Ω): {elements['omega']:.2f}°")
            print(f"Argument of periapsis (ω): {elements['w']:.2f}°")
            print(f"True anomaly (ν): {elements['nu']:.2f}°")
        else:
            elements = self.orbital_elements
            print("Type: Modified Equinoctial Elements")
            print(f"Semi-latus rectum (p): {elements['p']/1000:.1f} km")
            print(f"f: {elements['f']:.6f}")
            print(f"g: {elements['g']:.6f}")
            print(f"h: {elements['h']:.6f}")
            print(f"k: {elements['k']:.6f}")
            print(f"True longitude (L): {elements['L']:.2f}°")
            
            print("\nConverted to Keplerian:")
            keplerian = elements['keplerian']
            print(f"Semi-major axis (a): {keplerian['a']/1000:.1f} km")
            print(f"Eccentricity (e): {keplerian['e']:.6f}")
            print(f"Inclination (i): {keplerian['i']:.2f}°")
            print(f"Longitude of ascending node (Ω): {keplerian['omega']:.2f}°")
            print(f"Argument of periapsis (ω): {keplerian['w']:.2f}°")
            print(f"True anomaly (ν): {keplerian['nu']:.2f}°")
        
        print(f"\nOrbital period: {self.orbital_period/3600:.2f} hours")
        print(f"Animation speed: {self.animation_speed}x")


def main():
    """Main function to demonstrate the enhanced orbital simulation"""
    print("Enhanced Orbital Simulation with Satellite Visualization")
    print("=" * 60)
    
    # Create simulation
    sim = EnhancedOrbitSimulation()
    
    # Example 1: Keplerian elements (ISS-like orbit)
    print("\nExample 1: ISS-like orbit using Keplerian elements")
    sim.set_orbital_elements('keplerian',
        a=6.78e6,      # Semi-major axis (m) - ~400 km altitude
        e=0.0001,      # Eccentricity (nearly circular)
        i=51.6,        # Inclination (degrees)
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    sim.print_orbital_info()
    
    # Example 2: Modified equinoctial elements
    print("\nExample 2: Geostationary orbit using modified equinoctial elements")
    sim.set_orbital_elements('equinoctial',
        p=4.22e7,      # Semi-latus rectum (m) - geostationary
        f=0.0,         # e*cos(w+omega)
        g=0.0,         # e*sin(w+omega)
        h=0.0,         # tan(i/2)*cos(omega)
        k=0.0,         # tan(i/2)*sin(omega)
        L=0            # True longitude
    )
    sim.print_orbital_info()
    
    # Run simulation (uncomment to run)
    # sim.run_simulation(duration_hours=0.1, time_step_seconds=10)


if __name__ == "__main__":
    main()
