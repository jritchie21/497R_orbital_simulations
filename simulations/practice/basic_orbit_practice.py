import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math

class OrbitPlotter:
    def __init__(self):
        # Earth constants
        self.mu_earth = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        self.earth_radius = 6.371e6  # Earth radius (m)
        
        # Plot objects for updating
        self.fig = None
        self.ax = None
        self.orbit_line = None
        self.earth_surface = None
        self.periapsis_marker = None
        self.apoapsis_marker = None
        self.velocity_quivers = []
        
    def orbital_elements_to_cartesian(self, a, e, i, omega, w, nu):
        """
        Convert orbital elements to Cartesian coordinates
        
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
        # Standard orbital mechanics convention:
        # - X-axis: Points from Earth center to vernal equinox (First Point of Aries)
        # - Y-axis: 90° east of X-axis in equatorial plane
        # - Z-axis: Points to North Pole
        
        # R3(omega) - rotation around Z axis (longitude of ascending node)
        R3_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0],
            [np.sin(omega), np.cos(omega), 0],
            [0, 0, 1]
        ])
        
        # R1(i) - rotation around X axis (inclination)
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(i), -np.sin(i)],
            [0, np.sin(i), np.cos(i)]
        ])
        
        # R3(w) - rotation around Z axis (argument of periapsis)
        R3_w = np.array([
            [np.cos(w), -np.sin(w), 0],
            [np.sin(w), np.cos(w), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix: R3(omega) * R1(i) * R3(w)
        # This transforms from perifocal frame to Earth-centered inertial frame
        R = R3_omega @ R1_i @ R3_w
        
        # Transform position and velocity to inertial frame
        pos_pf = np.array([x_pf, y_pf, z_pf])
        vel_pf = np.array([vx_pf, vy_pf, vz_pf])
        
        pos_inertial = R @ pos_pf
        vel_inertial = R @ vel_pf
        
        return pos_inertial, vel_inertial
    
    def generate_orbit_points(self, a, e, i, omega, w, num_points=1000):
        """Generate points along the orbit"""
        # Create true anomaly array from 0 to 2π
        nu_array = np.linspace(0, 2*np.pi, num_points)
        
        positions = []
        velocities = []
        
        for nu in nu_array:
            pos, vel = self.orbital_elements_to_cartesian(a, e, i, omega, w, np.degrees(nu))
            positions.append(pos)
            velocities.append(vel)
        
        return np.array(positions), np.array(velocities)
    
    def setup_plot(self):
        """Initialize the 3D plot window"""
        if self.fig is None:
            # Enable interactive mode
            plt.ion()
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X (km)')
            self.ax.set_ylabel('Y (km)')
            self.ax.set_zlabel('Z (km)')
            self.ax.grid(True)
            plt.tight_layout()
            plt.show(block=False)
            plt.draw()
            plt.pause(0.01)
    
    def clear_plot(self):
        """Clear all plot elements except the axes"""
        if self.ax is not None:
            # Clear all collections and lines
            self.ax.clear()
            self.ax.set_xlabel('X (km)')
            self.ax.set_ylabel('Y (km)')
            self.ax.set_zlabel('Z (km)')
            self.ax.grid(True)
            
            # Reset plot objects
            self.orbit_line = None
            self.earth_surface = None
            self.periapsis_marker = None
            self.apoapsis_marker = None
            self.velocity_quivers = []
    
    def plot_orbit(self, a, e, i, omega, w, num_points=1000, show_earth=True, show_velocity=False):
        """Plot the orbit in 3D (updates existing plot)"""
        # Setup plot if it doesn't exist
        self.setup_plot()
        
        # Clear previous orbit
        self.clear_plot()
        
        # Generate orbit points
        positions, velocities = self.generate_orbit_points(a, e, i, omega, w, num_points)
        
        # Convert to km for better visualization
        positions_km = positions / 1000
        velocities_km = velocities / 1000
        
        # Plot the orbit
        self.orbit_line = self.ax.plot(positions_km[:, 0], positions_km[:, 1], positions_km[:, 2], 
                                      'b-', linewidth=2, label='Orbit')[0]
        
        # Mark periapsis and apoapsis
        if e > 0:
            # Periapsis (closest point)
            pos_peri, _ = self.orbital_elements_to_cartesian(a, e, i, omega, w, 0)
            pos_peri_km = pos_peri / 1000
            self.periapsis_marker = self.ax.scatter(pos_peri_km[0], pos_peri_km[1], pos_peri_km[2], 
                                                   color='red', s=100, label='Periapsis')
            
            # Apoapsis (farthest point)
            pos_apo, _ = self.orbital_elements_to_cartesian(a, e, i, omega, w, 180)
            pos_apo_km = pos_apo / 1000
            self.apoapsis_marker = self.ax.scatter(pos_apo_km[0], pos_apo_km[1], pos_apo_km[2], 
                                                  color='green', s=100, label='Apoapsis')
        
        # Show Earth
        if show_earth:
            # Create Earth sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_earth = self.earth_radius/1000 * np.outer(np.cos(u), np.sin(v))
            y_earth = self.earth_radius/1000 * np.outer(np.sin(u), np.sin(v))
            z_earth = self.earth_radius/1000 * np.outer(np.ones(np.size(u)), np.cos(v))
            
            self.earth_surface = self.ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.3, color='lightblue')
            
            # Add Earth coordinate axes
            earth_axis_length = self.earth_radius/1000 * 1.5
            
            # North pole (Z-axis) - red
            self.ax.plot([0, 0], [0, 0], [0, earth_axis_length], 'r-', linewidth=4, label='North Pole')
            
            # Equatorial axes
            # X-axis (Greenwich meridian) - green
            self.ax.plot([0, earth_axis_length], [0, 0], [0, 0], 'g-', linewidth=2, alpha=0.7, label='Prime Meridian')
            
            # Y-axis (90°E meridian) - blue  
            self.ax.plot([0, 0], [0, earth_axis_length], [0, 0], 'b-', linewidth=2, alpha=0.7, label='90°E Meridian')
            
            # Add equator circle
            theta = np.linspace(0, 2*np.pi, 100)
            x_eq = (self.earth_radius/1000) * np.cos(theta)
            y_eq = (self.earth_radius/1000) * np.sin(theta)
            z_eq = np.zeros_like(theta)
            self.ax.plot(x_eq, y_eq, z_eq, 'k--', linewidth=1, alpha=0.5, label='Equator')
            
            # Add north pole marker
            self.ax.scatter([0], [0], [self.earth_radius/1000], color='red', s=50, marker='^', label='North Pole')
        
        # Show velocity vectors (optional)
        if show_velocity:
            # Clear previous velocity vectors
            for quiver in self.velocity_quivers:
                try:
                    quiver.remove()
                except:
                    pass
            self.velocity_quivers = []
            
            # Show velocity at every 50th point
            for i in range(0, len(velocities_km), 50):
                quiver = self.ax.quiver(positions_km[i, 0], positions_km[i, 1], positions_km[i, 2],
                                      velocities_km[i, 0], velocities_km[i, 1], velocities_km[i, 2],
                                      color='orange', alpha=0.6, length=1000)
                self.velocity_quivers.append(quiver)
        
        # Calculate orbital period
        period = 2 * np.pi * np.sqrt(a**3 / self.mu_earth) / 3600  # hours
        altitude = a - self.earth_radius  # m
        
        self.ax.set_title(f'Orbit Visualization\n'
                         f'Semi-major axis: {a/1000:.1f} km, Eccentricity: {e:.3f}\n'
                         f'Inclination: {i:.1f}°, Period: {period:.2f} hours\n'
                         f'Altitude: {altitude/1000:.1f} km')
        
        # Set equal aspect ratio with zoom out factor
        max_range = np.array([positions_km[:, 0].max() - positions_km[:, 0].min(),
                             positions_km[:, 1].max() - positions_km[:, 1].min(),
                             positions_km[:, 2].max() - positions_km[:, 2].min()]).max() / 2.0
        
        # Zoom out by 1.5x for better view
        zoom_factor = 1.5
        max_range *= zoom_factor
        
        mid_x = (positions_km[:, 0].max() + positions_km[:, 0].min()) * 0.5
        mid_y = (positions_km[:, 1].max() + positions_km[:, 1].min()) * 0.5
        mid_z = (positions_km[:, 2].max() + positions_km[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        self.ax.legend()
        
        # Update the plot
        plt.draw()
        plt.pause(0.01)  # Small pause to ensure the plot updates
        
        # Ensure the plot window stays interactive
        if self.fig is not None:
            self.fig.canvas.draw_idle()
        
        return positions, velocities
    
    def print_orbital_info(self, a, e, i, omega, w):
        """Print detailed orbital information"""
        print("\n" + "="*50)
        print("ORBITAL ELEMENTS")
        print("="*50)
        print(f"Semi-major axis (a): {a/1000:.1f} km")
        print(f"Eccentricity (e): {e:.6f}")
        print(f"Inclination (i): {i:.2f}°")
        print(f"Longitude of ascending node (Ω): {omega:.2f}°")
        print(f"Argument of periapsis (ω): {w:.2f}°")
        
        # Calculate derived parameters
        p = a * (1 - e**2)  # Semi-latus rectum
        period = 2 * np.pi * np.sqrt(a**3 / self.mu_earth)  # seconds
        period_hours = period / 3600
        altitude = a - self.earth_radius
        
        print(f"\nDERIVED PARAMETERS")
        print("-" * 30)
        print(f"Semi-latus rectum (p): {p/1000:.1f} km")
        print(f"Orbital period: {period_hours:.2f} hours ({period/60:.1f} minutes)")
        print(f"Average altitude: {altitude/1000:.1f} km")
        
        if e > 0:
            rp = a * (1 - e)  # Periapsis radius
            ra = a * (1 + e)  # Apoapsis radius
            print(f"Periapsis altitude: {(rp - self.earth_radius)/1000:.1f} km")
            print(f"Apoapsis altitude: {(ra - self.earth_radius)/1000:.1f} km")
        else:
            print("Circular orbit")

def interactive_mode():
    """Interactive command line interface for orbit plotting"""
    print("="*60)
    print("🌍 INTERACTIVE ORBIT PLOTTER")
    print("="*60)
    print("Plot 3D Earth orbits using orbital elements")
    print("Type 'help' for commands, 'quit' to exit")
    print()
    
    plotter = OrbitPlotter()
    current_orbit = None
    
    while True:
        try:
            command = input("Orbit> ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                print("Goodbye! 🚀")
                break
                
            elif command == 'help':
                print_help()
                
            elif command == 'plot' or command == 'p':
                if current_orbit is None:
                    print("❌ No orbit defined. Use 'set' command first.")
                    continue
                    
                print(f"\n🛰️  Plotting orbit...")
                positions, velocities = plotter.plot_orbit(
                    current_orbit['a'], current_orbit['e'], current_orbit['i'],
                    current_orbit['omega'], current_orbit['w'], 
                    current_orbit['points'], current_orbit['show_earth'], current_orbit['show_velocity']
                )
                print("✅ Orbit plot complete!")
                
            elif command == 'info' or command == 'i':
                if current_orbit is None:
                    print("❌ No orbit defined. Use 'set' command first.")
                    continue
                    
                plotter.print_orbital_info(
                    current_orbit['a'], current_orbit['e'], current_orbit['i'],
                    current_orbit['omega'], current_orbit['w']
                )
                
            elif command.startswith('set '):
                try:
                    # Parse orbital elements from command
                    parts = command.split()
                    if len(parts) < 2:
                        print("❌ Usage: set a=<semi-major-axis> [e=<eccentricity>] [i=<inclination>] [omega=<node>] [w=<periapsis>]")
                        continue
                    
                    # Default values
                    a = None
                    e = 0.0
                    i = 0.0
                    omega = 0.0
                    w = 0.0
                    points = 1000
                    show_earth = True
                    show_velocity = False
                    
                    # Parse parameters
                    for part in parts[1:]:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            if key == 'a':
                                a = float(value) * 1000  # Convert km to m
                            elif key == 'e':
                                e = float(value)
                            elif key == 'i':
                                i = float(value)
                            elif key == 'omega':
                                omega = float(value)
                            elif key == 'w':
                                w = float(value)
                            elif key == 'points':
                                points = int(value)
                            elif key == 'earth':
                                show_earth = value.lower() == 'true'
                            elif key == 'velocity':
                                show_velocity = value.lower() == 'true'
                    
                    if a is None:
                        print("❌ Semi-major axis (a) is required")
                        continue
                    
                    # Validate inputs
                    if e < 0 or e >= 1:
                        print("❌ Eccentricity must be 0 <= e < 1")
                        continue
                    
                    if a <= 6371000:  # Earth radius
                        print("❌ Semi-major axis must be greater than Earth radius (6371 km)")
                        continue
                    
                    # Set current orbit
                    current_orbit = {
                        'a': a, 'e': e, 'i': i, 'omega': omega, 'w': w,
                        'points': points, 'show_earth': show_earth, 'show_velocity': show_velocity
                    }
                    
                    print(f"✅ Orbit set:")
                    print(f"   Semi-major axis: {a/1000:.1f} km")
                    print(f"   Eccentricity: {e:.3f}")
                    print(f"   Inclination: {i:.1f}°")
                    print(f"   Longitude of ascending node: {omega:.1f}°")
                    print(f"   Argument of periapsis: {w:.1f}°")
                    
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
                    
            elif command.startswith('preset '):
                preset_name = command.split(' ', 1)[1].lower()
                current_orbit = get_preset_orbit(preset_name)
                if current_orbit:
                    print(f"✅ Loaded preset: {preset_name}")
                    print(f"   Semi-major axis: {current_orbit['a']/1000:.1f} km")
                    print(f"   Eccentricity: {current_orbit['e']:.3f}")
                    print(f"   Inclination: {current_orbit['i']:.1f}°")
                else:
                    print(f"❌ Unknown preset: {preset_name}")
                    print("Available presets: leo, geo, molniya, iss, polar")
                    
            elif command == 'presets':
                print_presets()
                
            elif command == 'clear':
                current_orbit = None
                print("✅ Current orbit cleared")
                
            elif command == 'status':
                if current_orbit is None:
                    print("📊 Status: No orbit defined")
                else:
                    print("📊 Current orbit:")
                    print(f"   Semi-major axis: {current_orbit['a']/1000:.1f} km")
                    print(f"   Eccentricity: {current_orbit['e']:.3f}")
                    print(f"   Inclination: {current_orbit['i']:.1f}°")
                    print(f"   Longitude of ascending node: {current_orbit['omega']:.1f}°")
                    print(f"   Argument of periapsis: {current_orbit['w']:.1f}°")
                    print(f"   Points: {current_orbit['points']}")
                    print(f"   Show Earth: {current_orbit['show_earth']}")
                    print(f"   Show Velocity: {current_orbit['show_velocity']}")
                    
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 🚀")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_help():
    """Print help information"""
    print("\n📋 AVAILABLE COMMANDS:")
    print("-" * 30)
    print("set a=<km> [e=<0-1>] [i=<deg>] [omega=<deg>] [w=<deg>] [points=<num>] [earth=<true/false>] [velocity=<true/false>]")
    print("  Set orbital elements (a is required, others optional)")
    print("  Example: set a=7000 e=0.1 i=45")
    print()
    print("preset <name>     - Load a preset orbit")
    print("presets           - Show available presets")
    print("plot (or p)       - Plot current orbit")
    print("info (or i)       - Show orbital information")
    print("status            - Show current orbit parameters")
    print("clear             - Clear current orbit")
    print("help              - Show this help")
    print("quit/exit         - Exit program")
    print()
    print("📚 ORBITAL ELEMENTS:")
    print("a (semi-major axis): Distance in km")
    print("e (eccentricity): 0 = circular, 0-1 = elliptical")
    print("i (inclination): 0° = equatorial, 90° = polar")
    print("omega (Ω): Longitude of ascending node")
    print("w (ω): Argument of periapsis")

def get_preset_orbit(preset_name):
    """Get preset orbital elements"""
    presets = {
        'leo': {'a': 7000*1000, 'e': 0.0, 'i': 0.0, 'omega': 0.0, 'w': 0.0, 'points': 1000, 'show_earth': True, 'show_velocity': False},
        'geo': {'a': 42164*1000, 'e': 0.0, 'i': 0.0, 'omega': 0.0, 'w': 0.0, 'points': 1000, 'show_earth': True, 'show_velocity': False},
        'molniya': {'a': 26500*1000, 'e': 0.7, 'i': 63.4, 'omega': 0.0, 'w': 270.0, 'points': 1000, 'show_earth': True, 'show_velocity': False},
        'iss': {'a': 6800*1000, 'e': 0.0, 'i': 51.6, 'omega': 0.0, 'w': 0.0, 'points': 1000, 'show_earth': True, 'show_velocity': False},
        'polar': {'a': 7000*1000, 'e': 0.0, 'i': 90.0, 'omega': 0.0, 'w': 0.0, 'points': 1000, 'show_earth': True, 'show_velocity': False}
    }
    return presets.get(preset_name)

def print_presets():
    """Print available presets"""
    print("\n🎯 AVAILABLE PRESETS:")
    print("-" * 25)
    print("leo     - Low Earth Orbit (7000 km, circular, equatorial)")
    print("geo     - Geostationary Orbit (42164 km, circular, equatorial)")
    print("molniya - Molniya Orbit (26500 km, e=0.7, i=63.4°)")
    print("iss     - ISS-like Orbit (6800 km, circular, i=51.6°)")
    print("polar   - Polar Orbit (7000 km, circular, i=90°)")

def main():
    # Check if command line arguments are provided
    import sys
    if len(sys.argv) > 1:
        # Command line mode
        parser = argparse.ArgumentParser(description='Plot 3D Earth orbits using orbital elements')
        parser.add_argument('--a', type=float, required=True, help='Semi-major axis (km)')
        parser.add_argument('--e', type=float, default=0.0, help='Eccentricity (0-1, default: 0.0)')
        parser.add_argument('--i', type=float, default=0.0, help='Inclination (degrees, default: 0.0)')
        parser.add_argument('--omega', type=float, default=0.0, help='Longitude of ascending node (degrees, default: 0.0)')
        parser.add_argument('--w', type=float, default=0.0, help='Argument of periapsis (degrees, default: 0.0)')
        parser.add_argument('--points', type=int, default=1000, help='Number of orbit points (default: 1000)')
        parser.add_argument('--no-earth', action='store_true', help='Hide Earth sphere')
        parser.add_argument('--velocity', action='store_true', help='Show velocity vectors')
        
        args = parser.parse_args()
        
        # Convert semi-major axis from km to m
        a = args.a * 1000
        
        # Validate inputs
        if args.e < 0 or args.e >= 1:
            print("Error: Eccentricity must be 0 <= e < 1")
            return
        
        if a <= 6371000:  # Earth radius
            print("Error: Semi-major axis must be greater than Earth radius (6371 km)")
            return
        
        # Create orbit plotter
        plotter = OrbitPlotter()
        
        # Print orbital information
        plotter.print_orbital_info(a, args.e, args.i, args.omega, args.w)
        
        # Plot the orbit
        print(f"\nGenerating orbit plot with {args.points} points...")
        positions, velocities = plotter.plot_orbit(
            a, args.e, args.i, args.omega, args.w, 
            args.points, not args.no_earth, args.velocity
        )
        
        print("Orbit plot complete!")
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
