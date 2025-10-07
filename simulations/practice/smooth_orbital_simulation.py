#!/usr/bin/env python3
"""
Smooth Orbital Simulation
Improved real-time updates that don't freeze
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

# Use better backend for real-time plotting
matplotlib.use('TkAgg')

class SmoothOrbitalSimulation(EnhancedOrbitSimulation):
    """Enhanced orbital simulation with smooth real-time updates"""
    
    def __init__(self, test_mode=False):
        super().__init__(test_mode)
        self.update_interval = 0.05  # 50ms between updates
        self.last_update_time = 0
        
    def setup_plot(self):
        """Setup plot with better performance settings"""
        if self.test_mode:
            return
            
        # Create figure with optimized settings
        self.fig = plt.figure(figsize=(12, 10), dpi=80)  # Lower DPI for better performance
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels and styling
        self.ax.set_xlabel('X (km)', fontsize=10)
        self.ax.set_ylabel('Y (km)', fontsize=10)
        self.ax.set_zlabel('Z (km)', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
        # Set plot limits
        if self.orbital_elements:
            if self.orbital_elements['type'] == 'keplerian':
                a = self.orbital_elements['a']
            else:
                a = self.orbital_elements['keplerian']['a']
            
            max_range = a * 1.2 / 1000
            self.ax.set_xlim([-max_range, max_range])
            self.ax.set_ylim([-max_range, max_range])
            self.ax.set_zlim([-max_range, max_range])
        
        # Set title
        self.ax.set_title('Smooth Orbital Simulation', fontsize=12, fontweight='bold')
        
        # Plot Earth and orbit
        self.plot_earth()
        self.plot_orbit_trail()
        
        # Configure for better performance
        plt.ion()
        self.fig.tight_layout()
        self.fig.show()
        
    def update_plot_smooth(self):
        """Update plot with smooth, non-freezing updates"""
        if self.test_mode or self.ax is None:
            return
        
        current_time = time.time()
        
        # Throttle updates to prevent freezing
        if current_time - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = current_time
        
        try:
            # Clear previous satellite and velocity
            if self.satellite_plot is not None:
                try:
                    self.satellite_plot.remove()
                except:
                    pass
                    
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
            
            if self.orbital_elements and np.linalg.norm(vel_km) > 0:
                if self.orbital_elements['type'] == 'keplerian':
                    a = self.orbital_elements['a']
                else:
                    a = self.orbital_elements['keplerian']['a']
                
                scale_factor = (a / 1000) * 0.05 / np.linalg.norm(vel_km)
                vel_scaled = vel_km * scale_factor
                
                self.velocity_quiver = self.ax.quiver(pos_km[0], pos_km[1], pos_km[2],
                                                    vel_scaled[0], vel_scaled[1], vel_scaled[2],
                                                    color='orange', alpha=0.8, linewidth=2)
            
            # Update plot with better method
            self.fig.canvas.draw_idle()  # Use draw_idle instead of draw
            self.fig.canvas.flush_events()  # Process events
            
        except Exception as e:
            print(f"Update error (non-critical): {e}")
    
    def run_smooth_simulation(self, duration_hours=1.0, time_step_seconds=1.0):
        """Run simulation with smooth updates"""
        if self.orbital_elements is None:
            raise ValueError("Orbital elements not set. Call set_orbital_elements() first.")
        
        print(f"⚡ Starting smooth simulation for {duration_hours} hours...")
        print(f"Time step: {time_step_seconds} seconds")
        print("This version should not freeze and updates smoothly!")
        
        # Setup plot
        self.setup_plot()
        
        # Initialize simulation
        self.current_time = 0
        self.trail_points = []
        self.satellite_position = np.array([0, 0, 0])
        self.satellite_velocity = np.array([0, 0, 0])
        self.satellite_quaternion = np.array([1, 0, 0, 0])
        
        # Calculate simulation parameters
        total_time_seconds = duration_hours * 3600
        num_steps = int(total_time_seconds / time_step_seconds)
        
        print(f"Total simulation steps: {num_steps}")
        print("Close the plot window to stop the simulation early.")
        
        try:
            # Run simulation loop
            for step in range(num_steps):
                # Update satellite position
                self.update_satellite_position(time_step_seconds)
                
                # Update plot (throttled)
                self.update_plot_smooth()
                
                # Update time
                self.current_time += time_step_seconds
                
                # Show progress every 100 steps
                if step % 100 == 0:
                    progress = (step + 1) / num_steps * 100
                    print(f"Progress: {progress:.1f}% ({step + 1}/{num_steps} steps)")
                
                # Check if plot window is still open
                if not plt.get_fignums():
                    print("\nPlot window closed. Simulation stopped.")
                    break
                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        except Exception as e:
            print(f"\nSimulation error: {e}")
        finally:
            print("✅ Simulation complete!")

def run_smooth_iss_simulation():
    """Run smooth ISS orbit simulation"""
    print("🚀 SMOOTH ISS ORBIT SIMULATION")
    print("="*50)
    
    sim = SmoothOrbitalSimulation(test_mode=False)
    
    # Set up ISS-like orbit
    sim.set_orbital_elements('keplerian',
        a=6.78e6,      # 400 km altitude
        e=0.0001,      # Nearly circular
        i=51.6,        # 51.6° inclination
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    
    sim.print_orbital_info()
    
    # Run smooth simulation
    sim.run_smooth_simulation(duration_hours=0.3, time_step_seconds=30)
    
    return sim

def run_smooth_elliptical_simulation():
    """Run smooth elliptical orbit simulation"""
    print("\n🛰️ SMOOTH ELLIPTICAL ORBIT SIMULATION")
    print("="*50)
    
    sim = SmoothOrbitalSimulation(test_mode=False)
    
    # Set up elliptical orbit
    sim.set_orbital_elements('keplerian',
        a=8.0e6,       # Semi-major axis
        e=0.3,         # Eccentricity
        i=45.0,        # Inclination
        omega=30.0,    # Longitude of ascending node
        w=60.0,        # Argument of periapsis
        nu=0           # True anomaly
    )
    
    sim.print_orbital_info()
    
    # Run smooth simulation
    sim.run_smooth_simulation(duration_hours=0.5, time_step_seconds=45)
    
    return sim

def main():
    """Main function"""
    print("⚡ SMOOTH ORBITAL SIMULATION")
    print("="*50)
    print("This version fixes the freezing issue with better real-time updates!")
    print("Choose your simulation:")
    print("1. ISS Orbit (Smooth)")
    print("2. Elliptical Orbit (Smooth)")
    print("3. Run Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_smooth_iss_simulation()
    elif choice == "2":
        run_smooth_elliptical_simulation()
    elif choice == "3":
        run_smooth_iss_simulation()
        run_smooth_elliptical_simulation()
    else:
        print("Invalid choice. Running ISS simulation...")
        run_smooth_iss_simulation()
    
    print("\n🎉 All simulations complete!")

if __name__ == "__main__":
    main()
