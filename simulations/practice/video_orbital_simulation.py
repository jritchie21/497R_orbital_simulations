#!/usr/bin/env python3
"""
Video Orbital Simulation
Exports orbital simulations as MP4 videos and provides better real-time updates
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from datetime import datetime

# Set matplotlib backend for better performance
matplotlib.use('Agg')  # Use non-interactive backend for video generation

class VideoOrbitalSimulation(EnhancedOrbitSimulation):
    """Enhanced orbital simulation with video export capabilities"""
    
    def __init__(self, test_mode=False):
        super().__init__(test_mode)
        self.frames = []
        self.animation = None
        
    def setup_plot_for_video(self):
        """Setup plot optimized for video generation"""
        if self.test_mode:
            return
            
        # Create figure with better settings for video
        self.fig = plt.figure(figsize=(12, 10), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels and styling
        self.ax.set_xlabel('X (km)', fontsize=12)
        self.ax.set_ylabel('Y (km)', fontsize=12)
        self.ax.set_zlabel('Z (km)', fontsize=12)
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
        self.ax.set_title('Orbital Simulation', fontsize=14, fontweight='bold')
        
        # Plot Earth
        self.plot_earth()
        
        # Plot orbit trail
        self.plot_orbit_trail()
        
    def capture_frame(self, frame_num, total_frames):
        """Capture current frame for video"""
        if self.test_mode or self.ax is None:
            return
            
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
        
        # Add progress text
        progress = (frame_num + 1) / total_frames * 100
        self.ax.text2D(0.02, 0.98, f'Progress: {progress:.1f}%', 
                      transform=self.ax.transAxes, fontsize=10, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Capture frame
        self.fig.canvas.draw()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)
        
        if frame_num % 10 == 0:
            print(f"Captured frame {frame_num + 1}/{total_frames} ({progress:.1f}%)")
    
    def export_video(self, filename=None, fps=30):
        """Export the simulation as an MP4 video"""
        if not self.frames:
            print("No frames to export. Run simulation first.")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orbital_simulation_{timestamp}.mp4"
        
        print(f"\n🎬 Exporting video: {filename}")
        print(f"Frames: {len(self.frames)}, FPS: {fps}")
        
        try:
            import cv2
            
            # Get frame dimensions
            height, width, layers = self.frames[0].shape
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            # Write frames
            for i, frame in enumerate(self.frames):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
                if i % 30 == 0:
                    print(f"Writing frame {i + 1}/{len(self.frames)}")
            
            out.release()
            print(f"✅ Video exported successfully: {filename}")
            print(f"File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
            
        except ImportError:
            print("❌ OpenCV not available. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "opencv-python"])
            print("Please run the simulation again after installation.")
        except Exception as e:
            print(f"❌ Error exporting video: {e}")
    
    def run_video_simulation(self, duration_hours=0.5, time_step_seconds=30, fps=30):
        """Run simulation and export as video"""
        if self.orbital_elements is None:
            raise ValueError("Orbital elements not set. Call set_orbital_elements() first.")
        
        print(f"🎬 Starting video simulation for {duration_hours} hours...")
        print(f"Time step: {time_step_seconds} seconds, Target FPS: {fps}")
        
        # Calculate total frames
        total_time_seconds = duration_hours * 3600
        total_frames = int(total_time_seconds / time_step_seconds)
        
        print(f"Total frames to capture: {total_frames}")
        
        # Setup plot
        self.setup_plot_for_video()
        
        # Initialize simulation
        self.current_time = 0
        self.trail_points = []
        self.satellite_position = np.array([0, 0, 0])
        self.satellite_velocity = np.array([0, 0, 0])
        self.satellite_quaternion = np.array([1, 0, 0, 0])
        
        # Run simulation and capture frames
        for frame_num in range(total_frames):
            # Update satellite position
            self.update_satellite_position(time_step_seconds)
            
            # Capture frame
            self.capture_frame(frame_num, total_frames)
            
            # Update time
            self.current_time += time_step_seconds
        
        print(f"\n✅ Simulation complete! Captured {len(self.frames)} frames.")
        
        # Export video
        self.export_video(fps=fps)
        
        # Clean up
        plt.close(self.fig)

def run_iss_video():
    """Create ISS orbit video"""
    print("🚀 ISS ORBIT VIDEO SIMULATION")
    print("="*50)
    
    sim = VideoOrbitalSimulation(test_mode=False)
    
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
    
    # Run video simulation
    sim.run_video_simulation(duration_hours=0.25, time_step_seconds=20, fps=24)
    
    return sim

def run_elliptical_video():
    """Create elliptical orbit video"""
    print("\n🛰️ ELLIPTICAL ORBIT VIDEO SIMULATION")
    print("="*50)
    
    sim = VideoOrbitalSimulation(test_mode=False)
    
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
    
    # Run video simulation
    sim.run_video_simulation(duration_hours=0.5, time_step_seconds=30, fps=24)
    
    return sim

def run_improved_realtime_simulation():
    """Run improved real-time simulation that doesn't freeze"""
    print("\n⚡ IMPROVED REAL-TIME SIMULATION")
    print("="*50)
    
    # Use interactive backend
    matplotlib.use('TkAgg')
    
    sim = EnhancedOrbitSimulation(test_mode=False)
    
    # Set up orbit
    sim.set_orbital_elements('keplerian',
        a=7.0e6,       # Semi-major axis
        e=0.1,         # Eccentricity
        i=30.0,        # Inclination
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    
    sim.print_orbital_info()
    
    print(f"\n🌍 Starting improved real-time simulation...")
    print("This version should not freeze and updates smoothly!")
    
    # Run simulation with better settings
    sim.run_simulation(duration_hours=0.2, time_step_seconds=60)
    
    print("\n✅ Real-time simulation complete!")

def main():
    """Main function to run different simulation types"""
    print("🎬 ORBITAL SIMULATION - VIDEO & IMPROVED REAL-TIME")
    print("="*60)
    print("Choose your simulation type:")
    print("1. ISS Orbit Video")
    print("2. Elliptical Orbit Video") 
    print("3. Improved Real-time Simulation")
    print("4. Run All")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        run_iss_video()
    elif choice == "2":
        run_elliptical_video()
    elif choice == "3":
        run_improved_realtime_simulation()
    elif choice == "4":
        run_iss_video()
        run_elliptical_video()
        run_improved_realtime_simulation()
    else:
        print("Invalid choice. Running ISS video by default...")
        run_iss_video()
    
    print("\n🎉 All simulations complete!")

if __name__ == "__main__":
    main()
