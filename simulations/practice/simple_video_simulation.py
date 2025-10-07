#!/usr/bin/env python3
"""
Simple Video Orbital Simulation
Creates MP4 videos of orbital simulations without freezing
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from datetime import datetime

# Use non-interactive backend for video generation
matplotlib.use('Agg')

class SimpleVideoSimulation(EnhancedOrbitSimulation):
    """Simple video orbital simulation"""
    
    def __init__(self, test_mode=False):
        super().__init__(test_mode)
        self.frames = []
        
    def create_video_animation(self, duration_hours=0.5, time_step_seconds=30, fps=24):
        """Create animation and export as video"""
        if self.orbital_elements is None:
            raise ValueError("Orbital elements not set. Call set_orbital_elements() first.")
        
        print(f"🎬 Creating orbital animation...")
        print(f"Duration: {duration_hours} hours, Time step: {time_step_seconds}s, FPS: {fps}")
        
        # Calculate total frames
        total_time_seconds = duration_hours * 3600
        total_frames = int(total_time_seconds / time_step_seconds)
        
        print(f"Total frames: {total_frames}")
        
        # Setup plot
        self.setup_plot_for_animation()
        
        # Generate orbital positions
        positions = self.generate_orbit_trail(total_frames)
        
        # Create animation
        def animate(frame):
            if frame >= len(positions):
                return
            
            # Clear previous satellite
            if hasattr(self, 'satellite_plot') and self.satellite_plot is not None:
                try:
                    self.satellite_plot.remove()
                except:
                    pass
            
            # Update satellite position
            self.satellite_position = positions[frame]
            
            # Calculate velocity (approximate)
            if frame < len(positions) - 1:
                self.satellite_velocity = (positions[frame + 1] - positions[frame]) / time_step_seconds
            else:
                self.satellite_velocity = np.array([0, 0, 0])
            
            # Draw satellite
            self.draw_satellite(self.satellite_position, self.satellite_quaternion)
            
            # Draw velocity vector
            pos_km = self.satellite_position / 1000
            vel_km = self.satellite_velocity / 1000
            
            if np.linalg.norm(vel_km) > 0:
                if self.orbital_elements:
                    if self.orbital_elements['type'] == 'keplerian':
                        a = self.orbital_elements['a']
                    else:
                        a = self.orbital_elements['keplerian']['a']
                    
                    scale_factor = (a / 1000) * 0.05 / np.linalg.norm(vel_km)
                    vel_scaled = vel_km * scale_factor
                    
                    # Clear previous velocity vector
                    if hasattr(self, 'velocity_quiver') and self.velocity_quiver is not None:
                        try:
                            self.velocity_quiver.remove()
                        except:
                            pass
                    
                    self.velocity_quiver = self.ax.quiver(pos_km[0], pos_km[1], pos_km[2],
                                                        vel_scaled[0], vel_scaled[1], vel_scaled[2],
                                                        color='orange', alpha=0.8, linewidth=2)
            
            # Add progress text
            progress = (frame + 1) / total_frames * 100
            if hasattr(self, 'progress_text'):
                self.progress_text.remove()
            self.progress_text = self.ax.text2D(0.02, 0.98, f'Progress: {progress:.1f}%', 
                                              transform=self.ax.transAxes, fontsize=10, 
                                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            return []
        
        # Create animation
        print("Creating animation frames...")
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=total_frames, 
            interval=1000/fps, blit=False, repeat=False
        )
        
        return self.animation
    
    def setup_plot_for_animation(self):
        """Setup plot for animation"""
        if self.test_mode:
            return
            
        # Create figure
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
        self.ax.set_title('Orbital Simulation Animation', fontsize=14, fontweight='bold')
        
        # Plot Earth and orbit
        self.plot_earth()
        self.plot_orbit_trail()
        
    def save_video(self, filename=None, fps=24):
        """Save animation as MP4 video"""
        if not hasattr(self, 'animation') or self.animation is None:
            print("No animation to save. Create animation first.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"orbital_simulation_{timestamp}.mp4"
        
        print(f"🎬 Saving video: {filename}")
        print(f"FPS: {fps}")
        
        try:
            # Save as MP4
            self.animation.save(filename, writer='ffmpeg', fps=fps, bitrate=1800)
            print(f"✅ Video saved successfully: {filename}")
            
            # Check file size
            if os.path.exists(filename):
                file_size = os.path.getsize(filename) / (1024*1024)
                print(f"File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ Error saving video: {e}")
            print("Trying alternative format...")
            
            # Try saving as GIF instead
            gif_filename = filename.replace('.mp4', '.gif')
            try:
                self.animation.save(gif_filename, writer='pillow', fps=fps)
                print(f"✅ GIF saved successfully: {gif_filename}")
            except Exception as e2:
                print(f"❌ Error saving GIF: {e2}")

def create_iss_video():
    """Create ISS orbit video"""
    print("🚀 ISS ORBIT VIDEO")
    print("="*40)
    
    sim = SimpleVideoSimulation(test_mode=False)
    
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
    
    # Create animation
    sim.create_video_animation(duration_hours=0.25, time_step_seconds=30, fps=20)
    
    # Save video
    sim.save_video("iss_orbit_simulation.mp4", fps=20)
    
    return sim

def create_elliptical_video():
    """Create elliptical orbit video"""
    print("\n🛰️ ELLIPTICAL ORBIT VIDEO")
    print("="*40)
    
    sim = SimpleVideoSimulation(test_mode=False)
    
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
    
    # Create animation
    sim.create_video_animation(duration_hours=0.5, time_step_seconds=45, fps=20)
    
    # Save video
    sim.save_video("elliptical_orbit_simulation.mp4", fps=20)
    
    return sim

def main():
    """Main function"""
    print("🎬 SIMPLE VIDEO ORBITAL SIMULATION")
    print("="*50)
    print("This creates MP4 videos of orbital simulations!")
    print("No freezing issues - everything runs smoothly.")
    
    print("\nCreating ISS orbit video...")
    create_iss_video()
    
    print("\nCreating elliptical orbit video...")
    create_elliptical_video()
    
    print("\n🎉 All videos created successfully!")
    print("Check the current directory for the MP4 files.")

if __name__ == "__main__":
    main()
