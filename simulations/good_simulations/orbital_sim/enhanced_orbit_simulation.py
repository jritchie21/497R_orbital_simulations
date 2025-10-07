#!/usr/bin/env python3
"""
Enhanced Orbital Simulation with Multiple Satellites
Main simulation file with GUI controls and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# Import supporting modules
from orbital_mechanics import OrbitalMechanics
from coordinate_frames import CoordinateFrames
from attitude_dynamics import AttitudeDynamics, SatelliteState
from tle_handler import TLEHandler
from validation_tools import ValidationTools


@dataclass
class SatelliteConfig:
    """Configuration for a satellite"""
    name: str
    color: str = 'red'
    size: float = 300.0
    trail_length: int = 200
    show_velocity: bool = True
    show_attitude: bool = True
    orbital_elements: Dict = field(default_factory=dict)


class EnhancedOrbitSimulation:
    def __init__(self, test_mode: bool = False):
        self.mu_earth = 3.986004418e14
        self.earth_radius = 6.371e6
        self.omega_earth = 7.2921159e-5
        self.test_mode = test_mode
        self.animation_speed = 100
        self.current_time = 0.0
        self.epoch = datetime.utcnow()
        self.satellites = {}
        self.orbital_mech = OrbitalMechanics()
        self.coord_frames = CoordinateFrames()
        self.tle_handler = TLEHandler()
        self.validator = ValidationTools()
        self.fig = None
        self.ax = None
        self.gui_controls = {}
        self.animation = None
        self.is_paused = False
        self.show_earth = True
        self.show_grid = True
        self.frame_type = 'ECI'
        self.earth_surface = None
    
    def add_satellite(self, config: SatelliteConfig) -> None:
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
                'attitude_axes': None
            }
        }
        self._update_satellite_state(config.name, 0)
        print(f"Added satellite: {config.name}")
    
    def remove_satellite(self, name: str) -> None:
        if name in self.satellites:
            sat_data = self.satellites[name]
            for element in sat_data['visual_elements'].values():
                if element is not None:
                    try:
                        element.remove()
                    except:
                        pass
            del self.satellites[name]
            print(f"Removed satellite: {name}")
    
    def save_configuration(self, filename: str) -> None:
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
        with open(filename, 'r') as f:
            config = json.load(f)
        self.epoch = datetime.fromisoformat(config['epoch'])
        for name in list(self.satellites.keys()):
            self.remove_satellite(name)
        for sat_config in config['satellites'].values():
            self.add_satellite(SatelliteConfig(
                name=sat_config['name'],
                color=sat_config['color'],
                size=sat_config['size'],
                trail_length=sat_config['trail_length'],
                orbital_elements=sat_config['orbital_elements']
            ))
        print(f"Configuration loaded from {filename}")
    
    def export_to_tle(self, satellite_name: str, filename: str) -> str:
        if satellite_name not in self.satellites:
            raise ValueError(f"Satellite {satellite_name} not found")
        sat_data = self.satellites[satellite_name]
        tle = self.tle_handler.keplerian_to_tle(
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
        elements = self.tle_handler.tle_to_keplerian(tle_lines)
        if name is None:
            if len(tle_lines) == 3:
                name = tle_lines[0].strip()
            else:
                name = f"SAT_{len(self.satellites)+1}"
        config = SatelliteConfig(
            name=name,
            color=color,
            orbital_elements=elements
        )
        self.add_satellite(config)
        print(f"Imported satellite {name} from TLE")
    
    def _update_satellite_state(self, name: str, time: float) -> None:
        sat_data = self.satellites[name]
        elements = sat_data['orbital_elements']
        a = elements['a']
        period = 2 * np.pi * np.sqrt(a**3 / self.mu_earth)
        n = 2 * np.pi / period
        M0 = np.radians(elements.get('M', 0))
        M = M0 + n * time
        current_elements = elements.copy()
        current_elements['M'] = np.degrees(M % (2 * np.pi))
        position, velocity = self.orbital_mech.keplerian_to_cartesian(
            current_elements, self.mu_earth
        )
        sat_data['state'].position = position
        sat_data['state'].velocity = velocity
        sat_data['state'].time = time
        if sat_data['config'].show_attitude:
            gg_torque = sat_data['attitude_dynamics'].gravity_gradient_torque(
                sat_data['state'].quaternion,
                position,
                self.mu_earth
            )
            control_torque = np.array([0.001, 0.0005, 0.0002])
            total_torque = gg_torque + control_torque
            dt = 1.0
            sat_data['state'] = sat_data['attitude_dynamics'].propagate_attitude(
                sat_data['state'],
                total_torque,
                dt
            )
        sat_data['trail_points'].append(position.copy())
        if len(sat_data['trail_points']) > sat_data['config'].trail_length:
            sat_data['trail_points'].pop(0)
    
    def setup_plot(self) -> None:
        if self.test_mode:
            return
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(5, 3, width_ratios=[1, 3, 1], 
                                   height_ratios=[1, 10, 1, 1, 1])
        self.ax = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax.set_xlabel('X (km)', fontsize=10)
        self.ax.set_ylabel('Y (km)', fontsize=10)
        self.ax.set_zlabel('Z (km)', fontsize=10)
        self.ax.set_title('Multi-Satellite Orbit Simulation', fontsize=14, fontweight='bold')
        self.ax.set_box_aspect([1, 1, 1])
        self._setup_gui_controls(gs)
        plt.tight_layout()
    
    def _setup_gui_controls(self, gs) -> None:
        ax_speed = self.fig.add_subplot(gs[2, 1])
        self.gui_controls['speed_slider'] = Slider(
            ax_speed, 'Speed', 0.1, 1000, valinit=self.animation_speed, 
            valstep=10, color='lightblue'
        )
        self.gui_controls['speed_slider'].on_changed(self._on_speed_change)
        ax_time = self.fig.add_subplot(gs[3, 1])
        ax_time.axis('off')
        self.gui_controls['time_text'] = ax_time.text(
            0.5, 0.5, '', transform=ax_time.transAxes,
            ha='center', va='center', fontsize=11
        )
        ax_pause = self.fig.add_subplot(gs[4, 0])
        self.gui_controls['pause_button'] = Button(ax_pause, 'Pause/Resume', color='lightgreen')
        self.gui_controls['pause_button'].on_clicked(self._on_pause_click)
        ax_reset = self.fig.add_subplot(gs[4, 1])
        self.gui_controls['reset_button'] = Button(ax_reset, 'Reset', color='lightcoral')
        self.gui_controls['reset_button'].on_clicked(self._on_reset_click)
        ax_frame = self.fig.add_subplot(gs[4, 2])
        self.gui_controls['frame_button'] = Button(
            ax_frame, f'Frame: {self.frame_type}', color='lightyellow'
        )
        self.gui_controls['frame_button'].on_clicked(self._on_frame_click)
        ax_checks = self.fig.add_subplot(gs[0:2, 0])
        ax_checks.axis('off')
        self.gui_controls['check_boxes'] = CheckButtons(
            ax_checks,
            ['Earth', 'Grid', 'Trails', 'Velocity', 'Attitude'],
            [True, True, True, True, False]
        )
        self.gui_controls['check_boxes'].on_clicked(self._on_check_click)
        ax_sats = self.fig.add_subplot(gs[0:2, 2])
        ax_sats.axis('off')
        ax_sats.text(0.5, 0.95, 'Satellites:', transform=ax_sats.transAxes,
                    ha='center', fontsize=11, fontweight='bold')
        self.gui_controls['sat_list'] = ax_sats
    
    def _on_speed_change(self, val) -> None:
        self.animation_speed = val
    
    def _on_pause_click(self, event) -> None:
        self.is_paused = not self.is_paused
        if self.animation:
            if self.is_paused:
                self.animation.pause()
            else:
                self.animation.resume()
    
    def _on_reset_click(self, event) -> None:
        self.current_time = 0
        for sat_data in self.satellites.values():
            sat_data['trail_points'] = []
            sat_data['state'].quaternion = np.array([1, 0, 0, 0])
            sat_data['state'].angular_velocity = np.zeros(3)
    
    def _on_frame_click(self, event) -> None:
        frames = ['ECI', 'ECEF', 'LVLH']
        idx = frames.index(self.frame_type)
        self.frame_type = frames[(idx + 1) % len(frames)]
        self.gui_controls['frame_button'].label.set_text(f'Frame: {self.frame_type}')
    
    def _on_check_click(self, label) -> None:
        if label == 'Earth':
            self.show_earth = not self.show_earth
            if self.earth_surface:
                self.earth_surface.set_visible(self.show_earth)
        elif label == 'Grid':
            self.show_grid = not self.show_grid
            self.ax.grid(self.show_grid)
    
    def plot_earth(self) -> None:
        if self.test_mode or self.ax is None:
            return
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
        axis_length = self.earth_radius/1000 * 1.5
        self.ax.plot([0, 0], [0, 0], [0, axis_length], 'r-', linewidth=2, 
                    alpha=0.7, label='Z-axis')
        self.ax.plot([0, axis_length], [0, 0], [0, 0], 'g-', linewidth=2, 
                    alpha=0.7, label='X-axis')
        self.ax.plot([0, 0], [0, axis_length], [0, 0], 'b-', linewidth=2, 
                    alpha=0.7, label='Y-axis')
        theta = np.linspace(0, 2*np.pi, 100)
        x_eq = self.earth_radius/1000 * np.cos(theta)
        y_eq = self.earth_radius/1000 * np.sin(theta)
        z_eq = np.zeros_like(theta)
        self.ax.plot(x_eq, y_eq, z_eq, 'k-', linewidth=0.5, alpha=0.3)
        phi = np.linspace(-np.pi/2, np.pi/2, 50)
        x_mer = self.earth_radius/1000 * np.cos(phi)
        y_mer = np.zeros_like(phi)
        z_mer = self.earth_radius/1000 * np.sin(phi)
        self.ax.plot(x_mer, y_mer, z_mer, 'k-', linewidth=0.5, alpha=0.3)
    
    def draw_satellite(self, name: str) -> None:
        if self.test_mode or self.ax is None or name not in self.satellites:
            return
        sat_data = self.satellites[name]
        config = sat_data['config']
        state = sat_data['state']
        visuals = sat_data['visual_elements']
        for element in visuals.values():
            if element is not None:
                try:
                    element.remove()
                except:
                    pass
        if self.frame_type == 'ECI':
            pos = state.position
        elif self.frame_type == 'ECEF':
            pos = self.coord_frames.eci_to_ecef(state.position, self.current_time)
        else:
            if len(self.satellites) > 1 and name != list(self.satellites.keys())[0]:
                ref_sat = list(self.satellites.values())[0]
                pos = state.position - ref_sat['state'].position
            else:
                pos = state.position
        pos_km = pos / 1000
        visuals['body'] = self.ax.scatter(
            pos_km[0], pos_km[1], pos_km[2],
            s=config.size, c=config.color, marker='o', alpha=0.9
        )
        if config.show_velocity and self.gui_controls['check_boxes'].get_status()[3]:
            vel_scaled = state.velocity / 1000 * 0.3
            visuals['velocity_vector'] = self.ax.quiver(
                pos_km[0], pos_km[1], pos_km[2],
                vel_scaled[0], vel_scaled[1], vel_scaled[2],
                color='orange', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7
            )
        if config.show_attitude and self.gui_controls['check_boxes'].get_status()[4]:
            R = sat_data['attitude_dynamics'].quaternion_to_rotation_matrix(state.quaternion)
            axis_length = 500
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                axis_dir = R[:, i] * axis_length
                self.ax.quiver(
                    pos_km[0], pos_km[1], pos_km[2],
                    axis_dir[0], axis_dir[1], axis_dir[2],
                    color=color, arrow_length_ratio=0.1, linewidth=1, alpha=0.6
                )
        if len(sat_data['trail_points']) > 1 and self.gui_controls['check_boxes'].get_status()[2]:
            trail_array = np.array(sat_data['trail_points'])
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
        if self.is_paused:
            return
        dt = 10 * self.animation_speed / 100
        self.current_time += dt
        for name in self.satellites:
            self._update_satellite_state(name, self.current_time)
            self.draw_satellite(name)
        elapsed_hours = self.current_time / 3600
        elapsed_days = elapsed_hours / 24
        current_date = self.epoch + timedelta(seconds=self.current_time)
        time_text = (f"Elapsed: {elapsed_hours:.2f} hours ({elapsed_days:.3f} days)\\n"
                    f"Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')} UTC\\n"
                    f"Frame: {self.frame_type}")
        self.gui_controls['time_text'].set_text(time_text)
        self._update_axis_limits()
    
    def _update_axis_limits(self) -> None:
        if not self.satellites:
            return
        all_positions = []
        for sat_data in self.satellites.values():
            if sat_data['state'].position is not None:
                all_positions.append(sat_data['state'].position / 1000)
        if all_positions:
            all_positions = np.array(all_positions)
            max_range = np.max(np.abs(all_positions)) * 1.3
            max_range = max(max_range, self.earth_radius/1000 * 1.5)
            self.ax.set_xlim([-max_range, max_range])
            self.ax.set_ylim([-max_range, max_range])
            self.ax.set_zlim([-max_range, max_range])
    
    def run_animation(self, duration_hours: float = 2.0) -> None:
        if not self.satellites:
            raise ValueError("No satellites added. Use add_satellite() first.")
        self.setup_plot()
        self.plot_earth()
        self.update_satellite_list_display()
        fps = 30
        interval = 1000 / fps
        total_frames = int(duration_hours * 3600 * fps / (self.animation_speed / 100))
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=total_frames,
            interval=interval, blit=False, repeat=True
        )
        plt.show()
    
    def validate_satellite(self, name: str) -> Dict:
        if name not in self.satellites:
            raise ValueError(f"Satellite {name} not found")
        sat_data = self.satellites[name]
        results = {'satellite': name}
        our_pos = sat_data['state'].position
        our_vel = sat_data['state'].velocity
        print(f"\\nValidation for satellite: {name}")
        print(f"  Position: {our_pos/1000} km")
        print(f"  Velocity: {our_vel/1000} km/s")
        tle_str = self.export_to_tle(name, f"temp_{name}.tle")
        validation_results = self.validator.validate_propagation(
            tle_str, self.epoch, self.current_time, our_pos, our_vel
        )
        return validation_results


def main():
    print("="*70)
    print("ENHANCED MULTI-SATELLITE ORBITAL SIMULATION")
    print("="*70)
    sim = EnhancedOrbitSimulation()
    satellites = [
        SatelliteConfig(
            name="ISS",
            color='red',
            size=300,
            orbital_elements={
                'a': 6.78e6,
                'e': 0.0001,
                'i': 51.6,
                'omega': 0,
                'w': 0,
                'M': 0
            }
        ),
        SatelliteConfig(
            name="POLAR-1",
            color='blue',
            size=250,
            orbital_elements={
                'a': 7.2e6,
                'e': 0.001,
                'i': 90,
                'omega': 0,
                'w': 90,
                'M': 45
            }
        ),
        SatelliteConfig(
            name="MOLNIYA-1",
            color='green',
            size=250,
            show_attitude=True,
            orbital_elements={
                'a': 2.66e7,
                'e': 0.74,
                'i': 63.4,
                'omega': 45,
                'w': 270,
                'M': 0
            }
        )
    ]
    for config in satellites:
        sim.add_satellite(config)
    sim.save_configuration("satellite_constellation.json")
    print("\\nSatellites added:")
    for name in sim.satellites:
        print(f"  - {name}")
    print("\\nGUI Controls:")
    print("  - Speed slider: Adjust animation speed")
    print("  - Pause/Resume: Control animation")
    print("  - Reset: Reset to initial state")
    print("  - Frame selector: Switch ECI/ECEF/LVLH")
    print("  - Checkboxes: Toggle display elements")
    print("\\nStarting animation...")
    print("="*70)
    sim.run_animation(duration_hours=2.0)

if __name__ == "__main__":
    main()
