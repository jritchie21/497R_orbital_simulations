"""
COMPLETE ORBITAL SIMULATION PACKAGE
====================================

This file contains all modules combined. For better organization,
you should split this into separate files:

1. enhanced_orbit_simulation.py (lines 30-650)
2. orbital_mechanics.py (lines 652-900)
3. coordinate_frames.py (lines 902-1150)
4. attitude_dynamics.py (lines 1152-1500)
5. tle_handler.py (lines 1502-1750)
6. validation_tools.py (lines 1752-2100)
7. example_usage.py (lines 2102-2400)

Or use the file splitter script at the end of this file.
"""

# ============================================================================
# FILE SPLITTER SCRIPT - Run this to automatically create all files
# ============================================================================

import os

def create_simulation_files():
    """
    Automatically creates all simulation files in the current directory.
    Run this script to split the combined file into individual modules.
    """
    
    files_content = {
        'enhanced_orbit_simulation.py': '''#!/usr/bin/env python3
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
''',

        'orbital_mechanics.py': '''#!/usr/bin/env python3
"""
Orbital Mechanics Module
Handles Keplerian orbital calculations and propagation
"""

import numpy as np
from typing import Dict, Tuple


class OrbitalMechanics:
    def __init__(self):
        self.mu_earth = 3.986004418e14
    
    def keplerian_to_cartesian(self, elements: Dict, mu: float = None) -> Tuple[np.ndarray, np.ndarray]:
        if mu is None:
            mu = self.mu_earth
        a = elements['a']
        e = elements['e']
        i = np.radians(elements['i'])
        omega = np.radians(elements['omega'])
        w = np.radians(elements['w'])
        if 'nu' in elements:
            nu = np.radians(elements['nu'])
        elif 'M' in elements:
            M = np.radians(elements['M'])
            E = self.solve_kepler_equation(M, e)
            nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        else:
            nu = 0
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(nu))
        r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        h = np.sqrt(mu * p)
        v_pf = np.array([
            -(mu / h) * np.sin(nu),
            (mu / h) * (e + np.cos(nu)),
            0
        ])
        cos_o, sin_o = np.cos(omega), np.sin(omega)
        cos_i, sin_i = np.cos(i), np.sin(i)
        cos_w, sin_w = np.cos(w), np.sin(w)
        R = np.array([
            [cos_o*cos_w - sin_o*sin_w*cos_i, -cos_o*sin_w - sin_o*cos_w*cos_i, sin_o*sin_i],
            [sin_o*cos_w + cos_o*sin_w*cos_i, -sin_o*sin_w + cos_o*cos_w*cos_i, -cos_o*sin_i],
            [sin_w*sin_i, cos_w*sin_i, cos_i]
        ])
        return R @ r_pf, R @ v_pf
    
    def solve_kepler_equation(self, M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
        E = M if e < 0.8 else np.pi
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E_new = E - f / f_prime
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        return E
    
    def cartesian_to_keplerian(self, position: np.ndarray, velocity: np.ndarray, 
                              mu: float = None) -> Dict:
        if mu is None:
            mu = self.mu_earth
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h)
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        e_vec = ((v**2 - mu/r) * position - np.dot(position, velocity) * velocity) / mu
        e = np.linalg.norm(e_vec)
        energy = v**2 / 2 - mu / r
        if abs(e - 1) > 1e-10:
            a = -mu / (2 * energy)
        else:
            a = np.inf
        i = np.arccos(h[2] / h_mag)
        if n_mag != 0:
            omega = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0
        if n_mag != 0 and e > 1e-10:
            w = np.arccos(np.dot(n, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                w = 2 * np.pi - w
        else:
            w = 0
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, position) / (e * r))
            if np.dot(position, velocity) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = 0
        E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
        M = E - e * np.sin(E)
        return {
            'a': a,
            'e': e,
            'i': np.degrees(i),
            'omega': np.degrees(omega),
            'w': np.degrees(w),
            'nu': np.degrees(nu),
            'M': np.degrees(M)
        }
    
    def propagate_orbit(self, initial_elements: Dict, dt: float, mu: float = None) -> Dict:
        if mu is None:
            mu = self.mu_earth
        elements = initial_elements.copy()
        a = elements['a']
        n = np.sqrt(mu / a**3)
        M0 = np.radians(elements.get('M', 0))
        M = (M0 + n * dt) % (2 * np.pi)
        elements['M'] = np.degrees(M)
        if 'nu' in elements:
            del elements['nu']
        return elements
    
    def calculate_orbital_period(self, a: float, mu: float = None) -> float:
        if mu is None:
            mu = self.mu_earth
        return 2 * np.pi * np.sqrt(a**3 / mu)
    
    def calculate_orbital_velocity(self, r: float, a: float, mu: float = None) -> float:
        if mu is None:
            mu = self.mu_earth
        return np.sqrt(mu * (2/r - 1/a))
''',

        'coordinate_frames.py': '''#!/usr/bin/env python3
"""
Coordinate Frame Transformations Module
Handles transformations between different coordinate systems
"""

import numpy as np
from typing import Tuple


class CoordinateFrames:
    def __init__(self):
        self.earth_radius = 6378137.0
        self.earth_flattening = 1/298.257223563
        self.omega_earth = 7.2921159e-5
    
    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def eci_to_ecef(self, position_eci: np.ndarray, time: float, 
                    omega_earth: float = None) -> np.ndarray:
        if omega_earth is None:
            omega_earth = self.omega_earth
        theta = omega_earth * time
        R = self.rotation_matrix_z(-theta)
        return R @ position_eci
    
    def ecef_to_eci(self, position_ecef: np.ndarray, time: float, 
                    omega_earth: float = None) -> np.ndarray:
        if omega_earth is None:
            omega_earth = self.omega_earth
        theta = omega_earth * time
        R = self.rotation_matrix_z(theta)
        return R @ position_ecef
    
    @staticmethod
    def eci_to_lvlh(position_eci: np.ndarray, velocity_eci: np.ndarray) -> np.ndarray:
        z_lvlh = -position_eci / np.linalg.norm(position_eci)
        h = np.cross(position_eci, velocity_eci)
        y_lvlh = -h / np.linalg.norm(h)
        x_lvlh = np.cross(y_lvlh, z_lvlh)
        return np.array([x_lvlh, y_lvlh, z_lvlh])
    
    def geodetic_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        a = self.earth_radius
        f = self.earth_flattening
        e2 = 2*f - f**2
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)
        return np.array([x, y, z])
    
    def ecef_to_geodetic(self, position_ecef: np.ndarray) -> Tuple[float, float, float]:
        x, y, z = position_ecef
        a = self.earth_radius
        f = self.earth_flattening
        e2 = 2*f - f**2
        lon = np.arctan2(y, x)
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - e2))
        for _ in range(5):
            N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
            alt = p / np.cos(lat) - N
            lat = np.arctan2(z, p * (1 - e2 * N / (N + alt)))
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        return np.degrees(lat), np.degrees(lon), alt
    
    @staticmethod
    def eci_to_rae(observer_pos: np.ndarray, target_pos: np.ndarray) -> Tuple[float, float, float]:
        rel_pos = target_pos - observer_pos
        range_val = np.linalg.norm(rel_pos)
        east = np.array([0, 1, 0])
        north = np.array([-1, 0, 0])
        up = np.array([0, 0, 1])
        e = np.dot(rel_pos, east)
        n = np.dot(rel_pos, north)
        u = np.dot(rel_pos, up)
        azimuth = np.degrees(np.arctan2(e, n)) % 360
        elevation = np.degrees(np.arcsin(u / range_val))
        return range_val, azimuth, elevation
    
    @staticmethod
    def hill_equations(state: np.ndarray, n: float) -> np.ndarray:
        x, y, z, vx, vy, vz = state
        ax = 3 * n**2 * x + 2 * n * vy
        ay = -2 * n * vx
        az = -n**2 * z
        return np.array([vx, vy, vz, ax, ay, az])
    
    @staticmethod
    def perifocal_to_eci(r_pf: np.ndarray, v_pf: np.ndarray,
                        omega: float, i: float, w: float) -> Tuple[np.ndarray, np.ndarray]:
        R3_omega = CoordinateFrames.rotation_matrix_z(omega)
        R1_i = CoordinateFrames.rotation_matrix_x(i)
        R3_w = CoordinateFrames.rotation_matrix_z(w)
        R = R3_omega @ R1_i @ R3_w
        r_eci = R @ r_pf
        v_eci = R @ v_pf
        return r_eci, v_eci
''',

        'attitude_dynamics.py': '''#!/usr/bin/env python3
"""
Attitude Dynamics Module
Handles satellite attitude propagation and control
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SatelliteState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    time: float = 0.0


class AttitudeDynamics:
    def __init__(self, inertia_tensor: Optional[np.ndarray] = None):
        if inertia_tensor is None:
            self.inertia = np.diag([100, 100, 50])
        else:
            self.inertia = inertia_tensor
        self.inertia_inv = np.linalg.inv(self.inertia)
    
    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def quaternion_derivative(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        return q_dot
    
    def euler_equations(self, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
        L = self.inertia @ omega
        gyro_torque = np.cross(omega, L)
        omega_dot = self.inertia_inv @ (torque - gyro_torque)
        return omega_dot
    
    def gravity_gradient_torque(self, q: np.ndarray, position: np.ndarray, 
                               mu: float) -> np.ndarray:
        r = np.linalg.norm(position)
        r_hat_inertial = position / r
        R = self.quaternion_to_rotation_matrix(q)
        r_hat_body = R.T @ r_hat_inertial
        torque = (3 * mu / r**3) * np.cross(r_hat_body, self.inertia @ r_hat_body)
        return torque
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        return R
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])
    
    @staticmethod
    def euler_angles_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])
    
    @staticmethod
    def quaternion_to_euler_angles(q: np.ndarray) -> Tuple[float, float, float]:
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
    
    def propagate_attitude(self, state: SatelliteState, torque: np.ndarray, 
                          dt: float) -> SatelliteState:
        q0 = state.quaternion
        omega0 = state.angular_velocity
        q_dot1 = self.quaternion_derivative(q0, omega0)
        omega_dot1 = self.euler_equations(omega0, torque)
        q2 = q0 + 0.5 * dt * q_dot1
        q2 = q2 / np.linalg.norm(q2)
        omega2 = omega0 + 0.5 * dt * omega_dot1
        q_dot2 = self.quaternion_derivative(q2, omega2)
        omega_dot2 = self.euler_equations(omega2, torque)
        q3 = q0 + 0.5 * dt * q_dot2
        q3 = q3 / np.linalg.norm(q3)
        omega3 = omega0 + 0.5 * dt * omega_dot2
        q_dot3 = self.quaternion_derivative(q3, omega3)
        omega_dot3 = self.euler_equations(omega3, torque)
        q4 = q0 + dt * q_dot3
        q4 = q4 / np.linalg.norm(q4)
        omega4 = omega0 + dt * omega_dot3
        q_dot4 = self.quaternion_derivative(q4, omega4)
        omega_dot4 = self.euler_equations(omega4, torque)
        state.quaternion = q0 + (dt / 6) * (q_dot1 + 2*q_dot2 + 2*q_dot3 + q_dot4)
        state.quaternion = state.quaternion / np.linalg.norm(state.quaternion)
        state.angular_velocity = omega0 + (dt / 6) * (omega_dot1 + 2*omega_dot2 + 
                                                       2*omega_dot3 + omega_dot4)
        return state
    
    def nadir_pointing_control(self, q: np.ndarray, omega: np.ndarray, 
                               position: np.ndarray, velocity: np.ndarray,
                               kp: float = 0.01, kd: float = 0.1) -> np.ndarray:
        z_desired = -position / np.linalg.norm(position)
        h = np.cross(position, velocity)
        y_desired = -h / np.linalg.norm(h)
        x_desired = np.cross(y_desired, z_desired)
        R_desired = np.column_stack([x_desired, y_desired, z_desired]).T
        R_current = self.quaternion_to_rotation_matrix(q)
        R_error = R_desired @ R_current.T
        angle = np.arccos((np.trace(R_error) - 1) / 2)
        if abs(angle) > 1e-6:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            error_vector = angle * axis
        else:
            error_vector = np.zeros(3)
        control_torque = -kp * error_vector - kd * omega
        return control_torque
    
    def reaction_wheel_dynamics(self, wheel_speeds: np.ndarray, 
                               commanded_torque: np.ndarray,
                               wheel_inertias: np.ndarray,
                               max_wheel_speed: float = 6000.0,
                               max_wheel_torque: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        wheel_speeds_rad = wheel_speeds * 2 * np.pi / 60
        commanded_torque = np.clip(commanded_torque, -max_wheel_torque, max_wheel_torque)
        max_speed_rad = max_wheel_speed * 2 * np.pi / 60
        saturated = np.abs(wheel_speeds_rad) > max_speed_rad * 0.95
        wheel_torques = commanded_torque.copy()
        wheel_torques[saturated] *= 0.1
        wheel_accelerations = wheel_torques / wheel_inertias
        wheel_accelerations_rpm = wheel_accelerations * 60 / (2 * np.pi)
        return wheel_torques, wheel_accelerations_rpm
''',

        'tle_handler.py': '''#!/usr/bin/env python3
"""
TLE Handler Module
Handles Two-Line Element set operations and conversions
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class TLEHandler:
    @staticmethod
    def keplerian_to_tle(name: str, orbital_elements: Dict, 
                         epoch: datetime = None, 
                         bstar: float = 0.0,
                         sat_num: int = 99999,
                         classification: str = 'U',
                         launch_year: int = None,
                         launch_num: int = 1,
                         piece: str = 'A') -> str:
        if epoch is None:
            epoch = datetime.utcnow()
        if launch_year is None:
            launch_year = epoch.year
        a = orbital_elements['a'] / 1000
        e = orbital_elements['e']
        i = orbital_elements['i']
        omega = orbital_elements['omega']
        w = orbital_elements['w']
        M = orbital_elements.get('M', 0)
        mu_earth = 398600.4418
        n = 86400 / (2 * np.pi * np.sqrt(a**3 / mu_earth))
        year = epoch.year % 100
        day_of_year = epoch.timetuple().tm_yday
        fraction_of_day = (epoch.hour * 3600 + epoch.minute * 60 + 
                          epoch.second + epoch.microsecond / 1e6) / 86400
        line0 = f"{name[:24]:24s}"
        line1 = "1 "
        line1 += f"{sat_num:05d}{classification} "
        line1 += f"{launch_year % 100:02d}{launch_num:03d}{piece:3s} "
        line1 += f"{year:02d}{day_of_year:03d}.{int(fraction_of_day * 100000000):08d} "
        line1 += " .00000000 "
        line1 += " 00000-0 "
        if bstar == 0:
            line1 += " 00000-0"
        else:
            exp = int(np.floor(np.log10(abs(bstar))))
            mantissa = bstar / (10 ** exp)
            sign = '+' if bstar >= 0 else '-'
            line1 += f" {sign}{abs(mantissa):.5f}".replace('.', '')[0:7]
            line1 += f"{exp:+d}"[-2:]
        line1 += " 0"
        line1 += "  999"
        line2 = "2 "
        line2 += f"{sat_num:05d} "
        line2 += f"{i:8.4f} "
        line2 += f"{omega:8.4f} "
        line2 += f"{int(e * 10000000):07d} "
        line2 += f"{w:8.4f} "
        line2 += f"{M:8.4f} "
        line2 += f"{n:11.8f}"
        line2 += "    1"
        line1 = TLEHandler._add_checksum(line1)
        line2 = TLEHandler._add_checksum(line2)
        return f"{line0}\\n{line1}\\n{line2}"
    
    @staticmethod
    def _add_checksum(line: str) -> str:
        checksum = 0
        for char in line:
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        return line + str(checksum % 10)
    
    @staticmethod
    def tle_to_keplerian(tle_lines: List[str]) -> Dict:
        if len(tle_lines) < 2:
            raise ValueError("Need at least 2 TLE lines")
        if len(tle_lines) == 3:
            line1, line2 = tle_lines[1], tle_lines[2]
            name = tle_lines[0].strip()
        else:
            line1, line2 = tle_lines[0], tle_lines[1]
            name = "UNKNOWN"
        if not line1.startswith('1 ') or not line2.startswith('2 '):
            raise ValueError("Invalid TLE format")
        sat_num = int(line1[2:7])
        classification = line1[7]
        launch_year = int(line1[9:11])
        launch_num = int(line1[11:14])
        piece = line1[14:17].strip()
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        bstar_str = line1[53:61]
        if bstar_str.strip() == '00000-0':
            bstar = 0.0
        else:
            mantissa = float(bstar_str[0:6]) / 100000
            exp = int(bstar_str[6:8])
            bstar = mantissa * (10 ** exp)
        i = float(line2[8:16])
        omega = float(line2[17:25])
        e_str = line2[26:33]
        e = float('0.' + e_str)
        w = float(line2[34:42])
        M = float(line2[43:51])
        n = float(line2[52:63])
        rev_num = int(line2[63:68])
        mu_earth = 398600.4418
        n_rad_per_sec = n * 2 * np.pi / 86400
        a = (mu_earth / n_rad_per_sec**2)**(1/3) * 1000
        if epoch_year < 57:
            full_year = 2000 + epoch_year
        else:
            full_year = 1900 + epoch_year
        elements = {
            'name': name,
            'sat_num': sat_num,
            'a': a,
            'e': e,
            'i': i,
            'omega': omega,
            'w': w,
            'M': M,
            'n': n,
            'bstar': bstar,
            'epoch_year': full_year,
            'epoch_day': epoch_day,
            'rev_num': rev_num
        }
        return elements
    
    @staticmethod
    def validate_tle(tle_lines: List[str]) -> bool:
        if len(tle_lines) < 2:
            return False
        if len(tle_lines) == 3:
            line1, line2 = tle_lines[1], tle_lines[2]
        else:
            line1, line2 = tle_lines[0], tle_lines[1]
        if len(line1) != 69 or len(line2) != 69:
            return False
        if not line1.startswith('1 ') or not line2.startswith('2 '):
            return False
        for line in [line1, line2]:
            checksum = 0
            for char in line[:-1]:
                if char.isdigit():
                    checksum += int(char)
                elif char == '-':
                    checksum += 1
            if str(checksum % 10) != line[-1]:
                return False
        return True
    
    @staticmethod
    def read_tle_file(filename: str) -> List[Dict]:
        satellites = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            if i + 2 < len(lines) and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
                tle_lines = [lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()]
                i += 3
            elif i + 1 < len(lines) and lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
                tle_lines = [lines[i].strip(), lines[i+1].strip()]
                i += 2
            else:
                i += 1
                continue
            if TLEHandler.validate_tle(tle_lines):
                try:
                    elements = TLEHandler.tle_to_keplerian(tle_lines)
                    satellites.append(elements)
                except Exception as e:
                    print(f"Error parsing TLE: {e}")
        return satellites
''',

        'validation_tools.py': '''#!/usr/bin/env python3
"""
Validation Tools Module
Tools for validating orbital calculations against standard libraries
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import warnings

try:
    from sgp4.api import Satrec, jday
    from sgp4.api import SGP4_ERRORS
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    warnings.warn("sgp4 not installed. Install with 'pip install sgp4' for validation")

try:
    from skyfield.api import load, EarthSatellite
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False
    warnings.warn("skyfield not installed. Install with 'pip install skyfield' for validation")


class ValidationTools:
    def __init__(self):
        self.sgp4_available = SGP4_AVAILABLE
        self.skyfield_available = SKYFIELD_AVAILABLE
        self.mu_earth = 3.986004418e14
        self.earth_radius = 6.371e6
    
    def validate_with_sgp4(self, tle_lines: List[str], epoch: datetime, 
                           minutes_after: float = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.sgp4_available:
            print("SGP4 not available for validation. Install with: pip install sgp4")
            return None
        try:
            if len(tle_lines) == 3:
                line1, line2 = tle_lines[1], tle_lines[2]
            else:
                line1, line2 = tle_lines[0], tle_lines[1]
            satellite = Satrec.twoline2rv(line1, line2)
            propagation_time = epoch + timedelta(minutes=minutes_after)
            jd, fr = jday(propagation_time.year, propagation_time.month, 
                         propagation_time.day, propagation_time.hour, 
                         propagation_time.minute, 
                         propagation_time.second + propagation_time.microsecond/1e6)
            error, position, velocity = satellite.sgp4(jd, fr)
            if error == 0:
                return np.array(position) * 1000, np.array(velocity) * 1000
            else:
                error_msg = SGP4_ERRORS.get(error, f"Unknown error {error}")
                print(f"SGP4 propagation error: {error_msg}")
                return None
        except Exception as e:
            print(f"Error in SGP4 validation: {e}")
            return None
    
    def validate_with_skyfield(self, tle_lines: List[str], 
                               epoch: datetime) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.skyfield_available:
            print("Skyfield not available for validation. Install with: pip install skyfield")
            return None
        try:
            ts = load.timescale()
            if len(tle_lines) == 3:
                satellite = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)
            else:
                satellite = EarthSatellite(tle_lines[0], tle_lines[1], "SATELLITE", ts)
            t = ts.utc(epoch.year, epoch.month, epoch.day, 
                      epoch.hour, epoch.minute, 
                      epoch.second + epoch.microsecond/1e6)
            geocentric = satellite.at(t)
            position = geocentric.position.m
            velocity = geocentric.velocity.m_per_s
            return position, velocity
        except Exception as e:
            print(f"Error in Skyfield validation: {e}")
            return None
    
    def compare_results(self, pos1: np.ndarray, vel1: np.ndarray,
                       pos2: np.ndarray, vel2: np.ndarray,
                       label1: str = "Method 1", 
                       label2: str = "Method 2") -> Dict:
        pos_diff = np.linalg.norm(pos1 - pos2)
        vel_diff = np.linalg.norm(vel1 - vel2)
        pos_mag1 = np.linalg.norm(pos1)
        vel_mag1 = np.linalg.norm(vel1)
        pos_error_pct = (pos_diff / pos_mag1) * 100 if pos_mag1 > 0 else 0
        vel_error_pct = (vel_diff / vel_mag1) * 100 if vel_mag1 > 0 else 0
        pos_diff_components = pos1 - pos2
        vel_diff_components = vel1 - vel2
        results = {
            'position_difference_m': pos_diff,
            'velocity_difference_m_s': vel_diff,
            'position_error_percent': pos_error_pct,
            'velocity_error_percent': vel_error_pct,
            'position_diff_components': pos_diff_components,
            'velocity_diff_components': vel_diff_components,
            'position_1': pos1,
            'velocity_1': vel1,
            'position_2': pos2,
            'velocity_2': vel2
        }
        print(f"\\nComparison: {label1} vs {label2}")
        print("-" * 50)
        print(f"Position difference: {pos_diff:.2f} m ({pos_error_pct:.4f}%)")
        print(f"  Components (m): [{pos_diff_components[0]:.2f}, "
              f"{pos_diff_components[1]:.2f}, {pos_diff_components[2]:.2f}]")
        print(f"Velocity difference: {vel_diff:.4f} m/s ({vel_error_pct:.4f}%)")
        print(f"  Components (m/s): [{vel_diff_components[0]:.4f}, "
              f"{vel_diff_components[1]:.4f}, {vel_diff_components[2]:.4f}]")
        return results
    
    def validate_orbital_elements(self, elements: Dict) -> Dict:
        results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        a = elements.get('a', 0)
        if a <= 0:
            results['valid'] = False
            results['errors'].append(f"Invalid semi-major axis: {a} m (must be positive)")
        elif a < self.earth_radius:
            results['valid'] = False
            results['errors'].append(f"Semi-major axis {a/1000:.1f} km is less than Earth radius")
        elif a < self.earth_radius + 100000:
            results['warnings'].append(f"Very low orbit: altitude = {(a-self.earth_radius)/1000:.1f} km")
        elif a > 1e9:
            results['warnings'].append(f"Very high orbit: a = {a/1e6:.1f} Mm")
        e = elements.get('e', 0)
        if e < 0:
            results['valid'] = False
            results['errors'].append(f"Invalid eccentricity: {e} (must be non-negative)")
        elif e >= 1:
            results['valid'] = False
            results['errors'].append(f"Hyperbolic/parabolic orbit: e = {e} (only elliptical supported)")
        elif e > 0.9:
            results['warnings'].append(f"Highly eccentric orbit: e = {e}")
        if a > 0 and 0 <= e < 1:
            periapsis = a * (1 - e)
            periapsis_alt = periapsis - self.earth_radius
            if periapsis_alt < 0:
                results['valid'] = False
                results['errors'].append(f"Periapsis below Earth surface: {periapsis_alt/1000:.1f} km")
            elif periapsis_alt < 100000:
                results['warnings'].append(f"Low periapsis altitude: {periapsis_alt/1000:.1f} km")
        i = elements.get('i', 0)
        if i < 0 or i > 180:
            results['valid'] = False
            results['errors'].append(f"Invalid inclination: {i}° (must be 0-180°)")
        for angle_name in ['omega', 'w', 'M', 'nu']:
            if angle_name in elements:
                angle = elements[angle_name]
                if not 0 <= angle < 360:
                    results['warnings'].append(f"{angle_name} = {angle}° is outside [0, 360)")
        if results['valid']:
            print("✅ Orbital elements are valid")
        else:
            print("❌ Invalid orbital elements detected:")
            for error in results['errors']:
                print(f"   ERROR: {error}")
        if results['warnings']:
            print("⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"   WARNING: {warning}")
        return results
    
    def validate_conservation_laws(self, position: np.ndarray, velocity: np.ndarray,
                                  elements: Dict, tolerance: float = 0.01) -> Dict:
        results = {
            'energy_valid': False,
            'angular_momentum_valid': False,
            'errors': []
        }
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        actual_energy = v**2 / 2 - self.mu_earth / r
        a = elements.get('a')
        if a and a > 0:
            expected_energy = -self.mu_earth / (2 * a)
            energy_error = abs((actual_energy - expected_energy) / expected_energy)
            results['energy_valid'] = energy_error < tolerance
            results['energy_error'] = energy_error
            results['actual_energy'] = actual_energy
            results['expected_energy'] = expected_energy
            if not results['energy_valid']:
                results['errors'].append(
                    f"Energy conservation violated: error = {energy_error*100:.2f}%"
                )
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h)
        e = elements.get('e', 0)
        if a and a > 0 and 0 <= e < 1:
            p = a * (1 - e**2)
            expected_h = np.sqrt(self.mu_earth * p)
            h_error = abs((h_mag - expected_h) / expected_h)
            results['angular_momentum_valid'] = h_error < tolerance
            results['h_error'] = h_error
            results['actual_h'] = h_mag
            results['expected_h'] = expected_h
            if not results['angular_momentum_valid']:
                results['errors'].append(
                    f"Angular momentum conservation violated: error = {h_error*100:.2f}%"
                )
        return results
    
    def validate_propagation(self, tle_str: str, epoch: datetime, 
                           current_time: float, our_pos: np.ndarray, 
                           our_vel: np.ndarray) -> Dict:
        results = {
            'our_position': our_pos,
            'our_velocity': our_vel,
            'comparisons': []
        }
        tle_lines = tle_str.strip().split('\\n')
        minutes_after = current_time / 60
        if self.sgp4_available:
            sgp4_result = self.validate_with_sgp4(tle_lines, epoch, minutes_after)
            if sgp4_result:
                sgp4_pos, sgp4_vel = sgp4_result
                comparison = self.compare_results(
                    our_pos, our_vel, sgp4_pos, sgp4_vel,
                    "Our Propagator", "SGP4"
                )
                comparison['method'] = 'SGP4'
                results['comparisons'].append(comparison)
        if self.skyfield_available:
            current_epoch = epoch + timedelta(seconds=current_time)
            skyfield_result = self.validate_with_skyfield(tle_lines, current_epoch)
            if skyfield_result:
                sky_pos, sky_vel = skyfield_result
                comparison = self.compare_results(
                    our_pos, our_vel, sky_pos, sky_vel,
                    "Our Propagator", "Skyfield"
                )
                comparison['method'] = 'Skyfield'
                results['comparisons'].append(comparison)
        if results['comparisons']:
            print("\\n" + "="*50)
            print("VALIDATION SUMMARY")
            print("="*50)
            avg_pos_error = np.mean([c['position_error_percent'] 
                                    for c in results['comparisons']])
            avg_vel_error = np.mean([c['velocity_error_percent'] 
                                    for c in results['comparisons']])
            print(f"Average position error: {avg_pos_error:.4f}%")
            print(f"Average velocity error: {avg_vel_error:.4f}%")
            if avg_pos_error < 1.0 and avg_vel_error < 1.0:
                print("✅ Excellent agreement with standard propagators")
            elif avg_pos_error < 5.0 and avg_vel_error < 5.0:
                print("✅ Good agreement with standard propagators")
            else:
                print("⚠️  Significant differences from standard propagators")
                print("   Note: This may be due to perturbation modeling differences")
        else:
            print("\\n⚠️  No validation libraries available")
            print("   Install sgp4 or skyfield for validation")
        return results
    
    def benchmark_performance(self, n_satellites: int = 10, 
                            duration_seconds: float = 3600) -> Dict:
        import time
        print(f"\\nBenchmarking with {n_satellites} satellites for {duration_seconds}s...")
        test_elements = []
        for i in range(n_satellites):
            a = 6.7e6 + i * 1e5
            e = 0.001 * (i + 1)
            i_deg = i * 10
            elements = {
                'a': a,
                'e': min(e, 0.9),
                'i': i_deg % 180,
                'omega': (i * 30) % 360,
                'w': (i * 45) % 360,
                'M': (i * 60) % 360
            }
            test_elements.append(elements)
        from orbital_mechanics import OrbitalMechanics
        om = OrbitalMechanics()
        start_time = time.time()
        n_steps = 100
        dt = duration_seconds / n_steps
        for step in range(n_steps):
            current_t = step * dt
            for elements in test_elements:
                propagated = om.propagate_orbit(elements, current_t)
                pos, vel = om.keplerian_to_cartesian(propagated)
        end_time = time.time()
        elapsed = end_time - start_time
        total_propagations = n_satellites * n_steps
        time_per_propagation = elapsed / total_propagations * 1000
        propagations_per_second = total_propagations / elapsed
        results = {
            'n_satellites': n_satellites,
            'n_steps': n_steps,
            'total_propagations': total_propagations,
            'elapsed_time': elapsed,
            'time_per_propagation_ms': time_per_propagation,
            'propagations_per_second': propagations_per_second
        }
        print(f"Performance Results:")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Propagations: {total_propagations}")
        print(f"  Time per propagation: {time_per_propagation:.3f} ms")
        print(f"  Propagations per second: {propagations_per_second:.0f}")
        if propagations_per_second > 1000:
            print("  ✅ Excellent performance")
        elif propagations_per_second > 100:
            print("  ✅ Good performance")
        else:
            print("  ⚠️  Performance may be slow for real-time animation")
        return results
'''
    }
    
    print("Creating orbital simulation files...")
    print("="*70)
    
    for filename, content in files_content.items():
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✅ Created: {filename}")
    
    print("="*70)
    print("All files created successfully!")
    print("\nTo run the simulation:")
    print("  python enhanced_orbit_simulation.py")
    print("\nRequired packages:")
    print("  pip install numpy matplotlib")
    print("\nOptional packages for validation:")
    print("  pip install sgp4 skyfield")

if __name__ == "__main__":
    create_simulation_files()
