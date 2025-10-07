#!/usr/bin/env python3
"""
Enhanced Multi-Satellite Orbital Simulation with Interactive Interface
Adds satellite management, orbital element input, and satellite selection features
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons, RadioButtons
import time
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Import the base simulation
from orbital_sim_improved_claude import (
    SatelliteState, SatelliteConfig, CoordinateFrames, AttitudeDynamics,
    TLEHandler, ValidationTools, EnhancedOrbitSimulation
)

class SatelliteManager:
    """Manages satellite addition, removal, and configuration"""
    
    def __init__(self, simulation):
        self.sim = simulation
        self.satellite_window = None
        self.selected_satellite = None
    
    def keplerian_to_modified_equinoctial(self, a, e, i, omega, w, nu):
        """Convert Keplerian elements to Modified Equinoctial elements"""
        # Convert angles to radians
        i_rad = np.radians(i)
        omega_rad = np.radians(omega)
        w_rad = np.radians(w)
        nu_rad = np.radians(nu)
        
        # Calculate semi-latus rectum
        p = a * (1 - e**2)
        
        # Calculate f and g parameters
        f = e * np.cos(w_rad + omega_rad)
        g = e * np.sin(w_rad + omega_rad)
        
        # Calculate h and k parameters
        h = np.tan(i_rad / 2) * np.cos(omega_rad)
        k = np.tan(i_rad / 2) * np.sin(omega_rad)
        
        # Calculate true longitude
        L = omega_rad + w_rad + nu_rad
        L = np.degrees(L) % 360
        
        return p, f, g, h, k, L
    
    def modified_equinoctial_to_keplerian(self, p, f, g, h, k, L):
        """Convert Modified Equinoctial elements to Keplerian elements"""
        # Calculate eccentricity
        e = np.sqrt(f**2 + g**2)
        
        # Calculate semi-major axis
        a = p / (1 - e**2) if e < 1 else p / (1 - e**2)
        
        # Calculate inclination
        i = 2 * np.arctan(np.sqrt(h**2 + k**2))
        i = np.degrees(i)
        
        # Calculate longitude of ascending node
        omega = np.arctan2(k, h)
        omega = np.degrees(omega) % 360
        
        # Calculate argument of periapsis
        w = np.arctan2(g, f) - omega
        w = np.degrees(w) % 360
        
        # Calculate true anomaly
        nu = L - omega - w
        nu = nu % 360
        
        return a, e, i, omega, w, nu
        
    def open_satellite_manager(self):
        """Open the satellite management window"""
        if self.satellite_window is not None:
            self.satellite_window.destroy()
        
        self.satellite_window = tk.Toplevel()
        self.satellite_window.title("Satellite Manager")
        self.satellite_window.geometry("800x600")
        self.satellite_window.configure(bg='#f0f0f0')
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.satellite_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add satellite tab
        self._create_add_satellite_tab(notebook)
        
        # Manage satellites tab
        self._create_manage_satellites_tab(notebook)
        
        # Satellite details tab
        self._create_satellite_details_tab(notebook)
        
    def _create_add_satellite_tab(self, notebook):
        """Create the add satellite tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Add Satellite")
        
        # Title
        title_label = tk.Label(frame, text="Add New Satellite", font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Satellite name
        name_frame = tk.Frame(frame, bg='#f0f0f0')
        name_frame.pack(fill='x', padx=20, pady=5)
        tk.Label(name_frame, text="Satellite Name:", bg='#f0f0f0').pack(side='left')
        self.name_entry = tk.Entry(name_frame, width=20)
        self.name_entry.pack(side='left', padx=10)
        
        # Color selection
        color_frame = tk.Frame(frame, bg='#f0f0f0')
        color_frame.pack(fill='x', padx=20, pady=5)
        tk.Label(color_frame, text="Color:", bg='#f0f0f0').pack(side='left')
        self.color_var = tk.StringVar(value='red')
        color_combo = ttk.Combobox(color_frame, textvariable=self.color_var, 
                                  values=['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta'])
        color_combo.pack(side='left', padx=10)
        
        # Size
        size_frame = tk.Frame(frame, bg='#f0f0f0')
        size_frame.pack(fill='x', padx=20, pady=5)
        tk.Label(size_frame, text="Size:", bg='#f0f0f0').pack(side='left')
        self.size_var = tk.DoubleVar(value=300.0)
        size_scale = tk.Scale(size_frame, from_=50, to=1000, orient='horizontal', 
                             variable=self.size_var, bg='#f0f0f0')
        size_scale.pack(side='left', padx=10)
        
        # Orbital elements type
        elements_frame = tk.Frame(frame, bg='#f0f0f0')
        elements_frame.pack(fill='x', padx=20, pady=10)
        tk.Label(elements_frame, text="Orbital Elements Type:", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack()
        
        self.elements_type = tk.StringVar(value='keplerian')
        keplerian_radio = tk.Radiobutton(elements_frame, text="Keplerian Elements", 
                                        variable=self.elements_type, value='keplerian', bg='#f0f0f0')
        keplerian_radio.pack(anchor='w', padx=20)
        equinoctial_radio = tk.Radiobutton(elements_frame, text="Modified Equinoctial Elements", 
                                          variable=self.elements_type, value='equinoctial', bg='#f0f0f0')
        equinoctial_radio.pack(anchor='w', padx=20)
        
        # Orbital elements input
        self.elements_frame = tk.Frame(frame, bg='#f0f0f0')
        self.elements_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create input fields
        self._create_orbital_elements_inputs()
        
        # Add button
        add_button = tk.Button(frame, text="Add Satellite", command=self._add_satellite, 
                              bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'))
        add_button.pack(pady=20)
        
        # Bind radio button change
        keplerian_radio.configure(command=self._update_elements_inputs)
        equinoctial_radio.configure(command=self._update_elements_inputs)
        
    def _create_orbital_elements_inputs(self):
        """Create orbital elements input fields"""
        # Clear existing widgets
        for widget in self.elements_frame.winfo_children():
            widget.destroy()
        
        if self.elements_type.get() == 'keplerian':
            self._create_keplerian_inputs()
        else:
            self._create_equinoctial_inputs()
    
    def _create_keplerian_inputs(self):
        """Create Keplerian elements input fields"""
        # Semi-major axis
        a_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        a_frame.pack(fill='x', pady=2)
        tk.Label(a_frame, text="Semi-major axis (a) [km]:", bg='#f0f0f0').pack(side='left')
        self.a_entry = tk.Entry(a_frame, width=15)
        self.a_entry.insert(0, "6780")  # Default ISS altitude
        self.a_entry.pack(side='right', padx=10)
        
        # Eccentricity
        e_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        e_frame.pack(fill='x', pady=2)
        tk.Label(e_frame, text="Eccentricity (e):", bg='#f0f0f0').pack(side='left')
        self.e_entry = tk.Entry(e_frame, width=15)
        self.e_entry.insert(0, "0.0001")
        self.e_entry.pack(side='right', padx=10)
        
        # Inclination
        i_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        i_frame.pack(fill='x', pady=2)
        tk.Label(i_frame, text="Inclination (i) [deg]:", bg='#f0f0f0').pack(side='left')
        self.i_entry = tk.Entry(i_frame, width=15)
        self.i_entry.insert(0, "51.6")
        self.i_entry.pack(side='right', padx=10)
        
        # Longitude of ascending node
        omega_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        omega_frame.pack(fill='x', pady=2)
        tk.Label(omega_frame, text="RAAN (Ω) [deg]:", bg='#f0f0f0').pack(side='left')
        self.omega_entry = tk.Entry(omega_frame, width=15)
        self.omega_entry.insert(0, "0")
        self.omega_entry.pack(side='right', padx=10)
        
        # Argument of periapsis
        w_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        w_frame.pack(fill='x', pady=2)
        tk.Label(w_frame, text="Argument of periapsis (ω) [deg]:", bg='#f0f0f0').pack(side='left')
        self.w_entry = tk.Entry(w_frame, width=15)
        self.w_entry.insert(0, "0")
        self.w_entry.pack(side='right', padx=10)
        
        # Mean anomaly
        M_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        M_frame.pack(fill='x', pady=2)
        tk.Label(M_frame, text="Mean anomaly (M) [deg]:", bg='#f0f0f0').pack(side='left')
        self.M_entry = tk.Entry(M_frame, width=15)
        self.M_entry.insert(0, "0")
        self.M_entry.pack(side='right', padx=10)
    
    def _create_equinoctial_inputs(self):
        """Create Modified Equinoctial elements input fields"""
        # Semi-latus rectum
        p_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        p_frame.pack(fill='x', pady=2)
        tk.Label(p_frame, text="Semi-latus rectum (p) [km]:", bg='#f0f0f0').pack(side='left')
        self.p_entry = tk.Entry(p_frame, width=15)
        self.p_entry.insert(0, "6712")
        self.p_entry.pack(side='right', padx=10)
        
        # f parameter
        f_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        f_frame.pack(fill='x', pady=2)
        tk.Label(f_frame, text="f = e*cos(ω+Ω):", bg='#f0f0f0').pack(side='left')
        self.f_entry = tk.Entry(f_frame, width=15)
        self.f_entry.insert(0, "0.0")
        self.f_entry.pack(side='right', padx=10)
        
        # g parameter
        g_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        g_frame.pack(fill='x', pady=2)
        tk.Label(g_frame, text="g = e*sin(ω+Ω):", bg='#f0f0f0').pack(side='left')
        self.g_entry = tk.Entry(g_frame, width=15)
        self.g_entry.insert(0, "0.0")
        self.g_entry.pack(side='right', padx=10)
        
        # h parameter
        h_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        h_frame.pack(fill='x', pady=2)
        tk.Label(h_frame, text="h = tan(i/2)*cos(Ω):", bg='#f0f0f0').pack(side='left')
        self.h_entry = tk.Entry(h_frame, width=15)
        self.h_entry.insert(0, "0.0")
        self.h_entry.pack(side='right', padx=10)
        
        # k parameter
        k_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        k_frame.pack(fill='x', pady=2)
        tk.Label(k_frame, text="k = tan(i/2)*sin(Ω):", bg='#f0f0f0').pack(side='left')
        self.k_entry = tk.Entry(k_frame, width=15)
        self.k_entry.insert(0, "0.0")
        self.k_entry.pack(side='right', padx=10)
        
        # True longitude
        L_frame = tk.Frame(self.elements_frame, bg='#f0f0f0')
        L_frame.pack(fill='x', pady=2)
        tk.Label(L_frame, text="True longitude (L) [deg]:", bg='#f0f0f0').pack(side='left')
        self.L_entry = tk.Entry(L_frame, width=15)
        self.L_entry.insert(0, "0")
        self.L_entry.pack(side='right', padx=10)
    
    def _update_elements_inputs(self):
        """Update orbital elements input fields when type changes"""
        self._create_orbital_elements_inputs()
    
    def _add_satellite(self):
        """Add satellite with entered parameters"""
        try:
            name = self.name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a satellite name")
                return
            
            if name in self.sim.satellites:
                messagebox.showerror("Error", f"Satellite '{name}' already exists")
                return
            
            color = self.color_var.get()
            size = self.size_var.get()
            
            # Get orbital elements
            if self.elements_type.get() == 'keplerian':
                elements = self._get_keplerian_elements()
            else:
                elements = self._get_equinoctial_elements()
            
            # Create satellite config
            config = SatelliteConfig(
                name=name,
                color=color,
                size=size,
                orbital_elements=elements
            )
            
            # Initialize visibility attribute
            config.visible = True
            
            # Add to simulation
            self.sim.add_satellite(config)
            
            # Update satellite list
            self._update_satellite_list()
            
            messagebox.showinfo("Success", f"Satellite '{name}' added successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add satellite: {str(e)}")
    
    def _get_keplerian_elements(self):
        """Get Keplerian elements from input fields"""
        return {
            'a': float(self.a_entry.get()) * 1000,  # Convert km to m
            'e': float(self.e_entry.get()),
            'i': float(self.i_entry.get()),
            'omega': float(self.omega_entry.get()),
            'w': float(self.w_entry.get()),
            'M': float(self.M_entry.get())
        }
    
    def _get_equinoctial_elements(self):
        """Get Modified Equinoctial elements from input fields"""
        return {
            'p': float(self.p_entry.get()) * 1000,  # Convert km to m
            'f': float(self.f_entry.get()),
            'g': float(self.g_entry.get()),
            'h': float(self.h_entry.get()),
            'k': float(self.k_entry.get()),
            'L': float(self.L_entry.get())
        }
    
    def _create_manage_satellites_tab(self, notebook):
        """Create the manage satellites tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Manage Satellites")
        
        # Title
        title_label = tk.Label(frame, text="Manage Existing Satellites", font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Satellite list
        list_frame = tk.Frame(frame, bg='#f0f0f0')
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Listbox with scrollbar
        listbox_frame = tk.Frame(list_frame, bg='#f0f0f0')
        listbox_frame.pack(fill='both', expand=True)
        
        self.satellite_listbox = tk.Listbox(listbox_frame, height=10, font=('Arial', 10))
        scrollbar = tk.Scrollbar(listbox_frame, orient='vertical', command=self.satellite_listbox.yview)
        self.satellite_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.satellite_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Bind selection
        self.satellite_listbox.bind('<<ListboxSelect>>', self._on_satellite_select)
        
        # Control buttons
        button_frame = tk.Frame(list_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)
        
        remove_button = tk.Button(button_frame, text="Remove Selected", command=self._remove_selected_satellite,
                                 bg='#f44336', fg='white', font=('Arial', 10, 'bold'))
        remove_button.pack(side='left', padx=5)
        
        toggle_button = tk.Button(button_frame, text="Toggle Visibility", command=self._toggle_satellite_visibility,
                                 bg='#ff9800', fg='white', font=('Arial', 10, 'bold'))
        toggle_button.pack(side='left', padx=5)
        
        details_button = tk.Button(button_frame, text="View Details", command=self._view_satellite_details,
                                  bg='#2196F3', fg='white', font=('Arial', 10, 'bold'))
        details_button.pack(side='left', padx=5)
        
        lvlh_ref_button = tk.Button(button_frame, text="Set LVLH Ref", command=self._set_lvlh_reference,
                                   bg='#9C27B0', fg='white', font=('Arial', 10, 'bold'))
        lvlh_ref_button.pack(side='left', padx=5)
        
        # Update list
        self._update_satellite_list()
    
    def _create_satellite_details_tab(self, notebook):
        """Create the satellite details tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Satellite Details")
        
        # Title
        title_label = tk.Label(frame, text="Satellite Information", font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Details display
        self.details_text = tk.Text(frame, height=20, width=80, font=('Courier', 10), wrap='word')
        scrollbar = tk.Scrollbar(frame, orient='vertical', command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=scrollbar.set)
        
        self.details_text.pack(side='left', fill='both', expand=True, padx=20, pady=10)
        scrollbar.pack(side='right', fill='y', pady=10)
        
        # Initial message
        self.details_text.insert('1.0', "Select a satellite from the 'Manage Satellites' tab to view its details.")
        self.details_text.config(state='disabled')
    
    def _update_satellite_list(self):
        """Update the satellite list display"""
        if hasattr(self, 'satellite_listbox'):
            self.satellite_listbox.delete(0, tk.END)
            for name, sat_data in self.sim.satellites.items():
                color = sat_data['config'].color
                status = "●" if hasattr(sat_data['config'], 'visible') and sat_data['config'].visible else "○"
                self.satellite_listbox.insert(tk.END, f"{status} {name} ({color})")
    
    def _on_satellite_select(self, event):
        """Handle satellite selection"""
        selection = self.satellite_listbox.curselection()
        if selection:
            index = selection[0]
            satellite_names = list(self.sim.satellites.keys())
            if index < len(satellite_names):
                self.selected_satellite = satellite_names[index]
                self._update_satellite_details()
    
    def _remove_selected_satellite(self):
        """Remove the selected satellite"""
        if self.selected_satellite:
            if messagebox.askyesno("Confirm", f"Remove satellite '{self.selected_satellite}'?"):
                self.sim.remove_satellite(self.selected_satellite)
                self.selected_satellite = None
                self._update_satellite_list()
                self._clear_details()
                messagebox.showinfo("Success", "Satellite removed successfully!")
        else:
            messagebox.showwarning("Warning", "Please select a satellite to remove")
    
    def _toggle_satellite_visibility(self):
        """Toggle satellite visibility"""
        if self.selected_satellite:
            sat_data = self.sim.satellites[self.selected_satellite]
            # Initialize visible attribute if it doesn't exist
            if not hasattr(sat_data['config'], 'visible'):
                sat_data['config'].visible = True
            # Toggle visibility
            sat_data['config'].visible = not sat_data['config'].visible
            # Force redraw of the satellite
            self.sim.draw_satellite(self.selected_satellite)
            self._update_satellite_list()
            status = "visible" if sat_data['config'].visible else "hidden"
            messagebox.showinfo("Success", f"Satellite '{self.selected_satellite}' is now {status}")
        else:
            messagebox.showwarning("Warning", "Please select a satellite to toggle")
    
    def _view_satellite_details(self):
        """View detailed information about selected satellite"""
        if self.selected_satellite:
            self._update_satellite_details()
        else:
            messagebox.showwarning("Warning", "Please select a satellite to view details")
    
    def _update_satellite_details(self):
        """Update the satellite details display with both element types"""
        if not self.selected_satellite or self.selected_satellite not in self.sim.satellites:
            return
        
        sat_data = self.sim.satellites[self.selected_satellite]
        config = sat_data['config']
        state = sat_data['state']
        elements = sat_data['orbital_elements']
        
        # Calculate current orbital elements from position and velocity
        current_keplerian = self._calculate_current_keplerian(state.position, state.velocity)
        current_equinoctial = self._calculate_current_equinoctial(state.position, state.velocity)
        
        # Calculate orbital characteristics
        a = current_keplerian['a'] / 1000  # Convert to km
        e = current_keplerian['e']
        i = current_keplerian['i']
        
        # Calculate orbital period
        if a > 0:
            period = 2 * np.pi * np.sqrt((a * 1000)**3 / self.sim.mu_earth) / 3600  # hours
            altitude = a - self.sim.earth_radius / 1000
        else:
            period = 0
            altitude = 0
        
        # Format details
        details = f"""
SATELLITE DETAILS: {config.name} (UPDATED EVERY TIMESTEP)
{'='*60}

BASIC INFORMATION:
  Name: {config.name}
  Color: {config.color}
  Size: {config.size} km
  Trail Length: {config.trail_length} points
  Visible: {getattr(config, 'visible', True)}

CURRENT STATE:
  Position: [{state.position[0]/1000:.2f}, {state.position[1]/1000:.2f}, {state.position[2]/1000:.2f}] km
  Velocity: [{state.velocity[0]/1000:.2f}, {state.velocity[1]/1000:.2f}, {state.velocity[2]/1000:.2f}] km/s
  Time: {state.time:.2f} seconds

CURRENT KEPLERIAN ELEMENTS:
  Semi-major axis (a): {current_keplerian['a']/1000:.2f} km
  Eccentricity (e): {current_keplerian['e']:.6f}
  Inclination (i): {current_keplerian['i']:.2f}°
  RAAN (Ω): {current_keplerian['omega']:.2f}°
  Argument of periapsis (ω): {current_keplerian['w']:.2f}°
  True anomaly (ν): {current_keplerian['nu']:.2f}°

CURRENT MODIFIED EQUINOCTIAL ELEMENTS:
  Semi-latus rectum (p): {current_equinoctial['p']/1000:.2f} km
  f = e*cos(ω+Ω): {current_equinoctial['f']:.6f}
  g = e*sin(ω+Ω): {current_equinoctial['g']:.6f}
  h = tan(i/2)*cos(Ω): {current_equinoctial['h']:.6f}
  k = tan(i/2)*sin(Ω): {current_equinoctial['k']:.6f}
  True longitude (L): {current_equinoctial['L']:.2f}°

ORBITAL CHARACTERISTICS:
  Altitude: {altitude:.2f} km
  Orbital Period: {period:.2f} hours ({period/24:.3f} days)
  Eccentricity: {e:.6f}
  Inclination: {i:.2f}°
  
  Position Magnitude: {np.linalg.norm(state.position)/1000:.2f} km
  Velocity Magnitude: {np.linalg.norm(state.velocity)/1000:.2f} km/s
  Orbital Energy: {-self.sim.mu_earth/(2*a*1000)/1000:.2f} MJ/kg
"""
        
        # Update details text
        self.details_text.config(state='normal')
        self.details_text.delete('1.0', tk.END)
        self.details_text.insert('1.0', details)
        self.details_text.config(state='disabled')
    
    def _calculate_current_keplerian(self, position, velocity):
        """Calculate current Keplerian elements from position and velocity"""
        # This is a simplified calculation - in practice you'd use more sophisticated methods
        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)
        
        # Calculate specific angular momentum
        h_vec = np.cross(position, velocity)
        h = np.linalg.norm(h_vec)
        
        # Calculate eccentricity vector
        mu = self.sim.mu_earth
        e_vec = (1/mu) * ((v**2 - mu/r) * position - np.dot(position, velocity) * velocity)
        e = np.linalg.norm(e_vec)
        
        # Calculate semi-major axis
        a = h**2 / (mu * (1 - e**2)) if e < 1 else h**2 / (mu * (e**2 - 1))
        
        # Calculate inclination
        i = np.arccos(h_vec[2] / h)
        i = np.degrees(i)
        
        # Calculate longitude of ascending node
        omega = np.arctan2(h_vec[0], -h_vec[1])
        omega = np.degrees(omega) % 360
        
        # Calculate argument of periapsis
        if e > 1e-10:  # Avoid division by zero
            w = np.arccos(np.dot(e_vec, [np.cos(omega), np.sin(omega), 0]) / e)
            if e_vec[2] < 0:
                w = 2 * np.pi - w
            w = np.degrees(w) % 360
        else:
            w = 0
        
        # Calculate true anomaly
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, position) / (e * r))
            if np.dot(position, velocity) < 0:
                nu = 2 * np.pi - nu
            nu = np.degrees(nu) % 360
        else:
            nu = 0
        
        return {
            'a': a,
            'e': e,
            'i': i,
            'omega': omega,
            'w': w,
            'nu': nu
        }
    
    def _calculate_current_equinoctial(self, position, velocity):
        """Calculate current Modified Equinoctial elements from position and velocity"""
        # First get Keplerian elements
        keplerian = self._calculate_current_keplerian(position, velocity)
        
        # Convert to Modified Equinoctial
        p, f, g, h, k, L = self.keplerian_to_modified_equinoctial(
            keplerian['a'], keplerian['e'], keplerian['i'],
            keplerian['omega'], keplerian['w'], keplerian['nu']
        )
        
        return {
            'p': p,
            'f': f,
            'g': g,
            'h': h,
            'k': k,
            'L': L
        }
    
    def _eci_to_lvlh(self, position, velocity, ref_position, ref_velocity):
        """Convert ECI position to LVLH frame relative to reference satellite"""
        # Calculate relative position in ECI
        rel_pos_eci = position - ref_position
        
        # Calculate reference satellite's orbital frame
        # Z-axis: points toward Earth center (negative radial direction)
        z_lvlh = -ref_position / np.linalg.norm(ref_position)
        
        # Y-axis: negative orbit normal (negative angular momentum)
        h_vec = np.cross(ref_position, ref_velocity)
        y_lvlh = -h_vec / np.linalg.norm(h_vec)
        
        # X-axis: completes right-handed system
        x_lvlh = np.cross(y_lvlh, z_lvlh)
        
        # Create transformation matrix from ECI to LVLH
        R_eci_to_lvlh = np.array([x_lvlh, y_lvlh, z_lvlh])
        
        # Transform relative position to LVLH frame
        rel_pos_lvlh = R_eci_to_lvlh @ rel_pos_eci
        
        return rel_pos_lvlh
    
    def _set_lvlh_reference(self):
        """Set the selected satellite as LVLH reference"""
        if self.selected_satellite:
            self.sim.lvlh_reference_satellite = self.selected_satellite
            messagebox.showinfo("Success", f"Satellite '{self.selected_satellite}' set as LVLH reference")
        else:
            messagebox.showwarning("Warning", "Please select a satellite to set as LVLH reference")
    
    def _clear_details(self):
        """Clear the details display"""
        self.details_text.config(state='normal')
        self.details_text.delete('1.0', tk.END)
        self.details_text.insert('1.0', "Select a satellite from the 'Manage Satellites' tab to view its details.")
        self.details_text.config(state='disabled')


class EnhancedOrbitSimulationWithInterface(EnhancedOrbitSimulation):
    """Enhanced orbital simulation with interactive satellite management interface"""
    
    def __init__(self, test_mode: bool = False):
        super().__init__(test_mode)
        self.satellite_manager = SatelliteManager(self)
        self.interface_buttons = {}
        self.lvlh_reference_satellite = None  # Reference satellite for LVLH frame
    
    def draw_satellite(self, name: str) -> None:
        """Draw a satellite and its associated elements (enhanced with visibility)"""
        if self.test_mode or self.ax is None or name not in self.satellites:
            return
        
        sat_data = self.satellites[name]
        config = sat_data['config']
        state = sat_data['state']
        visuals = sat_data['visual_elements']
        
        # Check visibility
        if not getattr(config, 'visible', True):
            # Clear visual elements if not visible
            for element in visuals.values():
                if element is not None:
                    try:
                        element.remove()
                    except:
                        pass
            return
        
        # Clear previous visual elements
        for element in visuals.values():
            if element is not None:
                try:
                    element.remove()
                except:
                    pass
        
        # Get position based on coordinate frame
        if self.frame_type == 'ECI':
            pos = state.position
        elif self.frame_type == 'ECEF':
            pos = self.coord_frames.eci_to_ecef(state.position, self.current_time)
        else:  # LVLH (proper transformation to Local-Vertical-Local-Horizontal frame)
            if len(self.satellites) > 1:
                # Use specified reference satellite or first satellite as default
                if self.lvlh_reference_satellite and self.lvlh_reference_satellite in self.satellites:
                    ref_name = self.lvlh_reference_satellite
                else:
                    ref_name = list(self.satellites.keys())[0]
                    self.lvlh_reference_satellite = ref_name
                
                if name != ref_name:
                    ref_sat = self.satellites[ref_name]
                    pos = self._eci_to_lvlh(state.position, state.velocity, 
                                          ref_sat['state'].position, ref_sat['state'].velocity)
                else:
                    # Reference satellite shows at origin in LVLH frame
                    pos = np.zeros(3)
            else:
                pos = state.position
        
        pos_km = pos / 1000
        
        # Draw satellite body with proper color
        visuals['body'] = self.ax.scatter(
            pos_km[0], pos_km[1], pos_km[2],
            s=config.size, c=config.color, marker='o', alpha=0.9, edgecolors='black', linewidth=0.5
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
                # Use specified reference satellite or first satellite as default
                if self.lvlh_reference_satellite and self.lvlh_reference_satellite in self.satellites:
                    ref_name = self.lvlh_reference_satellite
                else:
                    ref_name = list(self.satellites.keys())[0]
                    self.lvlh_reference_satellite = ref_name
                
                if name != ref_name:
                    ref_sat = self.satellites[ref_name]
                    ref_trail = ref_sat['trail_points']
                    if len(ref_trail) >= len(trail_array):
                        # Transform trail points to LVLH frame
                        trail_transformed = []
                        for i, point in enumerate(trail_array):
                            ref_point = ref_trail[-len(trail_array) + i]
                            ref_vel = ref_sat['state'].velocity  # Use current velocity as approximation
                            trail_transformed.append(self._eci_to_lvlh(point, state.velocity, ref_point, ref_vel))
                        trail_array = np.array(trail_transformed)
                else:
                    # Reference satellite trail shows at origin
                    trail_array = np.zeros_like(trail_array)
            
            trail_km = trail_array / 1000
            visuals['trail'] = self.ax.plot(
                trail_km[:, 0], trail_km[:, 1], trail_km[:, 2],
                color=config.color, linewidth=1, alpha=0.5
            )[0]
    
    def animate_frame(self, frame: int) -> None:
        """Animation update function (enhanced to handle all satellites)"""
        if self.is_paused:
            return
        
        # Update time
        dt = 10 * self.animation_speed / 100  # seconds
        self.current_time += dt
        
        # Update all satellites
        for name in self.satellites:
            self._update_satellite_state(name, self.current_time)
            self.draw_satellite(name)  # This will handle visibility
        
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
        
        # Update satellite list display
        self.update_satellite_list_display()
        
        # Update satellite details if a satellite is selected
        if hasattr(self.satellite_manager, 'selected_satellite') and self.satellite_manager.selected_satellite:
            self.satellite_manager._update_satellite_details()
        
    def setup_plot(self) -> None:
        """Initialize the 3D plot with enhanced GUI controls"""
        if self.test_mode:
            return
        
        # Create figure with gridspec for layout
        self.fig = plt.figure(figsize=(18, 12))
        gs = self.fig.add_gridspec(6, 4, width_ratios=[1, 3, 1, 1], height_ratios=[1, 10, 1, 1, 1, 1])
        
        # Main 3D plot
        self.ax = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax.set_xlabel('X (km)', fontsize=10)
        self.ax.set_ylabel('Y (km)', fontsize=10)
        self.ax.set_zlabel('Z (km)', fontsize=10)
        self.ax.set_title('Multi-Satellite Orbit Simulation with Interface', fontsize=14, fontweight='bold')
        self.ax.set_box_aspect([1, 1, 1])
        
        # Control panel
        self._setup_enhanced_gui_controls(gs)
        
        plt.tight_layout()
    
    def _setup_enhanced_gui_controls(self, gs) -> None:
        """Setup enhanced GUI control panels"""
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
        
        # NEW: Satellite management buttons
        ax_sat_manager = self.fig.add_subplot(gs[4, 3])
        self.interface_buttons['sat_manager'] = Button(ax_sat_manager, 'Satellite Manager', color='lightcyan')
        self.interface_buttons['sat_manager'].on_clicked(self._open_satellite_manager)
        
        # Checkboxes for display options
        ax_checks = self.fig.add_subplot(gs[0:2, 0])
        ax_checks.axis('off')
        self.gui_controls['check_boxes'] = CheckButtons(
            ax_checks,
            ['Earth', 'Grid', 'Trails', 'Velocity', 'Attitude'],
            [True, True, True, True, False]
        )
        self.gui_controls['check_boxes'].on_clicked(self._on_check_click)
        
        # Satellite selector with quick add button
        ax_sats = self.fig.add_subplot(gs[0:2, 2])
        ax_sats.axis('off')
        ax_sats.text(0.5, 0.95, 'Satellites:', transform=ax_sats.transAxes,
                    ha='center', fontsize=11, fontweight='bold')
        self.gui_controls['sat_list'] = ax_sats
        
        # Quick add satellite button (moved to satellite box)
        ax_quick_add = self.fig.add_subplot(gs[0:1, 3])
        self.interface_buttons['quick_add'] = Button(ax_quick_add, 'Quick Add', color='lightgreen')
        self.interface_buttons['quick_add'].on_clicked(self._quick_add_satellite)
    
    def _open_satellite_manager(self, event):
        """Open the satellite management interface"""
        self.satellite_manager.open_satellite_manager()
    
    def _quick_add_satellite(self, event):
        """Quickly add a satellite with default parameters"""
        # Get a unique name
        base_name = "SAT"
        counter = 1
        while f"{base_name}_{counter}" in self.satellites:
            counter += 1
        name = f"{base_name}_{counter}"
        
        # Default ISS-like orbit
        config = SatelliteConfig(
            name=name,
            color='blue',
            size=300,
            orbital_elements={
                'a': 6.78e6,  # ~400 km altitude
                'e': 0.0001,
                'i': 51.6,
                'omega': 0,
                'w': 0,
                'M': 0
            }
        )
        
        # Initialize visibility attribute
        config.visible = True
        
        self.add_satellite(config)
        self.update_satellite_list_display()
        print(f"Quick added satellite: {name}")
    
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
            visible = getattr(sat_data['config'], 'visible', True)
            status = "●" if visible else "○"
            ax.text(0.1, y_pos, status, transform=ax.transAxes,
                   color=color, fontsize=16)
            ax.text(0.3, y_pos, name[:12], transform=ax.transAxes,
                   fontsize=9, va='center')
            y_pos -= 0.1


def create_enhanced_simulation():
    """Create an enhanced simulation with interface"""
    print("Creating Enhanced Multi-Satellite Orbital Simulation with Interface")
    print("="*70)
    
    # Create simulation
    sim = EnhancedOrbitSimulationWithInterface()
    
    # Add some example satellites
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
    
    print("\n" + "="*70)
    print("ENHANCED SIMULATION FEATURES:")
    print("="*70)
    print("• Click 'Satellite Manager' to add/remove satellites")
    print("• Use 'Quick Add Satellite' for fast satellite addition")
    print("• Select satellites in the manager to view details")
    print("• Toggle satellite visibility on/off")
    print("• Support for both Keplerian and Modified Equinoctial elements")
    print("• Real-time orbital characteristics display")
    print("• Interactive satellite selection and management")
    print("="*70)
    
    # Run animation
    sim.run_animation(duration_hours=2.0)


if __name__ == "__main__":
    create_enhanced_simulation()
