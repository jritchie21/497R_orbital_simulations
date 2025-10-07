#!/usr/bin/env python3
"""
Run the Enhanced Orbital Simulation
Demonstrates satellite visualization with different orbit types
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import matplotlib.pyplot as plt

def run_iss_simulation():
    """Run simulation for ISS-like orbit"""
    print("="*60)
    print("ISS-LIKE ORBIT SIMULATION")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Set up ISS-like orbit (400 km altitude, 51.6° inclination)
    sim.set_orbital_elements('keplerian',
        a=6.78e6,      # Semi-major axis (400 km altitude)
        e=0.0001,      # Eccentricity (nearly circular)
        i=51.6,        # Inclination (degrees)
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    
    sim.print_orbital_info()
    
    # Run simulation for 1/4 of an orbit (about 23 minutes)
    print(f"\nRunning simulation for 0.25 orbital periods...")
    sim.run_simulation(duration_hours=0.25, time_step_seconds=30)
    
    return sim

def run_geostationary_simulation():
    """Run simulation for geostationary orbit"""
    print("\n" + "="*60)
    print("GEOSTATIONARY ORBIT SIMULATION")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Set up geostationary orbit
    sim.set_orbital_elements('keplerian',
        a=42.164e6,    # Semi-major axis (geostationary)
        e=0.0,         # Eccentricity (circular)
        i=0.0,         # Inclination (equatorial)
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    
    sim.print_orbital_info()
    
    # Run simulation for 1/8 of an orbit (about 3 hours)
    print(f"\nRunning simulation for 0.125 orbital periods...")
    sim.run_simulation(duration_hours=3.0, time_step_seconds=300)
    
    return sim

def run_elliptical_simulation():
    """Run simulation for elliptical orbit"""
    print("\n" + "="*60)
    print("ELLIPTICAL ORBIT SIMULATION")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Set up elliptical orbit
    sim.set_orbital_elements('keplerian',
        a=8.0e6,       # Semi-major axis
        e=0.3,         # Eccentricity (elliptical)
        i=45.0,        # Inclination (degrees)
        omega=30.0,    # Longitude of ascending node
        w=60.0,        # Argument of periapsis
        nu=0           # True anomaly
    )
    
    sim.print_orbital_info()
    
    # Run simulation for 1/4 of an orbit
    print(f"\nRunning simulation for 0.25 orbital periods...")
    sim.run_simulation(duration_hours=0.5, time_step_seconds=60)
    
    return sim

def run_equinoctial_simulation():
    """Run simulation using modified equinoctial elements"""
    print("\n" + "="*60)
    print("MODIFIED EQUINOCTIAL ELEMENTS SIMULATION")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Set up orbit using modified equinoctial elements
    sim.set_orbital_elements('equinoctial',
        p=7.0e6,       # Semi-latus rectum
        f=0.2,         # e*cos(w+omega)
        g=0.1,         # e*sin(w+omega)
        h=0.3,         # tan(i/2)*cos(omega)
        k=0.2,         # tan(i/2)*sin(omega)
        L=45.0         # True longitude
    )
    
    sim.print_orbital_info()
    
    # Run simulation for 1/4 of an orbit
    print(f"\nRunning simulation for 0.25 orbital periods...")
    sim.run_simulation(duration_hours=0.3, time_step_seconds=45)
    
    return sim

def main():
    """Run multiple orbital simulations"""
    print("ENHANCED ORBITAL SIMULATION DEMONSTRATION")
    print("="*60)
    print("This will run several orbital simulations with different parameters.")
    print("Each simulation will show a 3D visualization of the satellite orbit.")
    print("Close each plot window to proceed to the next simulation.")
    
    try:
        # Run different simulations
        print("\n1. ISS-like orbit (Low Earth Orbit)")
        sim1 = run_iss_simulation()
        
        print("\n2. Geostationary orbit")
        sim2 = run_geostationary_simulation()
        
        print("\n3. Elliptical orbit")
        sim3 = run_elliptical_simulation()
        
        print("\n4. Modified Equinoctial elements orbit")
        sim4 = run_equinoctial_simulation()
        
        print("\n" + "="*60)
        print("ALL SIMULATIONS COMPLETED!")
        print("="*60)
        print("The enhanced orbital simulation includes:")
        print("• 3D satellite visualization with quaternion-based orientation")
        print("• Real-time position and velocity tracking")
        print("• Orbital trail visualization")
        print("• Earth surface with coordinate axes")
        print("• Support for both Keplerian and Modified Equinoctial elements")
        print("• Configurable animation speed and time steps")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print("Make sure you have matplotlib installed: pip install matplotlib")

if __name__ == "__main__":
    main()
