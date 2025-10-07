#!/usr/bin/env python3
"""
Demo script for the Enhanced Orbital Simulation
Shows how to use both Keplerian and Modified Equinoctial elements
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import numpy as np

def demo_keplerian_elements():
    """Demonstrate Keplerian orbital elements"""
    print("="*60)
    print("DEMO: KEPLERIAN ORBITAL ELEMENTS")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # ISS-like orbit
    print("\n1. ISS-like orbit (400 km altitude, 51.6° inclination)")
    sim.set_orbital_elements('keplerian',
        a=6.78e6,      # Semi-major axis (m) - 400 km altitude
        e=0.0001,      # Eccentricity (nearly circular)
        i=51.6,        # Inclination (degrees)
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    sim.print_orbital_info()
    
    # Geostationary orbit
    print("\n2. Geostationary orbit (35,786 km altitude)")
    sim.set_orbital_elements('keplerian',
        a=42.164e6,    # Semi-major axis (m) - geostationary
        e=0.0,         # Eccentricity (circular)
        i=0.0,         # Inclination (equatorial)
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    sim.print_orbital_info()
    
    # Molniya orbit (highly elliptical)
    print("\n3. Molniya orbit (highly elliptical)")
    sim.set_orbital_elements('keplerian',
        a=26.6e6,      # Semi-major axis (m)
        e=0.74,        # Eccentricity (highly elliptical)
        i=63.4,        # Inclination (degrees)
        omega=0,       # Longitude of ascending node
        w=270,         # Argument of periapsis
        nu=0           # True anomaly
    )
    sim.print_orbital_info()

def demo_modified_equinoctial_elements():
    """Demonstrate Modified Equinoctial orbital elements"""
    print("\n" + "="*60)
    print("DEMO: MODIFIED EQUINOCTIAL ELEMENTS")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Circular equatorial orbit
    print("\n1. Circular equatorial orbit")
    sim.set_orbital_elements('equinoctial',
        p=6.78e6,      # Semi-latus rectum (m)
        f=0.0,         # e*cos(w+omega)
        g=0.0,         # e*sin(w+omega)
        h=0.0,         # tan(i/2)*cos(omega)
        k=0.0,         # tan(i/2)*sin(omega)
        L=0            # True longitude
    )
    sim.print_orbital_info()
    
    # Inclined elliptical orbit
    print("\n2. Inclined elliptical orbit")
    sim.set_orbital_elements('equinoctial',
        p=8.0e6,       # Semi-latus rectum (m)
        f=0.1,         # e*cos(w+omega)
        g=0.05,        # e*sin(w+omega)
        h=0.2,         # tan(i/2)*cos(omega)
        k=0.1,         # tan(i/2)*sin(omega)
        L=45           # True longitude
    )
    sim.print_orbital_info()

def demo_satellite_visualization():
    """Demonstrate satellite visualization capabilities"""
    print("\n" + "="*60)
    print("DEMO: SATELLITE VISUALIZATION")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Set up a simple orbit
    sim.set_orbital_elements('keplerian',
        a=6.78e6,      # 400 km altitude
        e=0.1,         # Slightly elliptical
        i=30,          # 30° inclination
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    
    print("Satellite visualization features:")
    print("• 3D satellite model with quaternion-based orientation")
    print("• Real-time position and velocity tracking")
    print("• Orbital trail visualization")
    print("• Earth surface with coordinate axes")
    print("• Configurable animation speed")
    
    # Show orbital parameters
    sim.print_orbital_info()
    
    print(f"\nAnimation speed: {sim.animation_speed}x")
    print("Satellite scale: {:.1f} km".format(sim.satellite_scale * 1000))

def demo_orbital_mechanics():
    """Demonstrate orbital mechanics calculations"""
    print("\n" + "="*60)
    print("DEMO: ORBITAL MECHANICS CALCULATIONS")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Test different orbits
    orbits = [
        ("Low Earth Orbit (LEO)", 6.78e6, 0.0, 0.0),
        ("Medium Earth Orbit (MEO)", 20.0e6, 0.1, 45.0),
        ("Geostationary Orbit (GEO)", 42.164e6, 0.0, 0.0),
        ("Highly Elliptical Orbit (HEO)", 26.6e6, 0.74, 63.4)
    ]
    
    print(f"{'Orbit Type':<25} {'Period (hrs)':<12} {'Altitude (km)':<15} {'Eccentricity':<12}")
    print("-" * 70)
    
    for name, a, e, i in orbits:
        sim.set_orbital_elements('keplerian', a=a, e=e, i=i, omega=0, w=0, nu=0)
        period_hours = sim.orbital_period / 3600
        altitude_km = (a - sim.earth_radius) / 1000
        
        print(f"{name:<25} {period_hours:<12.2f} {altitude_km:<15.0f} {e:<12.3f}")

def demo_conversion_between_elements():
    """Demonstrate conversion between element types"""
    print("\n" + "="*60)
    print("DEMO: ELEMENT CONVERSION")
    print("="*60)
    
    sim = EnhancedOrbitSimulation()
    
    # Start with Keplerian elements
    print("Original Keplerian elements:")
    keplerian = {'a': 6.78e6, 'e': 0.1, 'i': 30.0, 'omega': 45.0, 'w': 60.0, 'nu': 90.0}
    for key, value in keplerian.items():
        print(f"  {key}: {value}")
    
    # Convert to modified equinoctial
    p, f, g, h, k, L = sim.keplerian_to_modified_equinoctial(**keplerian)
    print(f"\nConverted to Modified Equinoctial:")
    print(f"  p: {p/1000:.1f} km")
    print(f"  f: {f:.6f}")
    print(f"  g: {g:.6f}")
    print(f"  h: {h:.6f}")
    print(f"  k: {k:.6f}")
    print(f"  L: {L:.1f}°")
    
    # Convert back to Keplerian
    a, e, i, omega, w, nu = sim.modified_equinoctial_to_keplerian(p, f, g, h, k, L)
    print(f"\nConverted back to Keplerian:")
    print(f"  a: {a/1000:.1f} km")
    print(f"  e: {e:.6f}")
    print(f"  i: {i:.1f}°")
    print(f"  omega: {omega:.1f}°")
    print(f"  w: {w:.1f}°")
    print(f"  nu: {nu:.1f}°")

def main():
    """Run all demonstrations"""
    print("ENHANCED ORBITAL SIMULATION DEMONSTRATION")
    print("="*60)
    print("This demo shows the capabilities of the enhanced orbital simulation")
    print("including both Keplerian and Modified Equinoctial elements,")
    print("satellite visualization, and orbital mechanics calculations.")
    
    # Run all demos
    demo_keplerian_elements()
    demo_modified_equinoctial_elements()
    demo_satellite_visualization()
    demo_orbital_mechanics()
    demo_conversion_between_elements()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("To run the actual simulation with visualization, use:")
    print("  sim = EnhancedOrbitSimulation()")
    print("  sim.set_orbital_elements('keplerian', a=6.78e6, e=0.1, i=30, omega=0, w=0, nu=0)")
    print("  sim.run_simulation(duration_hours=0.1, time_step_seconds=10)")

if __name__ == "__main__":
    main()
