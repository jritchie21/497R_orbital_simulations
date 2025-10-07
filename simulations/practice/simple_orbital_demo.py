#!/usr/bin/env python3
"""
Simple Orbital Simulation Demo
Shows orbital calculations without interactive plotting
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import numpy as np

def demo_orbital_calculations():
    """Demonstrate orbital calculations without plotting"""
    print("ENHANCED ORBITAL SIMULATION - CALCULATION DEMO")
    print("="*60)
    
    # Create simulation in test mode (no plotting)
    sim = EnhancedOrbitSimulation(test_mode=True)
    
    # Demo 1: ISS-like orbit
    print("\n1. ISS-LIKE ORBIT (Low Earth Orbit)")
    print("-" * 40)
    sim.set_orbital_elements('keplerian',
        a=6.78e6,      # 400 km altitude
        e=0.0001,      # Nearly circular
        i=51.6,        # 51.6° inclination
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    sim.print_orbital_info()
    
    # Calculate some orbital positions using the orbit trail
    print(f"\nOrbital positions over time:")
    trail = sim.generate_orbit_trail(6)  # Get 6 points around the orbit
    for i, pos in enumerate(trail):
        t = i / 5.0  # Time as fraction of orbit
        print(f"  t={t:.1f}T: pos=({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]/1000:.1f}) km")
    
    # Demo 2: Geostationary orbit
    print("\n\n2. GEOSTATIONARY ORBIT")
    print("-" * 40)
    sim.set_orbital_elements('keplerian',
        a=42.164e6,    # Geostationary altitude
        e=0.0,         # Circular
        i=0.0,         # Equatorial
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    sim.print_orbital_info()
    
    # Demo 3: Elliptical orbit
    print("\n\n3. ELLIPTICAL ORBIT")
    print("-" * 40)
    sim.set_orbital_elements('keplerian',
        a=8.0e6,       # Semi-major axis
        e=0.3,         # Eccentricity
        i=45.0,        # Inclination
        omega=30.0,    # Longitude of ascending node
        w=60.0,        # Argument of periapsis
        nu=0           # True anomaly
    )
    sim.print_orbital_info()
    
    # Demo 4: Modified Equinoctial elements
    print("\n\n4. MODIFIED EQUINOCTIAL ELEMENTS")
    print("-" * 40)
    sim.set_orbital_elements('equinoctial',
        p=7.0e6,       # Semi-latus rectum
        f=0.2,         # e*cos(w+omega)
        g=0.1,         # e*sin(w+omega)
        h=0.3,         # tan(i/2)*cos(omega)
        k=0.2,         # tan(i/2)*sin(omega)
        L=45.0         # True longitude
    )
    sim.print_orbital_info()
    
    # Demo 5: Orbital mechanics calculations
    print("\n\n5. ORBITAL MECHANICS CALCULATIONS")
    print("-" * 40)
    
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
    
    # Demo 6: Element conversion
    print("\n\n6. ELEMENT CONVERSION DEMO")
    print("-" * 40)
    
    # Test conversion between element types
    keplerian = {'a': 6.78e6, 'e': 0.1, 'i': 30.0, 'omega': 45.0, 'w': 60.0, 'nu': 90.0}
    print("Original Keplerian elements:")
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
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("The enhanced orbital simulation provides:")
    print("• Support for both Keplerian and Modified Equinoctial elements")
    print("• Real-time position and velocity calculations")
    print("• Orbital period and mechanics calculations")
    print("• Element conversion between different systems")
    print("• 3D visualization capabilities (when not in test mode)")
    print("• Comprehensive validation and error checking")

if __name__ == "__main__":
    demo_orbital_calculations()
