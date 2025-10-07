#!/usr/bin/env python3
"""
Robust Orbital Simulation with Better Plotting
Handles matplotlib issues on Windows
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib backend to avoid freezing issues
matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility

def run_robust_simulation():
    """Run orbital simulation with robust plotting"""
    print("🚀 ROBUST ORBITAL SIMULATION")
    print("="*50)
    
    try:
        # Create simulation
        sim = EnhancedOrbitSimulation(test_mode=False)
        
        # Set up a simple, stable orbit
        sim.set_orbital_elements('keplerian',
            a=7.0e6,       # 7,000 km semi-major axis
            e=0.1,         # 10% eccentricity
            i=30.0,        # 30° inclination
            omega=0,       # Longitude of ascending node
            w=0,           # Argument of periapsis
            nu=0           # True anomaly
        )
        
        print("Orbital Parameters:")
        sim.print_orbital_info()
        
        print(f"\n🌍 Starting 3D visualization...")
        print("This should show a stable 3D plot of the satellite orbit.")
        print("If the plot freezes, try closing it and running again.")
        
        # Run a shorter simulation to avoid freezing
        sim.run_simulation(duration_hours=0.1, time_step_seconds=60)
        
        print("\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        print("Trying alternative approach...")
        
        # Fallback: run without visualization
        run_calculation_only()

def run_calculation_only():
    """Run simulation calculations without plotting"""
    print("\n📊 CALCULATION-ONLY MODE")
    print("="*40)
    
    sim = EnhancedOrbitSimulation(test_mode=True)
    
    # Test different orbits
    orbits = [
        ("ISS-like Orbit", 6.78e6, 0.0001, 51.6),
        ("Elliptical Orbit", 8.0e6, 0.3, 45.0),
        ("Geostationary Orbit", 42.164e6, 0.0, 0.0)
    ]
    
    for name, a, e, i in orbits:
        print(f"\n{name}:")
        print("-" * 30)
        
        sim.set_orbital_elements('keplerian', a=a, e=e, i=i, omega=0, w=0, nu=0)
        
        # Calculate orbital positions
        trail = sim.generate_orbit_trail(8)
        print(f"  Period: {sim.orbital_period/3600:.2f} hours")
        print(f"  Altitude: {(a - sim.earth_radius)/1000:.0f} km")
        print(f"  Eccentricity: {e:.3f}")
        
        # Show first few positions
        print("  Sample positions:")
        for i, pos in enumerate(trail[:3]):
            print(f"    Point {i+1}: ({pos[0]/1000:.0f}, {pos[1]/1000:.0f}, {pos[2]/1000:.0f}) km")

def run_interactive_simulation():
    """Run simulation with interactive controls"""
    print("\n🎮 INTERACTIVE ORBITAL SIMULATION")
    print("="*50)
    
    sim = EnhancedOrbitSimulation(test_mode=False)
    
    # Set up orbit
    sim.set_orbital_elements('keplerian',
        a=6.5e6,       # 6,500 km semi-major axis
        e=0.05,        # 5% eccentricity (nearly circular)
        i=25.0,        # 25° inclination
        omega=0,       # Longitude of ascending node
        w=0,           # Argument of periapsis
        nu=0           # True anomaly
    )
    
    print("Orbital Parameters:")
    sim.print_orbital_info()
    
    print(f"\n🌍 Starting interactive simulation...")
    print("The plot should be responsive. Try rotating and zooming!")
    
    try:
        # Run simulation with longer duration
        sim.run_simulation(duration_hours=0.2, time_step_seconds=30)
        print("\n✅ Interactive simulation completed!")
        
    except Exception as e:
        print(f"\n❌ Interactive mode failed: {e}")
        print("Switching to calculation-only mode...")
        run_calculation_only()

def main():
    """Main function to run simulations"""
    print("ENHANCED ORBITAL SIMULATION - ROBUST VERSION")
    print("="*60)
    print("This version handles matplotlib issues better on Windows.")
    print("If 3D plots freeze, the simulation will fall back to calculations only.")
    
    # Try interactive simulation first
    try:
        run_interactive_simulation()
    except:
        print("\nFalling back to calculation-only mode...")
        run_calculation_only()
    
    print("\n" + "="*60)
    print("🎉 SIMULATION COMPLETE!")
    print("="*60)
    print("The orbital simulation includes:")
    print("• 3D satellite visualization with quaternion-based orientation")
    print("• Real-time position and velocity tracking")
    print("• Orbital trail visualization")
    print("• Earth surface with coordinate axes")
    print("• Support for both Keplerian and Modified Equinoctial elements")
    print("• Robust error handling and fallback modes")

if __name__ == "__main__":
    main()
