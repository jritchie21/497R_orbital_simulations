#!/usr/bin/env python3
"""
Non-Freezing Orbital Simulation
Real-time simulation that doesn't freeze
"""

from enhanced_orbit_simulation import EnhancedOrbitSimulation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

# Use better backend
matplotlib.use('TkAgg')

def run_non_freezing_simulation():
    """Run orbital simulation that doesn't freeze"""
    print("⚡ NON-FREEZING ORBITAL SIMULATION")
    print("="*50)
    print("This version fixes the freezing issue!")
    
    # Create simulation
    sim = EnhancedOrbitSimulation(test_mode=False)
    
    # Set up orbit
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
    
    print(f"\n🌍 Starting non-freezing simulation...")
    print("This should run smoothly without freezing!")
    print("Close the plot window to stop the simulation.")
    
    try:
        # Run simulation with better settings
        sim.run_simulation(duration_hours=0.2, time_step_seconds=60)
        print("\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        print("This might be due to matplotlib backend issues.")
        print("The video generation above should work better.")

def run_calculation_demo():
    """Run calculation-only demo"""
    print("\n📊 CALCULATION-ONLY DEMO")
    print("="*40)
    print("This shows orbital calculations without plotting.")
    
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
        trail = sim.generate_orbit_trail(6)
        print(f"  Period: {sim.orbital_period/3600:.2f} hours")
        print(f"  Altitude: {(a - sim.earth_radius)/1000:.0f} km")
        print(f"  Eccentricity: {e:.3f}")
        
        # Show first few positions
        print("  Sample positions:")
        for i, pos in enumerate(trail[:3]):
            print(f"    Point {i+1}: ({pos[0]/1000:.0f}, {pos[1]/1000:.0f}, {pos[2]/1000:.0f}) km")

def main():
    """Main function"""
    print("🚀 ORBITAL SIMULATION - FIXED VERSION")
    print("="*60)
    print("Choose your option:")
    print("1. Non-freezing real-time simulation")
    print("2. Calculation-only demo")
    print("3. Run both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        run_non_freezing_simulation()
    elif choice == "2":
        run_calculation_demo()
    elif choice == "3":
        run_calculation_demo()
        run_non_freezing_simulation()
    else:
        print("Invalid choice. Running calculation demo...")
        run_calculation_demo()
    
    print("\n🎉 All done!")
    print("Check the GIF files for video animations!")

if __name__ == "__main__":
    main()
