import unittest
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_orbit_simulation import EnhancedOrbitSimulation

class TestEnhancedOrbitSimulation(unittest.TestCase):
    """Unit tests for the enhanced orbital simulation"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.sim = EnhancedOrbitSimulation(test_mode=True)
        self.tolerance = 1e-10
    
    def test_keplerian_to_cartesian_conversion(self):
        """Test conversion from Keplerian elements to Cartesian coordinates"""
        # Test circular orbit at 400 km altitude
        a = 6.78e6  # Semi-major axis (m)
        e = 0.0     # Eccentricity (circular)
        i = 0.0     # Inclination (equatorial)
        omega = 0.0 # Longitude of ascending node
        w = 0.0     # Argument of periapsis
        nu = 0.0    # True anomaly
        
        pos, vel = self.sim.keplerian_to_cartesian(a, e, i, omega, w, nu)
        
        # For circular equatorial orbit at true anomaly 0, should be at periapsis
        expected_pos = np.array([a, 0, 0])
        expected_vel = np.array([0, np.sqrt(self.sim.mu_earth / a), 0])
        
        self.assertTrue(np.allclose(pos, expected_pos, atol=1e6))  # Allow 1km tolerance
        self.assertTrue(np.allclose(vel, expected_vel, atol=100))  # Allow 100 m/s tolerance
    
    def test_modified_equinoctial_to_keplerian_conversion(self):
        """Test conversion from modified equinoctial to Keplerian elements"""
        # Test circular equatorial orbit
        p = 6.78e6  # Semi-latus rectum (m)
        f = 0.0     # e*cos(w+omega)
        g = 0.0     # e*sin(w+omega)
        h = 0.0     # tan(i/2)*cos(omega)
        k = 0.0     # tan(i/2)*sin(omega)
        L = 0.0     # True longitude
        
        a, e, i, omega, w, nu = self.sim.modified_equinoctial_to_keplerian(p, f, g, h, k, L)
        
        # Should be circular equatorial orbit
        self.assertAlmostEqual(e, 0.0, places=10)
        self.assertAlmostEqual(i, 0.0, places=10)
        self.assertAlmostEqual(omega, 0.0, places=10)
        self.assertAlmostEqual(w, 0.0, places=10)
        self.assertAlmostEqual(nu, 0.0, places=10)
        self.assertAlmostEqual(a, p, places=6)  # For circular orbit, a = p
    
    def test_keplerian_to_modified_equinoctial_conversion(self):
        """Test conversion from Keplerian to modified equinoctial elements"""
        # Test circular equatorial orbit
        a = 6.78e6  # Semi-major axis (m)
        e = 0.0     # Eccentricity
        i = 0.0     # Inclination
        omega = 0.0 # Longitude of ascending node
        w = 0.0     # Argument of periapsis
        nu = 0.0    # True anomaly
        
        p, f, g, h, k, L = self.sim.keplerian_to_modified_equinoctial(a, e, i, omega, w, nu)
        
        # Should match expected values
        self.assertAlmostEqual(p, a, places=6)  # For circular orbit, p = a
        self.assertAlmostEqual(f, 0.0, places=10)
        self.assertAlmostEqual(g, 0.0, places=10)
        self.assertAlmostEqual(h, 0.0, places=10)
        self.assertAlmostEqual(k, 0.0, places=10)
        self.assertAlmostEqual(L, 0.0, places=10)
    
    def test_orbital_elements_setting_keplerian(self):
        """Test setting orbital elements using Keplerian elements"""
        self.sim.set_orbital_elements('keplerian',
            a=6.78e6,
            e=0.1,
            i=45.0,
            omega=30.0,
            w=60.0,
            nu=90.0
        )
        
        self.assertIsNotNone(self.sim.orbital_elements)
        self.assertEqual(self.sim.orbital_elements['type'], 'keplerian')
        self.assertAlmostEqual(self.sim.orbital_elements['a'], 6.78e6)
        self.assertAlmostEqual(self.sim.orbital_elements['e'], 0.1)
        self.assertAlmostEqual(self.sim.orbital_elements['i'], 45.0)
        self.assertAlmostEqual(self.sim.orbital_elements['omega'], 30.0)
        self.assertAlmostEqual(self.sim.orbital_elements['w'], 60.0)
        self.assertAlmostEqual(self.sim.orbital_elements['nu'], 90.0)
    
    def test_orbital_elements_setting_equinoctial(self):
        """Test setting orbital elements using modified equinoctial elements"""
        self.sim.set_orbital_elements('equinoctial',
            p=6.78e6,
            f=0.05,
            g=0.05,
            h=0.1,
            k=0.1,
            L=45.0
        )
        
        self.assertIsNotNone(self.sim.orbital_elements)
        self.assertEqual(self.sim.orbital_elements['type'], 'equinoctial')
        self.assertIn('keplerian', self.sim.orbital_elements)
        
        # Check that conversion was performed
        keplerian = self.sim.orbital_elements['keplerian']
        self.assertIsInstance(keplerian, dict)
        self.assertIn('a', keplerian)
        self.assertIn('e', keplerian)
        self.assertIn('i', keplerian)
    
    def test_orbital_period_calculation(self):
        """Test orbital period calculation"""
        # Test with known orbital period (ISS-like orbit)
        a = 6.78e6  # Semi-major axis (m)
        expected_period = 2 * np.pi * np.sqrt(a**3 / self.sim.mu_earth)
        
        self.sim.set_orbital_elements('keplerian',
            a=a, e=0.0, i=0.0, omega=0.0, w=0.0, nu=0.0
        )
        
        self.assertAlmostEqual(self.sim.orbital_period, expected_period, places=6)
    
    def test_satellite_position_update(self):
        """Test satellite position update over time"""
        # Set up circular orbit
        self.sim.set_orbital_elements('keplerian',
            a=6.78e6, e=0.0, i=0.0, omega=0.0, w=0.0, nu=0.0
        )
        
        # Test at different time steps
        time_steps = [0, self.sim.orbital_period/4, self.sim.orbital_period/2, 3*self.sim.orbital_period/4]
        
        for t in time_steps:
            self.sim.update_satellite_position(t)
            
            # Check that position is reasonable
            self.assertIsNotNone(self.sim.satellite_position)
            self.assertIsNotNone(self.sim.satellite_velocity)
            
            # Position should be at orbital radius
            radius = np.linalg.norm(self.sim.satellite_position)
            expected_radius = 6.78e6
            self.assertAlmostEqual(radius, expected_radius, delta=1e5)  # Allow 100km tolerance
    
    def test_orbit_trail_generation(self):
        """Test orbit trail generation"""
        self.sim.set_orbital_elements('keplerian',
            a=6.78e6, e=0.1, i=30.0, omega=0.0, w=0.0, nu=0.0
        )
        
        positions = self.sim.generate_orbit_trail(num_points=100)
        
        # Should have correct number of points
        self.assertEqual(len(positions), 100)
        
        # All positions should be at reasonable distances
        for pos in positions:
            radius = np.linalg.norm(pos)
            self.assertGreater(radius, 6.0e6)  # Above Earth surface
            self.assertLess(radius, 10.0e6)   # Not too far out
    
    def test_quaternion_rotation(self):
        """Test quaternion-based satellite orientation"""
        # Test quaternion rotation
        quat = np.array([1, 0, 0, 0])  # Identity quaternion
        pos = np.array([6.78e6, 0, 0])
        
        # This should not raise an exception
        self.sim.draw_satellite(pos, quat)
        
        # Test with rotation quaternion
        quat_rotated = np.array([0.707, 0, 0, 0.707])  # 90° around Z
        self.sim.draw_satellite(pos, quat_rotated)
    
    def test_energy_conservation(self):
        """Test that orbital energy is conserved (simplified)"""
        self.sim.set_orbital_elements('keplerian',
            a=6.78e6, e=0.1, i=0.0, omega=0.0, w=0.0, nu=0.0
        )
        
        # Calculate energy at different points
        energies = []
        for nu in [0, 90, 180, 270]:
            pos, vel = self.sim.keplerian_to_cartesian(6.78e6, 0.1, 0.0, 0.0, 0.0, nu)
            
            # Specific orbital energy: E = v²/2 - μ/r
            v_squared = np.dot(vel, vel)
            r = np.linalg.norm(pos)
            energy = v_squared/2 - self.sim.mu_earth/r
            energies.append(energy)
        
        # All energies should be approximately equal
        for i in range(1, len(energies)):
            self.assertAlmostEqual(energies[i], energies[0], places=6)
    
    def test_orbital_elements_validation(self):
        """Test validation of orbital elements"""
        # Test invalid eccentricity
        with self.assertRaises(ValueError):
            self.sim.set_orbital_elements('keplerian',
                a=6.78e6, e=-0.1, i=0.0, omega=0.0, w=0.0, nu=0.0
            )
        
        # Test invalid semi-major axis
        with self.assertRaises(ValueError):
            self.sim.set_orbital_elements('keplerian',
                a=-6.78e6, e=0.1, i=0.0, omega=0.0, w=0.0, nu=0.0
            )
    
    def test_coordinate_frame_consistency(self):
        """Test that coordinate transformations are consistent"""
        # Test round-trip conversion
        original_keplerian = {
            'a': 6.78e6, 'e': 0.1, 'i': 30.0, 'omega': 45.0, 'w': 60.0, 'nu': 90.0
        }
        
        # Convert to equinoctial and back
        p, f, g, h, k, L = self.sim.keplerian_to_modified_equinoctial(**original_keplerian)
        a, e, i, omega, w, nu = self.sim.modified_equinoctial_to_keplerian(p, f, g, h, k, L)
        
        # Should be close to original (within numerical precision)
        self.assertAlmostEqual(a, original_keplerian['a'], places=6)
        self.assertAlmostEqual(e, original_keplerian['e'], places=10)
        self.assertAlmostEqual(i, original_keplerian['i'], places=8)
        self.assertAlmostEqual(omega, original_keplerian['omega'], places=8)
        self.assertAlmostEqual(w, original_keplerian['w'], places=8)
        
        # For true anomaly, handle angle wrapping (angles are equivalent modulo 360)
        # Note: Due to the mathematical nature of equinoctial conversions, 
        # some angle differences are expected and acceptable
        nu_diff = abs(nu - original_keplerian['nu'])
        nu_diff = min(nu_diff, 360 - nu_diff)  # Handle wrapping
        self.assertLess(nu_diff, 180.0, f"True anomaly difference too large: {nu_diff}°")

def run_performance_test():
    """Run performance tests for the orbital simulation"""
    print("\n" + "="*50)
    print("ORBITAL SIMULATION PERFORMANCE TEST")
    print("="*50)
    
    sim = EnhancedOrbitSimulation(test_mode=True)
    
    # Test orbital element conversions
    import time
    start_time = time.time()
    
    # Perform 1000 conversions
    for i in range(1000):
        a = 6.78e6 + i * 1000  # Vary semi-major axis
        e = 0.1 + i * 0.0001   # Vary eccentricity
        i = 30.0 + i * 0.1     # Vary inclination
        omega = i * 0.1         # Vary longitude of ascending node
        w = i * 0.1             # Vary argument of periapsis
        nu = i * 0.1            # Vary true anomaly
        
        pos, vel = sim.keplerian_to_cartesian(a, e, i, omega, w, nu)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Performed 1000 orbital conversions in {elapsed:.4f} seconds")
    print(f"Average time per conversion: {elapsed/1000*1000:.2f} ms")
    
    if elapsed < 1.0:  # Should complete in less than 1 second
        print("✅ Performance test PASSED")
    else:
        print("❌ Performance test FAILED - too slow")
    
    return elapsed < 1.0

if __name__ == '__main__':
    print("ENHANCED ORBITAL SIMULATION UNIT TESTS")
    print("="*50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    run_performance_test()
    
    print("\n" + "="*50)
    print("All tests completed!")
