import unittest
import numpy as np
import sys
import os

# Add the current directory to the path so we can import the quaternion module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the quaternion functions from the main file
from basic_quat_practice import QuaternionArrowPlotter

class TestQuaternionSimulation(unittest.TestCase):
    """Unit tests for the quaternion simulation functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create plotter with plotting disabled for tests
        self.plotter = QuaternionArrowPlotter(test_mode=True)
        # Use a tolerance for floating point comparisons
        self.tolerance = 1e-10
    
    def test_quaternion_normalization(self):
        """Test quaternion normalization"""
        # Test with a non-normalized quaternion
        q = np.array([2, 3, 4, 5])
        normalized = q / np.linalg.norm(q)
        
        # Check that the normalized quaternion has unit magnitude
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0, places=10)
        
        # Check that the direction is preserved (proportional)
        self.assertTrue(np.allclose(q / np.linalg.norm(q), normalized))
    
    def test_axis_angle_to_quaternion_conversion(self):
        """Test conversion from axis-angle to quaternion"""
        # Test 90-degree rotation around Z-axis
        axis = np.array([0, 0, 1])
        angle = 90.0
        quat = self.plotter.axis_angle_to_quaternion(axis, angle)
        
        # Expected quaternion for 90° around Z-axis
        expected = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        
        # Check that the quaternion is normalized
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=10)
        
        # Check that the quaternion components match expected values
        self.assertTrue(np.allclose(quat, expected, atol=self.tolerance))
        
        # Test 180-degree rotation around X-axis
        axis = np.array([1, 0, 0])
        angle = 180.0
        quat = self.plotter.axis_angle_to_quaternion(axis, angle)
        
        # Expected quaternion for 180° around X-axis
        expected = np.array([0, 1, 0, 0])
        self.assertTrue(np.allclose(quat, expected, atol=self.tolerance))
        
        # Test 45-degree rotation around [1,1,1] axis
        axis = np.array([1, 1, 1])
        angle = 45.0
        quat = self.plotter.axis_angle_to_quaternion(axis, angle)
        
        # Check normalization
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=10)
        
        # Check that all components are positive (45° rotation)
        self.assertTrue(np.all(quat >= 0))
    
    def test_quaternion_multiplication(self):
        """Test quaternion multiplication"""
        # Test identity quaternion multiplication
        q1 = np.array([1, 0, 0, 0])  # Identity
        q2 = np.array([0.707, 0, 0, 0.707])  # 90° around Z
        result = self.plotter.multiply_quaternions(q1, q2)
        
        # Identity * anything = anything
        self.assertTrue(np.allclose(result, q2, atol=self.tolerance))
        
        # Test two 90° rotations around Z-axis (should give 180°)
        q_90 = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        result = self.plotter.multiply_quaternions(q_90, q_90)
        
        # Expected result for 180° around Z-axis
        expected = np.array([0, 0, 0, 1])
        self.assertTrue(np.allclose(result, expected, atol=self.tolerance))
    
    def test_quaternion_to_rotation_matrix(self):
        """Test conversion from quaternion to rotation matrix"""
        # Test identity quaternion
        q_identity = np.array([1, 0, 0, 0])
        R = self.plotter.quaternion_to_rotation_matrix(q_identity)
        
        # Should give identity matrix
        expected = np.eye(3)
        self.assertTrue(np.allclose(R, expected, atol=self.tolerance))
        
        # Test 90° rotation around Z-axis
        q_90z = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        R = self.plotter.quaternion_to_rotation_matrix(q_90z)
        
        # Expected rotation matrix for 90° around Z
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        self.assertTrue(np.allclose(R, expected, atol=self.tolerance))
        
        # Test that the rotation matrix is orthogonal (R^T * R = I)
        self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=self.tolerance))
        
        # Test that det(R) = 1 (proper rotation)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)
    
    def test_rotation_application(self):
        """Test applying quaternion rotation to vectors"""
        # Test rotating [1, 0, 0] by 90° around Z-axis
        original_vector = np.array([1, 0, 0])
        q_90z = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        R = self.plotter.quaternion_to_rotation_matrix(q_90z)
        rotated_vector = R @ original_vector
        
        # Should rotate to [0, 1, 0]
        expected = np.array([0, 1, 0])
        self.assertTrue(np.allclose(rotated_vector, expected, atol=self.tolerance))
        
        # Test rotating [0, 1, 0] by 90° around X-axis
        original_vector = np.array([0, 1, 0])
        q_90x = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
        R = self.plotter.quaternion_to_rotation_matrix(q_90x)
        rotated_vector = R @ original_vector
        
        # Should rotate to [0, 0, 1]
        expected = np.array([0, 0, 1])
        self.assertTrue(np.allclose(rotated_vector, expected, atol=self.tolerance))
    
    def test_slerp_quaternions(self):
        """Test spherical linear interpolation between quaternions"""
        # Test interpolation between identity and 90° Z rotation
        q1 = np.array([1, 0, 0, 0])  # Identity
        q2 = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])  # 90° Z
        
        # Test t=0 (should give q1)
        result = self.plotter.slerp_quaternions(q1, q2, 0.0)
        self.assertTrue(np.allclose(result, q1, atol=self.tolerance))
        
        # Test t=1 (should give q2)
        result = self.plotter.slerp_quaternions(q1, q2, 1.0)
        self.assertTrue(np.allclose(result, q2, atol=self.tolerance))
        
        # Test t=0.5 (should give 45° rotation)
        result = self.plotter.slerp_quaternions(q1, q2, 0.5)
        
        # Check that result is normalized
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)
        
        # Check that it's a valid quaternion (all components finite)
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_axis_normalization(self):
        """Test axis vector normalization"""
        # Test with non-normalized axis
        axis = np.array([3, 4, 0])
        normalized = axis / np.linalg.norm(axis)
        
        # Check that normalized axis has unit length
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0, places=10)
        
        # Check that direction is preserved
        self.assertTrue(np.allclose(axis / np.linalg.norm(axis), normalized))
    
    def test_quaternion_consistency(self):
        """Test that quaternion operations are consistent"""
        # Test that applying a quaternion and its inverse gives identity
        q = np.array([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])  # 90° Z
        q_inv = np.array([np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2])  # -90° Z
        
        # q * q_inv should be identity
        result = self.plotter.multiply_quaternions(q, q_inv)
        identity = np.array([1, 0, 0, 0])
        self.assertTrue(np.allclose(result, identity, atol=self.tolerance))
        
        # Test that rotation matrix is consistent with quaternion
        R = self.plotter.quaternion_to_rotation_matrix(q)
        R_inv = self.plotter.quaternion_to_rotation_matrix(q_inv)
        
        # R * R_inv should be identity
        result_matrix = R @ R_inv
        self.assertTrue(np.allclose(result_matrix, np.eye(3), atol=self.tolerance))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test zero angle rotation
        axis = np.array([1, 0, 0])
        angle = 0.0
        quat = self.plotter.axis_angle_to_quaternion(axis, angle)
        expected = np.array([1, 0, 0, 0])  # Identity
        self.assertTrue(np.allclose(quat, expected, atol=self.tolerance))
        
        # Test 360-degree rotation (should be -identity, which is equivalent to identity)
        axis = np.array([1, 0, 0])
        angle = 360.0
        quat = self.plotter.axis_angle_to_quaternion(axis, angle)
        expected = np.array([-1, 0, 0, 0])  # -Identity (equivalent to identity for rotations)
        # Note: 360° rotation gives -identity quaternion, which represents the same rotation
        self.assertTrue(np.allclose(quat, expected, atol=1e-6))
        
        # Test very small angle
        axis = np.array([1, 0, 0])
        angle = 0.001
        quat = self.plotter.axis_angle_to_quaternion(axis, angle)
        
        # Should be close to identity
        self.assertAlmostEqual(quat[0], 1.0, places=3)  # w component close to 1
        self.assertTrue(np.allclose(quat[1:], [0, 0, 0], atol=1e-3))  # x,y,z close to 0
    
    def test_animation_parameters(self):
        """Test animation parameter settings"""
        # Test setting animation speed
        original_speed = self.plotter.animation_speed
        self.plotter.set_animation_speed(0.05)
        self.assertEqual(self.plotter.animation_speed, 0.05)
        
        # Test setting animation steps
        original_steps = self.plotter.animation_steps
        self.plotter.set_animation_steps(50)
        self.assertEqual(self.plotter.animation_steps, 50)
        
        # Test toggling animation
        original_state = self.plotter.animation_enabled
        self.plotter.toggle_animation()
        self.assertEqual(self.plotter.animation_enabled, not original_state)
        self.plotter.toggle_animation()  # Toggle back
        self.assertEqual(self.plotter.animation_enabled, original_state)
    
    def test_axis_angle_input_validation(self):
        """Test axis-angle input validation and processing"""
        # Test setting axis-angle input
        axis = [1, 1, 1]
        angle = 60.0
        self.plotter.set_axis_angle_input(axis, angle)
        
        # Check that values are stored correctly
        self.assertTrue(np.allclose(self.plotter.axis_input, np.array(axis)))
        self.assertEqual(self.plotter.angle_input, angle)
        
        # Test that axis is normalized when converted to quaternion
        quat = self.plotter.axis_angle_to_quaternion(self.plotter.axis_input, self.plotter.angle_input)
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0, places=10)
    
    def test_expected_vs_actual_orientations(self):
        """Test expected orientation outputs against actual simulation results"""
        print("\n" + "="*60)
        print("EXPECTED VS ACTUAL ORIENTATION TESTING")
        print("="*60)
        
        # Define test cases with expected results
        test_cases = [
            {
                "name": "90° rotation around Z-axis",
                "axis": [0, 0, 1],
                "angle": 90.0,
                "input_vector": [1, 0, 0],  # Original arrow direction
                "expected_output": [0, 1, 0],  # Expected after rotation
                "tolerance": 1e-10
            },
            {
                "name": "90° rotation around Y-axis", 
                "axis": [0, 1, 0],
                "angle": 90.0,
                "input_vector": [1, 0, 0],
                "expected_output": [0, 0, -1],
                "tolerance": 1e-10
            },
            {
                "name": "90° rotation around X-axis",
                "axis": [1, 0, 0], 
                "angle": 90.0,
                "input_vector": [0, 1, 0],
                "expected_output": [0, 0, 1],
                "tolerance": 1e-10
            },
            {
                "name": "180° rotation around Z-axis",
                "axis": [0, 0, 1],
                "angle": 180.0,
                "input_vector": [1, 0, 0],
                "expected_output": [-1, 0, 0],
                "tolerance": 1e-10
            },
            {
                "name": "45° rotation around Z-axis",
                "axis": [0, 0, 1],
                "angle": 45.0,
                "input_vector": [1, 0, 0],
                "expected_output": [np.sqrt(2)/2, np.sqrt(2)/2, 0],
                "tolerance": 1e-10
            },
            {
                "name": "60° rotation around [1,1,1] axis",
                "axis": [1, 1, 1],
                "angle": 60.0,
                "input_vector": [1, 0, 0],
                "expected_output": [0.66666667, 0.66666667, -0.33333333],  # Calculated
                "tolerance": 1e-10
            },
            {
                "name": "120° rotation around [1,0,1] axis",
                "axis": [1, 0, 1],
                "angle": 120.0,
                "input_vector": [0, 1, 0],
                "expected_output": [-0.61237244, -0.5, 0.61237244],  # Calculated
                "tolerance": 1e-10
            }
        ]
        
        # Track test results
        passed_tests = 0
        total_tests = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print("-" * 50)
            
            try:
                # Convert axis-angle to quaternion
                quat = self.plotter.axis_angle_to_quaternion(
                    test_case['axis'], 
                    test_case['angle']
                )
                
                # Convert quaternion to rotation matrix
                R = self.plotter.quaternion_to_rotation_matrix(quat)
                
                # Apply rotation to input vector
                actual_output = R @ np.array(test_case['input_vector'])
                expected_output = np.array(test_case['expected_output'])
                
                # Compare results
                is_close = np.allclose(
                    actual_output, 
                    expected_output, 
                    atol=test_case['tolerance']
                )
                
                # Print detailed results
                print(f"Input vector:     {test_case['input_vector']}")
                print(f"Expected output:  {expected_output}")
                print(f"Actual output:    {actual_output}")
                print(f"Difference:       {actual_output - expected_output}")
                print(f"Max difference:   {np.max(np.abs(actual_output - expected_output)):.2e}")
                print(f"Tolerance:        {test_case['tolerance']:.2e}")
                print(f"Result:           {'✅ PASS' if is_close else '❌ FAIL'}")
                
                if is_close:
                    passed_tests += 1
                else:
                    print(f"❌ FAILED: Output doesn't match expected result within tolerance")
                    
            except Exception as e:
                print(f"❌ ERROR: {str(e)}")
        
        # Summary
        print("\n" + "="*60)
        print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        print("="*60)
        
        # Assert that all tests passed
        self.assertEqual(passed_tests, total_tests, 
                        f"Only {passed_tests}/{total_tests} orientation tests passed")
    
    def test_quaternion_commands_and_outputs(self):
        """Test specific quaternion commands and their expected outputs"""
        print("\n" + "="*60)
        print("QUATERNION COMMANDS AND OUTPUTS TESTING")
        print("="*60)
        
        # Reset to initial state
        self.plotter.reset_rotation()
        
        # Test command sequences with expected cumulative results
        command_sequences = [
            {
                "name": "Single 90° Z rotation",
                "commands": [
                    {"type": "axis_angle", "axis": [0, 0, 1], "angle": 90.0}
                ],
                "expected_final_arrow": [0, 1, 0],
                "expected_quaternion": [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2]
            },
            {
                "name": "Two 90° Z rotations (180° total)",
                "commands": [
                    {"type": "axis_angle", "axis": [0, 0, 1], "angle": 90.0},
                    {"type": "axis_angle", "axis": [0, 0, 1], "angle": 90.0}
                ],
                "expected_final_arrow": [-1, 0, 0],
                "expected_quaternion": [0, 0, 0, 1]
            },
            {
                "name": "90° Z then 90° Y rotation",
                "commands": [
                    {"type": "axis_angle", "axis": [0, 0, 1], "angle": 90.0},
                    {"type": "axis_angle", "axis": [0, 1, 0], "angle": 90.0}
                ],
                "expected_final_arrow": [0, 1, 0],  # Calculated
                "expected_quaternion": [0.5, 0.5, 0.5, 0.5]  # Calculated
            }
        ]
        
        for seq_idx, sequence in enumerate(command_sequences, 1):
            print(f"\nCommand Sequence {seq_idx}: {sequence['name']}")
            print("-" * 50)
            
            # Reset for each sequence
            self.plotter.reset_rotation()
            
            try:
                # Execute commands
                for cmd_idx, command in enumerate(sequence['commands'], 1):
                    print(f"  Command {cmd_idx}: {command['type']} - axis={command['axis']}, angle={command['angle']}°")
                    
                    if command['type'] == 'axis_angle':
                        quat = self.plotter.axis_angle_to_quaternion(command['axis'], command['angle'])
                        self.plotter.rotate_arrow_instant(quat)  # Use instant for testing
                    
                    print(f"    Current arrow: {self.plotter.current_arrow}")
                    print(f"    Current quat:  {self.plotter.cumulative_quaternion}")
                
                # Check final results
                final_arrow = self.plotter.current_arrow
                final_quat = self.plotter.cumulative_quaternion
                
                arrow_match = np.allclose(final_arrow, sequence['expected_final_arrow'], atol=1e-10)
                quat_match = np.allclose(final_quat, sequence['expected_quaternion'], atol=1e-10)
                
                print(f"\n  Final Results:")
                print(f"    Expected arrow: {sequence['expected_final_arrow']}")
                print(f"    Actual arrow:   {final_arrow}")
                print(f"    Arrow match:    {'✅' if arrow_match else '❌'}")
                print(f"    Expected quat:  {sequence['expected_quaternion']}")
                print(f"    Actual quat:    {final_quat}")
                print(f"    Quat match:     {'✅' if quat_match else '❌'}")
                
                # Assert results
                self.assertTrue(arrow_match, f"Final arrow doesn't match expected: {final_arrow} vs {sequence['expected_final_arrow']}")
                self.assertTrue(quat_match, f"Final quaternion doesn't match expected: {final_quat} vs {sequence['expected_quaternion']}")
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
                self.fail(f"Command sequence failed: {str(e)}")
        
        print(f"\n✅ All {len(command_sequences)} command sequences passed!")

def run_performance_test():
    """Run a performance test to ensure the simulation runs efficiently"""
    print("\n" + "="*50)
    print("PERFORMANCE TEST")
    print("="*50)
    
    plotter = QuaternionArrowPlotter(test_mode=True)
    
    # Test multiple rotations
    import time
    start_time = time.time()
    
    # Perform 100 quaternion operations
    for i in range(100):
        angle = i * 3.6  # 0 to 360 degrees
        axis = [1, 0, 0]
        quat = plotter.axis_angle_to_quaternion(axis, angle)
        R = plotter.quaternion_to_rotation_matrix(quat)
        rotated = R @ np.array([1, 0, 0])
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Performed 100 quaternion operations in {elapsed:.4f} seconds")
    print(f"Average time per operation: {elapsed/100*1000:.2f} ms")
    
    if elapsed < 1.0:  # Should complete in less than 1 second
        print("✅ Performance test PASSED")
    else:
        print("❌ Performance test FAILED - too slow")
    
    return elapsed < 1.0

if __name__ == '__main__':
    print("QUATERNION SIMULATION UNIT TESTS")
    print("="*50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    run_performance_test()
    
    print("\n" + "="*50)
    print("All tests completed!")
