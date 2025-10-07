#!/usr/bin/env python3
"""
Test runner for quaternion simulation unit tests.
This script provides an easy way to run specific test categories.
"""

import unittest
import sys
import os
from test_quaternion_simulation import TestQuaternionSimulation, run_performance_test

def run_all_tests():
    """Run all unit tests"""
    print("Running ALL unit tests...")
    unittest.main(module='test_quaternion_simulation', verbosity=2, exit=False)

def run_specific_test(test_name):
    """Run a specific test by name"""
    print(f"Running test: {test_name}")
    suite = unittest.TestSuite()
    suite.addTest(TestQuaternionSimulation(test_name))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_core_tests():
    """Run core quaternion operation tests"""
    core_tests = [
        'test_quaternion_normalization',
        'test_axis_angle_to_quaternion_conversion',
        'test_quaternion_multiplication',
        'test_quaternion_to_rotation_matrix',
        'test_rotation_application'
    ]
    
    print("Running CORE quaternion operation tests...")
    suite = unittest.TestSuite()
    for test_name in core_tests:
        suite.addTest(TestQuaternionSimulation(test_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_orientation_tests():
    """Run comprehensive orientation comparison tests"""
    orientation_tests = [
        'test_expected_vs_actual_orientations',
        'test_quaternion_commands_and_outputs'
    ]
    
    print("Running ORIENTATION COMPARISON tests...")
    suite = unittest.TestSuite()
    for test_name in orientation_tests:
        suite.addTest(TestQuaternionSimulation(test_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_animation_tests():
    """Run animation and UI tests"""
    animation_tests = [
        'test_animation_parameters',
        'test_axis_angle_input_validation',
        'test_slerp_quaternions'
    ]
    
    print("Running ANIMATION and UI tests...")
    suite = unittest.TestSuite()
    for test_name in animation_tests:
        suite.addTest(TestQuaternionSimulation(test_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_edge_case_tests():
    """Run edge case and boundary condition tests"""
    edge_tests = [
        'test_edge_cases',
        'test_quaternion_consistency',
        'test_axis_normalization'
    ]
    
    print("Running EDGE CASE tests...")
    suite = unittest.TestSuite()
    for test_name in edge_tests:
        suite.addTest(TestQuaternionSimulation(test_name))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def main():
    """Main test runner with menu options"""
    print("QUATERNION SIMULATION TEST RUNNER")
    print("=" * 40)
    print("1. Run all tests")
    print("2. Run core quaternion tests")
    print("3. Run orientation comparison tests")
    print("4. Run animation/UI tests")
    print("5. Run edge case tests")
    print("6. Run performance test only")
    print("7. Run specific test")
    print("8. List available tests")
    print("9. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == "1":
                run_all_tests()
                run_performance_test()
                
            elif choice == "2":
                success = run_core_tests()
                print(f"\nCore tests: {'PASSED' if success else 'FAILED'}")
                
            elif choice == "3":
                success = run_orientation_tests()
                print(f"\nOrientation tests: {'PASSED' if success else 'FAILED'}")
                
            elif choice == "4":
                success = run_animation_tests()
                print(f"\nAnimation tests: {'PASSED' if success else 'FAILED'}")
                
            elif choice == "5":
                success = run_edge_case_tests()
                print(f"\nEdge case tests: {'PASSED' if success else 'FAILED'}")
                
            elif choice == "6":
                run_performance_test()
                
            elif choice == "7":
                test_name = input("Enter test name: ").strip()
                success = run_specific_test(test_name)
                print(f"\nTest {test_name}: {'PASSED' if success else 'FAILED'}")
                
            elif choice == "8":
                print("\nAvailable tests:")
                test_methods = [method for method in dir(TestQuaternionSimulation) 
                              if method.startswith('test_')]
                for i, method in enumerate(test_methods, 1):
                    print(f"  {i}. {method}")
                    
            elif choice == "9":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-9.")
                
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
