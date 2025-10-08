"""
Basic quaternion practice with body-frame rotations.
This is a simplified version focusing on the core quaternion mathematics
for satellite attitude control using body-frame rotations.

Body Frame vs World Frame:
- World Frame: Each rotation is applied in world coordinates (fixed reference)
- Body Frame: Each rotation is applied relative to current orientation (like a satellite)
"""

import numpy as np
from quaternion_math import QuaternionMath, create_sample_quaternions, create_sample_axis_angles, QuaternionFrameComparison


class BodyFrameQuaternionDemo:
    """Simple demonstration of body-frame quaternion rotations with frame comparison"""
    
    def __init__(self):
        # Initial orientation: pointing along x-axis
        self.original_arrow = np.array([1, 0, 0])
        self.current_arrow = self.original_arrow.copy()
        
        # Use the comparison class to track both frame types
        self.frame_comparison = QuaternionFrameComparison()
        
        # Keep track of rotation history
        self.rotation_history = []
    
    def apply_rotation(self, new_rotation_quaternion):
        """
        Apply a rotation and show both body frame and world frame results.
        This demonstrates the difference between the two approaches.
        
        Args:
            new_rotation_quaternion: Quaternion representing the new rotation
        """
        # Store the rotation in history
        self.rotation_history.append(new_rotation_quaternion.copy())
        
        # Apply rotation using the comparison class (tracks both frame types)
        self.frame_comparison.apply_rotation(new_rotation_quaternion)
        
        # Update the current arrow direction using body frame result (satellite-like)
        R = QuaternionMath.quaternion_to_rotation_matrix(self.frame_comparison.body_frame_cumulative)
        self.current_arrow = R @ self.original_arrow
        
        print(f"Applied rotation: {new_rotation_quaternion}")
        print(self.frame_comparison.get_comparison_string())
        print(f"Current arrow direction (body frame): [{self.current_arrow[0]:.3f}, {self.current_arrow[1]:.3f}, {self.current_arrow[2]:.3f}]")
        print()
    
    def reset(self):
        """Reset to initial orientation"""
        self.current_arrow = self.original_arrow.copy()
        self.frame_comparison.reset()
        self.rotation_history = []
        print("Reset to initial orientation!")
        print(self.frame_comparison.get_comparison_string())
        print()


def demonstrate_body_vs_world_frame():
    """Demonstrate the difference between body and world frame rotations"""
    print("BODY FRAME vs WORLD FRAME DEMONSTRATION")
    print("=" * 60)
    print("We'll apply two 90° rotations around Z-axis and see the difference.")
    print("Notice how the cumulative quaternions differ between the two approaches!")
    print()
    
    # Create two 90-degree Z rotations
    rot_90_z = QuaternionMath.axis_angle_to_quaternion([0, 0, 1], 90)
    
    demo = BodyFrameQuaternionDemo()
    
    print("Applying first 90° Z rotation:")
    demo.apply_rotation(rot_90_z)
    
    print("Applying second 90° Z rotation:")
    demo.apply_rotation(rot_90_z)
    
    print("EXPLANATION:")
    print("- Body Frame: Each rotation is relative to current orientation (like a satellite)")
    print("- World Frame: Each rotation is in fixed world coordinates")
    print("- Notice how the quaternions and total angles differ!")
    print("- Body frame gives 180° total (90° + 90° = 180°)")
    print("- World frame gives a different result because the second rotation")
    print("  is applied around the world's Z-axis, not the satellite's Z-axis")


def interactive_demo():
    """Interactive demonstration of body-frame quaternions"""
    print("INTERACTIVE BODY-FRAME QUATERNION DEMO")
    print("=" * 50)
    print("This demonstrates how satellites rotate using body-frame quaternions.")
    print("Each rotation is applied relative to the current orientation.")
    print()
    
    demo = BodyFrameQuaternionDemo()
    samples = create_sample_quaternions()
    axis_samples = create_sample_axis_angles()
    
    print("Available sample rotations:")
    for i, (name, _) in enumerate(samples.items(), 1):
        print(f"{i}. {name}")
    print()
    
    print("Available axis-angle rotations:")
    for i, (name, _) in enumerate(axis_samples.items(), len(samples) + 1):
        print(f"{i}. {name}")
    print()
    
    while True:
        print("\nOptions:")
        print("1-7. Apply sample quaternion rotation")
        print("8-14. Apply axis-angle rotation")
        print("15. Show current status")
        print("16. Reset")
        print("17. Exit")
        
        try:
            choice = int(input("\nEnter choice (1-17): "))
            
            if 1 <= choice <= 7:
                # Sample quaternion
                sample_names = list(samples.keys())
                quat = samples[sample_names[choice - 1]]
                print(f"\nApplying: {sample_names[choice - 1]}")
                demo.apply_rotation(quat)
                
            elif 8 <= choice <= 14:
                # Axis-angle
                axis_names = list(axis_samples.keys())
                axis, angle = axis_samples[axis_names[choice - 8]]
                quat = QuaternionMath.axis_angle_to_quaternion(axis, angle)
                print(f"\nApplying: {axis_names[choice - 8]}")
                demo.apply_rotation(quat)
                
            elif choice == 15:
                print("Current status:")
                print(demo.frame_comparison.get_comparison_string())
                print(f"Current arrow direction: [{demo.current_arrow[0]:.3f}, {demo.current_arrow[1]:.3f}, {demo.current_arrow[2]:.3f}]")
                
            elif choice == 16:
                demo.reset()
                
            elif choice == 17:
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-17.")
                
        except ValueError:
            print("Error: Please enter a valid number.")
        except IndexError:
            print("Error: Invalid choice number.")


def test_quaternion_operations():
    """Test basic quaternion operations"""
    print("QUATERNION OPERATIONS TEST")
    print("=" * 40)
    
    # Test normalization
    q1 = np.array([2, 4, 6, 8])
    q1_norm = QuaternionMath.normalize_quaternion(q1)
    print(f"Original: {q1}")
    print(f"Normalized: {q1_norm}")
    print(f"Magnitude: {np.linalg.norm(q1_norm):.6f}")
    print()
    
    # Test quaternion multiplication
    q2 = np.array([1, 0, 0, 0])  # Identity
    q3 = np.array([0, 1, 0, 0])  # 180° around X
    result = QuaternionMath.multiply_quaternions(q2, q3)
    print(f"Identity * 180°X = {result}")
    print()
    
    # Test axis-angle conversion
    axis, angle = QuaternionMath.quaternion_to_axis_angle(q3)
    print(f"180° around X -> axis: {axis}, angle: {angle:.1f}°")
    print()
    
    # Test SLERP
    q_start = np.array([1, 0, 0, 0])
    q_end = np.array([0, 0, 0, 1])  # 180° around Z
    q_mid = QuaternionMath.slerp_quaternions(q_start, q_end, 0.5)
    print(f"SLERP halfway between identity and 180°Z: {q_mid}")
    
    axis_mid, angle_mid = QuaternionMath.quaternion_to_axis_angle(q_mid)
    print(f"Midpoint rotation: {angle_mid:.1f}° around {axis_mid}")


def main():
    """Main function"""
    print("BODY-FRAME QUATERNION PRACTICE")
    print("=" * 50)
    print("This program demonstrates quaternion mathematics for satellite attitude control.")
    print("Body-frame rotations are how real satellites rotate - about their own axes!")
    print()
    
    while True:
        print("\nChoose a demonstration:")
        print("1. Body Frame vs World Frame comparison")
        print("2. Interactive body-frame quaternion demo")
        print("3. Test quaternion operations")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            demonstrate_body_vs_world_frame()
        elif choice == "2":
            interactive_demo()
        elif choice == "3":
            test_quaternion_operations()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")
    

if __name__ == "__main__":
    main()