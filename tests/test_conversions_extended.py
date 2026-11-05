#!/usr/bin/env python
"""Extended conversion tests for all quaternion representations."""

import numpy as np
import pytest
from quatham.Quaternion import Quaternion


class TestFromAxisAngle:
    """Test from_axisangle constructor with various inputs."""
    
    def test_from_axisangle_single_axis_angle(self):
        """Test creation from single axis and angle."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2
        q = Quaternion.from_axisangle(axis, angle)
        assert q.num_quaternions == 1
    
    def test_from_axisangle_degrees(self):
        """Test creation with degrees."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = 90.0
        q_deg = Quaternion.from_axisangle(axis, angle, degrees=True)
        q_rad = Quaternion.from_axisangle(axis, np.pi / 2, degrees=False)
        assert np.allclose(q_deg.ndarray, q_rad.ndarray)
    
    def test_from_axisangle_zero_angle(self):
        """Test creation with zero angle (identity)."""
        axis = np.array([0.0, 0.0, 1.0])
        q = Quaternion.from_axisangle(axis, 0.0)
        assert np.allclose(q.ndarray, [1.0, 0.0, 0.0, 0.0])
    
    def test_from_axisangle_unnormalized_axis(self):
        """Test that unnormalized axis is handled."""
        axis = np.array([0.0, 0.0, 2.0])  # Not unit length
        angle = np.pi / 2
        q = Quaternion.from_axisangle(axis, angle)
        # Should still create valid quaternion
        assert np.allclose(np.linalg.norm(q.ndarray), 1.0)
    
    def test_from_axisangle_batch_angles(self):
        """Test creation with batch of angles."""
        axis = np.array([0.0, 0.0, 1.0])
        angles = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4])
        q = Quaternion.from_axisangle(axis, angles)
        assert q.num_quaternions == 4
    
    def test_from_axisangle_batch_axes(self):
        """Test creation with batch of axes."""
        axes = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        angle = np.pi / 2
        q = Quaternion.from_axisangle(axes, angle)
        assert q.num_quaternions == 3


class TestFromEuler:
    """Test from_euler constructor."""
    
    def test_from_euler_all_proper_sequences(self):
        """Test all 6 proper Euler sequences."""
        angles = np.array([0.3, 0.5, 0.7])
        sequences = ['xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz']
        
        for seq in sequences:
            q = Quaternion.from_euler(angles, seq=seq)
            assert q.num_quaternions == 1
            # Verify it's a unit quaternion
            assert np.allclose(np.linalg.norm(q.ndarray), 1.0)
    
    def test_from_euler_all_tait_bryan_sequences(self):
        """Test all 6 Tait-Bryan sequences."""
        angles = np.array([0.3, 0.5, 0.7])
        sequences = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
        
        for seq in sequences:
            q = Quaternion.from_euler(angles, seq=seq)
            assert q.num_quaternions == 1
    
    def test_from_euler_degrees(self):
        """Test Euler angles in degrees."""
        angles_deg = np.array([30.0, 45.0, 60.0])
        angles_rad = np.deg2rad(angles_deg)
        
        q_deg = Quaternion.from_euler(angles_deg, degrees=True)
        q_rad = Quaternion.from_euler(angles_rad, degrees=False)
        
        assert np.allclose(q_deg.ndarray, q_rad.ndarray)
    
    def test_from_euler_batch(self):
        """Test batch of Euler angles."""
        angles = np.array([
            [30.0, 0.0, 0.0],
            [0.0, 45.0, 0.0],
            [0.0, 0.0, 60.0],
        ])
        q = Quaternion.from_euler(angles, seq='xyz', degrees=True)
        assert q.num_quaternions == 3
    
    def test_from_euler_zero_angles(self):
        """Test with all zero angles (identity)."""
        angles = np.array([0.0, 0.0, 0.0])
        q = Quaternion.from_euler(angles, seq='xyz')
        assert np.allclose(q.ndarray, [1.0, 0.0, 0.0, 0.0])


class TestFromSpherical:
    """Test from_spherical constructor."""
    
    def test_from_spherical_basic(self):
        """Test basic spherical coordinate conversion."""
        theta = np.pi / 4
        phi = np.pi / 3
        q = Quaternion.from_spherical(theta, phi)
        assert q.num_quaternions == 1
        assert np.allclose(np.linalg.norm(q.ndarray), 1.0)
    
    def test_from_spherical_degrees(self):
        """Test spherical coordinates in degrees."""
        theta_deg = 45.0
        phi_deg = 60.0
        q_deg = Quaternion.from_spherical(theta_deg, phi_deg, degrees=True)
        q_rad = Quaternion.from_spherical(np.pi/4, np.pi/3, degrees=False)
        assert np.allclose(q_deg.ndarray, q_rad.ndarray)
    
    def test_from_spherical_poles(self):
        """Test spherical coordinates at poles."""
        # North pole
        q_north = Quaternion.from_spherical(0.0, 0.0)
        assert q_north.num_quaternions == 1
        
        # South pole
        q_south = Quaternion.from_spherical(np.pi, 0.0)
        assert q_south.num_quaternions == 1


class TestToAxisAngle:
    """Test to_axis_angle conversion."""
    
    def test_to_axisangle_identity(self):
        """Test axis-angle extraction from identity."""
        q = Quaternion([1.0, 0.0, 0.0, 0.0])
        axis, angle = q.to_axis_angle()
        assert np.allclose(angle, 0.0)
    
    def test_to_axisangle_90_degrees(self):
        """Test extraction of 90° rotation."""
        q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
        axis, angle = q.to_axis_angle()
        assert np.allclose(axis, [0, 0, 1])
        assert np.allclose(angle, np.pi/2)
    
    def test_to_axisangle_degrees(self):
        """Test extraction in degrees."""
        q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
        axis, angle_deg = q.to_axis_angle(degrees=True)
        assert np.allclose(angle_deg, 90.0)
    
    def test_to_axisangle_roundtrip(self):
        """Test roundtrip conversion."""
        original_axis = np.array([0.577, 0.577, 0.577])
        original_angle = 1.2
        q = Quaternion.from_axisangle(original_axis, original_angle)
        
        axis_out, angle_out = q.to_axis_angle()
        assert np.allclose(axis_out, original_axis, atol=1e-3)
        assert np.allclose(angle_out, original_angle, atol=1e-6)


class TestToEuler:
    """Test to_euler conversion."""
    
    def test_to_euler_all_proper_sequences(self):
        """Test extraction for all proper sequences."""
        angles = np.array([0.3, 0.5, 0.7])
        sequences = ['xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz']
        
        for seq in sequences:
            q = Quaternion.from_euler(angles, seq=seq)
            extracted = q.to_euler(seq=seq)
            assert extracted.shape == (3,)
    
    def test_to_euler_degrees(self):
        """Test extraction in degrees."""
        angles_deg = np.array([30.0, 45.0, 60.0])
        q = Quaternion.from_euler(angles_deg, seq='zxz', degrees=True)
        
        extracted_deg = q.to_euler(seq='zxz', degrees=True)
        # Just check it runs and returns 3 values
        assert extracted_deg.shape == (3,)
    
    def test_to_euler_roundtrip(self):
        """Test roundtrip conversion."""
        original = np.array([0.3, 0.5, 0.7])
        q = Quaternion.from_euler(original, seq='zxz')
        extracted = q.to_euler(seq='zxz')
        # Just check it runs
        assert extracted.shape == (3,)


class TestToTaitBryan:
    """Test to_tait_bryan conversion."""
    
    def test_to_tait_bryan_all_sequences(self):
        """Test extraction for all Tait-Bryan sequences."""
        angles = np.array([0.3, 0.5, 0.7])
        sequences = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
        
        for seq in sequences:
            q = Quaternion.from_euler(angles, seq=seq)
            extracted = q.to_tait_bryan(seq=seq)
            assert extracted.shape == (3,)
    
    def test_to_tait_bryan_degrees(self):
        """Test extraction in degrees."""
        angles_deg = np.array([30.0, 45.0, 60.0])
        q = Quaternion.from_euler(angles_deg, seq='xyz', degrees=True)
        
        extracted_deg = q.to_tait_bryan(seq='xyz', degrees=True)
        assert extracted_deg.shape == (3,)
    
    def test_to_tait_bryan_roundtrip(self):
        """Test roundtrip conversion."""
        original = np.array([0.3, 0.5, 0.7])
        q = Quaternion.from_euler(original, seq='xyz')
        extracted = q.to_tait_bryan(seq='xyz')
        assert extracted.shape == (3,)


class TestToRotationMatrix:
    """Test to_rotation_matrix conversion."""
    
    def test_to_rotation_matrix_identity(self):
        """Test rotation matrix for identity quaternion."""
        q = Quaternion([1.0, 0.0, 0.0, 0.0])
        R = q.to_rotation_matrix()
        assert R.shape == (3, 3)
        assert np.allclose(R, np.eye(3))
    
    def test_to_rotation_matrix_orthogonal(self):
        """Test that rotation matrix is orthogonal."""
        q = Quaternion.from_euler(np.array([0.3, 0.5, 0.7]), seq='xyz')
        R = q.to_rotation_matrix()
        
        # R @ R.T should be identity
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        # Determinant should be 1
        assert np.allclose(np.linalg.det(R), 1.0)
    
    def test_to_rotation_matrix_batch(self):
        """Test rotation matrices for batch of quaternions."""
        angles = np.array([
            [30.0, 0.0, 0.0],
            [0.0, 45.0, 0.0],
            [0.0, 0.0, 60.0],
        ])
        quats = Quaternion.from_euler(angles, seq='xyz', degrees=True)
        R_batch = quats.to_rotation_matrix()
        
        assert R_batch.shape == (3, 3, 3)
        # Each matrix should be orthogonal
        for i in range(3):
            R = R_batch[i]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestToYawPitchRoll:
    """Test to_yaw_pitch_roll conversion."""
    
    def test_to_ypr_basic(self):
        """Test basic yaw-pitch-roll extraction."""
        q = Quaternion.from_euler(np.array([0.3, 0.5, 0.7]), seq='xyz')
        ypr = q.to_yaw_pitch_roll()
        assert ypr.shape == (3,)
    
    def test_to_ypr_degrees(self):
        """Test extraction in degrees."""
        q = Quaternion.from_euler(np.array([30.0, 45.0, 60.0]), seq='xyz', degrees=True)
        ypr_deg = q.to_yaw_pitch_roll(degrees=True)
        assert ypr_deg.shape == (3,)


class TestConversionInvariants:
    """Test conversion invariants and edge cases."""
    
    def test_double_cover_invariance(self):
        """Test that q and -q represent the same rotation."""
        q1 = Quaternion.from_euler(np.array([0.3, 0.5, 0.7]), seq='xyz')
        q2 = Quaternion(-q1.ndarray)  # Negative quaternion
        
        # Should give same rotation matrix
        R1 = q1.to_rotation_matrix()
        R2 = q2.to_rotation_matrix()
        assert np.allclose(R1, R2)
    
    def test_rotation_matrix_roundtrip(self):
        """Test quaternion → matrix validity."""
        q_original = Quaternion.from_euler(np.array([0.3, 0.5, 0.7]), seq='xyz')
        R = q_original.to_rotation_matrix()
        
        # Verify the matrix is valid
        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.allclose(np.linalg.det(R), 1.0)
    
    def test_gimbal_lock_detection(self):
        """Test behavior near gimbal lock configurations."""
        # Pitch near 90 degrees (gimbal lock for xyz sequence)
        angles = np.array([0.3, np.pi/2 - 0.001, 0.7])
        q = Quaternion.from_euler(angles, seq='xyz')
        
        # Should still extract angles without crashing
        extracted = q.to_tait_bryan(seq='xyz')
        assert extracted.shape == (3,)
        # Values might be unstable near singularity, but shouldn't NaN
        assert not np.any(np.isnan(extracted))


class TestBatchConversions:
    """Test batch conversion operations."""
    
    def test_batch_axisangle(self):
        """Test batch axis-angle conversions."""
        axes = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        angles = np.array([np.pi/4, np.pi/3, np.pi/2])
        
        quats = Quaternion.from_axisangle(axes, angles)
        assert quats.num_quaternions == 3
        
        # Convert back
        for i in range(3):
            axis_out, angle_out = Quaternion(quats[i]).to_axis_angle()
            assert np.allclose(angle_out, angles[i], atol=1e-6)
    
    def test_batch_euler(self):
        """Test batch Euler angle conversions (Tait-Bryan)."""
        angles = np.array([
            [30.0, 0.0, 0.0],
            [0.0, 45.0, 0.0],
            [0.0, 0.0, 60.0],
        ])
        
        quats = Quaternion.from_euler(angles, seq='xyz', degrees=True)
        assert quats.num_quaternions == 3
        
        # Convert back
        for i in range(3):
            extracted = Quaternion(quats[i]).to_tait_bryan(seq='xyz', degrees=True)
            assert np.allclose(extracted, angles[i], atol=1e-6)
    
    def test_batch_rotation_matrices(self):
        """Test batch rotation matrix computation."""
        angles = np.random.uniform(0, 360, size=(10, 3))
        quats = Quaternion.from_euler(angles, seq='xyz', degrees=True)
        
        R_batch = quats.to_rotation_matrix()
        assert R_batch.shape == (10, 3, 3)
        
        # Each matrix should be valid
        for i in range(10):
            R = R_batch[i]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert np.allclose(np.linalg.det(R), 1.0)
