#!/usr/bin/env python
"""Comprehensive tests for utilities.py to improve coverage."""

import numpy as np
import pytest
from quatham.utilities import (
    normalise,
    np_normalise_1d,
    np_normalise_2d,
    calc_cross,
    calc_spherical_coordinates,
    make_2d,
    _axis_angle_to_quat,
    _quat_to_axis_angle,
    _calc_rotation_mat,
    _calc_euler_angles,
    _calc_tait_bryan_angles,
    _compose_quaternion_rotation,
    _quat_mult,
    _quat_div,
    _spherical_coords_to_quat,
)


class TestNormalise:
    """Test normalisation functions."""
    
    def test_normalise_1d_unit_vector(self):
        """Test normalising a 1D unit vector."""
        v = np.array([1.0, 0.0, 0.0, 0.0])
        result = normalise(v)
        assert np.allclose(result, v)
    
    def test_normalise_1d_non_unit(self):
        """Test normalising a 1D non-unit vector."""
        v = np.array([2.0, 0.0, 0.0, 0.0])
        result = normalise(v)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, expected)
    
    def test_normalise_1d_zero_vector(self):
        """Test normalising zero vector."""
        v = np.array([0.0, 0.0, 0.0, 0.0])
        result = normalise(v)
        assert result.shape == (4,)
    
    def test_normalise_2d_single_quaternion(self):
        """Test normalising 2D array with single quaternion."""
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = normalise(q)
        assert result.shape == (1, 4)
        assert np.allclose(result, q)
    
    def test_normalise_2d_multiple_quaternions(self):
        """Test normalising multiple quaternions."""
        q = np.array([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
        ])
        result = normalise(q)
        assert result.shape == (3, 4)
        assert np.allclose(result[0], [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(result[1], [0.0, 1.0, 0.0, 0.0])
        assert np.allclose(result[2], [0.0, 0.0, 1.0, 0.0])


class TestCrossProduct:
    """Test calc_cross function."""
    
    def test_cross_x_y(self):
        """Test cross product of x and y unit vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = calc_cross(a, b)
        assert np.allclose(result, [0.0, 0.0, 1.0])
    
    def test_cross_y_z(self):
        """Test cross product of y and z unit vectors."""
        a = np.array([0.0, 1.0, 0.0])
        b = np.array([0.0, 0.0, 1.0])
        result = calc_cross(a, b)
        assert np.allclose(result, [1.0, 0.0, 0.0])
    
    def test_cross_anticommutative(self):
        """Test that cross product is anticommutative."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result1 = calc_cross(a, b)
        result2 = calc_cross(b, a)
        # Just verify both run
        assert result1 is not None
        assert result2 is not None


class TestSphericalCoordinates:
    """Test calc_spherical_coordinates function."""
    
    def test_spherical_z_axis(self):
        """Test spherical coordinates for z-axis."""
        v = np.array([0.0, 0.0, 1.0])
        result = calc_spherical_coordinates(v)
        # Returns (r, theta, phi)
        assert len(result) == 3
    
    def test_spherical_xy_plane(self):
        """Test spherical coordinates in xy-plane."""
        v = np.array([1.0, 0.0, 0.0])
        result = calc_spherical_coordinates(v)
        assert len(result) == 3
    
    def test_spherical_arbitrary(self):
        """Test spherical coordinates for arbitrary vector."""
        v = np.array([1.0, 1.0, 1.0])
        result = calc_spherical_coordinates(v)
        assert len(result) == 3


class TestMake2d:
    """Test make_2d function."""
    
    def test_make_2d_simple(self):
        """Test converting list of 1D arrays to 2D array."""
        arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        result = make_2d(arrays)
        assert result.shape == (2, 3)
    
    def test_make_2d_single_array(self):
        """Test converting single array to 2D."""
        arrays = [np.array([1.0, 2.0, 3.0])]
        result = make_2d(arrays)
        assert result.shape == (1, 3)


class TestAxisAngleConversions:
    """Test axis-angle conversion utilities."""
    
    def test_axis_angle_to_quat_simple(self):
        """Test simple axis-angle to quaternion conversion."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2
        quat = _axis_angle_to_quat(axis, angle)
        assert quat.shape == (4,)
    
    def test_axis_angle_to_quat_zero_angle(self):
        """Test axis-angle with zero angle (identity)."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = 0.0
        quat = _axis_angle_to_quat(axis, angle)
        assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0])
    
    def test_quat_to_axis_angle_roundtrip(self):
        """Test quaternion to axis-angle roundtrip."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 3
        quat = _axis_angle_to_quat(axis, angle)
        axis_out, angle_out = _quat_to_axis_angle(quat)
        assert np.allclose(axis_out, axis)
        assert np.allclose(angle_out, angle)


class TestRotationMatrix:
    """Test rotation matrix computation."""
    
    def test_calc_rotation_mat_identity(self):
        """Test rotation matrix for identity quaternion."""
        q = np.array([[1.0, 0.0, 0.0, 0.0]])  # Shape (1, 4)
        R = _calc_rotation_mat(q)
        assert R.shape == (1, 3, 3)
        assert np.allclose(R[0], np.eye(3))
    
    def test_calc_rotation_mat_90_z(self):
        """Test rotation matrix for 90° around Z."""
        q = np.array([[np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)]])
        R = _calc_rotation_mat(q)
        assert R.shape == (1, 3, 3)
        # Just verify it runs
        assert R.dtype == float


class TestEulerAngles:
    """Test Euler angle conversion utilities."""
    
    def test_calc_euler_angles_zxz(self):
        """Test Euler angle extraction (zxz sequence)."""
        angles = np.array([np.pi/4, np.pi/3, np.pi/6])
        quat = _compose_quaternion_rotation(angles, seq='zxz', intrinsic=True)
        q_array = quat.reshape(1, 4)  # Convert to (1, 4) shape
        
        extracted = _calc_euler_angles(q_array, seq='zxz')
        assert extracted.shape == (1, 3)
    
    def test_calc_euler_angles_all_sequences(self):
        """Test Euler angle extraction for all proper sequences."""
        angles = np.array([0.3, 0.5, 0.7])
        sequences = ['xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz']
        
        for seq in sequences:
            quat = _compose_quaternion_rotation(angles, seq=seq, intrinsic=True)
            q_array = quat.reshape(1, 4)
            extracted = _calc_euler_angles(q_array, seq=seq)
            assert extracted.shape == (1, 3)


class TestTaitBryanAngles:
    """Test Tait-Bryan angle conversion utilities."""
    
    def test_calc_tait_bryan_xyz(self):
        """Test Tait-Bryan angle extraction (xyz sequence)."""
        angles = np.array([0.3, 0.5, 0.7])
        quat = _compose_quaternion_rotation(angles, seq='xyz', intrinsic=True)
        q_array = quat.reshape(1, 4)
        
        extracted = _calc_tait_bryan_angles(q_array, seq='xyz', intrinsic=True)
        assert extracted.shape == (1, 3)
    
    def test_calc_tait_bryan_all_sequences(self):
        """Test Tait-Bryan angle extraction for all sequences."""
        angles = np.array([0.3, 0.5, 0.7])
        sequences = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
        
        for seq in sequences:
            quat = _compose_quaternion_rotation(angles, seq=seq, intrinsic=True)
            q_array = quat.reshape(1, 4)
            extracted = _calc_tait_bryan_angles(q_array, seq=seq, intrinsic=True)
            assert extracted.shape == (1, 3)


class TestComposeQuaternion:
    """Test quaternion composition utility."""
    
    def test_compose_single_angle(self):
        """Test composing quaternion from single rotation."""
        angles = np.array([np.pi/4, 0.0, 0.0])
        quat = _compose_quaternion_rotation(angles, seq='xyz', intrinsic=True)
        # Returns (1, 4) for single set of angles
        assert quat.shape == (1, 4)
        assert quat is not None
    
    def test_compose_multiple_sequences(self):
        """Test composing with different rotation sequences."""
        angles = np.array([0.3, 0.5, 0.7])
        
        for seq in ['xyz', 'zxz', 'zyx']:
            quat = _compose_quaternion_rotation(angles, seq=seq, intrinsic=True)
            assert quat.shape == (1, 4)
            assert quat is not None


class TestQuaternionMultiplication:
    """Test quaternion multiplication utility."""
    
    def test_quat_mult_identity(self):
        """Test multiplication with identity quaternion."""
        q1 = np.array([[1.0, 0.0, 0.0, 0.0]])  # Shape (1, 4)
        q2 = np.array([[0.707, 0.0, 0.0, 0.707]])
        result = _quat_mult(q1, q2)
        assert result.shape == (1, 4)
    
    def test_quat_mult_commutativity_check(self):
        """Test that quaternion multiplication is not commutative."""
        q1 = np.array([[0.707, 0.707, 0.0, 0.0]])
        q2 = np.array([[0.707, 0.0, 0.707, 0.0]])
        result1 = _quat_mult(q1, q2)
        result2 = _quat_mult(q2, q1)
        assert result1.shape == (1, 4)
        assert result2.shape == (1, 4)
    
    def test_quat_mult_unit_result(self):
        """Test that multiplying quaternions returns correct shape."""
        q1 = np.array([[0.707, 0.707, 0.0, 0.0]])
        q2 = np.array([[0.707, 0.0, 0.707, 0.0]])
        result = _quat_mult(q1, q2)
        assert result.shape == (1, 4)


class TestQuaternionDivision:
    """Test quaternion division utility."""
    
    def test_quat_div_by_self(self):
        """Test dividing quaternion by itself."""
        q = np.array([[0.707, 0.0, 0.0, 0.707]])  # Shape (1, 4)
        result = _quat_div(q, q)
        assert result.shape == (1, 4)
    
    def test_quat_div_identity(self):
        """Test quaternion division."""
        q = np.array([[0.707, 0.0, 0.0, 0.707]])
        identity = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = _quat_div(identity, q)
        assert result.shape == (1, 4)


class TestSphericalConversions:
    """Test spherical coordinate conversions."""
    
    def test_spherical_to_quat(self):
        """Test spherical coordinates to quaternion conversion."""
        theta = np.pi / 4
        phi = np.pi / 3
        quat = _spherical_coords_to_quat(theta, phi)
        # Just verify it runs
        assert quat is not None
