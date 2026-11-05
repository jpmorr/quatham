#!/usr/bin/env python
"""Quaternion Module.

Simple module to allow quaternion-based calculations and conversions. Supports calculations on arrays of quaternions
also.

Allows translation between quaternions and Euler angles, Tait-Bryan angles, rotation matrices and axis-angle
representations.
"""
from __future__ import annotations

from typing import Self


__copyright__ = 'Copyright 2026'
__status__ = 'Development'

# Standard Library


# Imports
import numpy as np
from numpy.typing import NDArray

# Local Sources
from .utilities import (
    _axis_angle_to_quat,
    _calc_euler_angles,
    _calc_rotation_mat,
    _calc_tait_bryan_angles,
    _compose_quaternion_rotation,
    _quat_to_axis_angle,
    _spherical_coords_to_quat,
    normalise, _quat_mult, _quat_div,
)


class Quaternion:
    """
    Quaternion Class for handling quaternions and their transformations and translations.

    Parameters
    ----------
    value : NDArray or list or tuple or None, default None
        Quaternion data in scalar-first format [w, x, y, z]. Can be a single quaternion (shape (4,))
        or multiple quaternions (shape (N, 4)). If None, creates an empty quaternion object.

    Attributes
    ----------
    scalar : NDArray
        The real (scalar) component w of the quaternion(s).
    vector : NDArray
        The imaginary (vector) component [x, y, z] of the quaternion(s).
    w : NDArray
        Alias for scalar component.
    x : NDArray
        x component of the vector part.
    y : NDArray
        y component of the vector part.
    z : NDArray
        z component of the vector part.
    num_quaternions : int or None
        Number of quaternions stored, or None if empty.
    conjugate : Quaternion
        The conjugate quaternion (scalar negated vector part).
    magnitude : NDArray
        Magnitude (norm) of each quaternion.

    Examples
    --------
    Create a quaternion from a list::

        q = Quaternion([1.0, 0.0, 0.0, 0.0])

    Create from Euler angles::

        q = Quaternion.from_euler([0.0, 0.0, np.pi/2], seq='z', degrees=False)

    Get rotation matrix::

        rot_mat = q.get_rotation_matrix()
    """

    def __init__(self, value: NDArray | list | tuple | None = None) -> None:

        if value is not None:
            vect_array = np.asarray(value)

            if vect_array.ndim == 1:
                vect_array = vect_array[None, :]
            if vect_array.shape[1] == 1:
                vect_array = vect_array.T

            self.ndarray = vect_array
            self.num_quaternions = self.ndarray.shape[0]
            self._scalar = self.ndarray[:, 0]
            self._vector = self.ndarray[:, 1::] * 1.0
            self.w = self.scalar
            self.x = self.ndarray[:, 1]
            self.y = self.ndarray[:, 2]
            self.z = self.ndarray[:, 3]

        else:
            self.ndarray = None
            self.num_quaternions = None
            self._scalar = None
            self._vector = None
            self.w = None
            self.x = None
            self.y = None
            self.z = None

    @property
    def scalar(self) -> NDArray:
        """
        Return the real or scalar component of the quaternion object.

        Returns
        -------
        NDArray
            The scalar (w) component of the quaternion(s).
        """
        return self._scalar

    @property
    def vector(self) -> NDArray:
        """
        Return the imaginary or vector component of the quaternion object.

        Returns
        -------
        NDArray
            A numpy array of shape (N, 3) containing [x, y, z] components.
            NOT guaranteed to be a unit vector.
        """
        return self._vector

    @property
    def conjugate(self) -> Self:
        """
        Conjugate of quaternion.

        The conjugate negates the vector part while keeping the scalar part unchanged.

        Returns
        -------
        Quaternion
            A new Quaternion object representing the conjugate.
        """
        return Quaternion(self.ndarray * np.array([1, -1, -1, -1]))

    @property
    def magnitude(self) -> NDArray:
        """
        Magnitude (Euclidean norm) of the quaternion(s).

        Returns
        -------
        NDArray
            Array of magnitudes for each quaternion, or a scalar if single quaternion.
        """
        try:
            return np.linalg.norm(self.ndarray, axis=1)
        except np.AxisError:
            return np.linalg.norm(self.ndarray)

    @classmethod
    def from_axisangle(cls, v: NDArray | list | tuple, theta: float | NDArray | list | tuple,
                       degrees: bool = False) -> Self:
        """
        Create a quaternion from an axis-angle representation.

        Parameters
        ----------
        v : NDArray or list or tuple
            Rotation axis as a 3-element vector or array of shape (N, 3).
        theta : float or NDArray or list or tuple
            Rotation angle(s) around the axis.
        degrees : bool, default False
            If True, theta is interpreted as degrees; otherwise radians.

        Returns
        -------
        Quaternion
            A new Quaternion object.

        Examples
        --------
        Rotate 90° around the Z-axis::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> axis = np.array([0, 0, 1])
            >>> angle = np.pi / 2  # 90 degrees in radians
            >>> q = Quaternion.from_axisangle(axis, angle)
            >>> print(q)
            Quaternion(1.5708; [0. 0. 1.])

        Using degrees::

            >>> q = Quaternion.from_axisangle([0, 0, 1], 90, degrees=True)

        Batch creation with multiple axes::

            >>> axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> angles = np.array([45, 30, 60])
            >>> quats = Quaternion.from_axisangle(axes, angles, degrees=True)
            >>> quats.num_quaternions
            3
        """
        return cls(_axis_angle_to_quat(v, theta, degrees=degrees))

    @classmethod
    def from_euler(cls, angles: NDArray, seq: str = 'xyz', degrees: bool = False,
                   intrinsic: bool = False) -> Self:
        """
        Create a quaternion from Euler angles.

        Parameters
        ----------
        angles : NDArray
            Array of Euler angles.
        seq : str, default 'xyz'
            Sequence of axes for rotations (e.g., 'xyz', 'zxz', etc.).
        degrees : bool, default False
            If True, angles are in degrees; otherwise radians.
        intrinsic : bool, default False
            If True, rotations are intrinsic (local axes); otherwise extrinsic (fixed axes).

        Returns
        -------
        Quaternion
            A new Quaternion object.

        Examples
        --------
        Create from 90° rotations around each axis::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> angles = np.array([90, 45, 30])
            >>> q = Quaternion.from_euler(angles, degrees=True)
            >>> print(q)
            Quaternion(2.3562; [0.9239 0.3827 1.0607])

        Using radians (default)::

            >>> angles_rad = np.array([np.pi/2, np.pi/4, np.pi/6])
            >>> q = Quaternion.from_euler(angles_rad)

        Different rotation sequence (proper Euler angles)::

            >>> q = Quaternion.from_euler([30, 45, 60], seq='zxz', degrees=True)

        Intrinsic vs extrinsic rotations::

            # Intrinsic (rotations about moving axes)
            >>> q_intrinsic = Quaternion.from_euler([30, 45, 60], intrinsic=True, degrees=True)
            
            # Extrinsic (rotations about fixed axes)
            >>> q_extrinsic = Quaternion.from_euler([30, 45, 60], intrinsic=False, degrees=True)

        Batch creation::

            >>> angles_batch = np.array([[30, 0, 0], [0, 45, 0], [0, 0, 60]])
            >>> quats = Quaternion.from_euler(angles_batch, seq='xyz', degrees=True)
            >>> quats.num_quaternions
            3
        """
        return cls(_compose_quaternion_rotation(angles, seq=seq, degrees=degrees, intrinsic=intrinsic))

    @classmethod
    def from_spherical(cls, theta: float, phi: float, degrees: bool = False) -> Self:
        """
        Create a quaternion from spherical coordinates.

        Parameters
        ----------
        theta : float
            Polar angle (angle from z-axis).
        phi : float
            Azimuthal angle (angle in x-y plane from x-axis).
        degrees : bool, default False
            If True, angles are in degrees; otherwise radians.

        Returns
        -------
        Quaternion
            A new Quaternion object.

        Examples
        --------
        Create quaternion from spherical coordinates::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> theta = np.pi / 4  # 45° from z-axis
            >>> phi = np.pi / 3    # 60° from x-axis in xy-plane
            >>> q = Quaternion.from_spherical(theta, phi)
            >>> print(q)
            Quaternion(1.0472; [0.5 0.866 0.5])

        Using degrees::

            >>> q = Quaternion.from_spherical(45, 60, degrees=True)
        """
        return cls(_spherical_coords_to_quat(theta, phi, degrees=degrees))

    def to_axis_angle(self, degrees: bool = False) -> tuple[NDArray, NDArray | float]:
        """
        Convert quaternion to axis-angle representation.

        Parameters
        ----------
        degrees : bool, default False
            If True, returns angle(s) in degrees; otherwise radians.

        Returns
        -------
        tuple of (NDArray, NDArray or float)
            (axis, theta) where axis is the normalized rotation axis vector(s) of shape
            (3,) or (N, 3), and theta is the rotation angle(s) as a float or array of shape (N,).

        Examples
        --------
        Extract axis and angle from a quaternion::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
            >>> axis, angle = q.to_axis_angle()
            >>> print(f"Axis: {axis}, Angle: {angle:.4f} rad")
            Axis: [0. 0. 1.], Angle: 1.5708 rad

        Get angle in degrees::

            >>> axis, angle_deg = q.to_axis_angle(degrees=True)
            >>> print(f"Angle: {angle_deg:.1f}°")
            Angle: 90.0°

        Roundtrip conversion::

            >>> original_axis = np.array([1, 0, 0])
            >>> original_angle = np.pi / 3  # 60 degrees
            >>> q = Quaternion.from_axisangle(original_axis, original_angle)
            >>> axis_out, angle_out = q.to_axis_angle()
            >>> np.allclose(axis_out, original_axis)
            True
            >>> np.allclose(angle_out, original_angle)
            True
        """
        axes, thetas = _quat_to_axis_angle(self.ndarray, degrees=degrees)

        if self.num_quaternions == 1:
            return axes[0], thetas[0]
        return axes, thetas

    def get_rotation_vector(self) -> NDArray:
        """
        Convert quaternion to rotation vector (axis * angle).

        Returns
        -------
        NDArray
            Rotation vector of shape (3,) for a single quaternion or (N, 3) for multiple quaternions,
            where each row is axis * angle.
        """
        axis, theta = self.to_axis_angle()

        if self.num_quaternions == 1:
            return theta * axis
        return theta[:, None] * axis

    def to_rotation_matrix(self) -> NDArray:
        """
        Convert quaternion to a 3x3 rotation matrix.

        Returns
        -------
        NDArray
            A 3x3 rotation matrix. For multiple quaternions, returns shape (N, 3, 3).
            Currently, returns a single 3x3 matrix for the first quaternion.

        Examples
        --------
        Get rotation matrix from quaternion::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)  # 90° around Z
            >>> R = q.to_rotation_matrix()
            >>> print(R)
            [[ 6.123e-17 -1.000e+00  0.000e+00]
             [ 1.000e+00  6.123e-17  0.000e+00]
             [ 0.000e+00  0.000e+00  1.000e+00]]

        Verify it's a valid rotation matrix::

            >>> # Should be orthogonal: R @ R.T = I
            >>> np.allclose(R @ R.T, np.eye(3))
            True
            >>> # Determinant should be 1
            >>> np.allclose(np.linalg.det(R), 1.0)
            True

        Batch rotation matrices::

            >>> angles = np.array([[30, 0, 0], [0, 45, 0], [0, 0, 60]])
            >>> quats = Quaternion.from_euler(angles, degrees=True)
            >>> R_batch = quats.to_rotation_matrix()
            >>> R_batch.shape
            (3, 3, 3)
        """
        mats = _calc_rotation_mat(self.ndarray)

        if self.num_quaternions == 1:
            return mats[0]
        return mats

    def get_rotation_matrix_flat(self) -> NDArray:
        """
        Convert quaternion to a flat 9-element rotation matrix array.

        Returns
        -------
        NDArray
            A flat array of shape (N, 9) containing rotation matrix elements.
            For a single quaternion, returns shape (9,).
        """
        mats = _calc_rotation_mat(self.ndarray)
        return mats.reshape(mats.shape[0], 9)

    def to_yaw_pitch_roll(self, degrees: bool = False) -> NDArray:
        """
        Get the intrinsic Tait-Bryan angles following the z-y-x convention (yaw-pitch-roll).

        Parameters
        ----------
        degrees : bool, default False
            If True, returns angles in degrees; otherwise radians.

        Returns
        -------
        NDArray
            Array of shape (N, 3) containing [yaw, pitch, roll] angles.

        Examples
        --------
        Extract yaw, pitch, roll from quaternion::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> q = Quaternion.from_euler([30, 45, 60], seq='xyz', degrees=True)
            >>> ypr = q.to_yaw_pitch_roll()
            >>> print(f"Yaw: {ypr[0]:.1f}°, Pitch: {ypr[1]:.1f}°, Roll: {ypr[2]:.1f}°")
            Yaw: 30.0°, Pitch: 45.0°, Roll: 60.0°

        Get angles in radians::

            >>> ypr_rad = q.to_yaw_pitch_roll(degrees=False)
        """
        """
        return self.to_tait_bryan(seq='zyx', degrees=degrees)


    def to_euler(self, seq: str = 'zxz', degrees: bool = False,
                 intrinsic: bool = True) -> NDArray:
        """
        Calculate Euler rotation angles from the quaternion.

        Euler angles refer to sequences of rotations about 2 distinct axes.
        The supported sequences are: x-y-x, x-z-x, y-x-y, y-z-y, z-y-z, z-x-z.

        Intrinsic refers to rotations about the local coordinate system, while extrinsic
        rotations are about the fixed global axes.

        Parameters
        ----------
        seq : str, default 'zxz'
            The Euler angle sequence (one of 'xzx', 'xyx', 'yxy', 'yzy', 'zxz', 'zyz').
        degrees : bool, default False
            If True, returns angles in degrees; otherwise radians.
        intrinsic : bool, default True
            If True, uses intrinsic rotations; otherwise extrinsic.

        Returns
        -------
        NDArray
            Array of shape (N, 3) containing [alpha, beta, gamma] Euler angles.

        Raises
        ------
        ValueError
            If ``seq`` is not one of the supported sequences.

        Examples
        --------
        Extract Euler angles (proper sequence)::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> q = Quaternion.from_euler([30, 45, 60], seq='zxz', degrees=True)
            >>> euler = q.to_euler(seq='zxz', degrees=True)
            >>> print(f"α={euler[0]:.1f}°, β={euler[1]:.1f}°, γ={euler[2]:.1f}°")
            α=30.0°, β=45.0°, γ=60.0°

        Different Euler sequence::

            >>> q = Quaternion.from_euler([30, 45, 60], seq='xyx', degrees=True)
            >>> euler = q.to_euler(seq='xyx', degrees=True)

        Extrinsic rotations::

            >>> euler_extrinsic = q.to_euler(seq='zxz', intrinsic=False, degrees=True)
        """
        valid = {"xzx", "xyx", "yxy", "yzy", "zxz", "zyz"}
        if seq.lower() not in valid:
            raise ValueError(f"Invalid Euler angle sequence '{seq}'. "
                             f"Supported sequences: {', '.join(sorted(valid))}")

        angles = _calc_euler_angles(self.ndarray, seq)

        if not intrinsic:
            angles = angles[:, ::-1]

        if degrees:
            angles = np.degrees(angles)

        if self.num_quaternions == 1:
            return angles[0]

        return angles

    def to_tait_bryan(self, seq: str = 'xyz', degrees: bool = False,
                      intrinsic: bool = True) -> NDArray:
        """
        Calculate Tait-Bryan rotation angles from the quaternion.

        Tait-Bryan angles (also called Cardan angles, nautical angles, or yaw-pitch-roll)
        refer to sequences of rotations about 3 distinct axes.
        The supported sequences are: x-y-z, y-z-x, z-x-y, x-z-y, z-y-x, y-x-z.

        Intrinsic refers to rotations about the local coordinate system, while extrinsic
        rotations are about the fixed global axes.

        Parameters
        ----------
        seq : str, default 'xyz'
            The Tait-Bryan angle sequence (one of 'xyz', 'yzx', 'zxy', 'xzy', 'zyx', 'yxz').
        degrees : bool, default False
            If True, returns angles in degrees; otherwise radians.
        intrinsic : bool, default True
            If True, uses intrinsic rotations; otherwise extrinsic.

        Returns
        -------
        NDArray
            Array of shape (N, 3) containing [alpha, beta, gamma] Tait-Bryan angles.

        Raises
        ------
        ValueError
            If ``seq`` is not one of the supported sequences.

        Examples
        --------
        Extract Tait-Bryan angles::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> q = Quaternion.from_euler([30, 45, 60], seq='xyz', degrees=True)
            >>> tb = q.to_tait_bryan(seq='xyz', degrees=True)
            >>> print(f"α={tb[0]:.1f}°, β={tb[1]:.1f}°, γ={tb[2]:.1f}°")
            α=30.0°, β=45.0°, γ=60.0°

        Different sequence (zyx = yaw-pitch-roll)::

            >>> tb = q.to_tait_bryan(seq='zyx', degrees=True)
            >>> print(f"Yaw={tb[0]:.1f}°, Pitch={tb[1]:.1f}°, Roll={tb[2]:.1f}°")

        Extrinsic rotations::

            >>> tb_extrinsic = q.to_tait_bryan(seq='xyz', intrinsic=False, degrees=True)
        """
        valid = {"xzy", "xyz", "yxz", "yzx", "zyx", "zxy"}
        if seq.lower() not in valid:
            raise ValueError(
                f"Invalid Tait-Bryan intrinsic sequence '{seq}'. "
                f"Supported sequences: {', '.join(sorted(valid))}"
            )
        angles = _calc_tait_bryan_angles(self.ndarray, seq, intrinsic)

        if degrees:
            angles = np.degrees(angles)

        if self.num_quaternions == 1:
            return angles[0]

        return angles

    def vector_norm(self) -> NDArray | float:
        """
        L2 norm of the vector component of the quaternion.

        Returns
        -------
        NDArray or float
            The norm of the vector (imaginary) part of each quaternion.
        """
        if self.num_quaternions == 1:
            return float(np.linalg.norm(self._vector[0]))
        return np.linalg.norm(self._vector, axis=1)

    def normalise(self) -> Self:
        """
        Return a normalised (unit) quaternion.

        Returns
        -------
        Quaternion
            A new Quaternion object normalized to unit length.

        Examples
        --------
        Normalize a non-unit quaternion::

            >>> from quatham import Quaternion
            >>> q = Quaternion([2.0, 0.0, 0.0, 0.0])
            >>> q.magnitude
            2.0
            >>> q_unit = q.normalise()
            >>> q_unit.magnitude
            1.0

        Normalize batch of quaternions::

            >>> quats = Quaternion([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0]])
            >>> quats_unit = quats.normalise()
            >>> quats_unit.magnitude
            array([1., 1., 1.])
        """
        return Quaternion(normalise(self.ndarray))

    def normalize(self) -> Self:
        """
        Return a normalised (unit) quaternion (alias for normalise).

        Returns
        -------
        Quaternion
            A new Quaternion object normalized to unit length.
        """
        return self.normalise()

    def tolist(self) -> list:
        """
        Convert quaternion data to a Python list.

        Returns
        -------
        list
            List representation of the quaternion array.
        """
        return self.ndarray.tolist()

    def _divide_by_quaternion(self, q2: Self) -> Self:
        """
        Divide this quaternion by another quaternion.

        Parameters
        ----------
        q2 : Quaternion
            The divisor quaternion.

        Returns
        -------
        Quaternion
            Result of quaternion division.
        """
        return Quaternion(_quat_div(self.ndarray, q2.ndarray))

    def _multiply_with_quaternion(self, q2: Self) -> Self:
        """
        Multiply this quaternion with another quaternion.

        Parameters
        ----------
        q2 : Quaternion
            The quaternion to multiply with.

        Returns
        -------
        Quaternion
            Result of quaternion multiplication.
        """
        return Quaternion(_quat_mult(self.ndarray, q2.ndarray))

    def _multiply_with_vector(self, vec: NDArray) -> NDArray:
        """
        Rotate a vector by this quaternion.

        Parameters
        ----------
        vec : NDArray
            A 3D vector to rotate, shape (3,) or (N, 3).

        Returns
        -------
        NDArray
            The rotated 3D vector(s), shape (3,) or (N, 3).
        """
        q2 = Quaternion(vec)
        return (self * q2 * self.conjugate).ndarray[:, 1:]

    def __getitem__(self, item: int | slice) -> NDArray:
        """
        Get item(s) from the quaternion array by index or slice.

        Parameters
        ----------
        item : int or slice
            Index or slice to apply to the underlying quaternion array.

        Returns
        -------
        NDArray
            The selected quaternion data.
        """
        return self.ndarray[item]

    def __array__(self) -> NDArray:
        """
        Convert the quaternion to a numpy array.

        Returns
        -------
        NDArray
            The quaternion data as a numpy array of shape (N, 4).
        """
        return np.array(self.ndarray)

    def __str__(self):
        """
        Return a string representation of the quaternion(s).

        Returns
        -------
        str
            Formatted string showing angle and axis for each quaternion.
        """
        if self.ndarray is None:
            return "Quaternion(empty)"
        axes, thetas = _quat_to_axis_angle(self.ndarray)
        if self.num_quaternions == 1:
            return f"Quaternion({thetas[0]:.6f}; {axes[0]})\n"
        return str([f"Quaternion({theta:.6f}; {axis})\n" for axis, theta in zip(axes, thetas, strict=True)])

    def __repr__(self):
        """
        Return a developer-friendly string representation.

        Returns
        -------
        str
            Array representation of the quaternion data.
        """
        return np.array_repr(self.ndarray, precision=6, suppress_small=True)

    def __add__(self, other: Self | NDArray | list | tuple) -> Self:
        """
        Add another quaternion or array to this quaternion.

        Parameters
        ----------
        other : Quaternion or array-like
            Quaternion or array of components to add.

        Returns
        -------
        Quaternion
            Sum of the quaternions.

        Examples
        --------
        Add two quaternions::

            >>> from quatham import Quaternion
            >>> q1 = Quaternion([1, 0, 0, 0])
            >>> q2 = Quaternion([0, 1, 0, 0])
            >>> q3 = q1 + q2
            >>> print(q3)
            Quaternion(1.000000; [1. 0. 0.])

        Add quaternion and array::

            >>> q = Quaternion([1, 0, 0, 0]) + [0.5, 0.5, 0, 0]
        """
        if isinstance(other, Quaternion):
            return Quaternion(self.ndarray + other.ndarray)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Quaternion(self.ndarray + np.asarray(other))
        raise TypeError(f"Cannot add Quaternion and {type(other)}")

    def __radd__(self, other: Self | NDArray | list | tuple) -> Self:
        """
        Right addition for quaternion.

        Parameters
        ----------
        other : Quaternion or array-like
            Quaternion or array of components to add.

        Returns
        -------
        Quaternion
            Sum of the quaternions.
        """
        return self.__add__(other)

    def __sub__(self, other: Self | NDArray | list | tuple) -> Self:
        """
        Subtract another quaternion or array from this quaternion.

        Parameters
        ----------
        other : Quaternion or array-like
            Quaternion or array of components to subtract.

        Returns
        -------
        Quaternion
            Difference of the quaternions.
        """
        if isinstance(other, Quaternion):
            return Quaternion(self.ndarray - other.ndarray)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Quaternion(self.ndarray - np.asarray(other))
        raise TypeError(f"Cannot subtract {type(other)} from Quaternion")

    def __rsub__(self, other: Self | NDArray | list | tuple) -> Self:
        """
        Right subtraction for quaternion.

        Parameters
        ----------
        other : Quaternion or array-like
            Quaternion or array of components to subtract from.

        Returns
        -------
        Quaternion
            Difference of the quaternions.
        """
        if isinstance(other, Quaternion):
            return Quaternion(other.ndarray - self.ndarray)
        elif isinstance(other, (list, tuple, np.ndarray)):
            return Quaternion(np.asarray(other) - self.ndarray)
        raise TypeError(f"Cannot subtract Quaternion from {type(other)}")

    def __neg__(self) -> Self:
        """
        Negate all components of the quaternion(s).

        Returns
        -------
        Quaternion
            Negated quaternion.
        """
        return Quaternion(-self.ndarray)

    def __mul__(self, other: Self | NDArray | list | tuple | float | int) -> Self | NDArray:
        """
        Multiply quaternion by another quaternion, scalar, or rotate a vector.

        Parameters
        ----------
        other : Quaternion or NDArray or list or tuple or float or int
            If Quaternion, performs quaternion multiplication.
            If float or int, performs scalar multiplication.
            If array-like of length 3, rotates the vector by this quaternion.

        Returns
        -------
        Quaternion or NDArray
            Result quaternion, scaled quaternion, or rotated vector.

        Raises
        ------
        TypeError or ValueError
            If other is an unknown type or vector has invalid length.

        Examples
        --------
        Quaternion multiplication (Hamilton product)::

            >>> from quatham import Quaternion
            >>> import numpy as np
            >>> q1 = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
            >>> q2 = Quaternion.from_axisangle([1, 0, 0], np.pi/2)
            >>> q3 = q1 * q2

        Scalar multiplication::

            >>> q = Quaternion([1, 0, 0, 0])
            >>> q_scaled = q * 2.0

        Rotate a vector::

            >>> q = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
            >>> v = np.array([1.0, 0.0, 0.0])
            >>> v_rotated = q * v
            >>> print(v_rotated)
            [0. 1. 0.]
        """
        if isinstance(other, Quaternion):
            return self._multiply_with_quaternion(other)

        elif isinstance(other, (int, float, np.number)):
            return Quaternion(self.ndarray * other)

        elif isinstance(other, (list, tuple, np.ndarray)):
            arr = np.asarray(other)
            if arr.ndim == 0:
                return Quaternion(self.ndarray * arr)
            if arr.shape[-1] == 3 and (arr.ndim == 1 or arr.ndim == 2):
                if arr.ndim == 1:
                    new_other = np.zeros(4)
                    new_other[1:] = arr
                else:
                    new_other = np.zeros((arr.shape[0], 4))
                    new_other[:, 1:] = arr
                return self._multiply_with_vector(new_other)
            elif arr.shape[-1] == 4 and (arr.ndim == 1 or arr.ndim == 2):
                return self._multiply_with_quaternion(Quaternion(arr))
            else:
                raise ValueError(f"Input vector/array has invalid shape {arr.shape}")

        else:
            raise TypeError(f"Multiplication with unknown type {type(other)}")

    def __rmul__(self, other: float | int | np.number | NDArray) -> Self:
        """
        Multiply a scalar by this quaternion (scalar * q).

        Parameters
        ----------
        other : float or int or np.number or NDArray
            Scalar multiplier.

        Returns
        -------
        Quaternion
            Scaled quaternion.
        """
        if isinstance(other, (int, float, np.number)):
            return Quaternion(self.ndarray * other)
        arr = np.asarray(other)
        if arr.ndim == 0:
            return Quaternion(self.ndarray * arr)
        raise TypeError(f"Cannot multiply {type(other)} with Quaternion")

    def __truediv__(self, other: Self | float | int | np.number | NDArray) -> Self:
        """
        Divide this quaternion by another quaternion or scalar.

        Parameters
        ----------
        other : Quaternion or float or int or np.number or NDArray
            The divisor quaternion or scalar.

        Returns
        -------
        Quaternion
            Result of quaternion division.

        Raises
        ------
        TypeError or ValueError
            If other is not a Quaternion, scalar, or 4-element array.

        Examples
        --------
        Divide by scalar::

            >>> from quatham import Quaternion
            >>> q = Quaternion([2, 0, 0, 0])
            >>> q_half = q / 2.0

        Divide by quaternion::

            >>> q1 = Quaternion.from_axisangle([0, 0, 1], np.pi/2)
            >>> q2 = Quaternion.from_axisangle([0, 0, 1], np.pi/4)
            >>> q_diff = q1 / q2
        """
        if isinstance(other, Quaternion):
            return self._divide_by_quaternion(other)

        elif isinstance(other, (int, float, np.number)):
            return Quaternion(self.ndarray / other)

        elif isinstance(other, (list, tuple, np.ndarray)):
            arr = np.asarray(other)
            if arr.ndim == 0:
                return Quaternion(self.ndarray / arr)
            elif arr.shape[-1] == 4 and (arr.ndim == 1 or arr.ndim == 2):
                return self._divide_by_quaternion(Quaternion(arr))
            else:
                raise ValueError(f"Input divisor array has invalid shape {arr.shape}")

        else:
            raise TypeError(f"Division with unknown type {type(other)}")

    def __rtruediv__(self, other: float | int | np.number) -> Self:
        """
        Divide a scalar by this quaternion (scalar / q = scalar * q^-1).

        Parameters
        ----------
        other : float or int or np.number
            Scalar numerator.

        Returns
        -------
        Quaternion
            Result of scalar division by quaternion.
        """
        if isinstance(other, (int, float, np.number)):
            norm_sq = np.sum(self.ndarray**2, axis=1, keepdims=True)
            return Quaternion((other * self.conjugate.ndarray) / norm_sq)
        raise TypeError(f"Cannot divide {type(other)} by Quaternion")