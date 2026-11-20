#!/usr/bin/env python
"""
Quaternion math utilities.

Provides low-level quaternion operations including normalisation,
axis-angle conversion, Euler/Tait-Bryan angle computation,
rotation matrices, and quaternion multiplication with Numba acceleration.
"""


# Standard Library
import math

# Imports
import numpy as np
from numba import njit
from numba.typed import List
from numpy.typing import NDArray

# Local Sources


# Global Constants
X_UNIT_AXIS = (1, 0, 0)
Y_UNIT_AXIS = (0, 1, 0)
Z_UNIT_AXIS = (0, 0, 1)

__all__ = ['normalise', 'rotate_vectors_by_quaternion', 'calc_spherical_coordinates', 'make_2d']

@njit(cache=True)
def np_normalise_2d(row_array: NDArray) -> NDArray:
    """
    2D row-wise normalisation without intermediate slice overhead.

    Parameters
    ----------
    row_array : NDArray
        2D array containing rows to normalise.

    Returns
    -------
    NDArray
        Normalised 2D array.
    """
    n_rows, n_cols = row_array.shape
    out = np.empty((n_rows, n_cols), dtype=row_array.dtype)
    for i in range(n_rows):
        sum_sq = 0.0
        for j in range(n_cols):
            val = row_array[i, j]
            sum_sq += val * val

        if sum_sq > 0.0:
            inv_norm = 1.0 / np.sqrt(sum_sq)
            for j in range(n_cols):
                out[i, j] = row_array[i, j] * inv_norm
        else:
            for j in range(n_cols):
                out[i, j] = 0.0

    return out


@njit(cache=True)
def np_normalise_1d(vect: NDArray) -> NDArray:
    """
    Normalise a 1D vector using L2 norm.

    Parameters
    ----------
    vect : NDArray
        1D array containing a vector to normalise.

    Returns
    -------
    NDArray
        Normalised vector of the same shape as input.
    """
    return vect / (np.sqrt(np.nansum(vect**2)))


def normalise(quat: NDArray) -> NDArray:
    """
    Normalise a 4-element numpy array for use as a quaternion.

    Handles both 1D (single quaternion) and 2D (multiple quaternions) arrays.

    Parameters
    ----------
    quat : NDArray
        Array of shape (4,) or (N, 4) containing quaternion data.

    Returns
    -------
    NDArray
        Normalised quaternion array of the same shape as input.
    """
    if quat.ndim == 1:
        return np_normalise_1d(quat)
    else:
        return np_normalise_2d(quat)


@njit(cache=True)
def _axis_angle_to_quat_nb(vec: NDArray, theta: NDArray) -> NDArray:
    """Compute (N, 4) quaternions [w, x, y, z] from (N, 3) axes and (N,) angles (Numba-jitted)."""
    n = vec.shape[0]
    out = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        vx = vec[i, 0]
        vy = vec[i, 1]
        vz = vec[i, 2]
        v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        half_theta = 0.5 * theta[i]
        sin_half = math.sin(half_theta)
        cos_half = math.cos(half_theta)
        if v_norm > 0.0:
            inv_norm = sin_half / v_norm
            out[i, 0] = cos_half
            out[i, 1] = vx * inv_norm
            out[i, 2] = vy * inv_norm
            out[i, 3] = vz * inv_norm
        else:
            out[i, 0] = 1.0
            out[i, 1] = 0.0
            out[i, 2] = 0.0
            out[i, 3] = 0.0
    return out


def _axis_angle_to_quat(vec: NDArray | list | tuple, theta: float | NDArray | list | tuple,
                        degrees: bool = False) -> NDArray:
    """
    Convert axis-angle representation to a quaternion.

    Supports both single axis/angle inputs and batched arrays, including broadcasting.

    Parameters
    ----------
    vec : NDArray or list or tuple
        Rotation axis as a 3-element vector or array of shape (N, 3). Will be normalised internally.
    theta : float or NDArray or list or tuple
        Rotation angle(s) around the axis.
    degrees : bool, default False
        If True, theta is interpreted as degrees; otherwise radians.

    Returns
    -------
    NDArray
        Quaternion array of shape (4,) for single inputs or (N, 4) for batched inputs
        in scalar-first format [w, x, y, z].
    """
    v_arr = np.asarray(vec, dtype=np.float64)
    t_arr = np.asarray(theta, dtype=np.float64)

    if degrees:
        t_arr = np.radians(t_arr)

    is_single = (v_arr.ndim == 1 and t_arr.ndim == 0)

    if v_arr.ndim == 1:
        if t_arr.ndim == 0:
            v_2d = v_arr[None, :]
            t_1d = np.array([float(t_arr)])
        elif t_arr.ndim == 1:
            v_2d = np.broadcast_to(v_arr, (t_arr.shape[0], 3))
            t_1d = t_arr
        else:
            raise ValueError(f"Angles array has unsupported dimensions: {t_arr.shape}")
    elif v_arr.ndim == 2:
        if t_arr.ndim == 0:
            v_2d = v_arr
            t_1d = np.full(v_arr.shape[0], float(t_arr))
        elif t_arr.ndim == 1:
            if v_arr.shape[0] != t_arr.shape[0]:
                raise ValueError(
                    f"Mismatched shapes for axis ({v_arr.shape}) and angle ({t_arr.shape})"
                )
            v_2d = v_arr
            t_1d = t_arr
        elif t_arr.ndim == 2 and t_arr.shape[1] == 1:
            v_2d = v_arr
            t_1d = t_arr[:, 0]
        else:
            raise ValueError(f"Angles array has unsupported dimensions: {t_arr.shape}")
    else:
        raise ValueError(f"Axis array has unsupported dimensions: {v_arr.shape}")

    res = _axis_angle_to_quat_nb(np.ascontiguousarray(v_2d), np.ascontiguousarray(t_1d))
    if is_single:
        return res[0]
    return res

@njit(cache=True)
def _calc_rotation_mat(qarr: NDArray) -> NDArray:
    """
    Compute (N, 3, 3) rotation matrices from (N, 4) quaternions [w, x, y, z].

    Numba-jitted for best performance.

    Parameters
    ----------
    qarr : NDArray (N, 9)
        Array of quaternions (scalar, x, y, z parts).

    Returns
    -------
    NDArray
        Array of shape (N, 9) containing rotation matrices in flat row-major order.
    """
    n_quats = qarr.shape[0]
    m = np.empty((n_quats, 3, 3), dtype=qarr.dtype)

    for i in range(n_quats):
        w = qarr[i, 0]
        x = qarr[i, 1]
        y = qarr[i, 2]
        z = qarr[i, 3]

        n = w * w + x * x + y * y + z * z
        inv_n = 2.0 / n if n > 0.0 else 0.0

        # Precompute products to minimise multiplications
        xx = x * x * inv_n
        yy = y * y * inv_n
        zz = z * z * inv_n
        xy = x * y * inv_n
        xz = x * z * inv_n
        yz = y * z * inv_n
        wx = w * x * inv_n
        wy = w * y * inv_n
        wz = w * z * inv_n

        # Diagonal elements
        m[i, 0, 0] = 1.0 - (yy + zz)
        m[i, 1, 1] = 1.0 - (xx + zz)
        m[i, 2, 2] = 1.0 - (xx + yy)

        # Off-diagonal elements
        m[i, 0, 1] = xy - wz
        m[i, 0, 2] = xz + wy
        m[i, 1, 0] = xy + wz
        m[i, 1, 2] = yz - wx
        m[i, 2, 0] = xz - wy
        m[i, 2, 1] = yz + wx

    return m


@njit(cache=True)
def _calc_euler_angles(qarr: NDArray, seq: str) -> NDArray:
    """
    Compute Euler angles from quaternion components (Numba-jitted).

    Supports proper Euler angle sequences: xzx, xyx, yxy, yzy, zxz, zyz.

    Parameters
    ----------
    qarr : NDArray
        Array of quaternions of shape (N, 4) in scalar-first format [w, x, y, z].
    seq : str
        Euler angle sequence (one of 'xzx', 'xyx', 'yxy', 'yzy', 'zxz', 'zyz').

    Returns
    -------
    NDArray
        Euler angles of shape (N, 3) containing [alpha, beta, gamma].
        Returns NaN for invalid sequences.
    """
    n_quats = qarr.shape[0]
    m = np.full((n_quats, 3), np.nan, dtype=qarr.dtype)
    seq_key = seq.lower()

    for i in range(n_quats):
        q0 = qarr[i, 0]
        q1 = qarr[i, 1]
        q2 = qarr[i, 2]
        q3 = qarr[i, 3]

        q0_sq = q0 * q0
        q1_sq = q1 * q1
        q2_sq = q2 * q2
        q3_sq = q3 * q3

        n = q0_sq + q1_sq + q2_sq + q3_sq
        if n == 0.0:
            continue

        inv_n = 2.0 / n

        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q3 = q2 * q3

        if seq_key == 'xzx':
            cos_beta = min(1.0, max(-1.0, (q0_sq + q1_sq) * inv_n - 1.0))
            m[i, 0] = np.arctan2(q1q3 - q0q2, q1q2 + q0q3)
            m[i, 1] = np.arccos(cos_beta)
            m[i, 2] = np.arctan2(q1q3 + q0q2, -(q1q2 - q0q3))

        elif seq_key == 'xyx':
            cos_beta = min(1.0, max(-1.0, (q0_sq + q1_sq) * inv_n - 1.0))
            m[i, 0] = np.arctan2(q1q2 + q0q3, -(q1q3 - q0q2))
            m[i, 1] = np.arccos(cos_beta)
            m[i, 2] = np.arctan2(q1q2 - q0q3, q1q3 + q0q2)

        elif seq_key == 'yxy':
            cos_beta = min(1.0, max(-1.0, (q0_sq + q2_sq) * inv_n - 1.0))
            m[i, 0] = np.arctan2(q1q2 - q0q3, q2q3 + q0q1)
            m[i, 1] = np.arccos(cos_beta)
            m[i, 2] = np.arctan2(q1q2 + q0q3, -(q2q3 - q0q1))

        elif seq_key == 'yzy':
            cos_beta = min(1.0, max(-1.0, (q0_sq + q2_sq) * inv_n - 1.0))
            m[i, 0] = np.arctan2(q2q3 + q0q1, -(q1q2 - q0q3))
            m[i, 1] = np.arccos(cos_beta)
            m[i, 2] = np.arctan2(q2q3 - q0q1, q1q2 + q0q3)

        elif seq_key == 'zxz':
            cos_beta = min(1.0, max(-1.0, (q0_sq + q3_sq) * inv_n - 1.0))
            m[i, 0] = np.arctan2(q1q3 + q0q2, -(q2q3 - q0q1))
            m[i, 1] = np.arccos(cos_beta)
            m[i, 2] = np.arctan2(q1q3 - q0q2, q2q3 + q0q1)

        elif seq_key == 'zyz':
            cos_beta = min(1.0, max(-1.0, (q0_sq + q3_sq) * inv_n - 1.0))
            m[i, 0] = np.arctan2(q2q3 - q0q1, q1q3 + q0q2)
            m[i, 1] = np.arccos(cos_beta)
            m[i, 2] = np.arctan2(q2q3 + q0q1, -(q1q3 - q0q2))

    return m


@njit(cache=True)
def _calc_tait_bryan_angles(qarr: NDArray, seq: str, intrinsic: bool = True) -> NDArray:
    """
    Compute Tait-Bryan angles from (N, 4) quaternions [w, x, y, z].

    Numba-jitted for performance.

    """
    n_quats = qarr.shape[0]
    m = np.full((n_quats, 3), np.nan, dtype=qarr.dtype)
    seq_key = seq.lower()

    for i in range(n_quats):
        q0 = qarr[i, 0]
        q1 = qarr[i, 1]
        q2 = qarr[i, 2]
        q3 = qarr[i, 3]

        q0_sq = q0 * q0
        q1_sq = q1 * q1
        q2_sq = q2 * q2
        q3_sq = q3 * q3

        n = q0_sq + q1_sq + q2_sq + q3_sq
        if n == 0.0:
            continue

        inv_n = 2.0 / n

        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q3 = q2 * q3

        if seq_key == 'zyx':
            if intrinsic:
                # Intrinsic ZYX (Yaw-Pitch-Roll)
                sin_beta = min(1.0, max(-1.0, -2.0 * (q1q3 - q0q2) / n))
                m[i, 0] = np.arctan2(2.0 * (q1q2 + q0q3) / n, (q0_sq + q1_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(2.0 * (q2q3 + q0q1) / n, (q0_sq + q3_sq) * inv_n - 1.0)
            else:
                sin_beta = min(1.0, max(-1.0, 2.0 * (q1q3 + q0q2) / n))
                m[i, 0] = np.arctan2(-2.0 * (q1q2 - q0q3) / n, (q0_sq + q1_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(-2.0 * (q2q3 - q0q1) / n, (q0_sq + q3_sq) * inv_n - 1.0)

        elif seq_key == 'yzx':
            if intrinsic:
                sin_beta = min(1.0, max(-1.0, 2.0 * (q1q2 + q0q3) / n))
                m[i, 0] = np.arctan2(-2.0 * (q1q3 - q0q2) / n, (q0_sq + q1_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(-2.0 * (q2q3 - q0q1) / n, (q0_sq + q2_sq) * inv_n - 1.0)
            else:
                sin_beta = min(1.0, max(-1.0, -2.0 * (q1q2 - q0q3) / n))
                m[i, 0] = np.arctan2(2.0 * (q1q3 + q0q2) / n, (q0_sq + q1_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(2.0 * (q2q3 + q0q1) / n, (q0_sq + q2_sq) * inv_n - 1.0)

        elif seq_key == 'zxy':
            if intrinsic:
                sin_beta = min(1.0, max(-1.0, 2.0 * (q2q3 + q0q1) / n))
                m[i, 0] = np.arctan2(-2.0 * (q1q2 - q0q3) / n, (q0_sq + q2_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(-2.0 * (q1q3 - q0q2) / n, (q0_sq + q3_sq) * inv_n - 1.0)
            else:
                sin_beta = min(1.0, max(-1.0, -2.0 * (q2q3 - q0q1) / n))
                m[i, 0] = np.arctan2(2.0 * (q1q2 + q0q3) / n, (q0_sq + q2_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(2.0 * (q1q3 + q0q2) / n, (q0_sq + q3_sq) * inv_n - 1.0)

        elif seq_key == 'xzy':
            if intrinsic:
                sin_beta = min(1.0, max(-1.0, -2.0 * (q1q2 - q0q3) / n))
                m[i, 0] = np.arctan2(2.0 * (q2q3 + q0q1) / n, (q0_sq + q2_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(2.0 * (q1q3 + q0q2) / n, (q0_sq + q1_sq) * inv_n - 1.0)
            else:
                sin_beta = min(1.0, max(-1.0, 2.0 * (q1q2 + q0q3) / n))
                m[i, 0] = np.arctan2(-2.0 * (q2q3 - q0q1) / n, (q0_sq + q2_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(-2.0 * (q1q3 - q0q2) / n, (q0_sq + q1_sq) * inv_n - 1.0)

        elif seq_key == 'yxz':
            if intrinsic:
                sin_beta = min(1.0, max(-1.0, -2.0 * (q2q3 - q0q1) / n))
                m[i, 0] = np.arctan2(2.0 * (q1q3 + q0q2) / n, (q0_sq + q3_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(2.0 * (q1q2 + q0q3) / n, (q0_sq + q2_sq) * inv_n - 1.0)
            else:
                sin_beta = min(1.0, max(-1.0, 2.0 * (q2q3 + q0q1) / n))
                m[i, 0] = np.arctan2(-2.0 * (q1q3 - q0q2) / n, (q0_sq + q3_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(-2.0 * (q1q2 - q0q3) / n, (q0_sq + q2_sq) * inv_n - 1.0)

        elif seq_key == 'xyz':
            if intrinsic:
                sin_beta = min(1.0, max(-1.0, 2.0 * (q1q3 + q0q2) / n))
                m[i, 0] = np.arctan2(-2.0 * (q2q3 - q0q1) / n, (q0_sq + q3_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(-2.0 * (q1q2 - q0q3) / n, (q0_sq + q1_sq) * inv_n - 1.0)
            else:
                sin_beta = min(1.0, max(-1.0, -2.0 * (q1q3 - q0q2) / n))
                m[i, 0] = np.arctan2(2.0 * (q2q3 + q0q1) / n, (q0_sq + q3_sq) * inv_n - 1.0)
                m[i, 1] = np.arcsin(sin_beta)
                m[i, 2] = np.arctan2(2.0 * (q1q2 + q0q3) / n, (q0_sq + q1_sq) * inv_n - 1.0)

    return m


@njit(cache=True)
def rotate_vectors_by_quaternion(quat: NDArray, vectors: NDArray) -> NDArray:
    """
    Rotate vectors by quaternion(s).

    Supports:
    - Single quaternion (4,) × multiple vectors (N, 3) → (N, 3)
    - Single quaternion (1, 4) × multiple vectors (N, 3) → (N, 3)
    - Multiple quaternions (N, 4) × single vector (1, 3) → (N, 3)
    - Multiple quaternions (N, 4) × multiple vectors (N, 3) → (N, 3)

    Parameters
    ----------
    quat : NDArray (4,), (1, 4), or (N, 4)
        Quaternion(s) [w, x, y, z]. 1D arrays are automatically reshaped to 2D.
    vectors : NDArray (N, 3)
        Vectors to rotate

    Returns
    -------
    NDArray (N, 3)
        Rotated vectors
    """
    # Handle 1D quaternion - reshape to 2D (N, 4)
    if quat.ndim == 1:
        quat = quat[None, :]
    
    n_quats = quat.shape[0]
    n_vectors = vectors.shape[0]

    # Determine if we're broadcasting
    if n_quats == 1 and n_vectors > 1:
        # Single quaternion, multiple vectors (original case)
        w, x, y, z = quat[0, 0], quat[0, 1], quat[0, 2], quat[0, 3]

        # Precompute rotation matrix elements
        w2, x2, y2, z2 = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        r00 = w2 + x2 - y2 - z2
        r01 = 2 * (xy - wz)
        r02 = 2 * (xz + wy)
        r10 = 2 * (xy + wz)
        r11 = w2 - x2 + y2 - z2
        r12 = 2 * (yz - wx)
        r20 = 2 * (xz - wy)
        r21 = 2 * (yz + wx)
        r22 = w2 - x2 - y2 + z2

        result = np.empty((n_vectors, 3), dtype=np.float64)
        for i in range(n_vectors):
            vx, vy, vz = vectors[i, 0], vectors[i, 1], vectors[i, 2]
            result[i, 0] = r00 * vx + r01 * vy + r02 * vz
            result[i, 1] = r10 * vx + r11 * vy + r12 * vz
            result[i, 2] = r20 * vx + r21 * vy + r22 * vz

        return result

    elif n_quats > 1 and n_vectors == 1:
        # Multiple quaternions, single vector (test case)
        result = np.empty((n_quats, 3), dtype=np.float64)

        for q in range(n_quats):
            w, x, y, z = quat[q, 0], quat[q, 1], quat[q, 2], quat[q, 3]
            vx, vy, vz = vectors[0, 0], vectors[0, 1], vectors[0, 2]

            # Precompute rotation matrix elements
            w2, x2, y2, z2 = w * w, x * x, y * y, z * z
            wx, wy, wz = w * x, w * y, w * z
            xy, xz, yz = x * y, x * z, y * z

            r00 = w2 + x2 - y2 - z2
            r01 = 2 * (xy - wz)
            r02 = 2 * (xz + wy)
            r10 = 2 * (xy + wz)
            r11 = w2 - x2 + y2 - z2
            r12 = 2 * (yz - wx)
            r20 = 2 * (xz - wy)
            r21 = 2 * (yz + wx)
            r22 = w2 - x2 - y2 + z2

            result[q, 0] = r00 * vx + r01 * vy + r02 * vz
            result[q, 1] = r10 * vx + r11 * vy + r12 * vz
            result[q, 2] = r20 * vx + r21 * vy + r22 * vz

        return result

    elif n_quats == n_vectors:
        # Multiple quaternions, multiple vectors (one-to-one)
        result = np.empty((n_quats, 3), dtype=np.float64)

        for i in range(n_quats):
            w, x, y, z = quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3]
            vx, vy, vz = vectors[i, 0], vectors[i, 1], vectors[i, 2]

            # Precompute rotation matrix elements
            w2, x2, y2, z2 = w * w, x * x, y * y, z * z
            wx, wy, wz = w * x, w * y, w * z
            xy, xz, yz = x * y, x * z, y * z

            r00 = w2 + x2 - y2 - z2
            r01 = 2 * (xy - wz)
            r02 = 2 * (xz + wy)
            r10 = 2 * (xy + wz)
            r11 = w2 - x2 + y2 - z2
            r12 = 2 * (yz - wx)
            r20 = 2 * (xz - wy)
            r21 = 2 * (yz + wx)
            r22 = w2 - x2 - y2 + z2

            result[i, 0] = r00 * vx + r01 * vy + r02 * vz
            result[i, 1] = r10 * vx + r11 * vy + r12 * vz
            result[i, 2] = r20 * vx + r21 * vy + r22 * vz

        return result

    else:
        raise ValueError(f"Shape mismatch: quat {quat.shape}, vectors {vectors.shape}")


def _spherical_coords_to_quat(theta: float | NDArray, phi: float | NDArray, degrees: bool = False) -> NDArray:
    """
    Convert spherical coordinates to a quaternion.

    Parameters
    ----------
    theta : float or NDArray
        Polar angle (angle from z-axis).
    phi : float or NDArray
        Azimuthal angle (angle in x-y plane from x-axis).
    degrees : bool, default False
        If True, input angles are in degrees; otherwise radians.

    Returns
    -------
    NDArray
        Quaternion of shape (1, 4) in scalar-first format [w, x, y, z].
    """
    if degrees:
        theta = np.radians(theta)
        phi = np.radians(phi)

    cos_theta = np.cos(theta / 2)
    cos_phi = np.cos(phi / 2)
    sin_theta = np.sin(theta / 2)
    sin_phi = np.sin(phi / 2)

    rot = np.array([cos_phi * cos_theta, -sin_phi * sin_theta, sin_theta * cos_phi, sin_phi * cos_theta])

    return rot[None, :]


@njit(cache=True)
def calc_cross(a, b):
    """
    Compute the cross-product of two 3D vectors (Numba-jitted).

    Parameters
    ----------
    a, b : array-like
        3-element vectors.

    Returns
    -------
        tuple of float
        (x, y, z) components of the cross-product.
    """
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return x, y, z


def calc_spherical_coordinates(vec):
    """
    Convert a 3D vector to spherical coordinates.

    Parameters
    ----------
    vec : array-like
        3-element vector [x, y, z].

    Returns
    -------
    tuple
        (r, theta, phi) where r is the magnitude, theta is the azimuthal
        angle, and phi is the polar angle.
    """
    r = np.linalg.norm(vec)

    theta = math.atan2(vec[1], vec[0])
    phi = math.acos(vec[2] / r)

    return r, theta, phi


@njit(cache=True)
def _quat_single_multiplication(p, q):
    """
    Multiply two single quaternions (Numba-jitted).

    Parameters
    ----------
    p, q : NDArray
        Quaternions of shape (4,) in scalar-first format.

    Returns
    -------
    NDArray
        Product quaternion of shape (4,).
    """
    cross = calc_cross(p[1:], q[1:])
    r = np.zeros(4)

    r[1] = p[0] * q[1] + q[0] * p[1] + cross[0]
    r[2] = p[0] * q[2] + q[0] * p[2] + cross[1]
    r[3] = p[0] * q[3] + q[0] * p[3] + cross[2]
    r[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]

    return r


@njit(cache=True)
def _quat_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = max(p.shape[0], q.shape[0])
    out = np.empty((n, 4), dtype=np.float64)
    p_is_single = p.shape[0] == 1
    q_is_single = q.shape[0] == 1

    for i in range(n):
        pi = 0 if p_is_single else i
        qi = 0 if q_is_single else i

        w1, x1, y1, z1 = p[pi, 0], p[pi, 1], p[pi, 2], p[pi, 3]
        w2, x2, y2, z2 = q[qi, 0], q[qi, 1], q[qi, 2], q[qi, 3]

        n2 = w2*w2 + x2*x2 + y2*y2 + z2*z2
        inv_n2 = 1.0 / n2 if n2 > 0.0 else 0.0

        out[i, 0] = (w1*w2 + x1*x2 + y1*y2 + z1*z2) * inv_n2
        out[i, 1] = (-w1*x2 + x1*w2 - y1*z2 + z1*y2) * inv_n2
        out[i, 2] = (-w1*y2 + x1*z2 + y1*w2 - z1*x2) * inv_n2
        out[i, 3] = (-w1*z2 - x1*y2 + y1*x2 + z1*w2) * inv_n2

    return out

@njit(cache=True)
def _quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = max(p.shape[0], q.shape[0])
    out = np.empty((n, 4), dtype=np.float64)
    p_is_single = p.shape[0] == 1
    q_is_single = q.shape[0] == 1

    for i in range(n):
        pi = 0 if p_is_single else i
        qi = 0 if q_is_single else i

        w1, x1, y1, z1 = p[pi, 0], p[pi, 1], p[pi, 2], p[pi, 3]
        w2, x2, y2, z2 = q[qi, 0], q[qi, 1], q[qi, 2], q[qi, 3]

        out[i, 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        out[i, 1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        out[i, 2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        out[i, 3] = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return out


@njit(cache=True)
def _quat_element_calc(axis, angles):
    """
    Compute quaternion elements for a single rotation axis (Numba-jitted).

    Parameters
    ----------
    axis : str
        Rotation axis ('x', 'y', or 'z').
    angles : NDArray
        Array of rotation angles.

    Returns
    -------
    NDArray
        Quaternion array of shape (N, 4) representing rotations about the given axis.
    """
    n = angles.shape[0]
    quat = np.zeros((n, 4))

    if axis == 'x':
        axis_ind = 1
    elif axis == 'y':
        axis_ind = 2
    elif axis == 'z':
        axis_ind = 3

    for ind in range(n):
        quat[ind, 0] = np.cos(angles[ind] / 2)
        quat[ind, axis_ind] = np.sin(angles[ind] / 2)

    return quat


def _compose_quaternion_rotation(angles: NDArray, seq: str = 'xyz', degrees: bool = False,
                                 intrinsic: bool = False) -> NDArray:
    """
    Compose a quaternion from a sequence of rotations (Euler/Tait-Bryan angles).

    Parameters
    ----------
    angles : NDArray
        Array of rotation angles of shape (N, len(seq)) or (len(seq),).
    seq : str, default 'xyz'
        Sequence of rotation axes.
    degrees : bool, default False
        If True, input angles are in degrees; otherwise radians.
    intrinsic : bool, default False
        If True, rotations are intrinsic (local axes); otherwise extrinsic (fixed axes).

    Returns
    -------
    NDArray
        Quaternion array of shape (N, 4) in scalar-first format.
    """
    if degrees:
        angles = np.radians(angles)

    if angles.ndim == 1:
        angles = angles[None, :]

    result = _quat_element_calc(seq[0], angles[:, 0])

    for idx in range(1, len(seq)):
        if intrinsic:
            result = _quat_mult(result, _quat_element_calc(seq[idx], angles[:, idx]))
        else:
            result = _quat_mult(_quat_element_calc(seq[idx], angles[:, idx]), result)

    return result


@njit(cache=True)
def _quat_to_axis_angle_nb(qarr: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute rotation axes (N, 3) and angles (N,) from (N, 4) quaternions [w, x, y, z] (Numba-jitted).
    """
    n_quats = qarr.shape[0]
    axes = np.empty((n_quats, 3), dtype=np.float64)
    thetas = np.empty(n_quats, dtype=np.float64)

    for i in range(n_quats):
        w = qarr[i, 0]
        x = qarr[i, 1]
        y = qarr[i, 2]
        z = qarr[i, 3]

        norm_sq = w * w + x * x + y * y + z * z
        if norm_sq == 0.0:
            axes[i, 0] = 1.0
            axes[i, 1] = 0.0
            axes[i, 2] = 0.0
            thetas[i] = 0.0
            continue

        norm = math.sqrt(norm_sq)
        cos_half = min(1.0, max(-1.0, w / norm))
        thetas[i] = 2.0 * math.acos(cos_half)

        v_norm_sq = x * x + y * y + z * z
        if v_norm_sq > 0.0:
            inv_v_norm = 1.0 / math.sqrt(v_norm_sq)
            axes[i, 0] = x * inv_v_norm
            axes[i, 1] = y * inv_v_norm
            axes[i, 2] = z * inv_v_norm
        else:
            axes[i, 0] = 1.0
            axes[i, 1] = 0.0
            axes[i, 2] = 0.0

    return axes, thetas


def _quat_to_axis_angle(q: NDArray | list | tuple, degrees: bool = False) -> tuple[NDArray, NDArray | float]:
    """
    Convert quaternion(s) to axis-angle representation.

    Supports both single quaternion (shape (4,)) and batched quaternions (shape (N, 4)).

    Parameters
    ----------
    q : NDArray or list or tuple
        Quaternion data of shape (4,) or (N, 4) in scalar-first format [w, x, y, z].
    degrees : bool, default False
        If True, angles are returned in degrees; otherwise radians.

    Returns
    -------
    tuple
        (axis, theta) where axis is the normalized rotation axis (shape (3,) or (N, 3))
        and theta is the rotation angle (float for single quaternion or shape (N,) for batched).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    is_single = (q_arr.ndim == 1)

    if is_single:
        q_2d = q_arr[None, :]
    elif q_arr.ndim == 2:
        q_2d = q_arr
    else:
        raise ValueError(f"Quaternion array has unsupported dimensions: {q_arr.shape}")

    axes, thetas = _quat_to_axis_angle_nb(np.ascontiguousarray(q_2d))

    if degrees:
        thetas = np.degrees(thetas)

    if is_single:
        return axes[0], float(thetas[0])
    return axes, thetas


@njit
def make_2d(arraylist: List[NDArray]) -> NDArray:
    """
    Stack a list of 1D arrays into a 2D array.

    Parameters
    ----------
    arraylist : list of NDArray
        List of 1D arrays of equal length.

    Returns
    -------
    NDArray
        2D array of shape (len(arraylist), len(arraylist[0])).
    """
    n = len(arraylist)
    k = arraylist[0].shape[0]
    a2d = np.empty((n, k), dtype=arraylist[0].dtype)
    for i in range(n):
        for j in range(k):
            a2d[i, j] = arraylist[i][j]
    return a2d