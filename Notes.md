# Fundamental Matrix vs. Essential Matrix

## Fundamental Matrix (F)

- Encodes the epipolar geometry between two uncalibrated camera views.
- Relates corresponding points in the two images (pixel coordinates).
- Works without knowing the cameras' intrinsic parameters (e.g., focal length).
- **Equation**:
  ```
  x2^T * F * x1 = 0
  ```
  where `x1` and `x2` are image points in pixels.
- Defines the epipolar lines in the second image for a point in the first image.

## Essential Matrix (E)

- Encodes the epipolar geometry for calibrated cameras.
- Relates corresponding points in normalized image coordinates.
- Requires knowledge of the cameras' intrinsic parameters.
- Encodes the relative rotation (R) and translation (T) between cameras:
  ```
  E = R * [T]_x
  ```
  where `[T]_x` is the skew-symmetric matrix of the translation vector `T`.
- **Equation**:
  ```
  x2^T * E * x1 = 0
  ```
  where `x1` and `x2` are normalized coordinates.

## Key Differences

- **Calibration**: The fundamental matrix works with uncalibrated cameras, while the essential matrix assumes calibrated cameras.
- **Coordinates**: The fundamental matrix uses raw pixel coordinates, while the essential matrix uses normalized coordinates.
- **Motion Encoding**: The essential matrix explicitly encodes the relative motion (rotation and translation) between the two cameras.

---

# Recovering Focal Length from Fundamental Matrix

- The fundamental matrix (F) relates pixel coordinates between two images.
- It can be decomposed into the essential matrix (E) using camera intrinsics (K):
  ```
  F = K2^(-T) * E * K1^(-1)
  ```
- By assuming the principal point (`cx, cy`) or using other constraints, focal lengths (`fx, fy`) can be estimated from `F`.
- The essential matrix (E) must have two equal singular values, which can be used as a constraint to solve for focal length up to scale.
- Absolute focal length requires additional information, such as scene scale or baseline.

---

# Triangulation: Recovering 3D Points from Two Views

Triangulation is the process of determining the 3D coordinates of a point in space by intersecting lines of sight from multiple viewpoints. It requires:
1. The poses of the two cameras (`pose1` and `pose2`), which include rotation and translation information.
2. Corresponding points (`pts1` and `pts2`) in two images, which are projections of the same 3D points.

## Theoretical Background
For each point pair, triangulation involves solving:
```python
A * X = 0
```
Where:
- `A` is a \(4 \times 4\) matrix constructed from the projection equations of the two cameras.
- `X` is the homogeneous 3D point in space.

Each row of `A` is derived as follows:
- For the first camera:
  ```python
  A[0] = x1 * pose1[2,:] - pose1[0,:]
  A[1] = y1 * pose1[2,:] - pose1[1,:]
  ```
- For the second camera:
  ```python
  A[2] = x2 * pose2[2,:] - pose2[0,:]
  A[3] = y2 * pose2[2,:] - pose2[1,:]
  ```

The solution to this homogeneous system can be found using Singular Value Decomposition (SVD). The last row of the right-singular matrix (`V^T`) gives the solution that minimizes reprojection error.

## Practical Steps
1. Compute the inverse of the camera poses to transform from camera to world coordinates.
2. Construct the matrix `A` for each pair of corresponding points.
3. Solve the system using SVD to obtain the homogeneous 3D coordinates.
4. Normalize the homogeneous coordinates to obtain 3D points in Euclidean space.

This method assumes accurate correspondences between the two images and calibrated camera poses. Errors in correspondences or camera calibration can affect the accuracy of the reconstructed 3D points.

