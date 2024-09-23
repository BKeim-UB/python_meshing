import numpy as np
import matplotlib.pyplot as plt


def angle_between_vectors(A, B):
    """Compute the angle between two vectors A and B."""
    # Dot product of A and B
    dot_product = np.dot(A, B)

    # Magnitudes (norms) of A and B
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    # Cosine of the angle
    cos_theta = dot_product / (norm_A * norm_B)

    # Ensure the value is within the valid range for arccos (-1 to 1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute the angle in radians
    theta = np.arccos(cos_theta)

    # Convert to degrees (optional)
    angle_in_degrees = np.degrees(theta)

    return angle_in_degrees


def compute_tans(X, deformation):
    """Compute the normal vectors for the sinusoidal bottom boundary."""
    dx = X[1] - X[0]  # Step size in the x-direction (uniform in this case)
    # Derivative of the sine function
    tangent_slope = -np.gradient(deformation, dx)

    return tangent_slope


def compute_convexity(X, slope):

    dx = X[1] - X[0]  # Step size in the x-direction (uniform in this case)
    # Derivative of the sine function
    convexity = -np.gradient(slope, dx)

    return convexity


def inverse_distance_interpolation(Xi, Zi, boundary_points, boundary_tan):
    """Interpolates displacement and normal vectors at internal point (Xi, Zi)."""
    distances = np.sqrt((boundary_points[:, 0] - Xi)**2 +
                        (boundary_points[:, 1] - Zi)**2)

    Ldef = 1.5
    alpha = 0.25
    a = 3
    b = 5

    weights = (Ldef / distances)**a + (alpha * Ldef / distances)**b

    # Normalize weights
    weights /= np.sum(weights)

    # Interpolate displacement
    interp_tan = np.sum(weights * boundary_tan)

    return interp_tan


# this parameter controls the orthogonality of the first cell:
# first_cell_coeff = 1.0   -> orthogonal
# first_cell_coeff = 0.0   -> vertical
# TODO: introduce local coeff proportional to average convexity
first_cell_coeff = 0.65

# these parameters define a buffer zone for the normals on the
# bottom face. The normal is vertical for a distance d from the
# size d <= d1, and is orthogonal for d>=d2, varying linearly
# between the two.
d1 = 0.5
d2 = 1.5

# Step 1: Generate a 2D grid of 50x50 points in the domain [0, 2pi] x [0, 2pi]
n = 50
x = np.linspace(0, 1, n)
z = np.linspace(0, 1, n)

x *= 4 * np.pi

# we add a vertical grading in the original grid
z = z**1.4
z *= 4 * np.pi

X, Z = np.meshgrid(x, z)

xmin = np.amin(x)
xmax = np.amax(x)
zmin = np.amin(z)
zmax = np.amax(z)

# this value represent the proportion between the vertical coordinate
# in the original grid and the horizontal displacement
tan = np.zeros_like(X)

# Step 2: Define boundary deformations
# Keep left, right, and top boundaries unaltered (tan=0)
left_boundary = np.column_stack((X[:, 0], Z[:, 0]))  # Left boundary points
left_tan = tan[:, 0].ravel()

right_boundary = np.column_stack((X[:, -1], Z[:, -1]))  # Right boundary points
right_tan = tan[:, -1].ravel()

top_boundary = np.column_stack((X[-1, :], Z[-1, :]))  # Top boundary points
top_tan = tan[-1, :].ravel()

# Step 3A: apply sinusoidal deformation to the bottom boundary

bottom_boundary = np.column_stack((X[0, :], Z[0, :]))
# Sinusoidal deformation for the bottom boundary
bottom_deformation = 1.5 * np.sin(X[0, :])

# Step3B: apply piecewise linear deformation to the bottom boundary

alfa = (X[0, :] - xmin) / (xmax - xmin)
for i, alfai in enumerate(alfa):

    bottom_deformation[i] = (alfai < 0.4) * (-4.0 * alfai) + \
        (0.4 <= alfai) * (alfai < 0.6) * (-1.6 + 16.0 * (alfai - 0.4)) + \
        (alfai >= 0.6) * (1.6 - 4.0 * (alfai - 0.6))

# Step 4: compute the normals for the bottom boundary
tan[0, :] = compute_tans(X[0, :], bottom_deformation)
slope = np.arctan(tan[0, :]) * 180.0 / np.pi
# print('max slope', np.amax(np.abs(slope)))

convexity = compute_convexity(X[0, :], tan[0, :])
# print('convexity',convexity)

curvature = convexity / (1 + tan[0, :]**2)**1.5
norm_curvature = np.clip(curvature, 0, None) * (X[0, 1] - X[0, 0])
# print('norm_curvature',norm_curvature)
# bottom_tan = first_cell_coeff * (1.0-8.0*norm_curvature)*tan[0, :].ravel()

bottom_tan = first_cell_coeff * tan[0, :].ravel()

# Step 5: bottom correction for distance form sides
# compute the distance of the bottom points from left and right
dist_bdry = np.minimum(X[0, :] - xmin, xmax - X[0, :])

# compute the correction coefficient for side distance
dist_coeff = np.minimum(
    np.ones_like(dist_bdry),
    np.maximum(np.zeros_like(dist_bdry), (dist_bdry - d1) / (d2 - d1)))

# set tan to zero close to the sides
bottom_tan *= dist_coeff

# Combine all boundary points
boundary_points = np.vstack(
    (left_boundary, right_boundary, top_boundary, bottom_boundary))
boundary_tan = np.hstack((left_tan, right_tan, top_tan, bottom_tan))

# Step 6: Deform the internal points
X_deformed = X.copy()
Z_deformed = Z.copy()

Z_deformed[0, :] += bottom_deformation

for i in range(1, n - 1):

    Zrel = (zmax - Z[i, 0]) / (zmax - zmin)
    Z_deformed[i, 0] += Z_deformed[0, 0] * Zrel

    for j in range(1, n - 1):
        # Interpolate displacement and normal vector for each internal point
        interp_tan = inverse_distance_interpolation(X[i, j], Z[i, j],
                                                    boundary_points,
                                                    boundary_tan)

        # the vertical deformation is a linear function of the elevation
        # in the original grid
        Zrel = (zmax - Z[i, j]) / (zmax - zmin)
        vert_deform = bottom_deformation[j] * Zrel

        # Displace the internal point along the interpolated normal direction
        X_deformed[i, j] += (interp_tan * (Z[i, j]))
        Z_deformed[i, j] += vert_deform

    Zrel = (zmax - Z[i, n - 1]) / (zmax - zmin)
    Z_deformed[i, n - 1] += Z_deformed[0, n - 1] * Zrel

print('deformation completed')

angles_vert = []
angles_deform = []

areas_vert = []
areas_deform = []

# Step 7: compute angles for the internal points

# increase this value to remove from the angle computation
# the points close to the sides (nx>=1)
nx = 1

# this number define the number of grid rows from the bottom
# used to compute the angles (nz<=n-1)
nz = 10

# Step 7: Plot original and deformed grid
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

for i in range(1, nz):

    for j in range(nx, n - nx):

        dx1 = X[i, j + 1] - X[i, j - 1]
        dz1 = Z_deformed[i, j + 1] - Z_deformed[i, j - 1]

        dx2 = X[i + 1, j] - X[i - 1, j]
        dz2 = Z_deformed[i + 1, j] - Z_deformed[i - 1, j]

        # Example vectors
        A = np.array([dx1, dz1])
        B = np.array([dx2, dz2])

        # Compute the angle
        angle = np.abs(angle_between_vectors(A, B) - 90.0)
        area = 4.0 * np.cross(A, B)

        angles_vert.append(angle)
        areas_vert.append(area)
        # ax[1].scatter(X[i, j],Z_deformed[i, j],c=[angle],vmin=0,vmax=60)

        dx1 = X_deformed[i, j + 1] - X_deformed[i, j - 1]
        dx2 = X_deformed[i + 1, j] - X_deformed[i - 1, j]

        # Example vectors
        A = np.array([dx1, dz1])
        B = np.array([dx2, dz2])

        # Compute the angle
        angle = np.abs(angle_between_vectors(A, B) - 90.0)
        area = 4.0 * np.cross(A, B)
        # ax[2].scatter(X_deformed[i, j],Z_deformed[i, j],c=angle,vmin=0,vmax=60)

        angles_deform.append(angle)
        areas_deform.append(area)

angle_vert_avg = np.mean(np.array(angles_vert))
angle_deform_avg = np.mean(np.array(angles_deform))

angle_vert_max = np.amax(np.array(angles_vert))
angle_deform_max = np.amax(np.array(angles_deform))

area_vert_min = np.amin(np.array(areas_vert))
area_deform_min = np.amin(np.array(areas_deform))

print('angle_vert_avg', angle_vert_avg)
print('angle_vert_max', angle_vert_max)
print('area_vert_min', area_vert_min)

print('angle_deform_avg', angle_deform_avg)
print('angle_deform_max', angle_deform_max)
print('area_deform_min', area_deform_min)

# Original grid
ax[0].plot(X, Z, 'k-', lw=0.5)
ax[0].plot(X.T, Z.T, 'k-', lw=0.5)
ax[0].set_title("Original Grid")
ax[0].set_aspect('equal')

# Deformed grid 1
ax[1].plot(X, Z_deformed, 'b-', lw=0.5)
ax[1].plot(X[0, :], Z_deformed[0, :], 'k.', lw=0.5)
ax[1].plot(X.T, Z_deformed.T, 'b-', lw=0.5)
ax[1].set_title("Deformed Grid (vertical only)")
ax[1].set_aspect('equal')

# Deformed grid 2
ax[2].plot(X_deformed, Z_deformed, 'b-', lw=0.5)
ax[2].plot(X_deformed[0, :], Z_deformed[0, :], 'k.', lw=0.5)
ax[2].plot(X_deformed.T, Z_deformed.T, 'b-', lw=0.5)
ax[2].set_title("Deformed Grid (Sinusoidal Bottom with Normals)")
ax[2].set_aspect('equal')

plt.tight_layout()
plt.savefig('grids.pdf')

fig, ax = plt.subplots(1, 2, figsize=(18, 6))

bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# density=False would make counts
ax[0].hist(angles_vert, density=True, bins=bins)
ax[0].set_ylabel('Probability')
ax[0].set_xlabel('Angles')
ax[0].set_title("Deformed Grid (vertical only)")

# density=False would make counts
ax[1].hist(angles_deform, density=True, bins=bins)
ax[1].set_ylabel('Probability')
ax[1].set_xlabel('Angles')
ax[1].set_title("Deformed Grid (vert and horiz)")

plt.savefig('angles.pdf')

plt.show()
