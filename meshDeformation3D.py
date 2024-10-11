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


def compute_tans(X, Y, deformation):
    """Compute the normal vectors for the sinusoidal bottom boundary."""
    dx = X[1, 0] - \
        X[0, 0]  # Step size in the x-direction (uniform in this case)
    # Step size in the x-direction (uniform in this case)
    dy = Y[0, 1] - Y[0, 0]

    print('dx,dy', dx, dy)
    # Derivative of the sine function
    tangent_x, tangent_y = np.gradient(deformation, dx, dy)

    return -tangent_x, -tangent_y


def compute_convexity(X, slope):

    dx = X[1] - X[0]  # Step size in the x-direction (uniform in this case)
    # Derivative of the sine function
    convexity = -np.gradient(slope, dx)

    return convexity


def inverse_distance_interpolation(Xi, Yi, Zi, boundary_points, boundary_tanx,
                                   boundary_tany):
    """Interpolates displacement and normal vectors at internal point (Xi, Zi)."""
    distances = np.sqrt((boundary_points[:, 0] - Xi)**2 +
                        (boundary_points[:, 1] - Yi)**2 +
                        (boundary_points[:, 2] - Zi)**2)

    Ldef = 1.5
    alpha = 0.25
    a = 3
    b = 5

    weights = (Ldef / distances)**a + (alpha * Ldef / distances)**b

    # Normalize weights
    weights /= np.sum(weights)

    # Interpolate displacement
    interp_tanx = np.sum(weights * boundary_tanx)
    interp_tany = np.sum(weights * boundary_tany)

    return interp_tanx, interp_tany


# this parameter controls the orthogonality of the first cell:
# first_cell_coeff = 1.0   -> orthogonal
# first_cell_coeff = 0.0   -> vertical
# TODO: introduce local coeff proportional to average convexity
first_cell_coeff = 0.5

# these parameters define a buffer zone for the normals on the
# bottom face. The normal is vertical for a distance d from the
# size d <= d1, and is orthogonal for d>=d2, varying linearly
# between the two.
d1 = 0.05
d2 = 0.1

# Step 1: Generate a 2D grid of 50x50 points in the domain [0, 2pi] x [0, 2pi]

xmin = -1500.0
xmax = 1500.0
ymin = -1500.0
ymax = 1500.0
zmin = 0.0
zmax = 1500.0

xmin = -np.pi
xmax = np.pi + 1
ymin = -np.pi
ymax = np.pi
zmin = 0.0
zmax = 2.0 * np.pi + 1

cellsize = 0.5

xvent = 0.0
yvent = 0.0

x1 = np.arange(xvent, xmin - cellsize, -cellsize)
x2 = np.arange(xvent, xmax + cellsize, cellsize)
x = np.concatenate((np.flip(x1[1:]), x2))

y1 = np.arange(yvent, ymin - cellsize, -cellsize)
y2 = np.arange(yvent, ymax + cellsize, cellsize)
y = np.concatenate((np.flip(y1[1:]), y2))

z = np.arange(zmin, zmax + cellsize, cellsize)

nx = x.size
ny = y.size
nz = z.size

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

print(X.shape)

# this value represent the proportion between the vertical coordinate
# in the original grid and the horizontal displacement
tanx = np.zeros_like(X)
tany = np.zeros_like(X)

# Step 2: Define boundary deformations
# Keep left, right, and top boundaries unaltered (tanx=tany=0)
west_boundary = np.column_stack(
    (X[0, :, :].ravel(), Y[0, :, :].ravel(), Z[0, :, :].ravel()))
west_tanx = tanx[0, :, :].ravel()
west_tany = tany[0, :, :].ravel()

east_boundary = np.column_stack(
    (X[-1, :, :].ravel(), Y[-1, :, :].ravel(), Z[-1, :, :].ravel()))
east_tanx = tanx[-1, :, :].ravel()
east_tany = tany[-1, :, :].ravel()

south_boundary = np.column_stack(
    (X[:, 0, :].ravel(), Y[:, 0, :].ravel(), Z[:, 0, :].ravel()))
south_tanx = tanx[:, 0, :].ravel()
south_tany = tany[:, 0, :].ravel()

north_boundary = np.column_stack(
    (X[:, -1, :].ravel(), Y[:, -1, :].ravel(), Z[:, -1, :].ravel()))
north_tanx = tanx[:, -1, :].ravel()
north_tany = tany[:, -1, :].ravel()

top_boundary = np.column_stack(
    (X[:, :, -1].ravel(), Y[:, :, -1].ravel(), Z[:, :, -1].ravel()))
top_tanx = tanx[:, :, -1].ravel()
top_tany = tany[:, :, -1].ravel()

# Step 3: apply sinusoidal deformation to the bottom boundary
bottom_boundary = np.column_stack(
    (X[:, :, 0].ravel(), Y[:, :, 0].ravel(), Z[:, :, 0].ravel()))
# Sinusoidal deformation for the bottom boundary
bottom_deformation = 1.5 * np.sin(X[:, :, 0])  # + 0.2*np.abs(Y[:,:,0])

# Step 4: compute the normals for the bottom boundary
tanx[:, :, 0], tany[:, :, 0] = compute_tans(X[:, :, 0], Y[:, :, 0],
                                            bottom_deformation)

bottom_tanx = first_cell_coeff * tanx[:, :, 0].ravel()
bottom_tany = first_cell_coeff * tany[:, :, 0].ravel()

# Step 5: bottom correction for distance form sides
# compute the distance of the bottom points from left and right
dist_bdryx = np.minimum(X[:, :, 0] - xmin, xmax - X[:, :, 0])
dist_bdryy = np.minimum(Y[:, :, 0] - ymin, ymax - Y[:, :, 0])

# compute the correction coefficient for side distance
dist_coeffx = np.minimum(
    np.ones_like(dist_bdryx),
    np.maximum(np.zeros_like(dist_bdryx), (dist_bdryx - d1) / (d2 - d1)))

dist_coeffy = np.minimum(
    np.ones_like(dist_bdryy),
    np.maximum(np.zeros_like(dist_bdryy), (dist_bdryy - d1) / (d2 - d1)))

# set tan to zero close to the sides
bottom_tanx *= dist_coeffx.ravel()
bottom_tany *= dist_coeffy.ravel()

# Combine all boundary points
boundary_points = np.vstack((east_boundary, west_boundary, south_boundary,
                             north_boundary, top_boundary, bottom_boundary))
boundary_tanx = np.hstack(
    (east_tanx, west_tanx, south_tanx, north_tanx, top_tanx, bottom_tanx))
boundary_tany = np.hstack(
    (east_tany, west_tany, south_tany, north_tany, top_tany, bottom_tany))

# Step 6: Deform the internal points
X_deformed = X.copy()
Y_deformed = Y.copy()
Z_deformed = Z.copy()

Z_deformed[:, :, 0] += bottom_deformation

for k in range(1, nz - 1):

    Zrel = (zmax - Z[0, :, k]) / (zmax - zmin)
    Z_deformed[0, :, k] += Z_deformed[0, :, 0] * Zrel

    for i in range(1, nx - 1):

        Zrel = (zmax - Z[i, 0, k]) / (zmax - zmin)
        Z_deformed[i, 0, k] += Z_deformed[i, 0, 0] * Zrel

        for j in range(1, ny - 1):

            # Interpolate displacement and normal vector for each internal point
            interp_tanx, interp_tany = inverse_distance_interpolation(
                X[i, j, k], Y[i, j, k], Z[i, j, k], boundary_points,
                boundary_tanx, boundary_tany)

            # the vertical deformation is a linear function of the elevation
            # in the original grid
            Zrel = (zmax - Z[i, j, k]) / (zmax - zmin)
            vert_deform = bottom_deformation[i, j] * Zrel

            # Displace the internal point along the interpolated normal direction
            X_deformed[i, j, k] += (interp_tanx * (Z[i, j, k]))
            Y_deformed[i, j, k] += (interp_tany * (Z[i, j, k]))
            Z_deformed[i, j, k] += vert_deform

        Zrel = (zmax - Z[i, ny - 1, k]) / (zmax - zmin)
        Z_deformed[i, ny - 1, k] += Z_deformed[i, ny - 1, 0] * Zrel

    Zrel = (zmax - Z[nx - 1, :, k]) / (zmax - zmin)
    Z_deformed[nx - 1, :, k] += Z_deformed[nx - 1, :, 0] * Zrel

print('deformation completed')

# Deformed grid
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_proj_type('ortho')
# ax.scatter(X_deformed[:, :, :], Y_deformed[:, :, :], Z_deformed[:, :, :])

for k in range(nz):

    ax.plot_wireframe(X_deformed[:, :, k], Y_deformed[:, :, k], Z_deformed[:, :, k])

for i in range(nx):

    ax.plot_wireframe(X_deformed[i, :, :], Y_deformed[i, :, :], Z_deformed[i, :, :])


ax.set_title("Deformed Grid")
plt.tight_layout()

plt.show()
