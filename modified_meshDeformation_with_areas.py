
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
    """Compute the tangent vectors for the bottom boundary."""
    dx = X[1] - X[0]  # Step size in the x-direction (uniform in this case)
    # Derivative of the deformation (slope at each point)
    tangent_slope = -np.gradient(deformation, dx)

    # Create the tangent vectors (dx, dy)
    tangents = np.array([np.ones_like(tangent_slope), tangent_slope]).T

    return tangents

def compute_normals(tangents):
    """Compute the normal vectors by rotating the tangents by 90 degrees."""
    # For each tangent, rotate by 90 degrees to get the normal vector
    normals = np.array([tangents[:, 1], tangents[:, 0]]).T
    return normals

def compute_areas(X, deformation):
    """Compute the area associated with each boundary point."""
    areas = np.zeros_like(X)
    
    for i in range(1, len(X) - 1):
        # Compute the length of the face formed between point i-1 and i, and i and i+1
        dx1 = X[i] - X[i - 1]
        dy1 = deformation[i] - deformation[i - 1]
        face_area1 = np.sqrt(dx1**2 + dy1**2)

        dx2 = X[i + 1] - X[i]
        dy2 = deformation[i + 1] - deformation[i]
        face_area2 = np.sqrt(dx2**2 + dy2**2)

        # Sum of areas of the adjacent faces
        areas[i] = (face_area1 + face_area2) / 2

    # Special case for the first and last points (only one face attached)
    areas[0] = np.sqrt((X[1] - X[0])**2 + (deformation[1] - deformation[0])**2) / 2
    areas[-1] = np.sqrt((X[-1] - X[-2])**2 + (deformation[-1] - deformation[-2])**2) / 2

    return areas

# Example usage
X = np.linspace(0, 10, 100)  # X coordinates
deformation = np.sin(X)  # Deformation function (sinusoidal as an example)

# Compute tangents
tangents = compute_tans(X, deformation)

# Compute normals from tangents
normals = compute_normals(tangents)

# Compute areas associated with each boundary point
areas = compute_areas(X, deformation)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(X, deformation, label='Deformation')

# Plot the normals for visualization
for i in range(0, len(X), 10):  # Plot every 10th normal for clarity
    plt.arrow(X[i], deformation[i], normals[i, 0] * 0.5, normals[i, 1] * 0.5, 
              head_width=0.1, head_length=0.2, fc='r', ec='r')

plt.title("Deformation and Normals")
plt.xlabel("X")
plt.ylabel("Deformation")
plt.legend()
plt.grid(True)
plt.show()

# Plot the areas associated with each boundary point
plt.figure(figsize=(8, 6))
plt.plot(X, areas, label='Areas associated with boundary points', color='g')
plt.title("Areas associated with Boundary Points")
plt.xlabel("X")
plt.ylabel("Area")
plt.legend()
plt.grid(True)
plt.show()
