
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
    normals = np.array([-tangents[:, 1], tangents[:, 0]]).T
    return normals

# Example usage
X = np.linspace(0, 10, 100)  # X coordinates
deformation = np.sin(X)  # Deformation function (sinusoidal as an example)

# Compute tangents
tangents = compute_tans(X, deformation)

# Compute normals from tangents
normals = compute_normals(tangents)

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
