"""
Code used to track the animal through time.

"""
__date__ = "January 2025"

import numpy as np



def track_principal_axes(means, covariances):
    """
    Parameters
    ----------
    means : np.ndarray of shape (T, n)
        The mean vectors of T Gaussians in n-dimensional space.
    covariances : np.ndarray of shape (T, n, n)
        The covariance matrices of T Gaussians in n-dimensional space.
    
    Returns
    -------
    principal_axes : np.ndarray of shape (T, n)
        The unit vectors describing the principal axes (largest principal component)
        at each time, with sign ambiguities resolved.
    """
    T, n = means.shape
    principal_axes = np.zeros((T, n))
    
    def largest_eigvec(cov):
        # Compute eigenvalues/eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        # Index of largest eigenvalue
        idx_max = np.argmax(vals)
        # Corresponding eigenvector
        v = vecs[:, idx_max]
        # Normalize
        v /= np.linalg.norm(v)
        return v
    
    def cov_sqrt(cov):
        """
        Returns the symmetric matrix square root of a positive-semidefinite
        covariance matrix cov.
        """
        # eigendecomposition
        vals, vecs = np.linalg.eigh(cov)
        # sqrt of eigenvalues
        sqrt_vals = np.sqrt(np.clip(vals, 0, None))
        # reconstruct sqrt(cov)
        return (vecs * sqrt_vals) @ vecs.T
    
    def optimal_transport_map(mean1, cov1, mean2, cov2, x):
        """
        Compute the W2-optimal transport map T_{1->2}(x) from N(mean1, cov1)
        to N(mean2, cov2), applied to point x.
        
        T_{1->2}(x) = mean2 + A (x - mean1),
        where A = Sigma2^(1/2) [ Sigma2^(1/2) Sigma1 Sigma2^(1/2) ]^(-1/2) Sigma2^(1/2).
        """
        # Matrix A
        sqrt_cov2 = cov_sqrt(cov2)
        inside = sqrt_cov2 @ cov1 @ sqrt_cov2
        # (inside)^(1/2) can again be computed by eigh
        vals_in, vecs_in = np.linalg.eigh(inside)
        sqrt_inside = (vecs_in * np.sqrt(np.clip(vals_in, 0, None))) @ vecs_in.T
        A = sqrt_cov2 @ np.linalg.inv(sqrt_inside) @ sqrt_cov2
        
        # Apply the affine map
        return mean2 + A @ (x - mean1)
    
    # 1) Compute principal axis for t=0
    v0 = largest_eigvec(covariances[0])
    principal_axes[0] = v0
    
    # We'll track a point p_t = mu_t + v_t
    p_t = means[0] + v0
    
    # 2) Step through time, use the OT map to fix the sign for subsequent axes
    for t in range(T - 1):
        # Compute principal axis for next time step
        v_next = largest_eigvec(covariances[t + 1])
        
        # Compute T_{t->t+1}(p_t)
        p_t_to_next = optimal_transport_map(
            means[t], covariances[t],
            means[t+1], covariances[t+1],
            p_t
        )
        
        # Compare distances to mean_{t+1} + v_next vs. mean_{t+1} - v_next
        plus = means[t+1] + v_next
        minus = means[t+1] - v_next
        
        d_plus = np.linalg.norm(p_t_to_next - plus)
        d_minus = np.linalg.norm(p_t_to_next - minus)
        
        # If going "minus" is closer, flip the sign
        if d_minus < d_plus:
            v_next = -v_next
        
        principal_axes[t+1] = v_next
        
        # Update p_{t+1} for next iteration
        p_t = means[t+1] + v_next
    
    # 3) Ensure the direction of motion of the means is positively correlated
    #    with the principal axes.
    displacements = np.diff(means, axis=0)

    # If dot < 0, flip them all.
    if np.sum(displacements * principal_axes[1:]) < 0:
        principal_axes = -principal_axes
    
    return principal_axes



