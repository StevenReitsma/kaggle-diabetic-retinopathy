import numpy as np

# Patches is a P x K matrix for one array
def pool(activations, n_pool_regions = 4, operator = np.sum):
    """
    Possible choices for operator:
        np.sum
        np.amax
        np.mean
    """
    
    if int(np.sqrt(n_pool_regions)) ** 2 is not n_pool_regions:
        raise ValueError("n_pool_regions should be a perfect square")
    
    patch_size = np.sqrt(activations.shape[0])
    n_features = activations.shape[1]
    
    # Reshape to 2D slabs
    if len(activations)*len(activations[0]) != (n_features*patch_size*patch_size):
        print len(activations)

    reshaped = np.reshape(activations.T, (n_features, patch_size, patch_size))
    # Pooling
    pool_regions_per_dimension = int(np.sqrt(n_pool_regions))
    half_patch = round(patch_size / pool_regions_per_dimension)
    feature_vector = np.zeros((n_pool_regions * n_features))
    
    for x in range(0, pool_regions_per_dimension):
        for y in range(0, pool_regions_per_dimension):
            start_index_row = x * half_patch
            end_index_row = start_index_row + half_patch

            start_index_col = y * half_patch
            end_index_col = start_index_col + half_patch

            q = operator(operator(reshaped[:, start_index_row:end_index_row, start_index_col:end_index_col], axis=1), axis=1)

            feature_index = (x*pool_regions_per_dimension + y) * n_features

            feature_vector[feature_index:feature_index + n_features] = q

    return feature_vector    