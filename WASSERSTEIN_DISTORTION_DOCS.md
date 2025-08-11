# Wasserstein Distortion Implementation Documentation

## Overview

This module implements **Wasserstein Distortion**, a perceptual metric that unifies fidelity and realism for image quality assessment. The implementation is based on JAX and provides efficient computation of multi-scale feature statistics for comparing images.

## Key Concepts

### Wasserstein Distortion
Wasserstein Distortion is a metric that measures the perceptual difference between two images by:
1. Computing multi-scale feature representations
2. Calculating local statistics (means and variances) at different scales
3. Comparing these statistics using a Wasserstein-inspired distance measure
4. Weighting the contributions based on a sigma map that indicates desired summarization levels

### Sigma Map (`log2_sigma`)
The sigma map is a key component that controls the trade-off between local and global comparisons:
- **Low values (≈0)**: Focus on fine-grained, local differences
- **High values (≈5)**: Focus on global, summarized differences
- **Shape**: `(height, width)` - same spatial dimensions as input images

## Core Functions

### `safe_sqrt(x, limit)`
A numerically stable square root function with gradient limiting.

**Parameters:**
- `x`: Input array
- `limit`: Maximum gradient value to prevent numerical instability

**Returns:**
- Square root of input with bounded gradients

### `lowpass(inputs, stride)`
Applies a 3×3 lowpass filter to reduce aliasing during downsampling.

**Parameters:**
- `inputs`: Array of shape `(batch, height, width)`
- `stride`: Convolution stride (typically 1 or 2)

**Returns:**
- Filtered array with same or reduced spatial dimensions

### `compute_multiscale_stats(features, num_levels)`
Computes local means and variances at multiple scales using a pyramid approach.

**Parameters:**
- `features`: Input feature array `(channels, height, width)`
- `num_levels`: Number of pyramid levels to compute

**Returns:**
- `means`: List of mean arrays at each scale
- `variances`: List of variance arrays at each scale

**Process:**
1. For each level, apply lowpass filtering
2. Compute local statistics
3. Downsample by factor of 2 for next level

## Main Functions

### `wasserstein_distortion(features_a, features_b, log2_sigma, **kwargs)`

Computes Wasserstein Distortion between two feature arrays.

**Parameters:**
- `features_a`, `features_b`: Feature arrays `(channels, height, width)`
- `log2_sigma`: Sigma map `(height, width)` - base-2 log of summarization levels
- `num_levels`: Number of multi-scale levels (default: 5)
- `sqrt_grad_limit`: Gradient limit for square root (default: 1e6)
- `return_intermediates`: Whether to return intermediate computations

**Returns:**
- Scalar distortion value
- Optional: Dictionary of intermediate computations

**Algorithm:**
1. Validate input shapes and parameters
2. Compute multi-scale statistics for both feature arrays
3. For each scale level:
   - Calculate squared differences of means
   - Calculate squared differences of standard deviations
   - Apply sigma-based weighting
4. Sum weighted contributions across all levels

### `multi_wasserstein_distortion(features_a, features_b, log2_sigma, **kwargs)`

Extends single-array distortion to multiple feature arrays with different resolutions.

**Key Features:**
- Handles feature arrays of different spatial resolutions
- Automatically resizes and rescales sigma maps
- Useful for multi-scale or multi-layer feature comparisons

**Sigma Map Adaptation:**
- Resizes sigma map to match each feature array's spatial dimensions
- Rescales values based on resolution ratio to maintain semantic meaning
- Lower resolution features get correspondingly adjusted sigma values

### `vgg16_wasserstein_distortion(image_a, image_b, log2_sigma, **kwargs)`

High-level function for image comparison using VGG-16 features.

**Parameters:**
- `image_a`, `image_b`: RGB images `(3, height, width)`
- `log2_sigma`: Sigma map for the images
- `num_scales`: Number of image scales to process (default: 3)
- Other parameters as in base distortion function

**Process:**
1. Extract VGG-16 features at multiple image scales
2. Apply multi-array Wasserstein Distortion
3. Return aggregate distortion value

## Usage Examples

### Basic Usage with Custom Features

```python
import jax.numpy as jnp
from your_module import wasserstein_distortion

# Example feature arrays (e.g., from a neural network)
features_a = jnp.random.normal(0, 1, (64, 128, 128))  # 64 channels, 128x128
features_b = jnp.random.normal(0, 1, (64, 128, 128))

# Sigma map: 0 for local comparisons, higher for global
log2_sigma = jnp.zeros((128, 128))  # Focus on local differences
# Or: log2_sigma = jnp.ones((128, 128)) * 3  # More global comparison

# Compute distortion
distortion = wasserstein_distortion(features_a, features_b, log2_sigma)
```

### Image Comparison with VGG-16

```python
from your_module import vgg16_wasserstein_distortion

# RGB images
image_a = jnp.random.uniform(0, 1, (3, 256, 256))
image_b = jnp.random.uniform(0, 1, (3, 256, 256))

# Create a sigma map that varies spatially
height, width = 256, 256
y, x = jnp.mgrid[:height, :width]
center_y, center_x = height // 2, width // 2
distance = jnp.sqrt((y - center_y)**2 + (x - center_x)**2)
log2_sigma = distance / jnp.max(distance) * 4  # 0-4 range

# Compute perceptual distortion
distortion = vgg16_wasserstein_distortion(image_a, image_b, log2_sigma)
```

### Multi-Resolution Feature Comparison

```python
from your_module import multi_wasserstein_distortion

# Different resolution feature maps
features_a = [
    jnp.random.normal(0, 1, (32, 64, 64)),   # High resolution
    jnp.random.normal(0, 1, (64, 32, 32)),   # Medium resolution  
    jnp.random.normal(0, 1, (128, 16, 16)),  # Low resolution
]

features_b = [
    jnp.random.normal(0, 1, (32, 64, 64)),
    jnp.random.normal(0, 1, (64, 32, 32)),
    jnp.random.normal(0, 1, (128, 16, 16)),
]

# Base sigma map (will be automatically adapted for each resolution)
log2_sigma = jnp.ones((64, 64)) * 2

distortion = multi_wasserstein_distortion(features_a, features_b, log2_sigma)
```

## Implementation Details

### Numerical Stability
- **Gradient Limiting**: The `safe_sqrt` function prevents infinite gradients near zero
- **Variance Clamping**: Negative variance estimates are clamped to zero with gradient preservation
- **Lower Bounds**: Uses `gradient.lower_limit` for smooth clamping operations

### Memory Efficiency
- **In-place Operations**: Minimizes intermediate array allocations
- **Pyramid Processing**: Processes scales sequentially to reduce memory usage
- **Selective Computation**: Only computes required intermediate values

### Gradient Compatibility
- **JAX Integration**: Full compatibility with JAX's automatic differentiation
- **Custom JVP**: Hand-optimized forward-mode differentiation for `safe_sqrt`
- **Smooth Operations**: All operations maintain smooth gradients for optimization

## Performance Considerations

### Computational Complexity
- **Time**: O(num_levels × spatial_resolution × channels)
- **Memory**: O(spatial_resolution × channels) per scale level
- **Parallelization**: Fully vectorized operations leverage GPU/TPU efficiently

### Parameter Tuning
- **`num_levels`**: More levels = finer scale analysis, but higher computation
- **`sqrt_grad_limit`**: Balance between numerical stability and gradient accuracy
- **`num_scales`** (VGG): More scales = better multi-resolution analysis

## Applications

### Image Compression
Use as a perceptual loss function that balances compression artifacts with visual quality:
```python
# In training loop
compressed_image = compression_model(original_image)
loss = vgg16_wasserstein_distortion(original_image, compressed_image, sigma_map)
```

### Style Transfer
Measure content preservation while allowing style changes:
```python
# Use lower sigma values to preserve fine details
# Higher sigma values allow more global style changes
```

### Image Generation
Evaluate generated image quality against references:
```python
generated = generator(noise)
quality_score = vgg16_wasserstein_distortion(reference, generated, sigma_map)
```

## References

1. **Y. Qiu, A. B. Wagner, J. Ballé, L. Theis**: "Wasserstein Distortion: Unifying Fidelity and Realism," 2024 58th Ann. Conf. on Information Sciences and Systems (CISS), 2024. [arXiv:2310.03629](https://arxiv.org/abs/2310.03629)

2. **J. Ballé, L. Versari, E. Dupont, H. Kim, M. Bauer**: "Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion," 2025 IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. [arXiv:2412.00505](https://arxiv.org/abs/2412.00505)

## Dependencies

- **JAX**: Core computation framework
- **JAX NumPy**: Array operations
- **codex.loss.pretrained_features**: VGG-16 feature extraction
- **codex.ops.gradient**: Gradient utilities

## Error Handling

The implementation includes comprehensive input validation:
- **Shape Compatibility**: Ensures matching dimensions between feature arrays
- **Parameter Validation**: Checks that `max(log2_sigma) <= num_levels`
- **Type Safety**: Uses proper type hints and overloads for different return types

## Thread Safety

The implementation is stateless and thread-safe, making it suitable for:
- Parallel batch processing
- Distributed training
- Concurrent evaluation pipelines
