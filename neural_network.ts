# Complete Neural Network Example
# This demonstrates a full forward pass through a simple network

# Layer 1: Input -> Hidden
# Input: batch_size x input_dim
# Weights1: input_dim x hidden_dim
hidden_linear = matmul(input_batch, weights1)
hidden_activated = relu(hidden_linear)

# Layer 2: Hidden -> Output  
# Weights2: hidden_dim x output_dim
output_linear = matmul(hidden_activated, weights2)

# Final activation: Softmax for classification
predictions = softmax(output_linear)

# The compiler will generate:
# - 2 optimized matmul kernels (with shared memory tiling)
# - 1 relu kernel (element-wise max with 0)
# - 1 softmax kernel (numerically stable with max subtraction)
