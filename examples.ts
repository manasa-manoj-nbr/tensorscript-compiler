# Example 1: Simple matrix multiplication
result = matmul(A, B)

# Example 2: Neural network forward pass
# Matrix multiply followed by ReLU activation
hidden = matmul(input, weights1)
activated = relu(hidden)
output = matmul(activated, weights2)

# Example 3: With optimizations
# Specify tile size for better shared memory usage
result = matmul(X, Y, tile_size=32)

# Example 4: Softmax classifier
logits = matmul(features, classifier_weights)
probabilities = softmax(logits)

# Example 5: Element-wise operations
sum_result = add(tensor1, tensor2)
final = relu(sum_result)
