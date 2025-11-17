import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query matrix of shape (n_queries, d_k)
        K: Key matrix of shape (n_keys, d_k)
        V: Value matrix of shape (n_keys, d_v)

    Returns:
        attention_weights: (n_queries, n_keys)
        context: (n_queries, d_v)
    """
    # 1. Compute raw scores: Q K^T
    scores = np.matmul(Q, K.T)          # shape: (n_queries, n_keys)

    # 2. Scale by sqrt(d_k)
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Apply softmax to get attention weights
    attention_weights = softmax(scaled_scores, axis=-1)

    # 4. Compute context vector as weighted sum of V
    context = np.matmul(attention_weights, V)  # shape: (n_queries, d_v)

    return attention_weights, context
# ---- Test Example ----
import numpy as np

Q = np.array([[1.0, 0.0],
              [0.0, 1.0]])

K = np.array([[1.0, 0.0],
              [0.0, 1.0]])

V = np.array([[1.0, 2.0],
              [3.0, 4.0]])

attn, context = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", attn)
print("\nContext Vector:\n", context)
