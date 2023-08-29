# Primitive Layers

In this page, we introduce basic neural network layers in the model.

---

## Linear

This layer applies a linear transformation to the incoming data: $y = x A^T + b$.

```
y = einsum('i,ij->j', x, w) + b
```

::: deepfold.model.alphafold.nn.primitives.Linear

## Layer normalization

## Attention

## Global attention
