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

This layer applies layer normalization on the channel dimensions with learnable per-channel scales $\gamma$ and biases $\beta$.
In this case, this layer will normalize over the last diemnsion which is expected to be of the specific size `c_in`.

$$ y = \frac{x - \left \langle x \right \rangle}{\sqrt{\mathrm{Var}(x) + \epsilon}} \odot \gamma + \beta $$

::: deepfold.model.alphafold.nn.primitives.LayerNorm

## Attention

This layer is a gated (and biased) multi-head attention used in AlphaFold.


::: deepfold.model.alphafold.nn.primitives.Attention

## Global attention

