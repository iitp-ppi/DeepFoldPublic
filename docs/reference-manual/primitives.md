# Primitive Layers

In this page, we introduce basic neural network layers in the model.

---

## Linear

This layer applies a linear transformation to the incoming data: $y = x A^T + b$.

```{r}
Input: x
y = einsum('bi,ij->bj', x, w) + b
```

::: deepfold.model.alphafold.nn.primitives.Linear

## Layer normalization

This layer applies layer normalization on the channel dimensions with learnable per-channel scales $\gamma$ and biases $\beta$.
In this case, this layer will normalize over the last diemnsion which is expected to be of the specific size `c_in`.

$$ y = \frac{x - \left \langle x \right \rangle}{\sqrt{\mathrm{Var}(x) + \epsilon}} \odot \gamma + \beta $$

::: deepfold.model.alphafold.nn.primitives.LayerNorm

## Attention

This layer is a gated (and biased) multi-head attention used in AlphaFold.

```{r}
Input:
    q_data  float[batch_size, n_queries, c_q]
    m_data  float[batch_size, n_keys, c_k]
    bias    float[batch_size, n_heads, n_queries, n_keys]

% query_w: (c_q, n_heads, c_hidden)
q = einsum('bqa,ahc->bqhc', q_data, query_w) * key_dim**(-0.5)
% key_w: (c_k, n_heads, c_hidden)
k = einsum('bqa,ahc->bqhc', m_data, key_w)
% value_w: (c_v, n_heads, c_hidden)
v = einsum('bqa,ahc->bqhc', m_data, value_w)

logits = einsum('bqhc,bkhc->bhqk', q, k)
logits += bias
weights = softmax(logits, -1)
weighted_avg = einsum('bhqk,bkhc->bqhc', weights, v)

% gating_w: (c_q, n_heads, c_hidden)
% gating_b: (n_heads, c_hidden)
gate_values = einsum('bqc,chv->bqhv', q_data, gating_w) +' gating_b
gate_values = sigmoid(gate_values)

weighted_avg *= gate_values

% output_w: (n_heads, c_hidden, c_output)
% output_b: (c_output)
output = einsum('bqhc,hco->bqo', weighetd_avg, output_w) +' output_b

return output
```

::: deepfold.model.alphafold.nn.primitives.Attention

## Global attention

This layer is global column-wise self-attention in AlphaFold. (Alg. 19)

```{r}
Input:
    m_data  float[batch_size, num_seq, num_res, c_in]
    mask    int[batch_size, num_seq, num_res]

% (batch_size, num_seq, num_res, 1)
mask = expand_dims(mask, -1)

% (*, num_res, c_in); Average on column dimension
q_avg = sum(m * mask, -3) / (sum(mask, -3) + eps)

% query_w: (channels, num_res, c_hidden)
q = einsum('bna,ahc->bnhc', q_avg, query_w) * c_hidden**(-0.5)
% key_w: (channels, c_hidden)
k = einsum('bsna,ac->bsnc', m_data, key_w)
% value_w: (channels, c_hidden)
v = einsum('bsna,ac->bsnc', m_data, value_w)

logits = einsum('bnhc,btnc->bnht', q, k)
logits += self.inf * (mask - 1)
weights = softmax(logit, -1)
weighted_avg = einsum('bnht,btnc->bnhc', weights, v)

% gating_w: (c_in, n_heads, c_hidden)
% gating_b: (n_heads, c_hidden)
gate_values = einsum('bsnc,chd->bsnhd', m_data, gating_w) +' gating_b
gate_values = sigmod(gate_values)

% output_w: (n_heads, c_hidden, c_output)
% output_b: (c_output)
output = einsum('bsnhc,hco->bsno', weighted_avg, output_w) +' output_b

return output
```

::: deepfold.model.alphafold.nn.primitives.GlobalAttention
