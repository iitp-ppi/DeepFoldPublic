# Proposure

## Introduction

We suggest distributed tensor primitives to make distributed computing in the SPMD (Single Program Multiple Device) paradigm easier.
The primitives used to express tensor distributions with both sharding and replication parallelism strategies should be simple yet powerful.
This could enable tensor parallelism as well as other advanced parallelism techniques.

```python
import torch
from distributed import DeviceMesh, Shard, distribute_tensor

mesh = DeviceMesh("cuda", list(range(world_size)))
local_tensor = torch.randn(16384, 512)
global_tensor = distribute_tensor(local_tensor, mesh, [Shard(0)])
```

## Motivation

Nowadays, the scale of neural network models keeps growing.
Large model parameters or activations exceed the memory capacity of a single state-of-the-art datacenter GPU.
Therefore, engineers begin to distribute models to multiple GPUs.

There are mainly three ways to scale up distributed training: data parallel, tensor parallel, and pipeline parallel.
When the strategies are put together independently, each of them operates on a distinct dimension.
Users would like to use these techniques concurrently when training or inferring extremely large models, despite the ineffectiveness of the extant solutions.

An ideal scenario is that users should be able to construct their models on a single device without having to worry about training distribution in a cluster.
For instance, researchers only need to construct their large transformer model, and the solution will automatically determine how to divide the model and run pipeline parallel across multiple nodes, as well as how to run data parallel and tesnor parallel within each node.

We present the DTensor concept to represent generic data distributions across nodes, inspired by
[Mesh TensorFlow](https://arxiv.org/abs/1811.02084),
[OneFlow](https://arxiv.org/abs/2110.15032),
[GSPMD](https://arxiv.org/abs/2105.04663),
and [Alpa](https://arxiv.org/abs/2201.12023).
With the DTensor abstraction, we hope that users can build efficient parallelism strategies in a easy way.

## Related works

### GSPMD

* The fundamental component of JAX/TensorFlow's distributed training.
* It enables various optimizations with the XLA compiler.
* It have three types of sharding strategies within a tensor: tield, replicated, and partially tiled to represent sharding and replication.

### OneFlow

* The `GlobalTensor` concept which is a variant form of GSPMD sharding.
* Split, broadcast, and partial sum concepts.
* They don't use partially tiled but have a concept of partial sum instead.

### Mesh TensorFlow

* `DTensor`
* `Mesh`
* `Layout`

---

## Goals

* DTensor should be the starting point of a SPMD programming model for researchers providing nice UX to mix up different types of parallelism.
* DTensor should be flexible enough for advanced users who want to mix sharding and replication.
* DTensor should offer a uniform way to store and load state dict checkpointing.

## `DTensor`

Deep learning models can be abstracted as a single expression of `output = model(inputs)`.
If we expand this expression, it could be treated as a computation graph that consists of the following expression.

> `output = op(inputs, params)`

We do not want to waste resources by executing the same computation multiple times, unless doing so is necessary to save memory or reduce communication overhead.

### Data distribution types

There are multiple ways to distribute the data and computation.

* **Distribute the parameter storage**: If model parameteres and gradients are too big to fit into a single device, or replication incurs too much communication overhead to sync parameters and gradients, we have to split the parameters and gradients across devices.
* **Distribute the activation storage**: If input and output activations are too big to fit into a single device, we have to split the data across devices. Note that output activation is highly correlated with parameter and input distribution, so we fold this with input distribution for simplicty.
* **Distribute the computation**: Given the input activation and parameter distributions, we could choose how to paralleize the computation.

Each combinations of the above three categories could result in different distributed computation strategies.

We choose to use three types of data distribution: shard, replicate, and partial.
These three types of data distribution can be applied to the global tensor on any mesh dimension.

* **shard** (by dimension): Shard or split the tensor on the specific tensor axes across devices.
* **replicate**: Replicate the tensor across devices, each rank gets the exact same tensor.
* **partial**: A type of tensor that has the same shape across devices, but has partial values on each device. It could be reduced (i.e., sum/min/max) to get the DTensor. Note that this is often useful as intermediate representations.

This is similar to OneFlow's representation.

The following table shows three types of distribution to represent all possible combinations.

|      Types      | Computation | Parameter | Activation |                         Examples                         |
| :-------------: | :---------: | :-------: | :--------: | :------------------------------------------------------: |
| `torch.Tensor`  |      X      |     X     |     X      |                  Single device training                  |
|        ?        |      O      |     X     |     X      |                         Useless                          |
|        ?        |      X      |     O     |     X      |                         Useless                          |
|  `I:S P:R O:S`  |      X      |     X     |     O      |                  Data parallelism(DDP)                   |
| `I:S P:S O:S/P` |      O      |     O     |     X      |                    Tensor parallelism                    |
|  `I:S P:S O:S`  |      O      |     O     |     O      | Shard the parameters and input, but do local computation |
| `I:S P:R O:S/P` |      O      |     X     |     O      |                            ?                             |
| `I:S P:S O:S/P` |      O      |     O     |     O      |                    Tensor parallelism                    |

`I: Input, P: Parameter, O: Output`

We plan to the DTensor support both foward and backward just like `torch.Tensor`.
This could be done by `DTensor` to be subclass of `Tensor` and utilize `__torch_dispatch__`.
See [the thread](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557).

## `DTensorSpec`

How to describe the layout?

### Logical device mesh

* We don't have to express a global view or topology of devices.
* **Each rank or process manages exactly one device.**
* **Assume homogeneity.**
* Use the `global_rank` as an device ID for each device.
* `local_device_id = rank % gpus_per_node`
* `DeviceMesh` describes the device and layout of the cluster.

`DeviceMesh` could be constructed in an arbitrary manner.

1. Same devices with same layout.
2. Same devices with different layout.
3. Different devices with same or different layouts.

We will support the first case.
(We wish to cover all cases.)
Note that JAX forces the first case, GSPMD/XLA allows the second case, OneFlow/TF allows the third case.
The third case enables flexible cases like pipeline parallelism, but it requires mesh send/recv communication.

### `PlacementSpec`

* `device_mesh`: A $n$-dimensional array specifies the ranks to place this tensor into.
* `placements`: An array that has the same rank as the `device_mesh` tensor. It is usee to describe how the DTensor data is distributed in the $i$th dimension of the `device_mesh`.

### Examples

```python
device_mesh = DeviceMesh([[0, 1], [2, 3]])
spec = [Replicate(), Shard(0)]
distributed.zeros((128, 64), device="cuda", device_mesh=device_mesh, placements=spec)
```
## Operations

We use `torch_dispatch` along with `torch_function` to implement operations.

### Dispatching

There are multiple ways to partition a Tensor according to the placement strategy, so when implementing a specific operator, there are also multiple implementations to be considered.
Therefore, we need to have some way to dispatch the operator implementation based on the *input sharding strategies*.

### Signature set

Let's consider a typical `matmul(a, b)` operator.

1. `a: Shard(0), b: Shard(1) -> Shard(0) and/or Shard(1)`
2. `a: Shard(1), b: Shard(0) -> Partial`
3. `a: Replicate, b: Shard(1) -> Shard(1)`
4. `a: Shard(0), b: Replicate -> Shard(0)`

The partial results in the second case could be converted to replicated.

### Sharding propagation

1. If there is sharding propagation rule for the operator, we just run the propagation to get the output sharding.
2. If only a certain set of operators are accepted, we do automatic resharding.
3. If there is no sharding propagation rule for the operator, we redistribute the inputs to replicate (`all_gather`) and compute.

We have to cover all existing PyTorch operators.

### Ambiguity

We choose a simple cost model (i.e., minimum communication volume) for ambiguous distributed operations because DTensor will only be able to make local decisions.
We left the chance of optimizations at a higher level stage for others (like a compiler).
Optimizing memory footprint can be another choice.

### Automatic resharding

What if inputs do not match any input signature of the operator?
We need to have some way to convert the mismatched inputs to legit inputs.

* We can redistribute the mismatched input before computation. It is similar to OneFlow's box operator.
* We can have some minimal distance algorithm or cost model.

## Case studies

Rules for `redistribute`.

|        Forward        |       Backward        |
| :-------------------: | :-------------------: |
| Sharded -> Replicated | Replicated -> Sharded |
| Replicated -> Sharded | Sharded -> Replicated |

## Checkpointing
