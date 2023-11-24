# Distributed Computing

> 백지장도 맞들면 낫다

Computer software has been written for serial computation.
To solve a problem, an algorithm is constructed and implemented as a serial stream of instructions.
These instructions are executed on a CPU on one computer.
Only one instruction may execute at a time.
Afther one instruction is fininshed, the next one is executed.

A distributed system is a system whose components are located on different networked computers, which communicate and coordinate their actions by passing messages to one another.

Consider NVIDIA's H100 GPUs (maybe the best on earth).

![H100](spec-H100.png)

- 80 GB of memory is not sufficient for traning large (language) models.
- AlphaFold model induce large memory consumption due to its activations.

It's why we should introduce distributed computing.

## Message passing interface

### Broadcast

### Scatter

### All-Gather

### All-Reduce

### All-to-All

## NCCL

## Distributed computing in DeepFold
