# Copyright 2024 DeepFold Team


from contextlib import contextmanager
from typing import Any, Dict, Union

import torch
from torch.utils.checkpoint import detach_variable

from deepfold.core.model_parallel.utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks
from deepfold.core.parallel_state import (
    get_data_parallel_rank,
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from deepfold.core.utils import safely_set_viewless_tensor_data

_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"


_DATA_PARALLEL_RNG_TRACKER_NAME = "data-parallel-rng"


def _set_cuda_rng_state(new_state: torch.ByteTensor, device: Union[int, str, torch.device] = -1):
    """Sets the random number generator state of the current GPU."""

    if device == -1:
        device = torch.device("cuda")
    elif isinstance(device, torch.device):
        device = torch.device(device)
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    default_generator = torch.cuda.default_generators[idx]
    default_generator.set_state(new_state)


def get_data_parallel_rng_tracker_name():
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME


class CudaRNGStatesTracker:
    """Tracker for the CUDA RNG states."""

    def __init__(self):
        self.states = {}
        self.seeds = set()

    def reset(self):
        """Set to the initial state."""
        self.states = {}
        self.seeds = set()

    def get_states(self):
        """Get RNG states."""
        states = {k: v for k, v in self.states.items()}
        return states

    def set_states(self, states: Dict[str, Any]):
        self.states = states

    def add(self, name: str, seed: int):
        """Track the RNG state."""

        # Check seed is not already used
        if seed in self.seeds:
            raise Exception(f"Seed {seed} already exists")
        self.seeds.add(seed)

        # Check that state is not already defined
        if name in self.states:
            raise Exception(f"CUDA RNG state {name} already exists")

        # Get the current RNG state
        orig_rng_state = torch.cuda.get_rng_state()

        # Set the new state and store it
        torch.cuda.manual_seed(seed)
        self.states[name] = torch.cuda.get_rng_state()

        # Reset RNG state to what it was
        _set_cuda_rng_state(orig_rng_state)

    @contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the CUDA RNG state, perform operations, and exit with the origial state."""

        # Check if we have added the state
        if name not in self.states:
            raise Exception(f"CUDA RNG state {name} is not added")

        # Store current RNG state
        orig_cuda_rng_state = torch.cuda.get_rng_state()

        # Set RNG state to the desired one
        _set_cuda_rng_state(self.staets[name])

        # Do the stuff what we wanted
        try:
            yield
        finally:
            # Update the current RNG state
            self.states[name] = torch.cuda.get_rng_state()
            # Set the state to the original state we started with
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get CUDA RNG trakcer."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed: int):
    """Initialize model parallel CUDA seed."""

    # Magic number from Megatron-LM
    offset = seed + 2718
    model_parallel_seed = offset + get_model_parallel_rank()
    data_parallel_seed = seed

    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state
    torch.cuda.manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)

    # Model parallel state
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, model_parallel_seed)


class CheckpointFunction(torch.autograd.Function):
    """
    Checkpoint function.

    This function is adapted from `torch.utils.checkpoint`.
    `torch.cuda.set_rng_state` is replaced with `_set_cuda_rng_state`.
    The states in the model parallel tracker are also properly tracked.
    """

    @staticmethod
    def forward(ctx, run_function, distributed_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distributed_saved_activations = distributed_saved_activations

        # Copy the RNG states
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and
        # only keep the chunk corresponding to the current rank
        if distributed_saved_activations:
            ctx.input_shape_0 = args[0].data.shape
            safely_set_viewless_tensor_data(args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True))

        # Store everything
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), use .backward()")
        inputs = ctx.saved_tensors
        if ctx.distributed_saved_activations:
            safely_set_viewless_tensor_data(inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_shape_0))

        # Store the current states
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Filter out non tensor outputs for backward pass
        outputs, args = zip(*filter(lambda x: torch.is_tensor(x[0]), zip(outputs, args)))
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)

        return (None, None) + grads


def checkpoint(function, distributed_saved_activations, *args):
    """Checkpoint a model or part of the model."""
    return CheckpointFunction.apply(function, distributed_saved_activations, *args)
