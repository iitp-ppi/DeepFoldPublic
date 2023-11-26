from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._ops import OpOverload

# Utility functions for working with nested python data structures
from torch.utils._pytree import TreeSpec, tree_map_only

from deepfold.distributed.tensor.device_mesh import DeviceMesh
from deepfold.distributed.tensor.placement_types import DTensorSpec, Placement

# Common type aliases
ArgsType = Tuple[Any, ...]
KwargsType = Dict[str, Any]
# ATen op can have output Tensor, Tuple[Tensor] and List[Tensor]
OutputSpecType = Optional[Union[DTensorSpec, Sequence[Optional[DTensorSpec]]]]


def _rebuild_tensor_from_dtensor_meta(spec: DTensorSpec) -> torch.Tensor:
    """This is used to propagate tensor metadata under fake mode."""
    assert spec.tensor_meta is not None, "DTensorSpec does not contain tensor_meta"
    return torch.empty_strided(spec.tensor_meta.shape, spec.tensor_meta.stride, dtype=spec.tensor_meta.dtype)


# TODO: make more concrete
def _is_inplace_op(op: OpOverload) -> bool:
    return op._schema.name[-1] == "_"


# TODO: make more concrete
def _is_out_variant_op(op: OpOverload) -> bool:
    return "out" in op._schema.overload_name


@dataclass
class PlacementStrategy:
    """
    Describes an acceptable sharding placements of the output and the tensor arguments of an operation.
    """

    output_sepc = DTensorSpec
    input_specs: Optional[Sequence[DTensorSpec]]

    # TODO: Redistribute costs

    def _pprint_placements(self, placements: Sequence[Placement]):
        return "".join([str(p) for p in placements])

    def __str__(self) -> str:
        if self.input_specs is None:
            input_specs_str = ""
        else:
            input_specs_str = "("
            input_specs_str += ", ".join([self._pprint_placements(spec.placements) for spec in self.input_specs])
            input_specs_str += ")"
        output_spec_str = self._pprint_placements(self.output_sepc.placements)
        return f"{input_specs_str} -> {output_spec_str}"


class StrategyType:
    pass


class OpStrategy(StrategyType):
    """A list of placement strategies associated with the op."""

    def __init__(self, strategies: List[PlacementStrategy]) -> None:
        super().__init__()
        self.strategies: List[PlacementStrategy] = strategies

    def __str__(self) -> str:
        strategy_list_str = ", ".join([str(strategy) for strategy in self.strategies])
        mesh_shape = self.strategies[0].output_sepc.mesh.shape
        return f"OpStrategy: [{strategy_list_str}] on mesh: {mesh_shape}"

    def max_num_shards(self) -> int:
        """Returns the maximum number of shards across all placement strategies."""
        return max([strategy.output_spec.num_shards for strategy in self.strategies])

    @property
    def output_shape(self):
        return self.strategies[0].output_sepc.shape

    @property
    def output_ndim(self):
        return self.strategies[0].output_sepc.ndim


class TupleStrategy(StrategyType):
    """
    Represents the output strategy of this op is a tuple of strategy, i.e., if the output of this op
    is a tuple of tensors or list of tensors with possibly different placement strategies, we should
    return a TupleStrategy that contains a tuple of OpStrategy.

    Note that if the output of the op is a List[Tensor] and they share the same placement strategy,
    then we should return a single OpStrategy instead of a TupleStrategy.
    """

    def __init__(self, childs: Sequence[StrategyType]) -> None:
        super().__init__()
        self.childs: Sequence[StrategyType] = childs

    def __str__(self) -> str:
        child_strategies_str = ", ".join([f"{str(strategy)}" for strategy in self.childs])
        return f"TupleStrategy({child_strategies_str})"


@dataclass
class RuntimeSchemaInfo:
    """
    Stores the operator schema related information for runtime execution.
    This is mainly used for two ways: (1) to generate hash for args to determine whether to re-rnun
    sharding prop or not not (2) to determine if we need pytree.
    """

    # Records static argument starting index for ops that have non-tensor args/kwargs
    # which would affect sharding propagation results.
    # All arguments after this index would be hashed to our sharding cache
    num_static_args: int = 64

    # Records static keyward arugment names which would affect sharding propagation
    static_kwarg_keys: Optional[List[str]] = None

    # To use pytree flatten/unflatten during operator execution. We don't need by default
    needs_pytree: bool = False


@dataclass
class OpSchema:
    """
    A data class that describes an operator input schemas, it includes DTensor, DTensorSpec and non-tensor
    args/kwargs (positional order preserved).
    It is mainly used by the dispatching logica below.

    Args:
        op: OpOverload
            the operator overload we are intercepting
        args_shcema:
            contains args except that the DTensor args have been replaced with its DTensorSpec
        kwargs_schema:
            contains kwargs except that the DTensor kwargs have been replaced with its DTensorSpec
    """

    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType

    schema_info: Optional[RuntimeSchemaInfo] = None

    @property
    def args_spec(self) -> Tuple[DTensorSpec, ...]:
        """
        Contains a clean list of args spec list with no non-DTensor positional arguments (i.e., int, float,
        tuple, etc) mainly used by sharding propagation to propagate the output spec.
        """
        # Filter out non-relavent values from args schema to get a clean spec list
        return tuple(item for item in self.args_schema if isinstance(item, DTensorSpec))

    def __repr__(self) -> str:
        return f"OpSchema(op={self.op}, args_schema={self.args_schema}, kwargs_schema={self.kwargs_schema})"

    def __str__(self) -> str:
        args_sharding: List[str] = []
        mesh_shape = None

        for arg in self.args_schema:
            if isinstance(arg, DTensorSpec):
                args_sharding.append(str(arg))
                mesh_shape = arg.mesh.shape
            elif isinstance(arg, OpStrategy):
                assert len(arg.strategies) == 1
                arg_spec = arg.strategies[0].output_sepc
                args_sharding.append(str(arg_spec))
                mesh_shape = arg_spec.mesh.shape
            elif isinstance(arg, TupleStrategy):
                first_op_strategy = arg.childs[0]
                assert isinstance(first_op_strategy, OpStrategy)
                mesh_shape = first_op_strategy.strategies[0].output_sepc.mesh.shape
                args_sharding.append(str(arg))
            else:
                args_sharding.append(str(arg))

        return f"Op(op={self.op}, args_sharding={', '.join(args_sharding)} on mesh: {mesh_shape})"

    def __post_init__(self) -> None:
        has_symints = False
        for a in self.args_schema:
            if isinstance(a, DTensorSpec) and a.tensor_meta is not None:
                if any(isinstance(s, torch.SymInt) for s in a.tensor_meta.shape):
                    has_symints = True
                    break
        self.has_symints = has_symints

    def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool:
        arg = self.args_schema[arg_idx]
        is_tensor = isinstance(arg, DTensorSpec)

        if is_tensor:
            return True
        if not isinstance(arg, list):
            return False

        return all(isinstance(a, DTensorSpec) or a is None for a in arg)

    def return_type_tuple_tensors(self) -> bool:
        return_types = self.op._schema.returns
        # All dispatch ops only return Tensor or Tuple[Tensor]
        return len(return_types) > 1 and isinstance(return_types[0].type, torch.TensorType)

    def return_type_tensor(self) -> bool:
        return_types = self.op._schema.returns
        # All dispatch ops only return Tensor or Tuple[Tensor] for tensor-like return types
        return isinstance(return_types[0].type, torch.TensorType)

    def __hash__(self) -> int:
        # Only hash args and kwargs
        if not self.schema_info:
            num_static_args = len(self.args_schema)
            static_kwarg_keys = None
        else:
            num_static_args = self.schema_info.num_static_args
            static_kwarg_keys = self.schema_info.static_kwarg_keys

        args_to_hash = tuple(
            tuple(a) if isinstance(a, list) else a
            for i, a in enumerate(self.args_schema)
            if self.arg_type_tensor_or_tensor_list_like(i) or i >= num_static_args
        )

        if static_kwarg_keys is not None:
            kwargs_to_hash = tuple(self.kwargs_schema.get(k, None) for k in static_kwarg_keys)
            return hash((self.op, args_to_hash, kwargs_to_hash))
        else:
            return hash((self.op, args_to_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpSchema):
            return False

        if self.op != other.op:
            return False

        if len(self.args_schema) != len(other.args_schema):
            return False

        if not self.schema_info:
            num_static_args = len(self.args_schema)
            static_kwrags_keys = None
        else:
            num_static_args = self.schema_info.num_static_args
            static_kwrags_keys = self.schema_info.static_kwarg_keys

        for i, (self_arg, other_arg) in enumerate(zip(self.args_schema, other.args_schema)):
            if isinstance(self_arg, DTensorSpec) and self_arg != other_arg:
                return False
            elif i >= num_static_args and self_arg != other_arg:
                return False

        if static_kwrags_keys:
            for key in static_kwrags_keys:
                if self.kwargs_schema.get(key, None) != other.kwargs_schema.get(key, None):
                    return False

        return True

    def gen_fake_args(self) -> ArgsType:
        return tree_map_only(DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.args_schema)

    def gen_fake_kwargs(self) -> KwargsType:
        return tree_map_only(DTensorSpec, _rebuild_tensor_from_dtensor_meta, self.kwargs_schema)

    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        suggestion_args_spec = self.args_spec
        new_arg_schema: List[Any] = []
        idx_of_args_spec = 0
        for arg in origin_schema.args_schema:
            if isinstance(arg, DTensorSpec):
                new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
                idx_of_args_spec += 1
            else:
                new_arg_schema.append(arg)
        self.args_schema = tuple(new_arg_schema)
        self.kwargs_schema = origin_schema.kwargs_schema


@dataclass
class OutputSharding:
    """
    A data class that is used by the sharding propagation rules. It could set the output_spec upon successful
    propagation, and if it failed, output_spec would become None and sharding propagation rules could give a
    list of suggestions for inupts to reshard.

    Note that the schema_suggestion generation by sharding propagation should be exactly the same as the
    operator OpSchema, except the DTensor and DTensorSpecs.
    """

    output_spec: OutputSpecType
    schema_suggestions: Optional[List[OpSchema]] = None
    failed_reason: Optional[str] = None
    needs_redistribute: bool = False


@dataclass
class OpInfo:
    """Operator execution info."""

    mesh: DeviceMesh
    schema: OpSchema
    flat_args_schema: List[Any]
    local_args: Sequence[Any]
    local_kwargs: Dict[str, Any]
    args_tree_spec: Optional[TreeSpec] = None
    output_sharding: Optional[OutputSharding] = None
