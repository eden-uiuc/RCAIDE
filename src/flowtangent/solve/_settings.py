from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .. import Module, TreePath, field, static_field, update
from ..utils import get_all_parents, get_all_targets
from ._mass import MassAnalysisSettings

# Energy Analysis --------------------------------------------------------------


class EnergyAnalysisSettings(Module):
    report_units: Literal["SI", "Imperial"] = field("SI", static=True)

    build_network: bool = field(True)
    clear_nodes: bool = field(True)


class AnalysisSettings[E_Type: EnergyAnalysisSettings](Module):
    aerodynamics: Optional[Module] = None
    energy: E_Type = field(EnergyAnalysisSettings)
    mass: MassAnalysisSettings = field(MassAnalysisSettings)


#  Numerical Settings --------------------------------------------------------------------------------------------------


class JacobianMap(Module):
    inputs: tuple[TreePath, ...]            = static_field(())
    outputs: tuple[TreePath, ...]           = static_field(())

    state_inputs: tuple[TreePath, ...]      = static_field(())
    state_outputs: tuple[TreePath, ...]     = static_field(())

    system_inputs: tuple[TreePath, ...]     = static_field(())
    system_outputs: tuple[TreePath, ...]    = static_field(())

    _n_st:  int = static_field(0)
    _n_sys: int = static_field(0)

    def __init__(
        self,
        inputs: tuple[TreePath | str | tuple, ...] = (),
        outputs: tuple[TreePath | str | tuple, ...] = (),
        state_inputs: Optional[tuple] = None,
        state_outputs: Optional[tuple] = None,
        system_inputs: Optional[tuple] = None,
        system_outputs: Optional[tuple] = None,
    ):
        self.inputs = tuple(TreePath(i) for i in inputs)
        self.outputs = tuple(TreePath(o) for o in outputs)

        _filter_in = lambda str: tuple(p for p in self.inputs if p.path[0].lower() == str)  # noqa: E731
        _filter_out = lambda str: tuple(p for p in self.inputs if p.path[0].lower() == str)  # noqa: E731

        # fmt: off
        self.state_inputs   = _filter_in("state") if state_inputs is None else state_inputs
        self.system_inputs  = _filter_in("system") if system_inputs is None else system_inputs
        self.state_outputs  = _filter_out("state") if state_outputs is None else state_outputs
        self.system_outputs = _filter_out("system") if system_outputs is None else system_outputs
        # fmt: on

        self._n_st = len(self.state_inputs)
        self._n_sys = len(self.system_inputs)

    def flatten_inputs(self, base_state, base_system):
        # Dynamically detect B from the base state (if any arrays are 3D)
        arr = next((l for l in jax.tree_util.tree_leaves(base_state) if isinstance(l, (jax.Array, np.ndarray))), None)
        has_B = arr is not None and arr.ndim == 3
        B = arr.shape[0] if has_B else None

        flat_st = []
        if self._n_st > 0:
            st_in = get_all_targets(base_state, self.state_inputs)
            flat_st = [x.reshape(B, -1) if has_B else x.reshape(-1) for x in st_in]
        flat_st_array = jnp.concatenate(flat_st, axis=-1) if flat_st else jnp.empty((B, 0) if has_B else (0,))

        flat_sys = []
        if self._n_sys > 0:
            sys_in = get_all_targets(base_system, self.system_inputs)
            flat_sys = [x.reshape(-1) for x in sys_in]
        flat_sys_array = jnp.concatenate(flat_sys, axis=-1) if flat_sys else jnp.empty((0,))

        return flat_st_array, flat_sys_array

    def update_inputs(self, flat_st, flat_sys, base_state, base_system):
        has_B = flat_st.ndim == 2
        B = flat_st.shape[0] if has_B else None

        st, sys = base_state, base_system

        # Update State
        if self._n_st > 0:
            st_in = get_all_targets(st, self.state_inputs)
            shapes = [x.shape[1:] if has_B else x.shape for x in st_in]
            sizes = [int(np.prod(s)) if s else 1 for s in shapes]

            splits = jnp.split(flat_st, np.cumsum(sizes)[:-1], axis=-1)
            new_slices = [s.reshape((B,) + shp) if has_B else s.reshape(shp) for s, shp in zip(splits, shapes)]

            parents = get_all_parents(st, self.state_inputs)
            updated = [
                p.at[pth.path_slice].set(n) if pth.path_slice != slice(None) else n
                for p, n, pth in zip(parents, new_slices, self.state_inputs)
            ]
            st = update(st, lambda t: get_all_parents(t, self.state_inputs), tuple(updated))

        # Update System
        if self._n_sys > 0:
            sys_in = get_all_targets(sys, self.system_inputs)
            shapes = [x.shape for x in sys_in]
            sizes = [int(np.prod(s)) if s else 1 for s in shapes]

            splits = jnp.split(flat_sys, np.cumsum(sizes)[:-1], axis=-1)
            new_slices = [s.reshape(shp) for s, shp in zip(splits, shapes)]

            parents = get_all_parents(sys, self.system_inputs)
            updated = [
                p.at[pth.path_slice].set(n) if pth.path_slice != slice(None) else n
                for p, n, pth in zip(parents, new_slices, self.system_inputs)
            ]
            sys = update(sys, lambda t: get_all_parents(t, self.system_inputs), tuple(updated))

        return st, sys

    def flatten_outputs(self, f_st, f_sys, f_setts):
        outputs = []
        if self.state_outputs:
            outputs.extend(get_all_targets(f_st, self.state_outputs))
        if self.system_outputs:
            outputs.extend(get_all_targets(f_sys, self.system_outputs))

        has_B = outputs[0].ndim == 3
        B = outputs[0].shape[0] if has_B else None

        if has_B:
            return jnp.concatenate([out.reshape(B, -1) for out in outputs], axis=-1)
        else:
            return jnp.concatenate([out.reshape(-1) for out in outputs], axis=-1)


class JacobianSettings(Module):
    calculate: bool = field(False, static=True)
    couple_time: bool = field(True, static=True)
    mapping: Optional[JacobianMap] = field(None, static=True)


class NumericalSettings(Module):
    relative_tolerance: float = field(1e-5, static=True)
    absolute_tolerance: float = field(1e-5, static=True)

    max_evaluations: int = field(100, static=True)
    step_size: float | None = field(None, static=True)

    batch_size: int = field(1, static=True)
    batch_mode: Literal["zip", "mesh"] = field("zip", static=True)

    number_of_control_points: int = field(1, static=True)
    maximum_graph_complexity: int = field(1e6, static=True)

    sum_residuals: bool = field(False, static=True)

    jacobian: JacobianSettings = field(JacobianSettings, static=True)
