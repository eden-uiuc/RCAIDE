# flowtangent/Framework/Analyses/Energy/__init__.py
# (c) Copyright 2023 Aerospace Research Community LLC

"""Flowtangent Package Setup"""

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

from ._energy_network import build_PACT_analysis as PACTAnalysis  # noqa: N812
from .jets import (
    JetSettings,
    build_turbofan_design,
    build_turbofan_performance,
    build_turbojet_design,
    build_turbojet_performance,
)

__all__ = [
    "PACTAnalysis",
    "JetSettings",
    "build_turbofan_design",
    "build_turbofan_performance",
    "build_turbojet_design",
    "build_turbojet_performance",
]
