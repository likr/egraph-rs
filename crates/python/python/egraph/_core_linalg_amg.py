from __future__ import annotations

from egraph._core import (
    AmgBatchSolveResult as AmgBatchSolveResult,
    AmgPcgSolver as AmgPcgSolver,
    AmgSolveResult as AmgSolveResult,
    EffectiveResistanceResult as EffectiveResistanceResult,
    ResistanceCurvatureFlowResult as ResistanceCurvatureFlowResult,
    compute_effective_resistances as compute_effective_resistances,
    resistance_curvature_flow as resistance_curvature_flow,
)

__all__ = [
    "AmgBatchSolveResult",
    "AmgPcgSolver",
    "AmgSolveResult",
    "EffectiveResistanceResult",
    "ResistanceCurvatureFlowResult",
    "compute_effective_resistances",
    "resistance_curvature_flow",
]
