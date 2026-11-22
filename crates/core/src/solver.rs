//! Cooperative iterative solver lifecycle trait for browser/UI-friendly stepping.

use crate::error::CoreResult;
use crate::layout::Layout2d;
use crate::scalar::Scalar;

/// Execution lifecycle status of a cooperative [`IterativeSolver`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SolverStatus {
    /// Solver has been configured and has not yet executed any iteration.
    Initialized,
    /// Solver has executed at least one iteration and has remaining work.
    Running,
    /// Solver stopped early because a convergence criterion was met.
    Converged,
    /// Solver finished all scheduled iterations.
    Completed,
}

impl SolverStatus {
    /// Returns `true` if the solver has reached a terminal state (`Converged` or `Completed`).
    #[inline]
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Converged | Self::Completed)
    }
}

/// Cooperative iterative layout solver supporting frame-budgeted `step()` execution.
///
/// Implementing `step()` lets WebAssembly `requestAnimationFrame` loops and Python
/// progressive visualizers advance a few iterations at a time without blocking a single
/// thread, while sharing identical math with batch `run()`.
pub trait IterativeSolver<S: Scalar = f64> {
    /// Executes a single iteration on `layout` and returns the updated [`SolverStatus`].
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus>;

    /// Executes up to `steps` iterations or stops early if the solver reaches a terminal status.
    fn run_steps(&mut self, layout: &mut Layout2d<S>, steps: usize) -> CoreResult<SolverStatus> {
        for _ in 0..steps {
            if self.status().is_finished() {
                break;
            }
            let status = self.step(layout)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status())
    }

    /// Runs the solver to completion (or early convergence).
    fn run(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        while !self.status().is_finished() {
            let status = self.step(layout)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status())
    }

    /// Returns the current lifecycle status.
    fn status(&self) -> SolverStatus;

    /// Returns the number of iterations executed so far.
    fn iteration(&self) -> usize;

    /// Returns the scheduled maximum number of iterations.
    fn total_iterations(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CoreError;

    struct DummyDecaySolver {
        iter: usize,
        total: usize,
        tol: f64,
        status: SolverStatus,
    }

    impl DummyDecaySolver {
        fn new(total: usize, tol: f64) -> Self {
            Self {
                iter: 0,
                total,
                tol,
                status: if total == 0 {
                    SolverStatus::Completed
                } else {
                    SolverStatus::Initialized
                },
            }
        }
    }

    impl IterativeSolver<f64> for DummyDecaySolver {
        fn step(&mut self, layout: &mut Layout2d<f64>) -> CoreResult<SolverStatus> {
            if self.status.is_finished() {
                return Ok(self.status);
            }
            if !layout.is_all_finite() {
                return Err(CoreError::non_finite("DummyDecaySolver"));
            }
            let mut max_abs = 0.0_f64;
            for v in layout.x_mut() {
                *v *= 0.5;
                max_abs = max_abs.max(v.abs());
            }
            for v in layout.y_mut() {
                *v *= 0.5;
                max_abs = max_abs.max(v.abs());
            }
            self.iter += 1;
            self.status = if max_abs <= self.tol {
                SolverStatus::Converged
            } else if self.iter >= self.total {
                SolverStatus::Completed
            } else {
                SolverStatus::Running
            };
            Ok(self.status)
        }

        fn status(&self) -> SolverStatus {
            self.status
        }

        fn iteration(&self) -> usize {
            self.iter
        }

        fn total_iterations(&self) -> usize {
            self.total
        }
    }

    #[test]
    fn test_iterative_solver_step_equivalence_and_convergence() {
        let mut batch_layout = Layout2d::from_xy(vec![4.0, -2.0], vec![2.0, 8.0]).unwrap();
        let mut step_layout = batch_layout.clone();

        let mut batch_solver = DummyDecaySolver::new(10, 0.0);
        let mut step_solver = DummyDecaySolver::new(10, 0.0);

        assert_eq!(step_solver.status(), SolverStatus::Initialized);
        assert!(!step_solver.status().is_finished());

        assert_eq!(
            step_solver.run_steps(&mut step_layout, 3).unwrap(),
            SolverStatus::Running
        );
        assert_eq!(step_solver.iteration(), 3);
        assert_eq!(
            step_solver.run(&mut step_layout).unwrap(),
            SolverStatus::Completed
        );
        assert_eq!(step_solver.iteration(), 10);
        assert_eq!(step_solver.total_iterations(), 10);

        assert_eq!(
            batch_solver.run(&mut batch_layout).unwrap(),
            SolverStatus::Completed
        );
        assert_eq!(batch_layout, step_layout);

        // Early convergence path.
        let mut conv_layout = Layout2d::from_xy(vec![1.0], vec![1.0]).unwrap();
        let mut conv_solver = DummyDecaySolver::new(10, 0.3);
        assert_eq!(
            conv_solver.run(&mut conv_layout).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(conv_solver.iteration(), 2);
        assert!(conv_solver.status().is_finished());
        // Calling step once finished is a clean no-op.
        assert_eq!(
            conv_solver.step(&mut conv_layout).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(conv_solver.iteration(), 2);
    }
}
