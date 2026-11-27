//! Cooperative iterative solver, one-shot generator, and in-place transform traits.

use crate::error::CoreResult;

/// Execution lifecycle status of a cooperative [`IterativeSolver`].
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SolverStatus {
    /// Solver has been configured and has not yet executed any iteration.
    Initialized,
    /// Solver has executed at least one iteration and has remaining work.
    Running,
    /// Solver stopped early because a convergence criterion was met.
    Converged,
    /// Solver finished all scheduled iterations.
    Completed,
    /// Solver encountered a fatal numerical or internal breakdown and cannot continue.
    Failed,
}

impl SolverStatus {
    /// Returns `true` if the solver has reached a terminal state
    /// (`Converged`, `Completed`, or `Failed`).
    #[inline]
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Converged | Self::Completed | Self::Failed)
    }

    /// Returns `true` if the solver terminated due to a fatal breakdown (`Failed`).
    #[inline]
    pub fn is_failed(self) -> bool {
        matches!(self, Self::Failed)
    }
}

/// Cooperative iterative process mutating `Self::Target` one step at a time.
///
/// Implementing `step()` lets WebAssembly `requestAnimationFrame` loops and Python
/// progressive visualizers advance a few iterations at a time without blocking a single
/// thread, while sharing identical math with batch `run()`.
pub trait IterativeSolver {
    /// Target layout or polyline buffer mutated by this solver.
    type Target: ?Sized;

    /// Executes a single iteration on `target` and returns the updated [`SolverStatus`].
    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus>;

    /// Executes up to `steps` iterations, stopping early if the solver reaches a terminal status.
    ///
    /// Even when `steps == 0`, implementations validate `target` compatibility (length,
    /// dimensionality, finiteness) before returning without mutating `target` or `status()`.
    fn run_steps(&mut self, target: &mut Self::Target, steps: usize) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;
        for _ in 0..steps {
            if self.status().is_finished() {
                break;
            }
            let status = self.step(target)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status())
    }

    /// Runs the solver to completion, early convergence, or failure.
    fn run(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;
        while !self.status().is_finished() {
            let status = self.step(target)?;
            if status.is_finished() {
                break;
            }
        }
        Ok(self.status())
    }

    /// Validates that `target` is compatible with this solver without mutating state.
    fn validate_target(&self, target: &Self::Target) -> CoreResult<()>;

    /// Returns the current lifecycle status.
    fn status(&self) -> SolverStatus;

    /// Returns the number of iterations executed so far.
    fn iteration(&self) -> usize;

    /// Returns the scheduled iteration budget, or `None` for open-ended convergence solvers.
    fn total_iterations(&self) -> Option<usize>;
}

impl<T: IterativeSolver + ?Sized> IterativeSolver for &mut T {
    type Target = T::Target;

    #[inline]
    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        (**self).step(target)
    }

    #[inline]
    fn run_steps(&mut self, target: &mut Self::Target, steps: usize) -> CoreResult<SolverStatus> {
        (**self).run_steps(target, steps)
    }

    #[inline]
    fn run(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        (**self).run(target)
    }

    #[inline]
    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        (**self).validate_target(target)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        (**self).status()
    }

    #[inline]
    fn iteration(&self) -> usize {
        (**self).iteration()
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        (**self).total_iterations()
    }
}

impl<T: IterativeSolver + ?Sized> IterativeSolver for Box<T> {
    type Target = T::Target;

    #[inline]
    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        (**self).step(target)
    }

    #[inline]
    fn run_steps(&mut self, target: &mut Self::Target, steps: usize) -> CoreResult<SolverStatus> {
        (**self).run_steps(target, steps)
    }

    #[inline]
    fn run(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        (**self).run(target)
    }

    #[inline]
    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        (**self).validate_target(target)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        (**self).status()
    }

    #[inline]
    fn iteration(&self) -> usize {
        (**self).iteration()
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        (**self).total_iterations()
    }
}

/// One-shot generator producing a newly allocated layout or edge polyline buffer.
pub trait LayoutGenerator {
    /// Generated layout or polyline container type.
    type Output;

    /// Computes and returns a newly generated output.
    fn generate(&mut self) -> CoreResult<Self::Output>;
}

impl<T: LayoutGenerator + ?Sized> LayoutGenerator for &mut T {
    type Output = T::Output;

    #[inline]
    fn generate(&mut self) -> CoreResult<Self::Output> {
        (**self).generate()
    }
}

impl<T: LayoutGenerator + ?Sized> LayoutGenerator for Box<T> {
    type Output = T::Output;

    #[inline]
    fn generate(&mut self) -> CoreResult<Self::Output> {
        (**self).generate()
    }
}

/// In-place one-shot transformation (e.g., overlap removal, decluttering, polyline smoothing).
pub trait LayoutTransform {
    /// Target layout or polyline buffer mutated by this transformation.
    type Target: ?Sized;

    /// Applies the transformation in place to `target`.
    fn apply(&mut self, target: &mut Self::Target) -> CoreResult<()>;
}

impl<T: LayoutTransform + ?Sized> LayoutTransform for &mut T {
    type Target = T::Target;

    #[inline]
    fn apply(&mut self, target: &mut Self::Target) -> CoreResult<()> {
        (**self).apply(target)
    }
}

impl<T: LayoutTransform + ?Sized> LayoutTransform for Box<T> {
    type Target = T::Target;

    #[inline]
    fn apply(&mut self, target: &mut Self::Target) -> CoreResult<()> {
        (**self).apply(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CoreError;
    use crate::layout::{Layout2d, LayoutNd};

    struct DecaySolver {
        expected_nodes: usize,
        iter: usize,
        total: Option<usize>,
        tol: f64,
        fail_at_step: Option<usize>,
        status: SolverStatus,
    }

    impl DecaySolver {
        fn new(expected_nodes: usize, total: Option<usize>, tol: f64) -> Self {
            Self {
                expected_nodes,
                iter: 0,
                total,
                tol,
                fail_at_step: None,
                status: if total == Some(0) {
                    SolverStatus::Completed
                } else {
                    SolverStatus::Initialized
                },
            }
        }

        fn with_failure_at(mut self, step: usize) -> Self {
            self.fail_at_step = Some(step);
            self
        }
    }

    impl IterativeSolver for DecaySolver {
        type Target = Layout2d<f64>;

        fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
            if target.len() != self.expected_nodes {
                return Err(CoreError::dimension_mismatch(
                    self.expected_nodes,
                    target.len(),
                    "DecaySolver::validate_target",
                ));
            }
            target.validate_finite()
        }

        fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
            self.validate_target(target)?;
            if self.status.is_finished() {
                return Ok(self.status);
            }
            if self.fail_at_step == Some(self.iter + 1) {
                self.status = SolverStatus::Failed;
                return Err(CoreError::numerical_breakdown("simulated breakdown"));
            }
            let mut max_abs = 0.0_f64;
            for v in target.x_mut() {
                *v *= 0.5;
                max_abs = max_abs.max(v.abs());
            }
            for v in target.y_mut() {
                *v *= 0.5;
                max_abs = max_abs.max(v.abs());
            }
            self.iter += 1;
            self.status = if max_abs <= self.tol {
                SolverStatus::Converged
            } else if self.total.is_some_and(|t| self.iter >= t) {
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

        fn total_iterations(&self) -> Option<usize> {
            self.total
        }
    }

    #[test]
    fn test_iterative_solver_lifecycle_zero_step_and_failure() {
        let mut batch_layout = Layout2d::from_xy(vec![4.0, -2.0], vec![2.0, 8.0]).unwrap();
        let mut step_layout = batch_layout.clone();

        let mut batch_solver = DecaySolver::new(2, Some(10), 0.0);
        let mut step_solver = DecaySolver::new(2, Some(10), 0.0);

        assert_eq!(step_solver.status(), SolverStatus::Initialized);
        assert!(!step_solver.status().is_finished());
        assert!(!step_solver.status().is_failed());
        assert_eq!(
            step_solver.run_steps(&mut step_layout, 0).unwrap(),
            SolverStatus::Initialized
        );
        assert_eq!(step_solver.iteration(), 0);
        assert_eq!(step_layout, batch_layout);

        // run_steps(target, 0) rejects invalid targets without mutating solver/target state.
        let mut wrong_len = Layout2d::zeros(3);
        assert!(matches!(
            step_solver.run_steps(&mut wrong_len, 0),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert_eq!(step_solver.status(), SolverStatus::Initialized);
        let mut non_finite = Layout2d::from_xy(vec![f64::NAN, 0.0], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            step_solver.run_steps(&mut non_finite, 0),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert_eq!(step_solver.status(), SolverStatus::Initialized);

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
        assert_eq!(step_solver.total_iterations(), Some(10));
        assert_eq!(
            batch_solver.run(&mut batch_layout).unwrap(),
            SolverStatus::Completed
        );
        assert_eq!(batch_layout, step_layout);

        // Open-ended convergence solver (`total_iterations == None`).
        let mut conv_layout = Layout2d::from_xy(vec![1.0], vec![1.0]).unwrap();
        let mut conv_solver = DecaySolver::new(1, None, 0.3);
        assert_eq!(conv_solver.total_iterations(), None);
        assert_eq!(
            conv_solver.run(&mut conv_layout).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(conv_solver.iteration(), 2);
        assert!(conv_solver.status().is_finished());
        assert!(!conv_solver.status().is_failed());
        assert_eq!(
            conv_solver.step(&mut conv_layout).unwrap(),
            SolverStatus::Converged
        );

        // Breakdown transitions to Failed terminal state.
        let mut fail_layout = Layout2d::from_xy(vec![2.0], vec![2.0]).unwrap();
        let mut fail_solver = DecaySolver::new(1, Some(5), 0.0).with_failure_at(2);
        assert!(matches!(
            fail_solver.run(&mut fail_layout),
            Err(CoreError::NumericalBreakdown(_))
        ));
        assert_eq!(fail_solver.status(), SolverStatus::Failed);
        assert!(fail_solver.status().is_finished());
        assert!(fail_solver.status().is_failed());
        assert_eq!(fail_solver.iteration(), 1);
        assert_eq!(
            fail_solver.step(&mut fail_layout).unwrap(),
            SolverStatus::Failed
        );
        assert_eq!(fail_solver.iteration(), 1);
    }

    #[test]
    fn test_solver_generator_transform_and_blanket_dispatch() {
        struct LineGenerator {
            n: usize,
        }
        impl LayoutGenerator for LineGenerator {
            type Output = LayoutNd<f64>;
            fn generate(&mut self) -> CoreResult<Self::Output> {
                let mut coords = Vec::with_capacity(self.n * 2);
                for i in 0..self.n {
                    coords.push(i as f64);
                    coords.push(0.0);
                }
                LayoutNd::from_flat(self.n, 2, &coords)
            }
        }

        struct ShiftTransform {
            dx: f64,
        }
        impl LayoutTransform for ShiftTransform {
            type Target = Layout2d<f64>;
            fn apply(&mut self, target: &mut Self::Target) -> CoreResult<()> {
                target.validate_finite()?;
                for x in target.x_mut() {
                    *x += self.dx;
                }
                Ok(())
            }
        }

        // Blanket impls for `&mut T` and `Box<dyn Trait>`.
        let mut gen_impl = LineGenerator { n: 3 };
        let mut gen_ref: &mut dyn LayoutGenerator<Output = LayoutNd<f64>> = &mut gen_impl;
        let nd = (&mut gen_ref).generate().unwrap();
        assert_eq!(nd.coords(), &[0.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
        let mut gen_box: Box<dyn LayoutGenerator<Output = LayoutNd<f64>>> =
            Box::new(LineGenerator { n: 2 });
        assert_eq!(gen_box.generate().unwrap().num_nodes(), 2);

        let mut layout = Layout2d::from_xy(vec![1.0, 2.0], vec![0.0, 0.0]).unwrap();
        let mut tr_impl = ShiftTransform { dx: 3.0 };
        let mut tr_ref: &mut dyn LayoutTransform<Target = Layout2d<f64>> = &mut tr_impl;
        (&mut tr_ref).apply(&mut layout).unwrap();
        assert_eq!(layout.x(), &[4.0, 5.0]);
        let mut tr_box: Box<dyn LayoutTransform<Target = Layout2d<f64>>> =
            Box::new(ShiftTransform { dx: -1.5 });
        tr_box.apply(&mut layout).unwrap();
        assert_eq!(layout.x(), &[2.5, 3.5]);

        let mut raw_solver = DecaySolver::new(2, Some(2), 0.0);
        let mut solver_ref: &mut dyn IterativeSolver<Target = Layout2d<f64>> = &mut raw_solver;
        assert_eq!(
            (&mut solver_ref).run_steps(&mut layout, 1).unwrap(),
            SolverStatus::Running
        );
        let mut solver_box: Box<dyn IterativeSolver<Target = Layout2d<f64>>> =
            Box::new(DecaySolver::new(2, Some(1), 0.0));
        assert_eq!(
            solver_box.run(&mut layout).unwrap(),
            SolverStatus::Completed
        );
        assert_eq!(layout.x(), &[0.625, 0.875]);
    }
}
