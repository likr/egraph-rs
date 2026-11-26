//! Arena-based 1D separation constraint projection and 2D rectangle overlap removal.
//!
//! Replaces the legacy `HashMap`/`HashSet` and recursive `ConstraintGraph` with a flat
//! `Vec`-backed arena (`ConstraintProjector`), fixing the legacy `split_blocks` early `break`,
//! `blocks[s]` overwrite, infinite `project()` loop, and `2 + 1e-6` / early-break scanline bugs.

mod block_ops;
mod clustered;
mod overlap_2d;
mod projector;

pub use clustered::project_clustered_rectangle_no_overlap_2d;
pub use overlap_2d::project_rectangle_no_overlap_2d;
pub use projector::{ConstraintProjector, SeparationConstraint, generate_layered_constraints};
