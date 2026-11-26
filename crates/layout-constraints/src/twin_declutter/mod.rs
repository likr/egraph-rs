//! Coincident and near-twin node decluttering via spatial hashing and golden-angle Vogel spirals.

mod config;
mod solver;
#[cfg(test)]
mod tests;

pub use config::TwinDeclutterConfig;
pub use solver::{TwinDeclutterer, declutter_twins_2d};
