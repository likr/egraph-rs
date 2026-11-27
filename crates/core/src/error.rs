//! Structured error types shared across `egraph-core` and downstream crates.

use std::fmt;

/// Convenience result alias for `egraph-core` operations.
pub type CoreResult<T> = Result<T, CoreError>;

/// Structured error variants returned by fallible `egraph-core` operations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// An input argument or configuration violated a domain constraint.
    InvalidInput(String),
    /// An array length, shape, or stride did not match expectations.
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: &'static str,
    },
    /// A vertex, edge, or coordinate index exceeded valid bounds.
    IndexOutOfBounds {
        index: usize,
        len: usize,
        context: &'static str,
    },
    /// A compressed sparse row (CSR) or edge-list representation was malformed.
    InvalidCsr(String),
    /// A floating-point value was `NaN`, `+inf`, or `-inf`.
    NonFiniteValue { context: &'static str },
    /// An operation is unsupported for the current graph or manifold configuration.
    UnsupportedOperation(String),
    /// An iterative solver encountered a numerical breakdown.
    NumericalBreakdown(String),
}

impl CoreError {
    /// Constructs an [`CoreError::InvalidInput`] error.
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Constructs a [`CoreError::DimensionMismatch`] error.
    pub fn dimension_mismatch(expected: usize, actual: usize, context: &'static str) -> Self {
        Self::DimensionMismatch {
            expected,
            actual,
            context,
        }
    }

    /// Constructs an [`CoreError::IndexOutOfBounds`] error.
    pub fn index_out_of_bounds(index: usize, len: usize, context: &'static str) -> Self {
        Self::IndexOutOfBounds {
            index,
            len,
            context,
        }
    }

    /// Constructs a [`CoreError::InvalidCsr`] error.
    pub fn invalid_csr(message: impl Into<String>) -> Self {
        Self::InvalidCsr(message.into())
    }

    /// Constructs a [`CoreError::NonFiniteValue`] error.
    pub fn non_finite(context: &'static str) -> Self {
        Self::NonFiniteValue { context }
    }

    /// Constructs a [`CoreError::UnsupportedOperation`] error.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::UnsupportedOperation(message.into())
    }

    /// Constructs a [`CoreError::NumericalBreakdown`] error.
    pub fn numerical_breakdown(message: impl Into<String>) -> Self {
        Self::NumericalBreakdown(message.into())
    }
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::DimensionMismatch {
                expected,
                actual,
                context,
            } => write!(
                f,
                "dimension mismatch in {context}: expected {expected}, got {actual}"
            ),
            Self::IndexOutOfBounds {
                index,
                len,
                context,
            } => write!(
                f,
                "index out of bounds in {context}: index {index} >= len {len}"
            ),
            Self::InvalidCsr(msg) => write!(f, "invalid CSR graph: {msg}"),
            Self::NonFiniteValue { context } => {
                write!(f, "non-finite floating-point value in {context}")
            }
            Self::UnsupportedOperation(msg) => write!(f, "unsupported operation: {msg}"),
            Self::NumericalBreakdown(msg) => write!(f, "numerical breakdown: {msg}"),
        }
    }
}

impl std::error::Error for CoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_and_std_error() {
        let cases: Vec<CoreError> = vec![
            CoreError::invalid_input("bad parameter"),
            CoreError::dimension_mismatch(4, 3, "Layout2d::from_xy"),
            CoreError::index_out_of_bounds(5, 4, "CsrGraph::neighbors"),
            CoreError::invalid_csr("indptr must start at 0"),
            CoreError::non_finite("Layout2d::center_and_normalize_std"),
            CoreError::unsupported("directed graph unsupported"),
            CoreError::numerical_breakdown("zero pivot norm"),
        ];

        for err in &cases {
            let msg = err.to_string();
            assert!(!msg.is_empty());
            let dyn_err: &dyn std::error::Error = err;
            assert!(dyn_err.source().is_none());
        }
    }
}
