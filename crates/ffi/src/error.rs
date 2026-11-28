//! Target-agnostic FFI error classification mapped from [`egraph_core::CoreError`].

use std::fmt;

use egraph_core::CoreError;

/// Coarse error category suitable for mapping into Python exceptions or JS `Error`s.
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ErrorKind {
    InvalidInput = 1,
    DimensionMismatch = 2,
    IndexOutOfBounds = 3,
    InvalidCsr = 4,
    NonFiniteValue = 5,
    UnsupportedOperation = 6,
    NumericalBreakdown = 7,
    BackendUnavailable = 8,
    DeviceLimitExceeded = 9,
}

impl ErrorKind {
    /// Maps a reference to [`CoreError`] into its stable [`ErrorKind`] discriminant.
    #[inline]
    pub fn from_core_error(err: &CoreError) -> Self {
        match err {
            CoreError::InvalidInput(_) => Self::InvalidInput,
            CoreError::DimensionMismatch { .. } => Self::DimensionMismatch,
            CoreError::IndexOutOfBounds { .. } => Self::IndexOutOfBounds,
            CoreError::InvalidCsr(_) => Self::InvalidCsr,
            CoreError::NonFiniteValue { .. } => Self::NonFiniteValue,
            CoreError::UnsupportedOperation(_) => Self::UnsupportedOperation,
            CoreError::NumericalBreakdown(_) => Self::NumericalBreakdown,
            CoreError::BackendUnavailable(_) => Self::BackendUnavailable,
            CoreError::DeviceLimitExceeded(_) => Self::DeviceLimitExceeded,
            _ => Self::InvalidInput,
        }
    }

    /// Returns the stable PascalCase error code string matching this [`ErrorKind`] variant.
    #[inline]
    pub fn code_str(self) -> &'static str {
        match self {
            Self::InvalidInput => "InvalidInput",
            Self::DimensionMismatch => "DimensionMismatch",
            Self::IndexOutOfBounds => "IndexOutOfBounds",
            Self::InvalidCsr => "InvalidCsr",
            Self::NonFiniteValue => "NonFiniteValue",
            Self::UnsupportedOperation => "UnsupportedOperation",
            Self::NumericalBreakdown => "NumericalBreakdown",
            Self::BackendUnavailable => "BackendUnavailable",
            Self::DeviceLimitExceeded => "DeviceLimitExceeded",
        }
    }
}

impl From<&CoreError> for ErrorKind {
    #[inline]
    fn from(err: &CoreError) -> Self {
        Self::from_core_error(err)
    }
}

/// Structured FFI error carrying an [`ErrorKind`] and human-readable diagnostic message.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfiError {
    pub kind: ErrorKind,
    pub message: String,
}

impl FfiError {
    /// Creates an [`FfiError`] with an explicit kind and message.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl From<CoreError> for FfiError {
    fn from(err: CoreError) -> Self {
        Self {
            kind: ErrorKind::from(&err),
            message: err.to_string(),
        }
    }
}

impl From<&CoreError> for FfiError {
    fn from(err: &CoreError) -> Self {
        Self {
            kind: ErrorKind::from(err),
            message: err.to_string(),
        }
    }
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FfiError {}

/// Convenience result alias for `egraph-ffi` operations.
pub type FfiResult<T> = Result<T, FfiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kind_mapping_covers_all_core_errors() {
        let cases = [
            (
                CoreError::invalid_input("x"),
                ErrorKind::InvalidInput,
                "InvalidInput",
            ),
            (
                CoreError::dimension_mismatch(2, 3, "test"),
                ErrorKind::DimensionMismatch,
                "DimensionMismatch",
            ),
            (
                CoreError::index_out_of_bounds(5, 4, "test"),
                ErrorKind::IndexOutOfBounds,
                "IndexOutOfBounds",
            ),
            (
                CoreError::invalid_csr("bad csr"),
                ErrorKind::InvalidCsr,
                "InvalidCsr",
            ),
            (
                CoreError::non_finite("test"),
                ErrorKind::NonFiniteValue,
                "NonFiniteValue",
            ),
            (
                CoreError::unsupported("directed"),
                ErrorKind::UnsupportedOperation,
                "UnsupportedOperation",
            ),
            (
                CoreError::numerical_breakdown("singular"),
                ErrorKind::NumericalBreakdown,
                "NumericalBreakdown",
            ),
            (
                CoreError::backend_unavailable("no GPU adapter"),
                ErrorKind::BackendUnavailable,
                "BackendUnavailable",
            ),
            (
                CoreError::device_limit_exceeded("buffer too large"),
                ErrorKind::DeviceLimitExceeded,
                "DeviceLimitExceeded",
            ),
        ];

        for (core_err, expected_kind, expected_code) in cases {
            assert_eq!(ErrorKind::from(&core_err), expected_kind);
            assert_eq!(expected_kind.code_str(), expected_code);
            let ffi_err = FfiError::from(core_err);
            assert_eq!(ffi_err.kind, expected_kind);
            assert_eq!(ffi_err.kind.code_str(), expected_code);
            assert!(!ffi_err.to_string().is_empty());
        }

        let manual = FfiError::new(ErrorKind::InvalidInput, "manual");
        assert_eq!(manual.kind.code_str(), "InvalidInput");
        assert_eq!(manual.to_string(), "manual");
    }
}
