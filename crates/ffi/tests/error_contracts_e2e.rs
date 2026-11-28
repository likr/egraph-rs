//! Opaque-box E2E tests for `egraph_ffi::ErrorKind` and `FfiError` parity with `CoreError`.

use std::collections::HashSet;

use egraph_core::CoreError;
use egraph_ffi::{ErrorKind, FfiError};

#[test]
fn ffi_error_kind_maps_all_core_variants_distinctly() {
    let cases = [
        (
            CoreError::invalid_input("bad arg"),
            ErrorKind::InvalidInput,
            1_u32,
            "InvalidInput",
        ),
        (
            CoreError::dimension_mismatch(4, 2, "ctx"),
            ErrorKind::DimensionMismatch,
            2_u32,
            "DimensionMismatch",
        ),
        (
            CoreError::index_out_of_bounds(5, 3, "ctx"),
            ErrorKind::IndexOutOfBounds,
            3_u32,
            "IndexOutOfBounds",
        ),
        (
            CoreError::invalid_csr("bad csr"),
            ErrorKind::InvalidCsr,
            4_u32,
            "InvalidCsr",
        ),
        (
            CoreError::non_finite("ctx"),
            ErrorKind::NonFiniteValue,
            5_u32,
            "NonFiniteValue",
        ),
        (
            CoreError::unsupported("unsupported"),
            ErrorKind::UnsupportedOperation,
            6_u32,
            "UnsupportedOperation",
        ),
        (
            CoreError::numerical_breakdown("singular"),
            ErrorKind::NumericalBreakdown,
            7_u32,
            "NumericalBreakdown",
        ),
        (
            CoreError::backend_unavailable("no external backend device"),
            ErrorKind::BackendUnavailable,
            8_u32,
            "BackendUnavailable",
        ),
        (
            CoreError::device_limit_exceeded("exceeds 16384 nodes"),
            ErrorKind::DeviceLimitExceeded,
            9_u32,
            "DeviceLimitExceeded",
        ),
    ];

    for (core_err, expected_kind, expected_disc, expected_code) in cases {
        let mapped = ErrorKind::from(&core_err);
        assert_eq!(mapped, expected_kind);
        assert_eq!(mapped as u32, expected_disc);
        assert_eq!(mapped.code_str(), expected_code);
    }
}

#[test]
fn ffi_error_kind_does_not_collapse_backend_errors_to_invalid_input() {
    let backend_kind = ErrorKind::from_core_error(&CoreError::backend_unavailable("missing"));
    let device_kind = ErrorKind::from_core_error(&CoreError::device_limit_exceeded("too large"));
    assert_ne!(backend_kind, ErrorKind::InvalidInput);
    assert_ne!(backend_kind, ErrorKind::UnsupportedOperation);
    assert_ne!(device_kind, ErrorKind::InvalidInput);
    assert_ne!(device_kind, ErrorKind::UnsupportedOperation);
    assert_ne!(backend_kind, device_kind);
}

#[test]
fn ffi_error_preserves_empty_and_unicode_diagnostics() {
    let empty_b = FfiError::from(CoreError::backend_unavailable(""));
    assert_eq!(empty_b.kind, ErrorKind::BackendUnavailable);
    assert!(!empty_b.message.is_empty());

    let empty_d = FfiError::from(CoreError::device_limit_exceeded(""));
    assert_eq!(empty_d.kind, ErrorKind::DeviceLimitExceeded);
    assert!(!empty_d.message.is_empty());

    let msg = "adapter unavailable: λ-backend-0";
    let uni_b = FfiError::from(CoreError::backend_unavailable(msg));
    assert_eq!(uni_b.kind, ErrorKind::BackendUnavailable);
    assert!(uni_b.message.contains(msg));

    let uni_d = FfiError::from(CoreError::device_limit_exceeded(msg));
    assert_eq!(uni_d.kind, ErrorKind::DeviceLimitExceeded);
    assert!(uni_d.message.contains(msg));
}

#[test]
fn ffi_error_roundtrips_display_and_std_error_trait() {
    let err = FfiError::new(ErrorKind::BackendUnavailable, "gpu adapter missing");
    assert_eq!(err.to_string(), "gpu adapter missing");
    let std_err: &dyn std::error::Error = &err;
    assert!(std_err.source().is_none());

    let dev = FfiError::new(ErrorKind::DeviceLimitExceeded, "max buffer size");
    assert_eq!(dev.to_string(), "max buffer size");
}

#[test]
fn ffi_error_kind_discriminants_are_unique_across_all_variants() {
    let all = [
        ErrorKind::InvalidInput,
        ErrorKind::DimensionMismatch,
        ErrorKind::IndexOutOfBounds,
        ErrorKind::InvalidCsr,
        ErrorKind::NonFiniteValue,
        ErrorKind::UnsupportedOperation,
        ErrorKind::NumericalBreakdown,
        ErrorKind::BackendUnavailable,
        ErrorKind::DeviceLimitExceeded,
    ];
    let discs: HashSet<u32> = all.iter().map(|&k| k as u32).collect();
    let codes: HashSet<&str> = all.iter().map(|k| k.code_str()).collect();
    assert_eq!(discs.len(), all.len());
    assert_eq!(codes.len(), all.len());
}

#[test]
fn ffi_error_from_borrowed_and_owned_core_error_match() {
    for core_err in [
        CoreError::backend_unavailable("no device"),
        CoreError::device_limit_exceeded("limit"),
    ] {
        let from_ref = FfiError::from(&core_err);
        let from_owned = FfiError::from(core_err);
        assert_eq!(from_ref, from_owned);
    }
}
