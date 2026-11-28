//! Opaque-box E2E tests for `CoreError` backend/device variants and
//! `RepulsionBackend` error propagation and adapter contracts.

use egraph_core::{
    CoreError, CoreResult, Layout2d, Layout3d, LayoutNd, LayoutStorage, RepulsionBackend, Scalar,
};

struct ScalarTestRepulsion {
    z_value: f64,
}

impl<L: LayoutStorage + ?Sized> RepulsionBackend<L> for ScalarTestRepulsion {
    fn accumulate_repulsion(&mut self, layout: &L, forces: &mut [f64]) -> CoreResult<Option<f64>> {
        if layout.is_empty() {
            return Ok(Some(0.0));
        }
        let n = layout.len();
        let stride = forces.len() / n;
        for node in 0..n {
            for axis in 0..stride {
                forces[node * stride + axis] += 1.0;
            }
        }
        Ok(Some(self.z_value))
    }
}

enum FailMode {
    BackendUnavailable(&'static str),
    DeviceLimitExceeded(&'static str),
}

struct FailingRepulsion {
    mode: FailMode,
}

impl<L: LayoutStorage + ?Sized> RepulsionBackend<L> for FailingRepulsion {
    fn accumulate_repulsion(
        &mut self,
        _layout: &L,
        _forces: &mut [f64],
    ) -> CoreResult<Option<f64>> {
        match self.mode {
            FailMode::BackendUnavailable(msg) => Err(CoreError::backend_unavailable(msg)),
            FailMode::DeviceLimitExceeded(msg) => Err(CoreError::device_limit_exceeded(msg)),
        }
    }
}

#[test]
fn core_error_backend_and_device_variants_display_and_traits() {
    let e_backend = CoreError::backend_unavailable("no external backend adapter available");
    let e_device = CoreError::device_limit_exceeded("buffer limit 16384 exceeded");
    assert!(!e_backend.to_string().is_empty());
    assert!(!e_device.to_string().is_empty());
    assert_ne!(e_backend, e_device);
    assert_eq!(e_backend.clone(), e_backend);
    let dyn_err: &dyn std::error::Error = &e_backend;
    assert!(dyn_err.source().is_none());
    let dyn_dev: &dyn std::error::Error = &e_device;
    assert!(dyn_dev.source().is_none());
    assert!(matches!(e_backend, CoreError::BackendUnavailable(_)));
    assert!(matches!(e_device, CoreError::DeviceLimitExceeded(_)));
}

#[test]
fn core_error_backend_and_device_empty_and_unicode_messages() {
    let empty_b = CoreError::backend_unavailable("");
    let empty_d = CoreError::device_limit_exceeded("");
    assert!(!empty_b.to_string().is_empty());
    assert!(!empty_d.to_string().is_empty());

    let unicode_msg = "adapter missing: αβγ-gpu-0";
    let unicode_b = CoreError::backend_unavailable(unicode_msg);
    let unicode_d = CoreError::device_limit_exceeded(unicode_msg);
    assert!(unicode_b.to_string().contains(unicode_msg));
    assert!(unicode_d.to_string().contains(unicode_msg));
}

#[test]
fn repulsion_backend_reference_and_box_adapters_work() {
    let layout = Layout2d::<f64>::zeros(3);
    let mut backend = ScalarTestRepulsion { z_value: 4.5 };
    let mut forces = vec![0.0; 6];
    let ref_adapter: &mut ScalarTestRepulsion = &mut backend;
    let z1 = ref_adapter
        .accumulate_repulsion(&layout, &mut forces)
        .unwrap();
    assert_eq!(z1, Some(4.5));
    assert!(forces.iter().all(|&f| Scalar::to_f64(f) == 1.0));

    let mut boxed: Box<dyn RepulsionBackend<Layout2d<f64>>> =
        Box::new(ScalarTestRepulsion { z_value: 2.25 });
    let z2 = boxed.accumulate_repulsion(&layout, &mut forces).unwrap();
    assert_eq!(z2, Some(2.25));
    assert!(forces.iter().all(|&f| f == 2.0));
}

#[test]
fn repulsion_backend_zero_node_layout_returns_cleanly() {
    let empty = Layout2d::<f64>::zeros(0);
    let mut backend = ScalarTestRepulsion { z_value: 1.0 };
    let mut forces = Vec::new();
    let z = backend.accumulate_repulsion(&empty, &mut forces).unwrap();
    assert_eq!(z, Some(0.0));
}

#[test]
fn repulsion_backend_propagates_backend_and_device_errors_across_adapters() {
    let l2 = Layout2d::<f64>::zeros(3);
    let l3 = Layout3d::<f32>::zeros(2);
    let lnd = LayoutNd::<f64>::zeros(2, 4).unwrap();

    let mut f2 = vec![3.5_f64; 6];
    let f2_snap = f2.clone();
    let mut b_unavail = FailingRepulsion {
        mode: FailMode::BackendUnavailable("no gpu adapter"),
    };
    let ref_unavail: &mut FailingRepulsion = &mut b_unavail;
    let err_b = ref_unavail.accumulate_repulsion(&l2, &mut f2).unwrap_err();
    assert!(matches!(
        err_b,
        CoreError::BackendUnavailable(ref msg) if msg == "no gpu adapter"
    ));
    assert_eq!(f2, f2_snap);

    let mut f3 = vec![-1.25_f64; 6];
    let f3_snap = f3.clone();
    let mut box_dev: Box<dyn RepulsionBackend<Layout3d<f32>>> = Box::new(FailingRepulsion {
        mode: FailMode::DeviceLimitExceeded("storage binding exceeds 128MiB"),
    });
    let err_d = box_dev.accumulate_repulsion(&l3, &mut f3).unwrap_err();
    assert!(matches!(
        err_d,
        CoreError::DeviceLimitExceeded(ref msg) if msg == "storage binding exceeds 128MiB"
    ));
    assert_eq!(f3, f3_snap);

    let mut fnd = vec![0.0_f64; 8];
    let mut nd_unavail = FailingRepulsion {
        mode: FailMode::BackendUnavailable("nd backend offline"),
    };
    assert!(matches!(
        nd_unavail.accumulate_repulsion(&lnd, &mut fnd),
        Err(CoreError::BackendUnavailable(_))
    ));
}
