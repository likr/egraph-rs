use ndarray::prelude::*;

fn cos(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let ab = a.dot(b);
    let aa = a.dot(a);
    let bb = b.dot(b);
    ab / (aa * bb).sqrt()
}

fn power_iteration(a: &Array2<f32>, eps: f32) -> (f32, Array1<f32>) {
    let n = a.shape()[0];
    let mut x = Array1::from_elem(n, 1. / n as f32);
    let mut x_next = a.dot(&x);
    for _ in 0..10 {
        if 1. - cos(&x_next, &x) < eps {
            break;
        }
        x_next /= x_next.dot(&x_next).sqrt();
        x = x_next;
        x_next = a.dot(&x);
    }
    let e = x_next.dot(&x_next) / x_next.dot(&x);
    x_next /= x_next.dot(&x_next).sqrt();
    (e, x_next)
}

pub fn eigendecomposition(a: &Array2<f32>, k: usize, eps: f32) -> (Array1<f32>, Array2<f32>) {
    let n = a.shape()[0];
    let mut b = a.clone();
    let mut e = Array1::zeros(k);
    let mut v = Array2::zeros((n, k));
    let (ei, vi) = power_iteration(&b, eps);
    e[0] = ei;
    v.slice_mut(s![.., 0]).assign(&vi);
    for i in 1..k {
        for r in 0..n {
            for c in 0..n {
                b[[r, c]] -= e[i - 1] * v[[r, i - 1]] * v[[c, i - 1]];
            }
        }
        let (ei, vi) = power_iteration(&b, eps);
        e[i] = ei;
        v.slice_mut(s![.., i]).assign(&vi);
    }
    (e, v)
}
