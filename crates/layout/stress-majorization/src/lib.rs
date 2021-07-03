fn line_search(a: &[f32], dx: &[f32], d: &[f32]) -> f32 {
  let n = dx.len();
  let mut alpha = -dot(d, &dx);
  let mut s = 0.;
  for i in 0..n {
    for j in 0..n {
      s += d[i] * d[j] * a[i * n + j];
    }
  }
  alpha /= s;
  alpha
}

fn delta_f(a: &[f32], b: &[f32], x: &[f32], dx: &mut [f32]) {
  let n = b.len();
  for i in 0..n {
    dx[i] = 0.;
    for j in 0..n {
      dx[i] += a[i * n + j] * x[j];
    }
    dx[i] -= b[i];
  }
}

fn dot(u: &[f32], v: &[f32]) -> f32 {
  let n = u.len();
  let mut s = 0.;
  for i in 0..n {
    s += u[i] * v[i];
  }
  s
}

pub fn conjugate_gradient(a: &[f32], b: &[f32], x: &mut [f32], epsilon: f32) {
  let n = b.len();
  let mut dx = vec![0.; n];
  let mut d = vec![0.; n];
  delta_f(a, b, &x, &mut dx);
  for i in 0..n {
    d[i] = -dx[i];
  }
  let mut dx_norm0 = dot(&dx, &dx);
  for _ in 0..n {
    let alpha = line_search(a, &dx, &d);
    for i in 0..n {
      x[i] += alpha * d[i];
    }
    delta_f(a, b, &x, &mut dx);
    let dx_norm = dot(&dx, &dx);
    if dx_norm < epsilon {
      break;
    }
    let beta = dx_norm / dx_norm0;
    dx_norm0 = dx_norm;
    for i in 0..n {
      d[i] = beta * d[i] - dx[i];
    }
  }
}

#[test]
fn test_conjugate_gradient() {
  let a = vec![3., 1., 1., 2.];
  let b = vec![6., 7.];
  let mut x = vec![2., 1.];
  let epsilon = 1e-4;
  conjugate_gradient(&a, &b, &mut x, epsilon);
  let x_exact = vec![1., 3.];
  let mut d = 0.;
  for i in 0..x.len() {
    let dx = x[i] - x_exact[i];
    d += dx * dx;
  }
  assert!(d < epsilon);
}

#[test]
fn test_kamada_kawai() {}
