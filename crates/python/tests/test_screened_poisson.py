from __future__ import annotations

import math
import unittest

import egraph as eg
import numpy as np


class TestScreenedPoisson(unittest.TestCase):
    def test_finite_difference_gradient_and_outward_sign(self) -> None:
        rng = np.random.default_rng(77)
        xs = []
        ys = []
        h = 12.0 / 128.0
        for i in range(10):
            ang = 0.31 + i * (2.0 * math.pi) / 10.0
            x_v = 0.17 + 0.25 * math.cos(ang) + 0.05 * (float(rng.random()) - 0.5)
            y_v = -0.13 + 0.25 * math.sin(ang) + 0.05 * (float(rng.random()) - 0.5)
            # Keep away from exact integer grid lines where bilinear splatting is only C^0.
            if min(((x_v + 6.0) / h) % 1.0, 1.0 - (((x_v + 6.0) / h) % 1.0)) < 0.1:
                x_v += 0.25 * h
            if min(((y_v + 6.0) / h) % 1.0, 1.0 - (((y_v + 6.0) / h) % 1.0)) < 0.1:
                y_v += 0.25 * h
            xs.append(x_v)
            ys.append(y_v)
        for i in range(4):
            ang = 0.43 + i * (2.0 * math.pi) / 4.0
            xs.append(0.17 + 0.70 * math.cos(ang))
            ys.append(-0.13 + 0.70 * math.sin(ang))
        xs.extend([3.23, -3.13])
        ys.extend([-2.91, 2.83])
        coords_base = np.column_stack([xs, ys])

        for dtype, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            for exact_adjoint, rho_max in ((True, 0.05), (False, 0.0)):
                field = eg.ScreenedPoissonField2d(
                    kernel="screened_poisson",
                    grid_size=128,
                    padding_factor=2,
                    sigma=0.35,
                    kappa=0.5,
                    rho_max=rho_max,
                    exact_adjoint=exact_adjoint,
                    dtype=dtype,
                )
                l_base = eg.Layout2d.from_coords(coords_base.astype(np_dtype))
                out = field.evaluate_in_bounds(l_base, -5.0, -5.0, 5.0, 5.0)
                self.assertFalse(out["early_exit_no_excess"])
                self.assertEqual(out["fft_2d_calls"], 6 if exact_adjoint else 4)

                eps = 2e-3
                max_err = 0.0
                max_grad = 0.0
                for i in range(len(xs)):
                    for col, anal_f in ((0, float(out["force_x"][i])), (1, float(out["force_y"][i]))):
                        cp = coords_base.copy()
                        cm = coords_base.copy()
                        cp[i, col] += eps
                        cm[i, col] -= eps
                        lp = field.evaluate_in_bounds(
                            eg.Layout2d.from_coords(cp.astype(np_dtype)), -5.0, -5.0, 5.0, 5.0
                        )["energy"]
                        lm = field.evaluate_in_bounds(
                            eg.Layout2d.from_coords(cm.astype(np_dtype)), -5.0, -5.0, 5.0, 5.0
                        )["energy"]
                        fd = (lp - lm) / (2.0 * eps)
                        max_grad = max(max_grad, abs(-anal_f))
                        max_err = max(max_err, abs((-anal_f) - fd))

                self.assertGreater(max_grad, 1e-4)
                self.assertLess(max_err / max_grad, 1e-2)
                self.assertLess(max_err, 2e-4)

                # Core must expand outward from core center.
                cx = float(np.mean(coords_base[:10, 0]))
                cy = float(np.mean(coords_base[:10, 1]))
                radial_dot = sum(
                    (float(coords_base[i, 0]) - cx) * float(out["force_x"][i])
                    + (float(coords_base[i, 1]) - cy) * float(out["force_y"][i])
                    for i in range(10)
                )
                self.assertGreater(radial_dot, 0.0)

    def test_unilateral_whitespace_invariance_and_plateau(self) -> None:
        # Early exit when rho_V <= rho_max everywhere.
        gx, gy = np.meshgrid(np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64))
        grid_coords = np.column_stack([gx.ravel(), gy.ravel()])
        field_ee = eg.ScreenedPoissonField2d(grid_size=128, rho_max=10.0)
        res_ee = field_ee.evaluate(eg.Layout2d.from_coords(grid_coords))
        self.assertTrue(res_ee["early_exit_no_excess"])
        self.assertEqual(res_ee["energy"], 0.0)
        self.assertEqual(res_ee["fft_2d_calls"], 2)
        np.testing.assert_array_equal(res_ee["force_x"], np.zeros(64))
        np.testing.assert_array_equal(res_ee["force_y"], np.zeros(64))

        # Plateau interior radial force comparison: ScreenedPoisson vs SingleGaussian.
        side = 36
        lin = np.linspace(-1.0, 1.0, side)
        px, py = np.meshgrid(lin, lin)
        mask = (px * px + py * py) <= 1.0
        plateau = np.column_stack([px[mask], py[mask]])
        layout = eg.Layout2d.from_coords(plateau)

        sp_field = eg.ScreenedPoissonField2d(
            kernel="screened_poisson", grid_size=256, sigma=0.08, kappa=0.8, rho_max=0.08
        )
        sg_field = eg.ScreenedPoissonField2d(
            kernel="single_gaussian", grid_size=256, sigma=0.08, kappa=0.8, rho_max=0.08
        )
        sp_out = sp_field.evaluate_in_bounds(layout, -3.0, -3.0, 3.0, 3.0)
        sg_out = sg_field.evaluate_in_bounds(layout, -3.0, -3.0, 3.0, 3.0)

        r = np.hypot(plateau[:, 0], plateau[:, 1])
        interior = (r > 0.1) & (r < 0.55)
        sp_rad = float(
            np.mean(
                (sp_out["force_x"][interior] * plateau[interior, 0]
                 + sp_out["force_y"][interior] * plateau[interior, 1])
                / r[interior]
            )
        )
        sg_rad = float(
            np.mean(
                (sg_out["force_x"][interior] * plateau[interior, 0]
                 + sg_out["force_y"][interior] * plateau[interior, 1])
                / r[interior]
            )
        )
        self.assertGreater(sp_rad, 1e-6)
        self.assertLess(abs(sg_rad), 0.05 * sp_rad)

    def test_refiner_step_run_graph_tether_and_memory(self) -> None:
        side = 20
        lin = np.linspace(-0.8, 0.8, side)
        px, py = np.meshgrid(lin, lin)
        mask = (px * px + py * py) <= 0.64
        coords = np.column_stack([px[mask], py[mask]])
        n = coords.shape[0]

        for dtype, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            l_run = eg.Layout2d.from_coords(coords.astype(np_dtype))
            l_step = eg.Layout2d.from_coords(coords.astype(np_dtype))
            kwargs = {
                "grid_size": 128,
                "padding_factor": 2,
                "sigma": 0.10,
                "kappa": 1.0,
                "rho_max": 0.15,
                "iterations": 6,
                "step_size": 0.3,
                "momentum": 0.4,
                "seed": 42,
                "dtype": dtype,
            }
            r_run = eg.ScreenedPoissonRefiner(n, **kwargs)  # type: ignore[arg-type]
            r_step = eg.ScreenedPoissonRefiner(n, **kwargs)  # type: ignore[arg-type]

            r_run.run(l_run)
            r_step.run_steps(l_step, 2)
            self.assertEqual(r_step.iteration, 2)
            r_step.run(l_step)
            np.testing.assert_allclose(l_run.coords(copy=True), l_step.coords(copy=True), atol=1e-6)
            self.assertGreater(float(np.ptp(l_run.coords(copy=True)[:, 0])), 1.8)

        # Graph anchors + coarse centroid tether + apply_epoch_step.
        g = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)], directed=False, sanitize=True)
        l4 = eg.Layout2d.from_coords(np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]))
        coarse = eg.Layout2d.from_coords(np.array([[5.0, 2.0], [-5.0, -2.0]]))
        ref_g = eg.ScreenedPoissonRefiner.from_graph(
            g,
            grid_size=64,
            rho_max=0.1,
            anchor_weight=0.2,
            gamma_tether=1.0,
            recenter=False,
        )
        ref_g.set_coarse_tether(coarse, [0, 0, 1, 1])
        st = ref_g.apply_epoch_step(l4)
        self.assertFalse(st["early_exit_no_excess"])
        c4 = l4.coords(copy=True)
        np.testing.assert_allclose(c4[:2].mean(axis=0), [5.0, 2.0], atol=1e-9)
        np.testing.assert_allclose(c4[2:].mean(axis=0), [-5.0, -2.0], atol=1e-9)

        # Default 512 grid workspace memory check and asymmetric bounds density grid test.
        default_field = eg.ScreenedPoissonField2d()
        self.assertEqual(default_field.grid_size, 512)
        self.assertEqual(default_field.density_grid_dim, 0)
        self.assertLessEqual(default_field.workspace_memory_bytes(), 10_000_000)

        asym_field = eg.ScreenedPoissonField2d(
            grid_size=64, padding_factor=2, sigma=0.2, kappa=0.5, rho_max=0.0
        )
        asym_coords = np.array([[2.0, -1.0], [2.2, -1.1], [2.1, -0.9], [1.9, -1.2]], dtype=np.float64)
        asym_lay = eg.Layout2d.from_coords(asym_coords)
        asym_out = asym_field.evaluate_in_bounds(asym_lay, -2.0, -5.0, 6.0, 3.0)
        self.assertGreater(asym_out["energy"], 0.0)
        self.assertEqual(asym_field.density_grid_dim, 64)
        dgrid = asym_field.density_grid()
        self.assertEqual(dgrid.shape, (64, 64))
        self.assertEqual(dgrid.dtype, np.float64)
        self.assertGreater(float(np.sum(dgrid)), 0.0)

    def test_adversarial_errors(self) -> None:
        with self.assertRaises(ValueError):
            eg.ScreenedPoissonField2d(kernel="unknown")
        with self.assertRaises(ValueError):
            eg.ScreenedPoissonField2d(grid_size=100)
        with self.assertRaises(ValueError):
            eg.ScreenedPoissonField2d(padding_factor=3)
        with self.assertRaises(ValueError):
            eg.ScreenedPoissonField2d(rho_max=-1.0)

        f64 = eg.ScreenedPoissonField2d(grid_size=64, dtype="float64")
        l32 = eg.Layout2d.zeros(4, dtype="float32")
        with self.assertRaises(ValueError):
            f64.evaluate(l32)
        with self.assertRaises(ValueError):
            f64.evaluate_in_bounds(eg.Layout2d.zeros(4), 1.0, 0.0, 0.0, 2.0)


if __name__ == "__main__":
    unittest.main()
