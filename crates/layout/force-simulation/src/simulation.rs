pub struct Simulation {
    pub alpha: f32,
    pub alpha_min: f32,
    pub alpha_target: f32,
    pub iterations: usize,
    count: usize,
}

impl Simulation {
    pub fn new() -> Simulation {
        Simulation {
            alpha: 1.,
            alpha_min: 0.001,
            alpha_target: 0.,
            iterations: 300,
            count: 0,
        }
    }

    pub fn run<F: FnMut(f32)>(&mut self, f: &mut F) {
        while !self.is_finished() {
            self.step(f);
        }
    }

    pub fn run_step<F: FnMut(f32)>(&mut self, n: usize, f: &mut F) {
        for _ in 0..n {
            self.step(f);
        }
    }

    pub fn step<F: FnMut(f32)>(&mut self, f: &mut F) {
        let alpha_decay = 1. - self.alpha_min.powf(1. / self.iterations as f32);
        self.alpha += (self.alpha_target - self.alpha) * alpha_decay;
        f(self.alpha);
        self.count += 1;
    }

    pub fn is_finished(&self) -> bool {
        self.count >= self.iterations
    }

    pub fn reset(&mut self, alpha_start: f32) {
        self.alpha = alpha_start;
        self.count = 0;
    }
}

#[test]
fn test_simulation() {
    let iter = 300;
    let mut count = 0;
    let mut simulation = Simulation::new();
    simulation.iterations = iter;
    simulation.run(&mut |_| {
        count += 1;
    });
    assert_eq!(iter, count);
}
