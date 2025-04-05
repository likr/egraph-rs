//! Random number generation for WebAssembly.
//!
//! This module provides a WebAssembly binding for random number generation
//! based on the Rust `rand` crate.

use rand::prelude::*;
use wasm_bindgen::prelude::*;

/// WebAssembly binding for random number generation.
///
/// This struct provides a JavaScript interface to Rust's random number generation,
/// exposing the capabilities of StdRng through WebAssembly.
#[wasm_bindgen(js_name = Rng)]
pub struct JsRng {
    rng: StdRng,
}

impl JsRng {
    /// Returns a mutable reference to the internal RNG.
    ///
    /// This method is intended for internal use by other Rust modules that need
    /// access to the random number generator.
    pub fn get_mut(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

impl Default for JsRng {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = Rng)]
impl JsRng {
    /// Creates a new random number generator using system entropy.
    ///
    /// This constructor creates a cryptographically secure random number generator
    /// that is suitable for most applications.
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsRng {
        JsRng {
            rng: StdRng::from_entropy(),
        }
    }

    /// Creates a new random number generator with a specific seed.
    ///
    /// This method allows for reproducible random number sequences by
    /// providing a seed value.
    ///
    /// @param {number} seed - A 64-bit unsigned integer to use as the seed
    /// @returns {Rng} A new seeded random number generator
    #[wasm_bindgen(js_name = "seedFrom")]
    pub fn seed_from(seed: u64) -> JsRng {
        JsRng {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}
