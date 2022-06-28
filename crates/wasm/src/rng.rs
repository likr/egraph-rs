use rand::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Rng)]
pub struct JsRng {
    rng: StdRng,
}

impl JsRng {
    pub fn get_mut(&mut self) -> &mut StdRng {
        &mut self.rng
    }
}

#[wasm_bindgen(js_class = Rng)]
impl JsRng {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsRng {
        JsRng {
            rng: StdRng::from_entropy(),
        }
    }

    #[wasm_bindgen(js_name = "seedFrom")]
    pub fn seed_from(seed: u64) -> JsRng {
        JsRng {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}
