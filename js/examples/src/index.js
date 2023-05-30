import "bulma/css/bulma.css";
import "./styles.css";
import init from "egraph/dist/web/egraph_wasm";
import wasm from "egraph/dist/web/egraph_wasm_bg.wasm?url";
import "eg-renderer/umd/eg-renderer.js";
import wasmUrl from "eg-renderer/umd/eg-renderer.wasm?url";
egRenderer(wasmUrl);

console.log(wasm);
await customElements.whenDefined("eg-renderer");
await init(wasm);
import("./App");
