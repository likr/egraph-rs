import "bulma/css/bulma.css";
import "./styles.css";
import egraph from "egraph/dist/web/egraph_wasm";
import egraphBinary from "egraph/dist/web/egraph_wasm_bg.wasm?url";
import egRenderer from "eg-renderer/umd/eg-renderer.js";
import egRendererBinary from "eg-renderer/umd/eg-renderer.wasm?url";

await egRenderer(egRendererBinary);
await egraph(egraphBinary);
import("./App");
