import "bulma/css/bulma.css";
import "./styles.css";
import init from "egraph/dist/web/egraph_wasm";
import wasm from "egraph/dist/web/egraph_wasm_bg.wasm?url";

await init(wasm);
import("./App");
