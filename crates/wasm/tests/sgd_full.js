const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of FullSgd class
 */
exports.testFullSgdConstructor = function () {
  // Create a FullSgd instance
  const sgd = new eg.FullSgd();

  // Verify that the FullSGD instance exists
  assert(sgd instanceof eg.FullSgd, "Should create an instance of FullSgd");
};
