const assert = require("assert");
const eg = require("wasm-bindgen-test");
const helpers = require("./util/test_helpers");

/**
 * Test basic instantiation of SparseSgd class
 */
exports.testSparseSgdConstructor = function () {
  // Create a SparseSgd instance with a simple length function and 1 pivot node
  const sgd = new eg.SparseSgd();

  // Verify that the SGD instance exists
  assert(sgd instanceof eg.SparseSgd, "Should create an instance of SparseSgd");
};
