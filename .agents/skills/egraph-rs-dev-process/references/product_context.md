# Product Context & Brief: egraph-rs

This document consolidates the project overview, requirements, and product goals for `egraph-rs` to provide context for developers working on the repository.

## Project Overview

`egraph-rs` is a high-performance graph manipulation and visualization library written in Rust. It offers a comprehensive toolkit for graph manipulation, drawing in various geometric spaces, clustering, and layout quality metrics.

## Why This Project Exists

- **Performance Gap**: Provides native, highly efficient Rust implementations of complex graph algorithms that outperform traditional interpreted or non-optimized libraries.
- **Language Barrier**: Bridges multiple programming environments (Rust, Python, JavaScript) by generating native bindings with consistent API interfaces.
- **Visualization Variety**: Offers layouts tailored to distinct geometric spaces (Euclidean, Spherical, Hyperbolic, Torus) for specialized research and application layouts.
- **Quality Assessment**: Incorporates metrics to quantitatively evaluate the aesthetic and layout qualities of drawings (e.g. edge crossings, stress minimization).

## Core Requirements & Scope

- **Data Structures**: Provide robust graph representations.
- **Algorithms**: Support layout techniques, layering, community detection, edge-bundling, and triangulation.
- **Geometry**: Draw in Euclidean (2D/nD), Spherical, Hyperbolic, and Torus spaces.
- **Language Integration**: Maintain PyO3 bindings for Python and wasm-bindgen bindings for WebAssembly/JavaScript.
- **Modularity**: Implement a modular crate-based workspace design (15+ specialized crates).

## User Experience Goals

- **Consistent API**: Ensure APIs are structurally parallel across Rust, Python, and JavaScript.
- **Performance**: Minimize memory allocations and CPU cycles, crucial for Large Graph drawings and WebAssembly context.
- **Extensibility**: Design traits to allow developers to compose or inject custom layout/layering/clustering algorithms.
- **Examples & Docs**: Provide thorough JSDoc/Docstring coverage and complete runnable example suites.
