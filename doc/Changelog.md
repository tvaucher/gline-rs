# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.9.3] - UNRELEASED

### Changed

- The generic pipeline framework has been externalized, see the [`composable`](https://github.com/fbilhaut/composable) and [`orp`](https://github.com/fbilhaut/orp) crates. The public API is left unchanged, beside minor import adaptations (see examples).


## [0.9.2] - 2025-01-26

### Fixed

- Fixed issue with **multi-word labels** (issue [#1](https://github.com/fbilhaut/gline-rs/issues/1)).

### Added

- Pipeline for **Relation Extraction**, with related example.
- Matrix-level documentation of pre- and post-processing steps (see `doc/Processing.typ` or `doc/Processing.pdf`).
- More unit-tests.

### Changed

- The `Pipeline` trait is now fully generic wrt. input, output and context types.
- The `Model` struct is more opaque and parametrized by a pipeline which it handles by itself.
- The `GLiNER` struct is now a light convenience wrapper around `Model`, `Pipeline` and `Parameters`.
- The `Composable` trait is now implemented for `Model`+`Pipeline`+`Parameters` combos, to facilitate re-use and combination of pipelines. See `Pipeline::to_composable()` or `Model::to_composable()`.
- Drop `num_traits` dependency (in favor of `ndarray`'s `NdFloat`).


## [0.9.1] - 2025-01-13

### Added

- GPU support: public API (`RuntimeParameters`), new example (`benchmark-gpu`), cargo features (mirroring `ort` backends), and documentation.

### Changed

- Minor source code linting and documentation refactoring.


## [0.9.0] - 2025-01-07

First public release.