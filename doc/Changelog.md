# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.2] - Unreleased

### Fixed

- Support for multi-word labels (issue [#1](https://github.com/fbilhaut/gline-rs/issues/1))

### Added

- Matrix-level documentation of pre- and post-processing steps (see `doc/Processing.typ` or `doc/Processing.pdf`)
- More unit tests

### Changed

- Make the `Pipeline` trait fully generic wrt. input and output types


## [0.9.1] - 2025-01-13

### Added

- GPU support: public API (`RuntimeParameters`), new example (`benchmark-gpu`), cargo features (mirroring `ort` backends), and documentation

### Changed

- Minor source code linting and documentation refactoring


## [0.9.0] - 2025-01-07

First public release