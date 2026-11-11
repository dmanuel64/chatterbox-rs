# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/dmanuel64/chatterbox-rs/compare/v0.1.1...v0.2.0) - 2026-11-11

### Fixed

- default sample rate to 24,000

### Other

- default to q4 and added cuda default
- pre-compute speech-token embeddings
- split voice preparation out of generate
- stop cache from crossing devices
- remove kv cache clone
- mention example and docstring formatting
- use environment for session builder
- *(deps)* bump the cargo-patch-minor group across 1 directory with 7 updates

## [0.1.1](https://github.com/dmanuel64/chatterbox-rs/compare/v0.1.0...v0.1.1) - 2026-09-04

### Other

- Merge pull request #1 from dmanuel64/dependabot/cargo/trait-set-0.5.0
- *(deps)* bump actions/checkout from 6 to 7
- added conventional commits and grouping
- added release-plz and updated docs for docs.rs
