# delta_struct [![Crates listing](https://img.shields.io/crates/v/delta-struct.svg)](https://crates.io/crates/delta-struct)

Delta struct provides a rust-lang `Derive`able trait, `Delta`, that can be used to compute the difference (aka delta) between two instances of a type.

This can be combined with `serde` to only transmit changes to structures, when updates are necessary.
