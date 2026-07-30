# delta_struct [![Crates listing](https://img.shields.io/crates/v/delta-struct.svg)](https://crates.io/crates/delta-struct) [![CI](https://github.com/Xaeroxe/delta-struct-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Xaeroxe/delta-struct-rs/actions/workflows/ci.yml)

Delta struct provides a rust-lang `Derive`able trait, `Delta`, that can be used to compute the difference (aka delta) between two instances of a type.

This can be combined with `serde` to only transmit changes to structures, when updates are necessary.

## Installation

```toml
[dependencies]
delta-struct = "0.2"
```

## Usage

Deriving `Delta` on a struct generates a companion struct holding only what changed, along with the implementation that produces and applies one.

```rust
use delta_struct::Delta;

#[derive(Delta)]
struct Config {
    host: String,
    port: u16,
}

let old = Config { host: "localhost".to_string(), port: 80 };
let new = Config { host: "localhost".to_string(), port: 8080 };

// `Config` gained a companion struct named `ConfigDelta`.
let delta = Delta::delta(old, new).expect("the port changed");
assert_eq!(delta.host, None);          // unchanged fields are `None`
assert_eq!(delta.port, Some(8080));

// Applying the delta to an older copy brings it up to date.
let mut current = Config { host: "localhost".to_string(), port: 80 };
current.apply_delta(delta);
assert_eq!(current.port, 8080);
```

`Delta::delta` returns `None` when nothing changed, which is what makes it worth reaching for: an update that would do nothing never has to be sent.

### Field types

Each field is diffed according to a *field type*, set with `#[delta_struct(field_type = "...")]`:

| Value | Delta representation | Notes |
| --- | --- | --- |
| `"scalar"` (default) | `Option<T>` | `Some(new)` when the values differ. |
| `"unordered"` | `{field}_add` and `{field}_remove`, both `Vec<Item>` | For collections whose order carries no meaning. |
| `"ordered"` | `SeqDelta<Item>`, a Myers edit script | For sequences where position matters. Items need `Hash + Eq`. |
| `"delta"` | `Option<<T as Delta>::Output>` | Diffs the field recursively; the field's type must derive `Delta` too. |

`#[delta_struct(default = "...")]` on the struct changes the default for its fields.

```rust
use delta_struct::Delta;

#[derive(Delta)]
struct Device {
    #[delta_struct(field_type = "unordered")]
    services: Vec<String>,
    online: bool,
}

let old = Device { services: vec!["ssh".to_string()], online: false };
let new = Device { services: vec!["mqtt".to_string()], online: true };

let delta = Delta::delta(old, new).unwrap();
assert_eq!(delta.services_add, vec!["mqtt".to_string()]);
assert_eq!(delta.services_remove, vec!["ssh".to_string()]);
assert_eq!(delta.online, Some(true));
```

### Decorating the generated struct

The generated struct derives nothing by default. `delta_leader` emits arbitrary tokens above it — or above an individual field — which is how derives, doc comments, and serde attributes get onto a type you never write by hand.

```rust
use delta_struct::Delta;

#[derive(Delta)]
#[delta_struct(delta_leader = "#[derive(serde::Serialize, serde::Deserialize)]")]
struct Config {
    host: String,
    port: u16,
}

let old = Config { host: "localhost".to_string(), port: 80 };
let new = Config { host: "localhost".to_string(), port: 8080 };

// Sender: there is no message to send at all when nothing changed.
let payload = Delta::delta(old, new).map(|delta| serde_json::to_string(&delta).unwrap());
assert_eq!(payload.as_deref(), Some(r#"{"host":null,"port":8080}"#));

// Receiver applies it to whatever it already had.
let mut config = Config { host: "localhost".to_string(), port: 80 };
config.apply_delta(serde_json::from_str::<ConfigDelta>(&payload.unwrap()).unwrap());
assert_eq!(config.port, 8080);
```

`#[serde(skip_serializing_if = "Option::is_none")]` through a field-level `delta_leader` keeps unchanged fields out of the payload entirely, rather than sending them as `null`.

### Ordered sequences

`ordered` diffs a field positionally with Myers' algorithm (via [`similar`](https://crates.io/crates/similar)), producing a minimal edit script instead of a membership delta:

```rust
use delta_struct::{Delta, Splice};

#[derive(Delta)]
struct Playlist {
    #[delta_struct(field_type = "ordered")]
    tracks: Vec<String>,
}

let old = Playlist { tracks: vec!["intro".to_string(), "b".to_string(), "outro".to_string()] };
let new = Playlist { tracks: vec!["intro".to_string(), "x".to_string(), "outro".to_string()] };

let delta = Delta::delta(old, new).unwrap();
assert_eq!(
    delta.tracks.splices,
    vec![Splice { at: 1, remove: 1, insert: vec!["x".to_string()] }],
);
```

Splice positions index the old sequence and arrive sorted and non-overlapping, so applying one is a single forward pass and reproduces the new sequence exactly. Items need `Hash + Eq` rather than `PartialEq`, which is what Myers requires — so a `Vec<f64>` can be `unordered` but not `ordered`. Enable the `serde` feature to serialize a delta containing an `ordered` field.

### Limitations

- Structs only — enums and unions are rejected.
- Applying an `unordered` delta preserves membership, not position; use `ordered` when position matters.
- `ordered` items must be `Hash + Eq`.

Full documentation, including trait bounds and the exact shape of the generated code, is on [docs.rs](https://docs.rs/delta-struct).

## License

Licensed under either of

- Apache License, Version 2.0
- MIT license

at your option.
