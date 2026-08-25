# delta_struct [![Crates listing](https://img.shields.io/crates/v/delta-struct.svg)](https://crates.io/crates/delta-struct) [![CI](https://github.com/Xaeroxe/delta-struct-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Xaeroxe/delta-struct-rs/actions/workflows/ci.yml)

## Agentic development disclosure

Large parts of this crate are developed in collaboration with Claude. All output from Claude is human reviewed and tested.

## About

Delta struct provides a rust-lang `Derive`able trait, `Delta`, that can be used to compute the difference (aka delta) between two instances of a type.

This can be combined with `serde` to only transmit changes to structures, when updates are necessary.

## Installation

```toml
[dependencies]
delta-struct = "0.4"
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
current.apply_delta(delta).unwrap();
assert_eq!(current.port, 8080);
```

`Delta::delta` returns `None` when nothing changed, which is what makes it worth reaching for: an update that would do nothing never has to be sent.

### Field types

Each field is diffed according to a *field type*, set with `#[delta_struct(field_type = "...")]`:

| Value | Delta representation | Notes |
| --- | --- | --- |
| `"scalar"` (default) | `Option<T>` | `Some(new)` when the values differ. |
| `"unordered"` | `<T as Unordered>::Delta` — a `BagDelta<Item>` for a set, an `EntryDelta<K, V>` for a map | For a **set or map** whose order carries no meaning. |
| `"unordered-delta"` | `MapDelta<K, V, D>`, an `add`, a `remove`, and a `change` | For a **map**: values under a surviving key are diffed rather than resent. |
| `"ordered"` | `SeqDelta<Item>`, a Myers edit script | For a sequence where position matters — the one field type that takes a `Vec`. Items need `Hash + Eq`. |
| `"delta"` | `Option<<T as Delta>::Output>` | Diffs the field recursively; the field's type must derive `Delta` too. |

`apply_delta` returns `Result<(), Mismatch>`. Only an enum can fail — a struct's delta always fits the struct — so the `unwrap` above is safe rather than lazy.

`#[delta_struct(default = "...")]` on the struct changes the default for its fields. `BagDelta`, `MapDelta`, and `SeqDelta` are types from this crate, so enable the `serde` feature to serialize a delta struct holding any of them.

```rust
use delta_struct::Delta;
use std::collections::HashSet;

#[derive(Delta)]
struct Device {
    #[delta_struct(field_type = "unordered")]
    services: HashSet<String>,
    online: bool,
}

let device = |service: &str, online| Device {
    services: vec![service.to_string()].into_iter().collect(),
    online,
};

let delta = Delta::delta(device("ssh", false), device("mqtt", true)).unwrap();
assert_eq!(delta.services.add, vec!["mqtt".to_string()]);
assert_eq!(delta.services.remove, vec!["ssh".to_string()]);
assert_eq!(delta.online, Some(true));
```

Both `unordered` field types diff through `TryIndex`, this crate's fallible answer to `std::ops::Index`, so each element is looked up once instead of scanned for. The cost of a diff is the cost of the collection you picked: **O(n)** for a `HashSet`/`HashMap`, O(n log n) for a `BTreeSet`/`BTreeMap`. `Vec` is deliberately not supported here — implementing `TryIndex` for it would only hide a quadratic scan behind an O(1)-looking call.

A map is an `unordered` field too, and this — not `unordered-delta` — is what to reach for when the values are scalars with no `Delta` impl of their own, as a map of labels, tags, or config usually is. A map's delta is a different shape from a set's, because a map has a rule a set has no equivalent of: no two entries share a key. So `add` carries whole entries, but `remove` carries **bare keys**:

```rust
use delta_struct::Delta;
use std::collections::BTreeMap;

#[derive(Delta)]
struct Deployment {
    #[delta_struct(field_type = "unordered")]
    labels: BTreeMap<String, String>,
}

let deployment = |labels: &[(&str, &str)]| Deployment {
    labels: labels.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
};

let delta = Delta::delta(
    deployment(&[("tier", "web"), ("zone", "a")]),
    deployment(&[("tier", "edge")]),
)
.unwrap();
// `tier` survived, so only what it holds now travels — the receiver keeps the
// old value it already has. `zone` left, and its key alone says so.
assert_eq!(delta.labels.add, vec![("tier".to_string(), "edge".to_string())]);
assert_eq!(delta.labels.remove, vec!["zone".to_string()]);
```

A key that survived with a new value under it is an *addition*, not a removal followed by one — applying an addition overwrites whatever the key held. Which of the two shapes a field gets is the collection's business: the delta field is declared as `<T as Unordered>::Delta`, and the collection's `Unordered` impl picks, which is what lets one field type cover both.

### Keyed collections

`unordered-delta` handles a map whose values are themselves worth diffing. Entries are paired by the collection's own key, so a value that merely changed a field travels as a delta rather than as a removal plus a full re-send:

```rust
use delta_struct::Delta;
use std::collections::HashMap;

#[derive(Delta)]
struct Service {
    port: u16,
    healthy: bool,
}

#[derive(Delta)]
struct Cluster {
    #[delta_struct(field_type = "unordered-delta")]
    services: HashMap<String, Service>,
}

let cluster = |port| Cluster {
    services: vec![("web".to_string(), Service { port, healthy: true })]
        .into_iter()
        .collect(),
};

let delta = Delta::delta(cluster(80), cluster(8080)).unwrap();
assert!(delta.services.add.is_empty());
assert!(delta.services.remove.is_empty());
assert_eq!(delta.services.change[0].key, "web");
assert_eq!(delta.services.change[0].delta.port, Some(8080));
assert_eq!(delta.services.change[0].delta.healthy, None);
```

The three parts are deliberately asymmetric: `add` carries whole entries because the receiver has never seen them, while `remove` carries bare keys and `change` carries deltas, because for those the receiver already holds the rest. That last part is the trade against `unordered` over the same map, which sends a changed value whole but asks nothing of the value type beyond `PartialEq`. The field must be a map — applying a change mutates a value where it sits, which is why this needs `TryIndexMut` and a set will not do. Enable the `serde` feature to serialize a delta containing an `unordered-delta` field.

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
config.apply_delta(serde_json::from_str::<ConfigDelta>(&payload.unwrap()).unwrap()).unwrap();
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

Splice positions index the old sequence and arrive sorted and non-overlapping, so applying one is a single forward pass and reproduces the new sequence exactly. Items need `Hash + Eq`, which is what Myers requires — so a `Vec<f64>` has nowhere to go but `scalar`. Enable the `serde` feature to serialize a delta containing an `ordered` field.

### Enums

An enum changes in two ways a struct cannot, and its delta says which. Same variant on both sides: diffed field by field, like a struct. Different variants: no difference to describe, so the whole value travels.

```rust
use delta_struct::{Delta, EnumDelta};

#[derive(Delta)]
#[delta_struct(delta_leader = "#[derive(Debug)]")]
enum Shape {
    Empty,
    Circle { r: u32 },
}

// Same variant: only the field that moved travels.
let delta = Delta::delta(Shape::Circle { r: 1 }, Shape::Circle { r: 2 }).unwrap();
match delta {
    EnumDelta::Delta(ShapeDelta::Circle { r }) => assert_eq!(r, Some(2)),
    _ => panic!("same variant"),
}

// Different variant: a replacement, not a difference.
let delta = Delta::delta(Shape::Empty, Shape::Circle { r: 3 }).unwrap();
assert!(matches!(delta, EnumDelta::Became(Shape::Circle { r: 3 })));
```

So `Output` is `EnumDelta<Self, {Self}Delta>` rather than the bare companion, and the generated enum carries one variant per *diffable* source variant — a field-less variant gets none, since two of those can never differ. Keeping `Became` on a crate type rather than as an arm of the generated enum is what lets you have a variant of your own called `Became`.

This is why `apply_delta` is fallible. A delta built while a value was one variant can arrive at a value that is now another, and that's real divergence:

```rust
use delta_struct::{Delta, Mismatch};

#[derive(Delta)]
enum Shape {
    Empty,
    Circle { r: u32 },
}

let delta = Delta::delta(Shape::Circle { r: 1 }, Shape::Circle { r: 2 }).unwrap();
let mut diverged = Shape::Empty;
assert_eq!(
    diverged.apply_delta(delta),
    Err(Mismatch { type_name: "Shape", expected: "Circle", found: "Empty" }),
);
```

Nested deltas propagate the innermost mismatch, so you get the enum that actually disagreed rather than the outermost struct you called `apply_delta` on.

### Checking that a delta belongs

`apply_delta` assumes the value it is handed equals the `old` the delta came from, and checks nothing — so over a wire, a dropped or duplicated message diverges the two sides in silence. `Versioned` is the opt-in fix:

```rust
use delta_struct::{Applied, Delta, Fingerprint, Versioned};

#[derive(Clone, Debug, Delta, Fingerprint, PartialEq)]
#[delta_struct(delta_leader = "#[derive(Clone)]")]
struct Config {
    host: String,
    port: u16,
}

let config = |port| Config { host: "localhost".to_string(), port };

let mut sender = Versioned::new(config(80));
let mut receiver = Versioned::new(config(80));

let message = sender.commit(config(8080)).expect("the port changed");

// Delivered twice: applied once, then recognised and ignored.
assert_eq!(receiver.apply(message.clone()), Ok(Applied::Updated));
assert_eq!(receiver.apply(message), Ok(Applied::Stale));
assert_eq!(receiver.get(), sender.get());
```

Each `VersionedDelta` carries four numbers, and each catches something the others can't:

| Field | Catches |
| --- | --- |
| `from`, `to` | A message dropped, reordered, or replayed. |
| `base` | A receiver whose state drifted for any reason, including one that never came through this stream. |
| `result` | The delta itself being wrong — mismatched schema versions, or a bug. |

A rejected delta leaves the receiver untouched and its version unmoved, so a later delta fails too rather than papering over the hole. The answer to any `Mismatch` is to resend the whole `Versioned`, which serializes as a unit and carries the version to resume from.

`Fingerprint` is its own derive because `std::hash::Hash` can't do the job — it isn't implemented for `HashSet` or `HashMap`, which are exactly what the `unordered` field types require, and its standard hasher may change between Rust releases. This one folds sets and maps commutatively and pins itself to FNV-1a constants, so the same value fingerprints identically on any platform and any Rust version. It derives on enums too.

None of this touches the `Delta` trait, the derive, or any generated struct. Diffing locally costs you nothing for it.

### Limitations

- Unions are rejected; structs and enums are both supported.
- An enum with no variants is rejected — an uninhabited type has no two values that could differ.
- Applying an `unordered` or `unordered-delta` delta preserves membership, not position; use `ordered` when position matters.
- `ordered` items must be `Hash + Eq`.
- A `Vec` cannot be an `unordered` field — use a `HashSet`/`BTreeSet`/`HashMap`/`BTreeMap`, or `ordered`.
- `unordered-delta` keys come from the collection, so there is no way to nominate a field of the value as the key.
- `Versioned` assumes one writer per stream; concurrent writers are detected, not reconciled.

Full documentation, including trait bounds and the exact shape of the generated code, is on [docs.rs](https://docs.rs/delta-struct).

## License

Licensed under either of

- Apache License, Version 2.0
- MIT license

at your option.
