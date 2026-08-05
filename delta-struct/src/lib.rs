//! Compute the difference (delta) between two instances of a type, and apply
//! that difference to a third.
//!
//! Deriving [`Delta`] on a struct generates a companion "delta struct" holding
//! only what changed, plus an implementation of the [`Delta`] trait that knows
//! how to produce one and how to apply it. Pair it with `serde` and you can
//! send updates over the wire without resending state that both sides already
//! agree on.
//!
//! # Quick start
//!
//! ```
//! use delta_struct::Delta;
//!
//! #[derive(Delta)]
//! struct Config {
//!     host: String,
//!     port: u16,
//! }
//!
//! let old = Config { host: "localhost".to_string(), port: 80 };
//! let new = Config { host: "localhost".to_string(), port: 8080 };
//!
//! // `Config` gained a companion struct named `ConfigDelta`.
//! let delta = Delta::delta(old, new).expect("the port changed");
//! assert_eq!(delta.host, None);          // unchanged fields are `None`
//! assert_eq!(delta.port, Some(8080));
//!
//! // Applying the delta to an older copy brings it up to date.
//! let mut current = Config { host: "localhost".to_string(), port: 80 };
//! current.apply_delta(delta);
//! assert_eq!(current.port, 8080);
//! ```
//!
//! Note that a single `use delta_struct::Delta;` imports both the trait and
//! the derive macro. The trait has to be in scope wherever you derive it — the
//! generated code refers to `Delta` by that name.
//!
//! [`Delta::delta`] returns [`None`] when nothing changed, so
//! `if let Some(delta) = Delta::delta(old, new)` is the usual way to skip
//! sending an empty update.
//!
//! # Field types
//!
//! Every field is diffed according to a *field type*, chosen with
//! `#[delta_struct(field_type = "...")]`. The default is `"scalar"`, which can
//! be changed per struct — see [Container attributes](#container-attributes).
//!
//! ## `scalar` (the default)
//!
//! The field is compared with `!=` and replaced wholesale. In the delta struct
//! it becomes `Option<T>`: `Some(new_value)` when the two differ, [`None`]
//! when they don't. Requires `T: PartialEq`.
//!
//! ## `unordered`
//!
//! The field is treated as a bag of elements whose order carries no meaning,
//! so the delta records only which elements came and went: a [`BagDelta`],
//! holding an `add` and a `remove`, both `Vec<Item>`.
//!
//! ```
//! use delta_struct::Delta;
//! use std::collections::HashSet;
//!
//! #[derive(Delta)]
//! struct Device {
//!     #[delta_struct(field_type = "unordered")]
//!     services: HashSet<String>,
//! }
//!
//! let device = |services: &[&str]| Device {
//!     services: services.iter().map(|s| s.to_string()).collect(),
//! };
//!
//! let delta = Delta::delta(device(&["ssh", "http"]), device(&["http", "mqtt"])).unwrap();
//! assert_eq!(delta.services.add, vec!["mqtt".to_string()]);
//! assert_eq!(delta.services.remove, vec!["ssh".to_string()]);
//! ```
//!
//! The field has to be a **set** — a [`HashSet`](std::collections::HashSet) or
//! a [`BTreeSet`](std::collections::BTreeSet). Formally it needs [`Extend`]
//! and [`TryIndex`], this crate's fallible answer to
//! [`Index`](std::ops::Index); the two std sets implement it, and you can
//! implement it for your own collection. A [`Vec`] deliberately does not
//! qualify — see [Limitations](#limitations).
//!
//! Every element of the old collection is looked up in the new one exactly
//! once, so the cost of a diff is the cost of n lookups in whichever
//! collection you picked: **O(n)** for a `HashSet`, O(n log n) for a
//! `BTreeSet`. Applying one costs the same, since each removal is a lookup
//! rather than a rebuild.
//!
//! `apply_delta` preserves membership but not position — additions land
//! wherever the collection decides to put them. Use `ordered` where that
//! matters.
//!
//! ## `unordered-delta`
//!
//! Like `unordered`, but for a collection of key/value entries whose values
//! are worth diffing rather than resending. An entry whose key is on both
//! sides is not a removal plus an addition: the two values are handed to
//! [`Delta::delta`] and only the difference is recorded. The delta is a
//! [`MapDelta`], holding an `add`, a `remove`, and a `change`.
//!
//! ```
//! use delta_struct::Delta;
//! use std::collections::HashMap;
//!
//! #[derive(Delta)]
//! struct Service {
//!     port: u16,
//!     healthy: bool,
//! }
//!
//! #[derive(Delta)]
//! struct Cluster {
//!     #[delta_struct(field_type = "unordered-delta")]
//!     services: HashMap<String, Service>,
//! }
//!
//! let cluster = |port| Cluster {
//!     services: vec![("web".to_string(), Service { port, healthy: true })]
//!         .into_iter()
//!         .collect(),
//! };
//!
//! let delta = Delta::delta(cluster(80), cluster(8080)).unwrap();
//! // `web` stayed put, so all that travels is the one field that moved.
//! assert!(delta.services.add.is_empty());
//! assert!(delta.services.remove.is_empty());
//! assert_eq!(delta.services.change[0].key, "web");
//! assert_eq!(delta.services.change[0].delta.port, Some(8080));
//! assert_eq!(delta.services.change[0].delta.healthy, None);
//! ```
//!
//! The key is the collection's own — the `K` of a `HashMap<K, V>` — not
//! something you nominate. The field has to be a **map**: a
//! [`HashMap`](std::collections::HashMap) or a
//! [`BTreeMap`](std::collections::BTreeMap). Formally it needs [`Extend`] and
//! [`TryIndexMut`], its entry type needs [`MapEntry`] (implemented for
//! `(K, V)`, which is what every std map iterates as), and its value type
//! needs [`Delta`].
//!
//! [`TryIndexMut`] rather than [`TryIndex`] is what excludes sets here, and
//! correctly so: applying a delta means mutating a value where it sits, which
//! a set cannot allow without letting you invalidate the hash or ordering it
//! filed the element under.
//!
//! Every key of the old collection is looked up in the new one exactly once,
//! so as with `unordered` the cost is n lookups — **O(n)** for a `HashMap`,
//! O(n log n) for a `BTreeMap`. Applying one preserves membership rather than
//! position, also the same as `unordered`.
//!
//! ## `ordered`
//!
//! The field is diffed positionally with Myers' algorithm, and the delta is a
//! minimal edit script: a [`SeqDelta`] holding [`Splice`]s that each say
//! "at this index, drop this many items and put these in their place".
//!
//! ```
//! use delta_struct::{Delta, Splice};
//!
//! #[derive(Delta)]
//! struct Playlist {
//!     #[delta_struct(field_type = "ordered")]
//!     tracks: Vec<String>,
//! }
//!
//! let old = Playlist { tracks: vec!["intro".to_string(), "b".to_string(), "outro".to_string()] };
//! let new = Playlist { tracks: vec!["intro".to_string(), "x".to_string(), "outro".to_string()] };
//!
//! let delta = Delta::delta(old, new).unwrap();
//! assert_eq!(
//!     delta.tracks.splices,
//!     vec![Splice { at: 1, remove: 1, insert: vec!["x".to_string()] }],
//! );
//! ```
//!
//! Splice positions index the *old* sequence and arrive sorted and
//! non-overlapping, so applying one is a single forward pass. Reordering is a
//! real change here where `unordered` would see none, and applying a delta
//! reproduces the new sequence exactly, position included.
//!
//! The collection needs `IntoIterator` and `FromIterator`, and its items need
//! `Hash + Eq`, because that is what indexing the sequences for Myers
//! requires. This is the one field type that takes a [`Vec`], and so the only
//! one that will diff a sequence at all — but `f64` is neither `Hash` nor
//! `Eq`, so a `Vec<f64>` still has nowhere to go but `scalar`.
//!
//! ## `delta`
//!
//! The field is itself diffed recursively, which keeps a nested change from
//! resending the whole subtree. Requires the field's type to implement
//! [`Delta`]; the delta struct holds `Option<<T as Delta>::Output>`.
//!
//! ```
//! use delta_struct::Delta;
//!
//! #[derive(Delta)]
//! struct Inner {
//!     a: i32,
//!     b: i32,
//! }
//!
//! #[derive(Delta)]
//! struct Outer {
//!     #[delta_struct(field_type = "delta")]
//!     inner: Inner,
//!     name: String,
//! }
//!
//! let old = Outer { inner: Inner { a: 1, b: 2 }, name: "x".to_string() };
//! let new = Outer { inner: Inner { a: 1, b: 3 }, name: "x".to_string() };
//!
//! let delta = Delta::delta(old, new).unwrap();
//! let inner_delta = delta.inner.expect("`b` changed");
//! assert_eq!(inner_delta.a, None);
//! assert_eq!(inner_delta.b, Some(3));
//! ```
//!
//! # Container attributes
//!
//! `#[delta_struct(...)]` on the struct itself accepts:
//!
//! - `default = "..."` — the field type used for fields without their own
//!   `field_type`. Defaults to `"scalar"`.
//! - `delta_leader = "..."` — tokens to emit immediately above the generated
//!   struct. This is how you attach derives, doc comments, or any other
//!   attribute to a type you never get to write by hand.
//!
//! ```
//! use delta_struct::Delta;
//! use std::collections::HashSet;
//!
//! #[derive(Delta)]
//! #[delta_struct(
//!     default = "unordered",
//!     delta_leader = "/// The changes to a `Tags`.\n#[derive(Debug)]"
//! )]
//! struct Tags {
//!     labels: HashSet<String>,
//!     // Opt an individual field back out of the container default.
//!     #[delta_struct(field_type = "scalar")]
//!     revision: u32,
//! }
//!
//! let old = Tags { labels: HashSet::new(), revision: 1 };
//! let new = Tags {
//!     labels: vec!["new".to_string()].into_iter().collect(),
//!     revision: 2,
//! };
//! let delta = Delta::delta(old, new).unwrap();
//! assert_eq!(format!("{:?}", delta.labels.add), r#"["new"]"#);
//! assert_eq!(delta.revision, Some(2));
//! ```
//!
//! `delta_leader` also works on individual fields, where it decorates the
//! generated field instead of the generated struct.
//!
//! ```
//! # use delta_struct::Delta;
//! #[derive(Delta)]
//! struct Host {
//!     #[delta_struct(delta_leader = "/// The new port, if it moved.")]
//!     port: u16,
//! }
//! ```
//!
//! # Working with serde
//!
//! For `scalar` and `delta` fields there is no serde integration to enable;
//! `delta_leader` is the whole story. Put the derives on the generated struct
//! and it serializes like anything else:
//!
//! ```
//! use delta_struct::Delta;
//!
//! #[derive(Delta)]
//! #[delta_struct(delta_leader = "#[derive(serde::Serialize, serde::Deserialize)]")]
//! struct Config {
//!     host: String,
//!     port: u16,
//! }
//!
//! let old = Config { host: "localhost".to_string(), port: 80 };
//! let new = Config { host: "localhost".to_string(), port: 8080 };
//!
//! // Sender: there is no message to send at all when nothing changed.
//! let payload = Delta::delta(old, new).map(|delta| serde_json::to_string(&delta).unwrap());
//! assert_eq!(payload.as_deref(), Some(r#"{"host":null,"port":8080}"#));
//!
//! // Receiver applies it to whatever it already had.
//! let mut config = Config { host: "localhost".to_string(), port: 80 };
//! config.apply_delta(serde_json::from_str::<ConfigDelta>(&payload.unwrap()).unwrap());
//! assert_eq!(config.port, 8080);
//! ```
//!
//! Field-level `delta_leader` carries serde attributes just as well, so
//! `skip_serializing_if` can keep unchanged fields out of the payload
//! entirely rather than sending them as `null`:
//!
//! ```
//! use delta_struct::Delta;
//!
//! #[derive(Delta)]
//! #[delta_struct(delta_leader = "#[derive(serde::Serialize)]")]
//! struct Config {
//!     #[delta_struct(delta_leader = "#[serde(skip_serializing_if = \"Option::is_none\")]")]
//!     host: String,
//!     #[delta_struct(delta_leader = "#[serde(skip_serializing_if = \"Option::is_none\")]")]
//!     port: u16,
//! }
//!
//! let old = Config { host: "localhost".to_string(), port: 80 };
//! let new = Config { host: "localhost".to_string(), port: 8080 };
//!
//! let delta = Delta::delta(old, new).unwrap();
//! assert_eq!(serde_json::to_string(&delta).unwrap(), r#"{"port":8080}"#);
//! ```
//!
//! # Checking that a delta belongs
//!
//! [`Delta::apply_delta`] assumes the value it is handed equals the `old` the
//! delta came from, and checks nothing. Over an unreliable transport that assumption breaks:
//! a message is dropped, delivered twice, or arrives at a receiver whose state
//! drifted for some other reason, and the two sides diverge in silence.
//!
//! [`Versioned`] is the opt-in fix. It pairs a value with a version counter
//! and a [`Fingerprint`] of its contents, and refuses any delta that does not
//! belong.
//!
//! ```
//! use delta_struct::{Applied, Delta, Fingerprint, Mismatch, Versioned};
//!
//! #[derive(Clone, Debug, Delta, Fingerprint, PartialEq)]
//! #[delta_struct(delta_leader = "#[derive(Clone)]")]
//! struct Config {
//!     host: String,
//!     port: u16,
//! }
//!
//! let config = |port| Config { host: "localhost".to_string(), port };
//!
//! let mut sender = Versioned::new(config(80));
//! let mut receiver = Versioned::new(config(80));
//!
//! let first = sender.commit(config(8080)).expect("the port changed");
//! let second = sender.commit(config(9090)).expect("the port changed again");
//!
//! // Delivered twice: recognised and ignored.
//! assert_eq!(receiver.apply(first.clone()), Ok(Applied::Updated));
//! assert_eq!(receiver.apply(first), Ok(Applied::Stale));
//! assert_eq!(receiver.apply(second), Ok(Applied::Updated));
//! assert_eq!(receiver.get(), sender.get());
//!
//! // A delta from a stream this receiver never joined is refused rather than
//! // half-applied.
//! let mut stranger = Versioned::new(config(80));
//! let orphan = Versioned::new(config(1)).commit(config(2)).unwrap();
//! assert!(matches!(stranger.apply(orphan), Err(Mismatch::Base { .. })));
//! ```
//!
//! Every [`VersionedDelta`] carries four numbers, each catching a failure the
//! others cannot:
//!
//! | Field | Catches |
//! | --- | --- |
//! | `from`, `to` | A message dropped, reordered, or replayed. |
//! | `base` | A receiver whose state drifted for any reason, including one that never came through this stream. |
//! | `result` | The delta itself being wrong — mismatched schema versions, or a bug. |
//!
//! A rejected delta leaves the receiver untouched and its version unmoved, so
//! a later delta in the same stream fails too rather than papering over the
//! hole. The answer to any [`Mismatch`] is to resend the whole [`Versioned`],
//! which serializes as a unit and carries the version the receiver resumes
//! from.
//!
//! None of this touches the [`Delta`] trait, the derive, or any generated
//! struct. If you are diffing locally rather than over a wire, you never name
//! anything in this section and pay for none of it.
//!
//! ## `Fingerprint`
//!
//! [`Fingerprint`] is a separate derive because [`std::hash::Hash`] cannot do
//! the job: it is not implemented for [`HashSet`](std::collections::HashSet)
//! or [`HashMap`](std::collections::HashMap) — exactly the collections the
//! `unordered` field types require — and its standard hasher is allowed to
//! change between Rust releases, which would make a toolchain upgrade on one
//! side of a connection look like corruption.
//!
//! So sets and maps fold commutatively, iteration order cannot reach the
//! result, and the hash is pinned to FNV-1a constants written down in the
//! source. The same value fingerprints identically on any platform and any
//! Rust version. Unlike [`Delta`], it derives on enums too.
//!
//! Checking costs a full traversal of the state on each `commit` and each
//! `apply` — cheaper than serializing it, but not free, which is the price of
//! the `base` and `result` guarantees.
//!
//! # What gets generated
//!
//! For `struct Foo`, deriving [`Delta`] emits `struct FooDelta` with the same
//! visibility as `Foo` and the same generic parameters, carrying over their
//! bounds and `where` clause as written. All of its fields are
//! `pub`, and by default it derives nothing at all — reach for `delta_leader`
//! whenever you need `Debug`, `Clone`, or serde on it. (Likewise if your crate
//! sets `#![deny(missing_docs)]`: the generated struct and its fields need doc
//! comments supplied through `delta_leader`.)
//!
//! Every field type maps one source field onto exactly one delta field, so a
//! delta struct always has the same fields in the same order as the struct it
//! came from — only their types differ. A tuple struct's delta is a tuple
//! struct in turn, so its fields keep their positions:
//!
//! ```
//! use delta_struct::Delta;
//!
//! #[derive(Delta)]
//! struct Meters(i32);
//!
//! let delta = Delta::delta(Meters(3), Meters(4)).unwrap();
//! assert_eq!(delta.0, Some(4));
//! ```
//!
//! # Limitations
//!
//! - **Structs only.** Enums and unions are rejected; there is no obvious
//!   delta for a value that changed variant.
//! - **Every type parameter gets a `PartialEq` bound** on the generated impl,
//!   whether or not the field that uses it needs one.
//! - **A unit struct's delta is always [`None`]**, as is that of a struct with
//!   no fields — there is nothing that could differ.
//! - **`ordered` items need `Hash + Eq`**, so float sequences are out. See
//!   that section above.
//! - **A [`Vec`] cannot be an `unordered` field.** Membership diffing goes
//!   through [`TryIndex`], and a `Vec` has no sub-linear lookup to offer —
//!   implementing it would only hide a quadratic scan behind an O(1)-looking
//!   call. Use a [`HashSet`](std::collections::HashSet) or a
//!   [`BTreeSet`](std::collections::BTreeSet), or `ordered` if position
//!   matters.
//! - **`unordered-delta` keys are the collection's own.** There is no way to
//!   nominate a field of the value as the key, so a `Vec<Record>` has to
//!   become a `HashMap<Id, Record>` to use it.
//! - **[`Versioned`] assumes one writer per stream.** Two senders committing
//!   against the same base both produce `from: 0`, and the second is rejected
//!   rather than merged. Divergence is detected, not reconciled — reach for a
//!   CRDT if you need concurrent writers.

#![warn(missing_docs)]

// The derive emits `::delta_struct::…` paths for the runtime items an
// `ordered` field needs. That path has to resolve inside this crate too, or
// the crate's own tests could not use its own derive.
extern crate self as delta_struct;

pub mod bag;
pub mod fingerprint;
pub mod index;
pub mod map;
pub mod seq;
pub mod version;

pub use bag::BagDelta;
pub use delta_struct_macros::{Delta, Fingerprint};
pub use fingerprint::{fingerprint_of, Fingerprint};
pub use index::{TryIndex, TryIndexMut};
pub use map::{KeyedDelta, MapDelta, MapEntry};
pub use seq::{SeqDelta, Splice};
pub use version::{Applied, Mismatch, Versioned, VersionedDelta};

/// Computing the difference between two values, and applying it to a third.
///
/// You will normally derive this rather than implement it — see the
/// [crate documentation](crate) for the derive's attributes and the shape of
/// the type it generates. Implement it by hand when you want custom diffing
/// for a type that other structs then reference with
/// `#[delta_struct(field_type = "delta")]`.
pub trait Delta {
    /// The type describing a difference between two `Self` values.
    ///
    /// The derive sets this to the generated `{Self}Delta` struct.
    type Output;

    /// Computes what it would take to turn `old` into `new`.
    ///
    /// Returns [`None`] when the two are equivalent, which lets callers skip
    /// sending or storing an update that would do nothing. Both values are
    /// consumed: the delta takes ownership of whatever it needs from `new`.
    fn delta(old: Self, new: Self) -> Option<Self::Output>;

    /// Applies a delta in place.
    ///
    /// Applying the delta from `delta(old, new)` to a value equal to `old`
    /// yields a value equal to `new` — with the caveat that `unordered` fields
    /// preserve membership rather than order.
    fn apply_delta(&mut self, delta: Self::Output);
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct UnitType;

    #[derive(Delta, Clone, Debug, PartialEq, Eq)]
    #[delta_struct(delta_leader = "#[derive(Clone, Debug, PartialEq, Eq)]")]
    struct NewType(i32);

    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct NewTypeWithGeneric<T>(T);

    // A tuple struct's delta is a tuple struct, which puts its `where` clause
    // after the fields rather than before them. Both spellings of a bound have
    // to survive that.
    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct InlineBoundNewType<T: Clone>(T);

    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct WhereClauseNewType<T>(T)
    where
        T: Clone;

    #[derive(Clone, Debug, Delta, PartialEq)]
    #[delta_struct(delta_leader = "#[derive(Debug, PartialEq)]")]
    struct Reading(
        #[delta_struct(field_type = "unordered")] BTreeSet<i32>,
        #[delta_struct(field_type = "ordered")] Vec<String>,
        #[delta_struct(field_type = "delta")] NewType,
        bool,
    );

    #[test]
    fn tuple_struct_delta_keeps_field_positions() {
        let old = Reading(
            vec![1, 2].into_iter().collect(),
            vec!["a".to_string()],
            NewType(7),
            false,
        );
        let new = Reading(
            vec![2, 3].into_iter().collect(),
            vec!["a".to_string(), "b".to_string()],
            NewType(8),
            true,
        );
        let delta = Delta::delta(old.clone(), new.clone()).unwrap();
        assert_eq!(delta.0.add, vec![3]);
        assert_eq!(delta.0.remove, vec![1]);
        assert_eq!(
            delta.1.splices,
            vec![Splice {
                at: 1,
                remove: 0,
                insert: vec!["b".to_string()],
            }]
        );
        assert_eq!(delta.2, Some(NewTypeDelta(Some(8))));
        assert_eq!(delta.3, Some(true));

        let mut applied = old;
        applied.apply_delta(delta);
        assert_eq!(applied, new);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn tuple_struct_delta_serializes_as_a_sequence() {
        #[derive(Delta)]
        #[delta_struct(delta_leader = "#[derive(serde::Serialize)]")]
        struct Meters(i32, i32);

        let delta = Delta::delta(Meters(1, 2), Meters(1, 3)).unwrap();
        assert_eq!(serde_json::to_string(&delta).unwrap(), "[null,3]");
    }

    #[derive(Delta)]
    struct InlineBoundGeneric<T: Clone> {
        foo: T,
        bar: bool,
    }

    #[derive(Delta)]
    struct WhereClauseGeneric<T>
    where
        T: Clone,
    {
        foo: T,
        bar: bool,
    }

    #[derive(Delta)]
    struct InlineBoundDeltaField<T: Delta> {
        #[delta_struct(field_type = "delta")]
        foo: T,
    }

    #[derive(Delta)]
    struct WhereClauseDeltaField<T>
    where
        T: Delta,
    {
        #[delta_struct(field_type = "delta")]
        foo: T,
    }

    #[derive(Delta)]
    struct SimpleType {
        #[delta_struct(delta_leader = "/// This is foo.")]
        foo: i32,
        bar: bool,
    }

    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct SimpleTypeWithGeneric<T> {
        foo: T,
        bar: bool,
    }

    #[derive(Delta)]
    struct SimpleCollectionWithGeneric<T: Ord> {
        #[delta_struct(
            field_type = "unordered",
            delta_leader = "/// This the foo type on the delta struct."
        )]
        foo: BTreeSet<T>,
        bar: bool,
    }

    #[derive(Delta)]
    struct DeltaRecursion {
        #[delta_struct(field_type = "delta")]
        foo: NewType,
        bar: bool,
    }

    #[derive(Delta)]
    #[delta_struct(default = "unordered")]
    struct AttributeTest {
        #[delta_struct(field_type = "scalar")]
        foo: i32,
        #[delta_struct(field_type = "scalar")]
        bar: i32,
        baz: BTreeSet<i32>,
    }

    #[derive(Delta, Clone, Debug, PartialEq, Eq)]
    struct AllFieldTypes {
        #[delta_struct(field_type = "scalar")]
        scalar: i32,
        #[delta_struct(field_type = "delta")]
        delta: NewType,
        #[delta_struct(field_type = "unordered")]
        unordered: HashSet<i32>,
    }

    #[derive(Clone, Debug, Delta, PartialEq)]
    #[allow(dead_code)] // The derive is itself the test
    struct DeviceConfig {
        #[delta_struct(field_type = "unordered")]
        pub services: HashSet<String>,
        #[delta_struct(field_type = "unordered")]
        pub settings: HashSet<String>,
        pub thumbnail_request: i32,
        pub speedtest_request: i32,
        #[delta_struct(field_type = "delta")]
        pub features: AllFieldTypes,
        pub deprovision: bool,
    }

    #[test]
    fn unordered_with_scalar() {
        let old = SimpleCollectionWithGeneric {
            foo: vec![1, 2, 3].into_iter().collect(),
            bar: false,
        };
        let new = SimpleCollectionWithGeneric {
            foo: vec![3, 4, 5].into_iter().collect(),
            bar: true,
        };
        let delta = Delta::delta(old, new).unwrap();
        assert_eq!(delta.foo.add, vec![4, 5]);
        assert_eq!(delta.foo.remove, vec![1, 2]);
        assert_eq!(delta.bar, Some(true));
    }

    #[test]
    fn unordered_apply_round_trips() {
        #[derive(Clone, Debug, Delta, PartialEq)]
        struct Tags {
            #[delta_struct(field_type = "unordered")]
            labels: HashSet<i32>,
        }

        let tags = |labels: &[i32]| Tags {
            labels: labels.iter().copied().collect(),
        };
        let cases: &[(&[i32], &[i32])] = &[
            (&[1, 2, 3], &[3, 4, 5]),
            (&[1, 2], &[1, 2, 3]),
            (&[1, 2, 3], &[1, 2]),
            (&[], &[1, 2, 3]),
            (&[1, 2, 3], &[]),
            (&[1, 2], &[3, 4]),
        ];
        for (old, new) in cases {
            let mut applied = tags(old);
            let delta = Delta::delta(tags(old), tags(new)).unwrap();
            applied.apply_delta(delta);
            assert_eq!(applied, tags(new), "{:?} -> {:?}", old, new);
        }
    }

    #[test]
    fn unordered_apply_ignores_absent_removals() {
        // `apply` drops each removal by lookup rather than rebuilding, so a
        // key that isn't there is a no-op — which makes applying the same
        // delta twice harmless.
        let old = AllFieldTypes {
            scalar: 1,
            delta: NewType(1),
            unordered: vec![1, 2].into_iter().collect(),
        };
        let new = AllFieldTypes {
            scalar: 1,
            delta: NewType(1),
            unordered: vec![2, 3].into_iter().collect(),
        };
        let mut applied = old.clone();
        applied.apply_delta(Delta::delta(old.clone(), new.clone()).unwrap());
        applied.apply_delta(Delta::delta(old, new.clone()).unwrap());
        assert_eq!(applied, new);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn unordered_delta_serializes_nested() {
        #[derive(Delta)]
        #[delta_struct(delta_leader = "#[derive(serde::Serialize)]")]
        struct Device {
            #[delta_struct(field_type = "unordered")]
            services: BTreeSet<String>,
        }

        let device = |services: &[&str]| Device {
            services: services.iter().map(|s| s.to_string()).collect(),
        };
        let delta = Delta::delta(device(&["ssh"]), device(&["mqtt"])).unwrap();
        assert_eq!(
            serde_json::to_string(&delta).unwrap(),
            r#"{"services":{"add":["mqtt"],"remove":["ssh"]}}"#
        );
    }

    #[test]
    fn delta_false_positive_check() {
        let old = NewType(5);
        let new = NewType(5);
        let delta = Delta::delta(old, new);
        assert!(delta.is_none());
    }

    #[test]
    fn scalar_delta_false_positive_check() {
        let old = SimpleType { foo: 5, bar: false };
        let new = SimpleType { foo: 5, bar: true };
        let delta = Delta::delta(old, new).unwrap();
        assert!(delta.foo.is_none());
        assert_eq!(delta.bar, Some(true));
    }

    #[test]
    fn delta_field() {
        let old = DeltaRecursion {
            foo: NewType(5),
            bar: false,
        };
        let new = DeltaRecursion {
            foo: NewType(6),
            bar: true,
        };
        let delta = Delta::delta(old, new).unwrap();
        assert_eq!(delta.foo, Some(NewTypeDelta(Some(6))));
        assert_eq!(delta.bar, Some(true));
    }

    #[test]
    fn default_type_respected() {
        let old = AttributeTest {
            foo: 5,
            bar: 4,
            baz: BTreeSet::new(),
        };
        let new = AttributeTest {
            foo: 5,
            bar: 4,
            baz: vec![9, 4, 5].into_iter().collect(),
        };
        let delta = Delta::delta(old, new).unwrap();
        assert!(delta.foo.is_none());
        assert!(delta.bar.is_none());
        assert_eq!(delta.baz.add, vec![4, 5, 9]);
        assert_eq!(delta.baz.remove, Vec::<i32>::new());
    }

    #[derive(Clone, Debug, Delta, PartialEq)]
    struct Playlist {
        #[delta_struct(field_type = "ordered")]
        tracks: Vec<String>,
        shuffle: bool,
    }

    #[derive(Delta)]
    #[delta_struct(default = "ordered")]
    struct OrderedByDefault {
        a: Vec<i32>,
        b: Vec<i32>,
    }

    fn playlist(tracks: &[&str], shuffle: bool) -> Playlist {
        Playlist {
            tracks: tracks.iter().map(|t| t.to_string()).collect(),
            shuffle,
        }
    }

    #[test]
    fn ordered_records_position() {
        let delta = Delta::delta(
            playlist(&["a", "b", "c"], false),
            playlist(&["a", "x", "c"], false),
        )
        .unwrap();
        assert_eq!(
            delta.tracks.splices,
            vec![Splice {
                at: 1,
                remove: 1,
                insert: vec!["x".to_string()],
            }]
        );
        assert_eq!(delta.shuffle, None);
    }

    #[test]
    fn ordered_distinguishes_reorder_from_unordered() {
        // Reordering is invisible to `unordered` but not to `ordered`.
        let delta = Delta::delta(playlist(&["a", "b"], false), playlist(&["b", "a"], false));
        assert!(delta.is_some());
    }

    #[test]
    fn ordered_false_positive_check() {
        let delta = Delta::delta(playlist(&["a", "b"], false), playlist(&["a", "b"], false));
        assert!(delta.is_none());
    }

    #[test]
    fn ordered_apply_round_trips() {
        let cases: &[(&[&str], &[&str])] = &[
            (&["a", "b", "c"], &["a", "x", "c"]),
            (&["a", "b"], &["a", "b", "c"]),
            (&["b", "c"], &["a", "b", "c"]),
            (&["a", "b", "c"], &[]),
            (&[], &["a", "b", "c"]),
            (&["a", "b", "c", "d", "e"], &["a", "x", "c", "y", "e"]),
            (&["a", "a", "a", "b"], &["a", "b", "a", "a"]),
            (&["a", "b", "c"], &["c", "b", "a"]),
        ];
        for (old, new) in cases {
            let mut applied = playlist(old, true);
            let delta = Delta::delta(playlist(old, false), playlist(new, true)).unwrap();
            applied.apply_delta(delta);
            assert_eq!(applied, playlist(new, true), "{:?} -> {:?}", old, new);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn ordered_delta_serializes() {
        let delta =
            Delta::delta(playlist(&["a", "b"], false), playlist(&["a", "c"], false)).unwrap();
        let json = serde_json::to_string(&delta.tracks).unwrap();
        assert_eq!(json, r#"{"splices":[{"at":1,"remove":1,"insert":["c"]}]}"#);
        let round_tripped: SeqDelta<String> = serde_json::from_str(&json).unwrap();
        let mut target = playlist(&["a", "b"], false);
        seq::apply(&mut target.tracks, round_tripped);
        assert_eq!(target.tracks, vec!["a".to_string(), "c".to_string()]);
    }

    #[test]
    fn ordered_as_container_default() {
        let delta = Delta::delta(
            OrderedByDefault {
                a: vec![1, 2],
                b: vec![3],
            },
            OrderedByDefault {
                a: vec![1, 2],
                b: vec![3, 4],
            },
        )
        .unwrap();
        assert!(delta.a.is_empty());
        assert_eq!(
            delta.b.splices,
            vec![Splice {
                at: 1,
                remove: 0,
                insert: vec![4],
            }]
        );
    }

    #[derive(Clone, Debug, Delta, PartialEq, serde::Serialize)]
    #[delta_struct(delta_leader = "#[derive(Debug, serde::Serialize)]")]
    struct Service {
        port: u16,
        healthy: bool,
    }

    #[derive(Clone, Debug, Delta, PartialEq)]
    struct Cluster {
        #[delta_struct(field_type = "unordered-delta")]
        services: HashMap<String, Service>,
        region: String,
    }

    #[derive(Delta)]
    #[delta_struct(default = "unordered-delta")]
    #[allow(dead_code)] // The derive is itself the test
    struct UnorderedDeltaByDefault {
        a: HashMap<u8, NewType>,
        b: BTreeMap<u8, NewType>,
    }

    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct UnorderedDeltaWithGeneric<K: std::hash::Hash + Eq, V: Delta> {
        #[delta_struct(
            field_type = "unordered-delta",
            delta_leader = "/// One part of the change to `foo`."
        )]
        foo: HashMap<K, V>,
    }

    /// A cluster's services in the compact `(name, port, healthy)` form the
    /// tests below are written in.
    type Services<'a> = &'a [(&'a str, u16, bool)];

    fn cluster(services: Services, region: &str) -> Cluster {
        Cluster {
            services: services
                .iter()
                .map(|(name, port, healthy)| {
                    (
                        name.to_string(),
                        Service {
                            port: *port,
                            healthy: *healthy,
                        },
                    )
                })
                .collect(),
            region: region.to_string(),
        }
    }

    #[test]
    fn unordered_delta_diffs_values_in_place() {
        let delta = Delta::delta(
            cluster(&[("web", 80, true), ("db", 5432, true)], "us"),
            cluster(&[("web", 8080, true), ("db", 5432, true)], "us"),
        )
        .unwrap();
        // `db` is untouched and `web` only moved its port, so neither entry is
        // resent in full.
        assert!(delta.services.add.is_empty());
        assert!(delta.services.remove.is_empty());
        assert_eq!(delta.services.change.len(), 1);
        assert_eq!(delta.services.change[0].key, "web");
        assert_eq!(delta.services.change[0].delta.port, Some(8080));
        assert_eq!(delta.services.change[0].delta.healthy, None);
        assert_eq!(delta.region, None);
    }

    #[test]
    fn unordered_delta_adds_and_removes_by_key() {
        let delta = Delta::delta(
            cluster(&[("web", 80, true)], "us"),
            cluster(&[("db", 5432, false)], "us"),
        )
        .unwrap();
        assert_eq!(
            delta.services.add,
            vec![(
                "db".to_string(),
                Service {
                    port: 5432,
                    healthy: false
                }
            )]
        );
        assert_eq!(delta.services.remove, vec!["web".to_string()]);
        assert!(delta.services.change.is_empty());
    }

    #[test]
    fn unordered_delta_false_positive_check() {
        let delta = Delta::delta(
            cluster(&[("web", 80, true), ("db", 5432, true)], "us"),
            cluster(&[("db", 5432, true), ("web", 80, true)], "us"),
        );
        assert!(delta.is_none());
    }

    #[test]
    fn unordered_delta_apply_round_trips() {
        let cases: &[(Services, Services)] = &[
            // A value changed under a stable key.
            (&[("web", 80, true)], &[("web", 8080, true)]),
            // Pure addition, pure removal, and both at once.
            (&[("web", 80, true)], &[("web", 80, true), ("db", 1, false)]),
            (&[("web", 80, true), ("db", 1, false)], &[("web", 80, true)]),
            (&[("web", 80, true)], &[("db", 1, false)]),
            // Every kind of change in one go.
            (
                &[("web", 80, true), ("db", 1, false), ("gone", 9, true)],
                &[("web", 8080, true), ("db", 1, false), ("new", 7, false)],
            ),
            (&[], &[("web", 80, true)]),
            (&[("web", 80, true)], &[]),
        ];
        for (old, new) in cases {
            let mut applied = cluster(old, "us");
            let delta = Delta::delta(cluster(old, "us"), cluster(new, "eu")).unwrap();
            applied.apply_delta(delta);
            assert_eq!(applied, cluster(new, "eu"), "{:?} -> {:?}", old, new);
        }
    }

    #[test]
    fn unordered_delta_over_a_btree_map() {
        // The field need not be a `HashMap`: any collection with a
        // `TryIndexMut` impl works, and a `BTreeMap`'s ordering makes the
        // three lists deterministic.
        #[derive(Clone, Debug, Delta, PartialEq)]
        struct Pairs {
            #[delta_struct(field_type = "unordered-delta")]
            entries: BTreeMap<u8, NewType>,
        }

        let pairs = |entries: &[(u8, i32)]| Pairs {
            entries: entries.iter().map(|(k, v)| (*k, NewType(*v))).collect(),
        };

        let mut applied = pairs(&[(1, 10), (2, 20)]);
        let delta = Delta::delta(pairs(&[(1, 10), (2, 20)]), pairs(&[(2, 21), (3, 30)])).unwrap();
        assert_eq!(delta.entries.add, vec![(3, NewType(30))]);
        assert_eq!(delta.entries.remove, vec![1]);
        assert_eq!(delta.entries.change.len(), 1);
        applied.apply_delta(delta);
        assert_eq!(applied, pairs(&[(2, 21), (3, 30)]));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn unordered_delta_serializes() {
        #[derive(Delta)]
        #[delta_struct(delta_leader = "#[derive(serde::Serialize)]")]
        struct Fleet {
            #[delta_struct(field_type = "unordered-delta")]
            services: BTreeMap<String, Service>,
        }

        let fleet = |port| Fleet {
            services: vec![(
                "web".to_string(),
                Service {
                    port,
                    healthy: true,
                },
            )]
            .into_iter()
            .collect(),
        };
        let delta = Delta::delta(fleet(80), fleet(8080)).unwrap();
        assert_eq!(
            serde_json::to_string(&delta).unwrap(),
            r#"{"services":{"add":[],"remove":[],"change":[{"key":"web","delta":{"port":8080,"healthy":null}}]}}"#
        );
    }

    #[derive(Clone, Debug, Delta, Fingerprint, PartialEq)]
    #[delta_struct(delta_leader = "#[derive(Clone, Debug)]")]
    struct Tracked {
        name: String,
        #[delta_struct(field_type = "unordered")]
        tags: HashSet<String>,
        revision: u32,
    }

    fn tracked(name: &str, tags: &[&str], revision: u32) -> Tracked {
        Tracked {
            name: name.to_string(),
            tags: tags.iter().map(|t| t.to_string()).collect(),
            revision,
        }
    }

    #[test]
    fn fingerprint_ignores_set_iteration_order() {
        // The whole point: two `HashSet`s built in different orders are the
        // same state and must fingerprint the same.
        let forwards = tracked("a", &["x", "y", "z"], 1);
        let backwards = tracked("a", &["z", "y", "x"], 1);
        assert_eq!(fingerprint_of(&forwards), fingerprint_of(&backwards));
        assert_ne!(
            fingerprint_of(&forwards),
            fingerprint_of(&tracked("a", &["x", "y"], 1))
        );
        assert_ne!(
            fingerprint_of(&forwards),
            fingerprint_of(&tracked("b", &["x", "y", "z"], 1))
        );
    }

    #[test]
    fn fingerprint_derives_on_enums_and_tuple_structs() {
        #[derive(Fingerprint)]
        enum Shape {
            Empty,
            Circle(u32),
            Rect { w: u32, h: u32 },
        }

        #[derive(Fingerprint)]
        struct Pair(u8, bool);

        assert_ne!(
            fingerprint_of(&Shape::Empty),
            fingerprint_of(&Shape::Circle(0))
        );
        // Same payload, different variant, so the discriminant has to count.
        assert_ne!(
            fingerprint_of(&Shape::Circle(1)),
            fingerprint_of(&Shape::Rect { w: 1, h: 0 })
        );
        assert_eq!(
            fingerprint_of(&Shape::Rect { w: 2, h: 3 }),
            fingerprint_of(&Shape::Rect { w: 2, h: 3 })
        );
        assert_ne!(
            fingerprint_of(&Pair(1, true)),
            fingerprint_of(&Pair(1, false))
        );
    }

    #[test]
    fn fingerprint_is_stable_across_runs() {
        // Pinned literals: if these ever change, every deployed sender and
        // receiver disagree until both are rebuilt.
        assert_eq!(fingerprint_of(&0u8), 0xaf63bd4c8601b7df);
        assert_eq!(fingerprint_of(&true), 0xaf63bc4c8601b62c);
        assert_eq!(fingerprint_of(&"delta"), 0x3035df3ae9e50ee6);
    }

    #[test]
    fn versioned_round_trips() {
        let mut sender = Versioned::new(tracked("a", &["x"], 1));
        let mut receiver = Versioned::new(tracked("a", &["x"], 1));

        let message = sender.commit(tracked("a", &["x", "y"], 2)).unwrap();
        assert_eq!((message.from, message.to), (0, 1));
        assert_eq!(receiver.apply(message), Ok(Applied::Updated));
        assert_eq!(receiver.get(), sender.get());
        assert_eq!(receiver.version(), sender.version());
    }

    #[test]
    fn versioned_no_change_burns_nothing() {
        let mut sender = Versioned::new(tracked("a", &["x"], 1));
        assert!(sender.commit(tracked("a", &["x"], 1)).is_none());
        assert_eq!(sender.version(), 0);
    }

    #[test]
    fn versioned_ignores_a_replayed_delta() {
        let mut sender = Versioned::new(tracked("a", &["x"], 1));
        let mut receiver = Versioned::new(tracked("a", &["x"], 1));

        let message = sender.commit(tracked("a", &["x", "y"], 2)).unwrap();
        assert_eq!(receiver.apply(message.clone()), Ok(Applied::Updated));
        // Duplicate delivery is a no-op rather than a corruption.
        assert_eq!(receiver.apply(message), Ok(Applied::Stale));
        assert_eq!(receiver.get(), sender.get());
    }

    #[test]
    fn versioned_catches_a_dropped_delta() {
        let mut sender = Versioned::new(tracked("a", &["x"], 1));
        let mut receiver = Versioned::new(tracked("a", &["x"], 1));

        let _lost = sender.commit(tracked("a", &["x", "y"], 2)).unwrap();
        let second = sender.commit(tracked("a", &["x", "y"], 3)).unwrap();

        assert_eq!(
            receiver.apply(second),
            Err(Mismatch::Gap {
                expected: 0,
                found: 1
            })
        );
        // A rejected delta leaves the receiver untouched.
        assert_eq!(receiver.version(), 0);
        assert_eq!(receiver.get(), &tracked("a", &["x"], 1));
    }

    #[test]
    fn versioned_catches_drift_from_outside_the_stream() {
        let mut sender = Versioned::new(tracked("a", &["x"], 1));
        // The receiver starts at the right version but the wrong contents,
        // which no sequence number could notice.
        let mut receiver = Versioned::new(tracked("a", &["tampered"], 1));

        let message = sender.commit(tracked("a", &["x", "y"], 2)).unwrap();
        match receiver.apply(message) {
            Err(Mismatch::Base { expected, found }) => assert_ne!(expected, found),
            other => panic!("expected a base mismatch, got {:?}", other),
        }
        assert_eq!(receiver.version(), 0);
    }

    #[test]
    fn versioned_resync_recovers() {
        // The documented answer to any `Mismatch`: send the whole thing.
        let mut sender = Versioned::new(tracked("a", &["x"], 1));
        let mut receiver = Versioned::new(tracked("a", &["wrong"], 1));

        let message = sender.commit(tracked("a", &["x", "y"], 2)).unwrap();
        assert!(receiver.apply(message).is_err());

        receiver = sender.clone();
        let next = sender.commit(tracked("a", &["x", "y"], 3)).unwrap();
        assert_eq!(receiver.apply(next), Ok(Applied::Updated));
        assert_eq!(receiver.get(), sender.get());
    }

    #[test]
    fn versioned_catches_a_wrong_result() {
        // A hand-built delta whose `result` does not describe what applying it
        // actually does — the case only the second fingerprint can catch.
        let mut sender = Versioned::new(tracked("a", &["x"], 1));
        let mut receiver = Versioned::new(tracked("a", &["x"], 1));

        let mut message = sender.commit(tracked("a", &["x"], 2)).unwrap();
        message.result ^= 1;

        match receiver.apply(message) {
            Err(Mismatch::Result { expected, found }) => assert_ne!(expected, found),
            other => panic!("expected a result mismatch, got {:?}", other),
        }
        // Version not advanced, so the corruption cannot be mistaken for
        // healthy state by the next delta either.
        assert_eq!(receiver.version(), 0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn versioned_delta_serializes() {
        #[derive(Clone, Delta, Fingerprint)]
        #[delta_struct(delta_leader = "#[derive(serde::Serialize, serde::Deserialize)]")]
        struct Config {
            port: u16,
        }

        let mut sender = Versioned::new(Config { port: 80 });
        let mut receiver = Versioned::new(Config { port: 80 });

        let payload =
            serde_json::to_string(&sender.commit(Config { port: 8080 }).unwrap()).unwrap();
        let message: VersionedDelta<ConfigDelta> = serde_json::from_str(&payload).unwrap();
        assert_eq!(receiver.apply(message), Ok(Applied::Updated));
        assert_eq!(receiver.get().port, 8080);
    }

    #[test]
    fn bounded_generics() {
        let delta = Delta::delta(
            InlineBoundGeneric { foo: 1, bar: false },
            InlineBoundGeneric { foo: 2, bar: false },
        )
        .unwrap();
        assert_eq!(delta.foo, Some(2));
        assert_eq!(delta.bar, None);

        let delta = Delta::delta(
            WhereClauseGeneric { foo: 1, bar: false },
            WhereClauseGeneric { foo: 2, bar: false },
        )
        .unwrap();
        assert_eq!(delta.foo, Some(2));
        assert_eq!(delta.bar, None);
    }

    #[test]
    fn bounded_generics_with_delta_field() {
        let delta = Delta::delta(
            InlineBoundDeltaField { foo: NewType(1) },
            InlineBoundDeltaField { foo: NewType(2) },
        )
        .unwrap();
        assert_eq!(delta.foo.unwrap().0, Some(2));

        let mut applied = WhereClauseDeltaField { foo: NewType(1) };
        let delta = Delta::delta(
            WhereClauseDeltaField { foo: NewType(1) },
            WhereClauseDeltaField { foo: NewType(2) },
        )
        .unwrap();
        applied.apply_delta(delta);
        assert_eq!(applied.foo, NewType(2));
    }

    #[test]
    fn apply_delta_all_field_types() {
        let old = AllFieldTypes {
            scalar: 1,
            delta: NewType(3),
            unordered: vec![1, 2, 3].into_iter().collect(),
        };
        let new = AllFieldTypes {
            scalar: 2,
            delta: NewType(4),
            unordered: vec![3, 4, 5].into_iter().collect(),
        };
        let new_clone = new.clone();
        let mut old_delta_applied = old.clone();
        let delta = Delta::delta(old, new);
        old_delta_applied.apply_delta(delta.unwrap());
        assert_eq!(new_clone, old_delta_applied);
    }
}
