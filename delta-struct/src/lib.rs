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
//! The field is treated as a bag of items whose order carries no meaning, so
//! the delta records only which items came and went. One field becomes two:
//! `{field}_add` and `{field}_remove`, both `Vec<Item>`.
//!
//! ```
//! use delta_struct::Delta;
//!
//! #[derive(Delta)]
//! struct Device {
//!     #[delta_struct(field_type = "unordered")]
//!     services: Vec<String>,
//! }
//!
//! let old = Device { services: vec!["ssh".to_string(), "http".to_string()] };
//! let new = Device { services: vec!["http".to_string(), "mqtt".to_string()] };
//!
//! let delta = Delta::delta(old, new).unwrap();
//! assert_eq!(delta.services_add, vec!["mqtt".to_string()]);
//! assert_eq!(delta.services_remove, vec!["ssh".to_string()]);
//! ```
//!
//! Duplicates are counted rather than deduplicated: diffing `[3, 3]` against
//! `[3]` removes a single `3`. Any collection works as long as it satisfies
//! `IntoIterator` (to compute the delta) plus `FromIterator` and `Extend` (to
//! apply one) and its items are `PartialEq` — [`Vec`] and
//! [`HashSet`](std::collections::HashSet) both qualify.
//!
//! Because matching items are found by linear search, computing a delta over
//! an `unordered` field costs O(n * m) comparisons. That is fine for the
//! handful-of-elements collections this is aimed at, and something to keep in
//! mind for large ones.
//!
//! `apply_delta` removes then appends, so for an ordered collection like
//! [`Vec`] the result is a permutation of the new value, not necessarily the
//! new value itself. Use `ordered` where that matters.
//!
//! ## `ordered`
//!
//! The field is diffed positionally with Myers' algorithm, and the delta is a
//! minimal edit script: a [`SeqDelta`] holding [`Splice`]s that each say
//! "at this index, drop this many items and put these in their place". Unlike
//! `unordered`, one source field stays one delta field.
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
//! **`Hash + Eq`** — a stricter bar than the `PartialEq` the other field types
//! ask for, because that is what indexing the sequences for Myers requires.
//! The practical consequence is that a `Vec<f64>` cannot be an `ordered`
//! field, though it can be `unordered`.
//!
//! Turn on the `serde` feature to get `Serialize` and `Deserialize` on
//! [`SeqDelta`] and [`Splice`]; without it, a delta struct containing an
//! `ordered` field cannot derive them.
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
//!
//! #[derive(Delta)]
//! #[delta_struct(
//!     default = "unordered",
//!     delta_leader = "/// The changes to a `Tags`.\n#[derive(Debug)]"
//! )]
//! struct Tags {
//!     labels: Vec<String>,
//!     // Opt an individual field back out of the container default.
//!     #[delta_struct(field_type = "scalar")]
//!     revision: u32,
//! }
//!
//! let old = Tags { labels: vec![], revision: 1 };
//! let new = Tags { labels: vec!["new".to_string()], revision: 2 };
//! let delta = Delta::delta(old, new).unwrap();
//! assert_eq!(format!("{:?}", delta.labels_add), r#"["new"]"#);
//! assert_eq!(delta.revision, Some(2));
//! ```
//!
//! `delta_leader` also works on individual fields, where it decorates the
//! generated field instead of the generated struct. On an `unordered` field it
//! is emitted above *both* the `_add` and the `_remove` field, so write it to
//! read sensibly on each.
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
//! For everything but `ordered` fields there is no serde integration to
//! enable; `delta_leader` is the whole story. Put the derives on the generated
//! struct and it serializes like anything else:
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
//! Tuple structs are supported; their delta fields are named `field_0`,
//! `field_1`, and so on, since tuple-struct syntax has nowhere to hang the
//! `_add`/`_remove` pairs an `unordered` field needs.
//!
//! ```
//! use delta_struct::Delta;
//!
//! #[derive(Delta)]
//! struct Meters(i32);
//!
//! let delta = Delta::delta(Meters(3), Meters(4)).unwrap();
//! assert_eq!(delta.field_0, Some(4));
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

#![warn(missing_docs)]

// The derive emits `::delta_struct::…` paths for the runtime items an
// `ordered` field needs. That path has to resolve inside this crate too, or
// the crate's own tests could not use its own derive.
extern crate self as delta_struct;

pub mod seq;

pub use delta_struct_macros::Delta;
pub use seq::{SeqDelta, Splice};

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

    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct UnitType;

    #[derive(Delta, Clone, Debug, PartialEq, Eq)]
    #[delta_struct(delta_leader = "#[derive(Clone, Debug, PartialEq, Eq)]")]
    struct NewType(i32);

    #[derive(Delta)]
    #[allow(dead_code)] // The derive is itself the test
    struct NewTypeWithGeneric<T>(T);

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
    struct SimpleCollectionWithGeneric<T> {
        #[delta_struct(
            field_type = "unordered",
            delta_leader = "/// This the foo type on the delta struct."
        )]
        foo: Vec<T>,
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
        baz: Vec<i32>,
    }

    #[derive(Delta, Clone, Debug, PartialEq, Eq)]
    struct AllFieldTypes {
        #[delta_struct(field_type = "scalar")]
        scalar: i32,
        #[delta_struct(field_type = "delta")]
        delta: NewType,
        #[delta_struct(field_type = "unordered")]
        unordered: Vec<i32>,
    }

    #[derive(Clone, Debug, Delta, PartialEq)]
    #[allow(dead_code)] // The derive is itself the test
    struct DeviceConfig {
        #[delta_struct(field_type = "unordered")]
        pub services: Vec<String>,
        #[delta_struct(field_type = "unordered")]
        pub settings: Vec<String>,
        pub thumbnail_request: i32,
        pub speedtest_request: i32,
        #[delta_struct(field_type = "delta")]
        pub features: AllFieldTypes,
        pub deprovision: bool,
    }

    #[test]
    fn unordered_with_scalar() {
        let old = SimpleCollectionWithGeneric {
            foo: vec![1, 2, 3],
            bar: false,
        };
        let new = SimpleCollectionWithGeneric {
            foo: vec![3, 4, 5],
            bar: true,
        };
        let delta = Delta::delta(old, new).unwrap();
        assert_eq!(delta.foo_add, vec![4, 5]);
        assert_eq!(delta.foo_remove, vec![1, 2]);
        assert_eq!(delta.bar, Some(true));
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
        // TODO: Use assert_eq when we build out delta struct
        // attributes.
        if let Some(NewTypeDelta { field_0: Some(6) }) = delta.foo {
            // Do nothing, this is the pass case.
        } else {
            panic!();
        }
        assert_eq!(delta.bar, Some(true));
    }

    #[test]
    fn default_type_respected() {
        let old = AttributeTest {
            foo: 5,
            bar: 4,
            baz: vec![],
        };
        let new = AttributeTest {
            foo: 5,
            bar: 4,
            baz: vec![9, 4, 5],
        };
        let delta = Delta::delta(old, new).unwrap();
        assert!(delta.foo.is_none());
        assert!(delta.bar.is_none());
        assert_eq!(delta.baz_add, vec![9, 4, 5]);
        assert_eq!(delta.baz_remove, Vec::<i32>::new());
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
        assert_eq!(delta.foo.unwrap().field_0, Some(2));

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
            unordered: vec![1, 2, 3, 3],
        };
        let new = AllFieldTypes {
            scalar: 2,
            delta: NewType(4),
            unordered: vec![3, 4, 5],
        };
        let new_clone = new.clone();
        let mut old_delta_applied = old.clone();
        let delta = Delta::delta(old, new);
        old_delta_applied.apply_delta(delta.unwrap());
        assert_eq!(new_clone, old_delta_applied);
    }
}
