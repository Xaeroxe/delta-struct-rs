//! Keyed diffing, behind the `unordered-delta` field type.
//!
//! A field marked `#[delta_struct(field_type = "unordered-delta")]` is treated
//! as a bag of key/value entries and represented as a [`MapDelta`]. Entries
//! are paired up by key, and a pair whose value changed is recorded as a
//! [`KeyedDelta`] — the value's own [`Delta`] rather than a removal followed by
//! a re-send of the whole thing. Reach for it when a map's values are
//! themselves large structs that tend to change a field at a time.
//!
//! The derive emits calls to [`diff`] and [`apply`]; you only need this module
//! directly to inspect or construct a delta by hand.

use crate::{Delta, Mismatch, TryIndex, TryIndexMut};

/// An entry that splits into a key and a value.
///
/// This is what lets the derive talk about the `K` and the `V` of a
/// `HashMap<K, V>` when all it can name is the collection's
/// [`Item`](IntoIterator::Item). It is implemented for `(K, V)`, which is what
/// every std map iterates as; implement it yourself only if you have a map
/// whose entry type is not a tuple.
pub trait MapEntry {
    /// The part identifying the entry. Entries with equal keys are the same
    /// entry, and so are diffed against each other rather than swapped.
    type Key;
    /// The part that gets diffed.
    type Value;

    /// Splits the entry into its parts.
    fn into_parts(self) -> (Self::Key, Self::Value);

    /// Reassembles an entry from parts, so it can be put back into a
    /// collection.
    fn from_parts(key: Self::Key, value: Self::Value) -> Self;
}

impl<K, V> MapEntry for (K, V) {
    type Key = K;
    type Value = V;

    fn into_parts(self) -> (K, V) {
        self
    }

    fn from_parts(key: K, value: V) -> Self {
        (key, value)
    }
}

/// A keyed diff between two collections of entries.
///
/// The three parts are deliberately asymmetric, and that asymmetry is the
/// whole point of the field type: [`add`] carries whole entries because the
/// receiver has never seen them, while [`remove`] carries bare keys and
/// [`change`] carries deltas, because for those the receiver already holds the
/// rest.
///
/// Nothing here records position — see [`SeqDelta`](crate::SeqDelta) for that.
///
/// [`add`]: MapDelta::add
/// [`remove`]: MapDelta::remove
/// [`change`]: MapDelta::change
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapDelta<K, V, D> {
    /// Entries whose keys are in the new collection but not the old one.
    pub add: Vec<(K, V)>,
    /// Keys that were in the old collection but are not in the new one.
    pub remove: Vec<K>,
    /// Keys in both collections whose values differ, and how.
    pub change: Vec<KeyedDelta<K, D>>,
}

impl<K, V, D> MapDelta<K, V, D> {
    /// Whether the two collections held the same entries, and so nothing needs
    /// sending.
    pub fn is_empty(&self) -> bool {
        self.add.is_empty() && self.remove.is_empty() && self.change.is_empty()
    }
}

impl<K, V, D> Default for MapDelta<K, V, D> {
    fn default() -> Self {
        MapDelta {
            add: Vec::new(),
            remove: Vec::new(),
            change: Vec::new(),
        }
    }
}

/// A change to the value stored under one key.
///
/// Turn on the `serde` feature to get `Serialize` and `Deserialize` on this,
/// as a delta struct with an `unordered-delta` field cannot derive them
/// otherwise.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyedDelta<K, D> {
    /// The key whose value changed. It is present in both the old and the new
    /// collection — a key on only one side is an addition or a removal
    /// instead.
    pub key: K,
    /// What changed about the value, as produced by [`Delta::delta`].
    pub delta: D,
}

/// Pairs the entries of `old` and `new` by key and diffs the values that
/// survived.
///
/// A key present on both sides with an equal value produces nothing at all, so
/// the [`MapDelta`] is empty when the collections agree.
///
/// Each key of `old` is looked up in `new` exactly once, through the
/// collection's own [`TryIndex`] implementation, so the cost is that of n
/// lookups: O(n) for a [`HashMap`](std::collections::HashMap), O(n log n) for
/// a [`BTreeMap`](std::collections::BTreeMap).
///
/// ```
/// use delta_struct::{map, Delta, ScalarDelta};
/// use std::collections::BTreeMap;
///
/// #[derive(Delta)]
/// #[delta_struct(delta_leader = "#[derive(Debug, PartialEq)]")]
/// struct Service {
///     port: u16,
///     healthy: bool,
/// }
///
/// let services = |port| {
///     vec![("web", Service { port, healthy: true })]
///         .into_iter()
///         .collect::<BTreeMap<&str, Service>>()
/// };
///
/// let delta = map::diff(services(80), services(8080));
/// assert!(delta.add.is_empty() && delta.remove.is_empty());
/// assert_eq!(delta.change[0].key, "web");
/// assert_eq!(delta.change[0].delta.port, ScalarDelta::Changed(8080));
/// assert_eq!(delta.change[0].delta.healthy, ScalarDelta::Unchanged);
/// ```
pub fn diff<C, E>(old: C, mut new: C) -> MapDelta<E::Key, E::Value, <E::Value as Delta>::Output>
where
    C: IntoIterator<Item = E> + TryIndex<E::Key, Output = E::Value>,
    E: MapEntry,
    E::Value: Delta,
{
    // Take each of `old`'s entries out of `new` as it is matched, so whatever
    // is still standing at the end is exactly what was added. Taking rather
    // than borrowing is also what makes the values below owned, which is what
    // `Delta::delta` needs.
    let mut remove = Vec::new();
    let mut change = Vec::new();
    for entry in old {
        let (key, old_value) = entry.into_parts();
        match new.try_remove(&key) {
            Some(new_value) => {
                if let Some(delta) = Delta::delta(old_value, new_value) {
                    change.push(KeyedDelta { key, delta });
                }
            }
            None => remove.push(key),
        }
    }
    MapDelta {
        add: new.into_iter().map(MapEntry::into_parts).collect(),
        remove,
        change,
    }
}

/// Applies a keyed diff to `target` in place.
///
/// Removals and changes are single lookups rather than scans, so this costs
/// the same as [`diff`] does, and a changed value is updated where it sits
/// rather than taken out and put back. As with an `unordered` field,
/// membership is preserved but position is not.
///
/// A key in `remove` or `change` that `target` does not have is ignored.
///
/// This is the one collection helper that can fail, because it is the one that
/// recurses into [`Delta::apply_delta`]: a map of enums can be handed a change
/// whose value has since moved to another variant. The error propagates rather
/// than being swallowed, and leaves the map partly updated.
///
/// ```
/// use delta_struct::{map, Delta};
/// use std::collections::BTreeMap;
///
/// #[derive(Delta)]
/// struct Service {
///     port: u16,
/// }
///
/// let services = |port| {
///     vec![("web", Service { port })]
///         .into_iter()
///         .collect::<BTreeMap<&str, Service>>()
/// };
///
/// let delta = map::diff(services(80), services(8080));
/// let mut target = services(80);
/// map::apply(&mut target, delta)?;
/// assert_eq!(target["web"].port, 8080);
/// # Ok::<(), delta_struct::Mismatch>(())
/// ```
pub fn apply<C, E>(
    target: &mut C,
    delta: MapDelta<E::Key, E::Value, <E::Value as Delta>::Output>,
) -> Result<(), Mismatch>
where
    C: IntoIterator<Item = E> + Extend<E> + TryIndexMut<E::Key, Output = E::Value>,
    E: MapEntry,
    E::Value: Delta,
{
    let MapDelta {
        add,
        remove,
        change,
    } = delta;
    for key in remove {
        target.try_remove(&key);
    }
    for KeyedDelta { key, delta } in change {
        if let Some(value) = target.try_index_mut(&key) {
            value.apply_delta(delta)?;
        }
    }
    target.extend(
        add.into_iter()
            .map(|(key, value)| E::from_parts(key, value)),
    );
    Ok(())
}
