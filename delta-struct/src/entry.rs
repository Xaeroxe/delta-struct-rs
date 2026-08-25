//! Keyed membership diffing, behind the `unordered` field type when the field
//! is a map.
//!
//! A map is a bag of entries, but a bag with a rule that the plain
//! [`bag`](crate::bag) knows nothing about: no two entries share a key. That
//! rule is what makes an [`EntryDelta`] the right shape for one — a bare key
//! is enough to say an entry left, and a value is enough to say what a key
//! holds now, because putting one back cannot leave the old one standing
//! beside it.
//!
//! Values are compared with `==` and replaced wholesale, which is the whole of
//! what separates this from [`map`](crate::map): reach for `unordered-delta`
//! and a [`MapDelta`](crate::MapDelta) when the values are big enough to be
//! worth diffing and implement [`Delta`](crate::Delta) so they can be.
//!
//! The derive emits calls to [`diff`] and [`apply`]; you only need this module
//! directly to inspect or construct a delta by hand.

use crate::{MapEntry, TryIndex};

/// A membership diff between two collections of key/value entries.
///
/// The two parts are asymmetric, and the map's one-value-per-key rule is what
/// makes them so: [`add`] carries whole entries because the receiver needs to
/// be told the value, while [`remove`] carries bare keys because a key names
/// an entry on its own.
///
/// A key that survived with a new value under it is an [`add`], not a removal
/// followed by one. Applying an addition overwrites whatever the key held, so
/// the removal would say nothing the addition does not already say.
///
/// Nothing here records position — see [`SeqDelta`](crate::SeqDelta) for that.
///
/// [`add`]: EntryDelta::add
/// [`remove`]: EntryDelta::remove
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryDelta<K, V> {
    /// What a key holds in the new collection that it did not hold in the old
    /// one — both keys that arrived and keys whose value moved.
    pub add: Vec<(K, V)>,
    /// Keys that were in the old collection but are not in the new one.
    pub remove: Vec<K>,
}

impl<K, V> EntryDelta<K, V> {
    /// Whether the two collections held the same entries, and so nothing needs
    /// sending.
    pub fn is_empty(&self) -> bool {
        self.add.is_empty() && self.remove.is_empty()
    }
}

impl<K, V> Default for EntryDelta<K, V> {
    fn default() -> Self {
        EntryDelta {
            add: Vec::new(),
            remove: Vec::new(),
        }
    }
}

/// Pairs the entries of `old` and `new` by key and records what the keys that
/// survived hold now.
///
/// Returns an empty [`EntryDelta`] when the two hold the same entries. Each
/// key of `old` is looked up in `new` exactly once, through the collection's
/// own [`TryIndex`] implementation, so the cost is that of n lookups: O(n) for
/// a [`HashMap`](std::collections::HashMap), O(n log n) for a
/// [`BTreeMap`](std::collections::BTreeMap).
///
/// ```
/// use delta_struct::entry::diff;
/// use std::collections::BTreeMap;
///
/// let labels = |tier: &'static str| {
///     vec![("tier", tier)].into_iter().collect::<BTreeMap<&str, &str>>()
/// };
///
/// let delta = diff(labels("web"), labels("edge"));
/// // The key survived, so only what it holds now travels — the old value
/// // stays where it is, on the receiver.
/// assert_eq!(delta.add, vec![("tier", "edge")]);
/// assert!(delta.remove.is_empty());
/// ```
pub fn diff<C, E>(old: C, mut new: C) -> EntryDelta<E::Key, E::Value>
where
    C: IntoIterator<Item = E> + TryIndex<E::Key, Output = E::Value>,
    E: MapEntry,
    E::Value: PartialEq,
{
    // Take each of `old`'s entries out of `new` as it is matched, so whatever
    // is still standing at the end is exactly the keys that arrived — and can
    // join the changed ones in `add`, since applying either means the same
    // thing to the receiver.
    let mut add = Vec::new();
    let mut remove = Vec::new();
    for entry in old {
        let (key, old_value) = entry.into_parts();
        match new.try_remove(&key) {
            Some(new_value) => {
                if new_value != old_value {
                    add.push((key, new_value));
                }
            }
            None => remove.push(key),
        }
    }
    add.extend(new.into_iter().map(MapEntry::into_parts));
    EntryDelta { add, remove }
}

/// Applies a keyed membership diff to `target` in place.
///
/// Each removal is a single lookup rather than a scan, so this costs the same
/// as [`diff`] does. Membership is preserved but position is not — additions
/// land wherever the collection decides to put them. Use `ordered` where that
/// matters.
///
/// A removal naming a key that `target` does not have is ignored, and an
/// addition overwrites whatever the key held, which together make applying the
/// same delta twice harmless.
///
/// ```
/// use delta_struct::entry::{apply, diff};
/// use std::collections::BTreeMap;
///
/// let labels = |entries: Vec<(&'static str, &'static str)>| {
///     entries.into_iter().collect::<BTreeMap<&str, &str>>()
/// };
///
/// let delta = diff(
///     labels(vec![("tier", "web"), ("zone", "a")]),
///     labels(vec![("tier", "edge")]),
/// );
/// let mut target = labels(vec![("tier", "web"), ("zone", "a")]);
/// apply(&mut target, delta);
/// assert_eq!(target, labels(vec![("tier", "edge")]));
/// ```
pub fn apply<C, E>(target: &mut C, delta: EntryDelta<E::Key, E::Value>)
where
    C: IntoIterator<Item = E> + Extend<E> + TryIndex<E::Key, Output = E::Value>,
    E: MapEntry,
{
    for key in delta.remove {
        target.try_remove(&key);
    }
    target.extend(
        delta
            .add
            .into_iter()
            .map(|(key, value)| E::from_parts(key, value)),
    );
}
