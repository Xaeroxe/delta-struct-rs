//! The bridge from a collection to the shape its membership diff takes.
//!
//! `#[delta_struct(field_type = "unordered")]` says "diff this by membership
//! and record nothing about order", and two kinds of collection answer that in
//! two different shapes. A set's answer is a [`BagDelta`]: these elements came,
//! these went. A map's answer is an [`EntryDelta`]: a bare key is enough to say
//! an entry left, because a map cannot hold the same key twice.
//!
//! [`Unordered`] is what lets the derive write the right one down without
//! knowing which it has. An `unordered` field's delta is declared as
//! `<T as Unordered>::Delta`, and the collection's impl picks the shape.
//!
//! Implement it for your own collection to make it eligible, delegating to
//! [`bag`], to [`entry`], or to something of your
//! own. A [`Vec`] deliberately has no impl — see the crate's
//! [Limitations](crate#limitations).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{BuildHasher, Hash};

use crate::{bag, entry, BagDelta, EntryDelta};

/// A collection that can be diffed by membership alone.
pub trait Unordered: Sized {
    /// The shape a membership diff of this collection takes.
    type Delta: Default;

    /// Computes what it would take to turn `old` into `new`, or [`None`] when
    /// the two hold the same elements.
    ///
    /// The [`None`] mirrors [`Delta::delta`](crate::Delta::delta): it is how a
    /// field says it has nothing to contribute, so that a struct whose fields
    /// all say so produces no delta at all.
    fn diff(old: Self, new: Self) -> Option<Self::Delta>;

    /// Applies a membership diff in place.
    ///
    /// Membership is preserved but position is not — additions land wherever
    /// the collection decides to put them. Unlike
    /// [`map::apply`](crate::map::apply) this cannot fail, because it never
    /// recurses into [`Delta::apply_delta`](crate::Delta::apply_delta).
    fn apply(&mut self, delta: Self::Delta);
}

impl<T, S> Unordered for HashSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    type Delta = BagDelta<T>;

    fn diff(old: Self, new: Self) -> Option<BagDelta<T>> {
        let delta = bag::diff(old, new);
        if delta.is_empty() {
            None
        } else {
            Some(delta)
        }
    }

    fn apply(&mut self, delta: BagDelta<T>) {
        bag::apply(self, delta)
    }
}

impl<T> Unordered for BTreeSet<T>
where
    T: Ord,
{
    type Delta = BagDelta<T>;

    fn diff(old: Self, new: Self) -> Option<BagDelta<T>> {
        let delta = bag::diff(old, new);
        if delta.is_empty() {
            None
        } else {
            Some(delta)
        }
    }

    fn apply(&mut self, delta: BagDelta<T>) {
        bag::apply(self, delta)
    }
}

impl<K, V, S> Unordered for HashMap<K, V, S>
where
    K: Hash + Eq,
    V: PartialEq,
    S: BuildHasher,
{
    type Delta = EntryDelta<K, V>;

    fn diff(old: Self, new: Self) -> Option<EntryDelta<K, V>> {
        let delta = entry::diff(old, new);
        if delta.is_empty() {
            None
        } else {
            Some(delta)
        }
    }

    fn apply(&mut self, delta: EntryDelta<K, V>) {
        entry::apply(self, delta)
    }
}

impl<K, V> Unordered for BTreeMap<K, V>
where
    K: Ord,
    V: PartialEq,
{
    type Delta = EntryDelta<K, V>;

    fn diff(old: Self, new: Self) -> Option<EntryDelta<K, V>> {
        let delta = entry::diff(old, new);
        if delta.is_empty() {
            None
        } else {
            Some(delta)
        }
    }

    fn apply(&mut self, delta: EntryDelta<K, V>) {
        entry::apply(self, delta)
    }
}
