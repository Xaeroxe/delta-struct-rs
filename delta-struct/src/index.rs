//! Fallible lookup, the operation the `unordered` field types diff through.
//!
//! [`std::ops::Index`] cannot express a diff: it panics when the element is
//! absent, and "is this element on the other side?" is the only question a
//! membership diff asks. [`TryIndex`] is that trait with the answer made
//! fallible, which is what lets [`bag`](crate::bag) and [`map`](crate::map)
//! push the lookup down into the collection instead of scanning it.
//!
//! The consequence is that a field's cost is the cost of the collection you
//! picked: a diff over a [`HashSet`] or [`HashMap`] is O(n), and over a
//! [`BTreeSet`] or [`BTreeMap`] it is O(n log n).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{BuildHasher, Hash};

/// A collection that can look an element up by key without panicking when it
/// is not there.
///
/// This is [`std::ops::Index`] with the failure made visible, plus the owning
/// counterpart that the index operator has no equivalent of.
pub trait TryIndex<Idx: ?Sized> {
    /// What is stored under a key: the value for a map, and the element
    /// itself for a set.
    type Output;

    /// Borrows the element stored under `index`, or [`None`] if there is none.
    fn try_index(&self, index: &Idx) -> Option<&Self::Output>;

    /// Removes the element stored under `index` and returns it, or [`None`] if
    /// there was none.
    fn try_remove(&mut self, index: &Idx) -> Option<Self::Output>;
}

/// A [`TryIndex`] whose elements can also be modified in place.
pub trait TryIndexMut<Idx: ?Sized>: TryIndex<Idx> {
    /// Mutably borrows the element stored under `index`, or [`None`] if there
    /// is none.
    fn try_index_mut(&mut self, index: &Idx) -> Option<&mut Self::Output>;
}

impl<K, V, S> TryIndex<K> for HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Output = V;

    fn try_index(&self, index: &K) -> Option<&V> {
        self.get(index)
    }

    fn try_remove(&mut self, index: &K) -> Option<V> {
        self.remove(index)
    }
}

impl<K, V, S> TryIndexMut<K> for HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn try_index_mut(&mut self, index: &K) -> Option<&mut V> {
        self.get_mut(index)
    }
}

impl<K, V> TryIndex<K> for BTreeMap<K, V>
where
    K: Ord,
{
    type Output = V;

    fn try_index(&self, index: &K) -> Option<&V> {
        self.get(index)
    }

    fn try_remove(&mut self, index: &K) -> Option<V> {
        self.remove(index)
    }
}

impl<K, V> TryIndexMut<K> for BTreeMap<K, V>
where
    K: Ord,
{
    fn try_index_mut(&mut self, index: &K) -> Option<&mut V> {
        self.get_mut(index)
    }
}

impl<T, S> TryIndex<T> for HashSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    type Output = T;

    fn try_index(&self, index: &T) -> Option<&T> {
        self.get(index)
    }

    fn try_remove(&mut self, index: &T) -> Option<T> {
        self.take(index)
    }
}

impl<T> TryIndex<T> for BTreeSet<T>
where
    T: Ord,
{
    type Output = T;

    fn try_index(&self, index: &T) -> Option<&T> {
        self.get(index)
    }

    fn try_remove(&mut self, index: &T) -> Option<T> {
        self.take(index)
    }
}
