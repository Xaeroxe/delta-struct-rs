//! Membership diffing, behind the `unordered` field type.
//!
//! A field marked `#[delta_struct(field_type = "unordered")]` is treated as a
//! bag of elements whose order carries no meaning, and represented as a
//! [`BagDelta`] — which elements came and which went, and nothing about where
//! they sit.
//!
//! The derive emits calls to [`diff`] and [`apply`]; you only need this module
//! directly to inspect or construct a delta by hand.

use crate::TryIndex;

/// A membership diff between two collections: what arrived and what left.
///
/// Nothing here records position — see [`SeqDelta`](crate::SeqDelta) for that.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BagDelta<T> {
    /// Elements in the new collection that the old one did not have.
    pub add: Vec<T>,
    /// Elements in the old collection that the new one does not have.
    pub remove: Vec<T>,
}

impl<T> BagDelta<T> {
    /// Whether the two collections held the same elements, and so nothing
    /// needs sending.
    pub fn is_empty(&self) -> bool {
        self.add.is_empty() && self.remove.is_empty()
    }
}

impl<T> Default for BagDelta<T> {
    fn default() -> Self {
        BagDelta {
            add: Vec::new(),
            remove: Vec::new(),
        }
    }
}

/// Computes which elements `new` gained and which `old` lost.
///
/// Returns an empty [`BagDelta`] when the two hold the same elements. Each
/// element of `old` is looked up in `new` exactly once, through the
/// collection's own [`TryIndex`] implementation, so the cost is that of n
/// lookups: O(n) for a [`HashSet`](std::collections::HashSet), O(n log n) for
/// a [`BTreeSet`](std::collections::BTreeSet).
///
/// ```
/// use delta_struct::bag::diff;
/// use std::collections::BTreeSet;
///
/// let old: BTreeSet<i32> = vec![1, 2, 3].into_iter().collect();
/// let new: BTreeSet<i32> = vec![3, 4, 5].into_iter().collect();
///
/// let delta = diff(old, new);
/// assert_eq!(delta.add, vec![4, 5]);
/// assert_eq!(delta.remove, vec![1, 2]);
/// ```
pub fn diff<C, T>(old: C, mut new: C) -> BagDelta<T>
where
    C: IntoIterator<Item = T> + TryIndex<T, Output = T>,
{
    // Take each of `old`'s elements out of `new` as it is matched, so whatever
    // is still standing at the end is exactly what was added, and no second
    // pass is needed to work that out.
    let remove = old
        .into_iter()
        .filter(|element| new.try_remove(element).is_none())
        .collect();
    BagDelta {
        add: new.into_iter().collect(),
        remove,
    }
}

/// Applies a membership diff to `target` in place.
///
/// Each removal is a single lookup rather than a scan, so this costs the same
/// as [`diff`] does. Membership is preserved but position is not — additions
/// land wherever the collection decides to put them. Use `ordered` where that
/// matters.
///
/// A removal that `target` does not have is ignored, which makes applying the
/// same delta twice harmless.
///
/// ```
/// use delta_struct::bag::{apply, diff};
/// use std::collections::BTreeSet;
///
/// let set = |items: Vec<i32>| items.into_iter().collect::<BTreeSet<i32>>();
///
/// let delta = diff(set(vec![1, 2, 3]), set(vec![2, 3, 4]));
/// let mut target = set(vec![1, 2, 3]);
/// apply(&mut target, delta);
/// assert_eq!(target, set(vec![2, 3, 4]));
/// ```
pub fn apply<C, T>(target: &mut C, delta: BagDelta<T>)
where
    C: IntoIterator<Item = T> + Extend<T> + TryIndex<T, Output = T>,
{
    for element in delta.remove {
        target.try_remove(&element);
    }
    target.extend(delta.add);
}
