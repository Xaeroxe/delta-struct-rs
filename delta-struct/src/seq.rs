//! Positional diffing, behind the `ordered` field type.
//!
//! A field marked `#[delta_struct(field_type = "ordered")]` is diffed with
//! Myers' algorithm and represented as a [`SeqDelta`] — a minimal edit script
//! rather than the membership-only add/remove pair that `unordered` produces.
//! Reach for it when a field's order carries meaning and you would rather send
//! two splices than the whole sequence.
//!
//! The derive emits calls to [`diff`] and [`apply`]; you only need this module
//! directly to inspect or construct a delta by hand.

use similar::algorithms::{myers, DiffHook, Replace};
use std::hash::Hash;
use std::iter::FromIterator;
use std::ops::Range;

/// A positional diff between two sequences: an ordered edit script.
///
/// Splices are sorted by [`Splice::at`] and never overlap, and every `at`
/// indexes the *old* sequence rather than the partially-rebuilt one. Holding
/// to old coordinates is what lets [`apply`] run in a single forward pass
/// without the index-shifting hazards an edit script usually brings.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeqDelta<T> {
    /// The edit script, in ascending order of [`Splice::at`].
    pub splices: Vec<Splice<T>>,
}

impl<T> SeqDelta<T> {
    /// Whether the two sequences were identical, and so nothing needs sending.
    pub fn is_empty(&self) -> bool {
        self.splices.is_empty()
    }
}

impl<T> Default for SeqDelta<T> {
    fn default() -> Self {
        SeqDelta {
            splices: Vec::new(),
        }
    }
}

/// One edit: drop `remove` items starting at `at`, then put `insert` in their
/// place.
///
/// A pure insertion has `remove == 0`, a pure deletion has an empty `insert`,
/// and `at` counts positions in the old sequence.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Splice<T> {
    /// Where the edit starts, as an index into the old sequence.
    pub at: usize,
    /// How many old items the edit drops.
    pub remove: usize,
    /// The items to put in their place.
    pub insert: Vec<T>,
}

/// Records the edit script as index ranges, so nothing is cloned or owned
/// until [`diff`] materializes the inserts from `new`.
#[derive(Default)]
struct RangeHook {
    ops: Vec<(usize, usize, Range<usize>)>,
}

impl DiffHook for RangeHook {
    type Error = std::convert::Infallible;

    fn equal(
        &mut self,
        _old_index: usize,
        _new_index: usize,
        _len: usize,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn delete(
        &mut self,
        old_index: usize,
        old_len: usize,
        new_index: usize,
    ) -> Result<(), Self::Error> {
        self.ops.push((old_index, old_len, new_index..new_index));
        Ok(())
    }

    fn insert(
        &mut self,
        old_index: usize,
        new_index: usize,
        new_len: usize,
    ) -> Result<(), Self::Error> {
        self.ops
            .push((old_index, 0, new_index..new_index + new_len));
        Ok(())
    }

    fn replace(
        &mut self,
        old_index: usize,
        old_len: usize,
        new_index: usize,
        new_len: usize,
    ) -> Result<(), Self::Error> {
        self.ops
            .push((old_index, old_len, new_index..new_index + new_len));
        Ok(())
    }
}

/// Computes a minimal edit script turning `old` into `new`.
///
/// Returns an empty [`SeqDelta`] when the two are identical. Items need
/// `Hash + Eq` rather than the `PartialEq` an `unordered` field asks for —
/// that is what Myers' implementation in `similar` requires to index the
/// sequences, and it is why a sequence of floats cannot be an `ordered` field.
///
/// ```
/// use delta_struct::seq::{diff, Splice};
///
/// let delta = diff(vec![1, 2, 3, 4], vec![1, 9, 3, 4]);
/// assert_eq!(
///     delta.splices,
///     vec![Splice { at: 1, remove: 1, insert: vec![9] }],
/// );
/// ```
pub fn diff<C, I>(old: C, new: C) -> SeqDelta<I>
where
    C: IntoIterator<Item = I>,
    I: Hash + Eq,
{
    let old: Vec<I> = old.into_iter().collect();
    let new: Vec<I> = new.into_iter().collect();

    // `Replace` coalesces an adjacent delete and insert into the single
    // `replace` call that maps onto one splice.
    let mut hook = Replace::new(RangeHook::default());
    match myers::diff(&mut hook, &old[..], 0..old.len(), &new[..], 0..new.len()) {
        Ok(()) => {}
        Err(never) => match never {},
    }
    let ops = hook.into_inner().ops;

    // The recorded ranges are ascending and non-overlapping in new coordinates
    // too, so the inserted items can be pulled out of `new` in one pass rather
    // than indexed out (which would demand `Clone`).
    let mut new = new.into_iter();
    let mut cursor = 0;
    let splices = ops
        .into_iter()
        .map(|(at, remove, range)| {
            new.by_ref().take(range.start - cursor).for_each(drop);
            let insert = new.by_ref().take(range.end - range.start).collect();
            cursor = range.end;
            Splice { at, remove, insert }
        })
        .collect();

    SeqDelta { splices }
}

/// Applies an edit script to `target` in place.
///
/// Because splice positions are old-coordinates and monotonically increasing,
/// this walks the old sequence once and rebuilds it; no random access, and no
/// index arithmetic that could drift as edits land.
///
/// A [`SeqDelta`] produced by [`diff`] always upholds the sorted,
/// non-overlapping invariant. One that was hand-built or arrived over a wire
/// might not, so out-of-order or overlong splices are clamped rather than
/// allowed to panic; the result in that case is unspecified but the call
/// still returns.
///
/// ```
/// use delta_struct::seq::{apply, diff};
///
/// let delta = diff(vec![1, 2, 3, 4], vec![1, 9, 3, 4]);
/// let mut target = vec![1, 2, 3, 4];
/// apply(&mut target, delta);
/// assert_eq!(target, vec![1, 9, 3, 4]);
/// ```
pub fn apply<C, I>(target: &mut C, delta: SeqDelta<I>)
where
    C: IntoIterator<Item = I> + FromIterator<I>,
{
    let old = std::mem::replace(target, std::iter::empty().collect());
    let mut old = old.into_iter();
    let mut out: Vec<I> = Vec::new();
    let mut cursor = 0;
    for Splice { at, remove, insert } in delta.splices {
        out.extend(old.by_ref().take(at.saturating_sub(cursor)));
        old.by_ref().take(remove).for_each(drop);
        out.extend(insert);
        cursor = at + remove;
    }
    out.extend(old);
    *target = out.into_iter().collect();
}
