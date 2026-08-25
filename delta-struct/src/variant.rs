//! The delta of a value that might have changed shape, and the failure that
//! comes with it.
//!
//! A struct's delta always fits the value it is applied to: the fields are the
//! fields, and nothing about them can disagree. An enum's cannot. Two values
//! of the same enum may be in different variants, in which case there is no
//! difference to describe — only a replacement — and a delta built for one
//! variant may arrive at a value sitting in another.
//!
//! [`EnumDelta`] is the first half of that: one arm for "same variant, here is
//! what changed inside it" and one for "different variant, here is the whole
//! thing". [`Mismatch`] is the second: what [`Delta::apply_delta`] returns when
//! the two disagree.
//!
//! [`Delta::apply_delta`]: crate::Delta::apply_delta

use std::fmt;

/// The delta of an enum: a change *within* a variant, or a change *of*
/// variant.
///
/// `T` is the source enum and `D` its generated `{Self}Delta`, so deriving
/// [`Delta`](crate::Delta) on `enum Shape` gives
/// `Output = EnumDelta<Shape, ShapeDelta>`. The wrapper is a type in this
/// crate rather than an arm of the generated enum so that [`Became`] cannot
/// collide with a variant you wrote.
///
/// Note that [`Became`] carries the source value whole. There is nothing
/// smaller it could carry — the receiver holds a different variant, so it
/// shares none of the new one — which is also why serializing an enum's delta
/// needs the enum itself to be serializable.
///
/// [`Became`]: EnumDelta::Became
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnumDelta<T, D> {
    /// The value moved to a different variant, so here it is in full.
    Became(T),
    /// The value stayed in its variant; this is what changed inside it.
    Delta(D),
}

/// A delta that does not fit the value it was applied to.
///
/// Only enums can produce this — a delta built while the value was one variant
/// arriving at a value that is now another — which means the two sides have
/// already diverged. A struct's `apply_delta` never fails, and neither does
/// one for an enum whose value is in the expected variant.
///
/// Nested deltas propagate the innermost mismatch rather than wrapping it, so
/// the type and variant named here are the ones that actually disagreed, not
/// the outermost thing being applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mismatch {
    /// The enum whose delta did not fit, as written in its definition.
    pub type_name: &'static str,
    /// The variant the delta was computed for.
    pub expected: &'static str,
    /// The variant the value was actually in.
    pub found: &'static str,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "delta does not fit: it was computed for {}::{}, but the value is {}::{}",
            self.type_name, self.expected, self.type_name, self.found
        )
    }
}

impl std::error::Error for Mismatch {}
