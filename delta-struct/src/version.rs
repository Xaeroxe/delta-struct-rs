//! Detecting a delta that is being applied to the wrong state.
//!
//! [`Delta::apply_delta`] assumes the value it is handed is equal to the `old`
//! the delta was computed from. Nothing checks that, so a dropped, reordered,
//! or replayed message silently diverges the two sides. [`Versioned`] wraps a
//! value with the bookkeeping to notice.
//!
//! This module is entirely opt-in. The [`Delta`] trait, the derive, and every
//! generated struct are unchanged by it — if you are diffing locally or over a
//! reliable transport, you never have to name anything in here.
//!
//! # What gets checked
//!
//! Each [`VersionedDelta`] carries four numbers, and they catch different
//! failures:
//!
//! - `from` and `to` — a sequence. A delta arriving out of order leaves a gap,
//!   and one arriving twice is recognised and ignored.
//! - `base` — the [`Fingerprint`] of the state the sender diffed against. This
//!   catches a receiver whose value drifted for any reason at all, including
//!   one that never came through this stream.
//! - `result` — the fingerprint the sender expects applying to produce. This
//!   catches the delta itself being wrong.
//!
//! ```
//! use delta_struct::{Applied, Delta, Fingerprint, Versioned};
//!
//! #[derive(Clone, Delta, Fingerprint)]
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
//! let message = sender.commit(config(8080)).expect("the port changed");
//! assert!(matches!(receiver.apply(message), Ok(Applied::Updated)));
//! assert_eq!(receiver.get().port, 8080);
//! ```

use crate::{fingerprint_of, Delta, Fingerprint, Mismatch};
use std::fmt;

/// A delta, plus everything needed to tell whether it belongs here.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionedDelta<D> {
    /// The version this delta was computed against.
    pub from: u64,
    /// The version applying it produces.
    pub to: u64,
    /// The [`Fingerprint`] of the state it was computed against.
    pub base: u64,
    /// The fingerprint applying it should produce.
    pub result: u64,
    /// The delta itself.
    pub delta: D,
}

/// A value and the version it is currently at.
///
/// Both ends of a connection hold one. The sender calls [`commit`], the
/// receiver calls [`apply`], and the version and fingerprints travel between
/// them inside a [`VersionedDelta`].
///
/// [`commit`]: Versioned::commit
/// [`apply`]: Versioned::apply
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Versioned<T> {
    value: T,
    version: u64,
}

impl<T> Versioned<T> {
    /// Starts a value at version 0.
    ///
    /// Both ends have to start from the same value; sending a `Versioned<T>`
    /// whole is how a receiver catches up after a [`Rejected`].
    pub fn new(value: T) -> Self {
        Versioned { value, version: 0 }
    }

    /// The version this value is at.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Borrows the value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Takes the value back out, discarding the version.
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T: Fingerprint> Versioned<T> {
    /// The fingerprint of the value as it stands.
    pub fn fingerprint(&self) -> u64 {
        fingerprint_of(&self.value)
    }
}

impl<T: Delta + Fingerprint + Clone> Versioned<T> {
    /// Moves to `new` and produces the delta that gets a peer here.
    ///
    /// Returns [`None`] when nothing changed, in which case the version does
    /// not advance either — an update that would do nothing costs no message
    /// and no number.
    ///
    /// `T: Clone` is needed because [`Delta::delta`] consumes both sides and
    /// the new value has to be kept as well as diffed.
    pub fn commit(&mut self, new: T) -> Option<VersionedDelta<T::Output>> {
        let base = fingerprint_of(&self.value);
        let result = fingerprint_of(&new);
        let old = std::mem::replace(&mut self.value, new.clone());
        Delta::delta(old, new).map(|delta| {
            let from = self.version;
            self.version += 1;
            VersionedDelta {
                from,
                to: self.version,
                base,
                result,
                delta,
            }
        })
    }
}

impl<T: Delta + Fingerprint> Versioned<T> {
    /// Applies a delta, or explains why it does not belong here.
    ///
    /// A delta that has already been applied is reported as
    /// [`Applied::Stale`] and does nothing, so duplicate delivery is safe. A
    /// [`Rejected`] leaves the version untouched, which means a later delta in
    /// the same stream will fail too rather than papering over the hole — the
    /// only way forward is to replace the whole value.
    pub fn apply(&mut self, delta: VersionedDelta<T::Output>) -> Result<Applied, Rejected> {
        if delta.to <= self.version {
            return Ok(Applied::Stale);
        }
        if delta.from != self.version {
            return Err(Rejected::Gap {
                expected: self.version,
                found: delta.from,
            });
        }
        let found = fingerprint_of(&self.value);
        if found != delta.base {
            return Err(Rejected::Base {
                expected: delta.base,
                found,
            });
        }
        // The base fingerprint matched, so the value is the one the delta was
        // built against and an enum's variants line up — this can only fire on
        // a fingerprint collision or a bug, which is worth being able to say.
        self.value
            .apply_delta(delta.delta)
            .map_err(Rejected::Apply)?;
        let found = fingerprint_of(&self.value);
        if found != delta.result {
            // The value is now wrong, and deliberately left that way: the
            // version has not advanced, so the next delta cannot be mistaken
            // for a clean apply.
            return Err(Rejected::Result {
                expected: delta.result,
                found,
            });
        }
        self.version = delta.to;
        Ok(Applied::Updated)
    }
}

/// What [`Versioned::apply`] did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Applied {
    /// The delta moved the value forward.
    Updated,
    /// The value already reflected this delta, so nothing was done.
    Stale,
}

/// Why a delta could not be applied.
///
/// Every variant means the same thing operationally — this receiver cannot
/// catch up from deltas and needs the whole value resent — but they say
/// different things about what went wrong.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rejected {
    /// A delta was missed: this one was computed against a version that was
    /// never reached. The stream lost or reordered a message.
    Gap {
        /// The version the receiver is at.
        expected: u64,
        /// The version the delta was computed against.
        found: u64,
    },
    /// The version lined up but the content did not, so the value drifted for
    /// a reason outside this stream — a direct mutation, a partly applied
    /// earlier delta, or two senders writing to one receiver.
    Base {
        /// The fingerprint the sender diffed against.
        expected: u64,
        /// The fingerprint the receiver actually holds.
        found: u64,
    },
    /// Applying the delta did not produce what the sender said it would. The
    /// two sides disagree about what the delta means: mismatched schema
    /// versions, or a bug.
    Result {
        /// The fingerprint the sender expected.
        expected: u64,
        /// The fingerprint applying actually produced.
        found: u64,
    },
    /// The delta did not fit the value's shape at all — an enum delta built
    /// for one variant meeting a value in another.
    ///
    /// The [`base`](VersionedDelta::base) fingerprint is checked first and
    /// would normally have caught that, so reaching this means a fingerprint
    /// collision or a bug rather than ordinary divergence.
    Apply(Mismatch),
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rejected::Gap { expected, found } => write!(
                f,
                "missed a delta: at version {}, but this one starts from {}",
                expected, found
            ),
            Rejected::Base { expected, found } => write!(
                f,
                "state has diverged: delta was computed against fingerprint {:#018x}, \
                 but this value is {:#018x}",
                expected, found
            ),
            Rejected::Apply(mismatch) => write!(f, "{}", mismatch),
            Rejected::Result { expected, found } => write!(
                f,
                "delta applied to the wrong result: expected fingerprint {:#018x}, \
                 got {:#018x}",
                expected, found
            ),
        }
    }
}

impl std::error::Error for Rejected {}
