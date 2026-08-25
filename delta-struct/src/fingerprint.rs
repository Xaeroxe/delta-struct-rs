//! Stable content hashing, so a receiver can check that a delta is being
//! applied to the state it was computed against.
//!
//! [`std::hash::Hash`] cannot do this job for two reasons. It is not
//! implemented for [`HashSet`] or [`HashMap`], which are important
//! collections for the `unordered` field type, and the hash it feeds a
//! [`DefaultHasher`](std::collections::hash_map::DefaultHasher) is explicitly
//! allowed to change between Rust releases — fine for a hash table that lives
//! and dies in one process, useless for a value two processes have to agree
//! on.
//!
//! [`Fingerprint`] fixes both. Sets and maps are folded commutatively so
//! iteration order cannot matter, and [`Hasher`] is FNV-1a with the constants
//! written down here, so the same value fingerprints identically on any
//! platform, any Rust version, forever.
//!
//! Derive it rather than writing it:
//!
//! ```
//! use delta_struct::{fingerprint_of, Fingerprint};
//! use std::collections::HashSet;
//!
//! #[derive(Fingerprint)]
//! struct Device {
//!     services: HashSet<String>,
//!     online: bool,
//! }
//!
//! let device = |online| Device {
//!     services: vec!["ssh".to_string(), "http".to_string()].into_iter().collect(),
//!     online,
//! };
//!
//! // Set iteration order does not reach the fingerprint.
//! assert_eq!(fingerprint_of(&device(true)), fingerprint_of(&device(true)));
//! assert_ne!(fingerprint_of(&device(true)), fingerprint_of(&device(false)));
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// The FNV-1a 64-bit offset basis.
const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The FNV-1a 64-bit prime.
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// The hasher [`Fingerprint`] writes into: FNV-1a, 64-bit.
///
/// Deliberately not [`std::hash::Hasher`]. That trait's implementations are
/// free to change their output between Rust releases, which would turn a
/// toolchain upgrade on one side of a connection into a stream of spurious
/// mismatches. This one is pinned to the constants above and will not move.
///
/// It is not a cryptographic hash and is not meant to survive an adversary —
/// it exists to catch accidental divergence.
#[derive(Clone, Debug)]
pub struct Hasher {
    state: u64,
}

impl Hasher {
    /// Starts a new hasher at the offset basis.
    pub fn new() -> Self {
        Hasher {
            state: OFFSET_BASIS,
        }
    }

    /// Folds `bytes` into the hash.
    pub fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(PRIME);
        }
    }

    /// Folds a whole sub-hash in, for combining nested fingerprints.
    pub fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    /// The hash so far.
    pub fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Hasher::new()
    }
}

/// A value whose contents can be reduced to a number that two processes will
/// agree on.
///
/// Derive it with `#[derive(Fingerprint)]`, which walks a struct's fields in
/// declaration order, or an enum's variant index followed by its fields. Every
/// field's type has to implement it too.
pub trait Fingerprint {
    /// Folds `self` into `hasher`.
    fn fingerprint(&self, hasher: &mut Hasher);
}

/// Fingerprints a value on its own, which is what you usually want.
///
/// ```
/// use delta_struct::fingerprint_of;
///
/// assert_eq!(fingerprint_of(&"hello"), fingerprint_of(&"hello"));
/// assert_ne!(fingerprint_of(&"hello"), fingerprint_of(&"hellp"));
/// ```
pub fn fingerprint_of<T: Fingerprint + ?Sized>(value: &T) -> u64 {
    let mut hasher = Hasher::new();
    value.fingerprint(&mut hasher);
    hasher.finish()
}

macro_rules! fingerprint_le_bytes {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Fingerprint for $ty {
                fn fingerprint(&self, hasher: &mut Hasher) {
                    hasher.write(&self.to_le_bytes());
                }
            }
        )*
    };
}

fingerprint_le_bytes!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

macro_rules! fingerprint_widened {
    ($($ty:ty => $wide:ty),* $(,)?) => {
        $(
            impl Fingerprint for $ty {
                fn fingerprint(&self, hasher: &mut Hasher) {
                    // Widened so that a 32-bit sender and a 64-bit receiver
                    // agree on the same value.
                    <$wide as Fingerprint>::fingerprint(&(*self as $wide), hasher);
                }
            }
        )*
    };
}

fingerprint_widened!(usize => u64, isize => i64);

impl Fingerprint for bool {
    fn fingerprint(&self, hasher: &mut Hasher) {
        hasher.write(&[u8::from(*self)]);
    }
}

impl Fingerprint for char {
    fn fingerprint(&self, hasher: &mut Hasher) {
        u32::from(*self).fingerprint(hasher);
    }
}

/// Floats fingerprint by their bit pattern, so `NaN` matches itself and `0.0`
/// does not match `-0.0` — the opposite of what `==` says in both cases. The
/// question a fingerprint answers is "are these the same state?", not "are
/// these numerically equal?".
impl Fingerprint for f32 {
    fn fingerprint(&self, hasher: &mut Hasher) {
        self.to_bits().fingerprint(hasher);
    }
}

impl Fingerprint for f64 {
    fn fingerprint(&self, hasher: &mut Hasher) {
        self.to_bits().fingerprint(hasher);
    }
}

impl Fingerprint for str {
    fn fingerprint(&self, hasher: &mut Hasher) {
        // Length first, so that ("ab", "c") cannot collide with ("a", "bc").
        hasher.write_u64(self.len() as u64);
        hasher.write(self.as_bytes());
    }
}

impl Fingerprint for String {
    fn fingerprint(&self, hasher: &mut Hasher) {
        self.as_str().fingerprint(hasher);
    }
}

impl<T: Fingerprint + ?Sized> Fingerprint for &T {
    fn fingerprint(&self, hasher: &mut Hasher) {
        (**self).fingerprint(hasher);
    }
}

impl<T: Fingerprint + ?Sized> Fingerprint for Box<T> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        (**self).fingerprint(hasher);
    }
}

impl<T: Fingerprint> Fingerprint for Option<T> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        match self {
            None => hasher.write(&[0]),
            Some(value) => {
                hasher.write(&[1]);
                value.fingerprint(hasher);
            }
        }
    }
}

impl<T: Fingerprint, E: Fingerprint> Fingerprint for Result<T, E> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        match self {
            Ok(value) => {
                hasher.write(&[0]);
                value.fingerprint(hasher);
            }
            Err(error) => {
                hasher.write(&[1]);
                error.fingerprint(hasher);
            }
        }
    }
}

impl<T: Fingerprint> Fingerprint for [T] {
    fn fingerprint(&self, hasher: &mut Hasher) {
        hasher.write_u64(self.len() as u64);
        for item in self {
            item.fingerprint(hasher);
        }
    }
}

impl<T: Fingerprint> Fingerprint for Vec<T> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        self.as_slice().fingerprint(hasher);
    }
}

impl Fingerprint for () {
    fn fingerprint(&self, _hasher: &mut Hasher) {}
}

macro_rules! fingerprint_tuples {
    ($(($($index:tt $param:ident),+))+) => {
        $(
            impl<$($param: Fingerprint),+> Fingerprint for ($($param,)+) {
                fn fingerprint(&self, hasher: &mut Hasher) {
                    $(self.$index.fingerprint(hasher);)+
                }
            }
        )+
    };
}

fingerprint_tuples! {
    (0 A)
    (0 A, 1 B)
    (0 A, 1 B, 2 C)
    (0 A, 1 B, 2 C, 3 D)
    (0 A, 1 B, 2 C, 3 D, 4 E)
    (0 A, 1 B, 2 C, 3 D, 4 E, 5 F)
}

/// Folds each element's own fingerprint together with `^`, so the order they
/// come out of the collection in cannot reach the result.
///
/// The length goes in too, which is what stops a set from colliding with a
/// differently-sized one whose element hashes happen to cancel.
fn fingerprint_unordered<I, F>(hasher: &mut Hasher, len: usize, items: I, mut each: F)
where
    F: FnMut(&mut Hasher, I::Item),
    I: Iterator,
{
    let mut combined = 0u64;
    for item in items {
        let mut element = Hasher::new();
        each(&mut element, item);
        combined ^= element.finish();
    }
    hasher.write_u64(len as u64);
    hasher.write_u64(combined);
}

impl<T: Fingerprint, S> Fingerprint for HashSet<T, S> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        fingerprint_unordered(hasher, self.len(), self.iter(), |h, item| {
            item.fingerprint(h)
        });
    }
}

impl<T: Fingerprint> Fingerprint for BTreeSet<T> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        fingerprint_unordered(hasher, self.len(), self.iter(), |h, item| {
            item.fingerprint(h)
        });
    }
}

impl<K: Fingerprint, V: Fingerprint, S> Fingerprint for HashMap<K, V, S> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        fingerprint_unordered(hasher, self.len(), self.iter(), |h, (key, value)| {
            key.fingerprint(h);
            value.fingerprint(h);
        });
    }
}

impl<K: Fingerprint, V: Fingerprint> Fingerprint for BTreeMap<K, V> {
    fn fingerprint(&self, hasher: &mut Hasher) {
        fingerprint_unordered(hasher, self.len(), self.iter(), |h, (key, value)| {
            key.fingerprint(h);
            value.fingerprint(h);
        });
    }
}
