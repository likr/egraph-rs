use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub(super) struct FastU64Hasher {
    state: u64,
}

impl Hasher for FastU64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        let z = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.state = z ^ (z >> 32);
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.state;
        for &b in bytes {
            h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(b);
        }
        self.state = h;
    }
}

pub(super) type PairSet = HashSet<u64, BuildHasherDefault<FastU64Hasher>>;

#[inline]
pub(super) fn pack_pair(a: usize, b: usize) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | (hi as u64)
}

#[inline]
pub(super) fn unpack_pair(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, (key & 0xFFFF_FFFF) as usize)
}

pub(super) struct InlinePairSet {
    inline: [u64; 8],
    inline_len: usize,
    overflow: PairSet,
}

impl InlinePairSet {
    pub(super) fn new() -> Self {
        Self {
            inline: [0; 8],
            inline_len: 0,
            overflow: PairSet::default(),
        }
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.inline_len == 0 && self.overflow.is_empty()
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        if self.overflow.is_empty() {
            self.inline_len
        } else {
            self.overflow.len()
        }
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        self.inline_len = 0;
        if !self.overflow.is_empty() {
            self.overflow.clear();
        }
    }

    #[inline]
    pub(super) fn contains(&self, pair: u64) -> bool {
        if self.overflow.is_empty() {
            self.inline[..self.inline_len].contains(&pair)
        } else {
            self.overflow.contains(&pair)
        }
    }

    #[inline]
    pub(super) fn insert(&mut self, pair: u64) -> bool {
        if self.overflow.is_empty() {
            if self.inline[..self.inline_len].contains(&pair) {
                return false;
            }
            if self.inline_len < self.inline.len() {
                self.inline[self.inline_len] = pair;
                self.inline_len += 1;
                return true;
            }
            for &p in &self.inline {
                self.overflow.insert(p);
            }
            self.inline_len = 0;
        }
        self.overflow.insert(pair)
    }

    #[inline]
    pub(super) fn remove(&mut self, pair: u64) {
        if self.overflow.is_empty() {
            if let Some(pos) = self.inline[..self.inline_len]
                .iter()
                .position(|&p| p == pair)
            {
                self.inline_len -= 1;
                self.inline[pos] = self.inline[self.inline_len];
            }
        } else {
            self.overflow.remove(&pair);
        }
    }
}
