//! Single-producer single-consumer lock-free ring buffer over raw memory:
//! cache-aligned header with `AtomicUsize` head/tail and Acquire/Release
//! ordering. Backing storage is either process-local aligned heap or a POSIX
//! shared-memory segment ([`crate::shm`]).

use crate::telemetry::Frame;
use std::alloc::{alloc_zeroed, Layout};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Ring control block shared between producer and consumer. One cache line.
#[repr(C, align(64))]
pub struct RingHeader {
    /// Monotonic write index (slot = `head % capacity`). Producer-owned.
    pub head: AtomicUsize,
    /// Monotonic read index. Consumer-owned.
    pub tail: AtomicUsize,
    /// Frames dropped by the producer because the ring was full.
    pub dropped: AtomicUsize,
    /// Number of frame slots (power of two).
    pub capacity: usize,
    /// Magic/version tag for shared-memory handshakes.
    pub magic: u64,
}

const _: () = assert!(std::mem::size_of::<RingHeader>() <= 64);

/// `0x5348_4254_5249_4e47` = "SHBTRING".
pub const RING_MAGIC: u64 = 0x5348_4254_5249_4e47;

/// Total bytes of a region holding `capacity` frames.
pub fn region_len(capacity: usize) -> usize {
    64 + capacity * std::mem::size_of::<Frame>()
}

/// Shared ownership of a mapped ring region (heap or POSIX shm).
pub struct Region {
    /// Points to the [`RingHeader`]; slots begin at `ptr + 64`.
    pub(crate) base: ptr::NonNull<u8>,
    pub(crate) len: usize,
    pub(crate) kind: RegionKind,
}

pub(crate) enum RegionKind {
    Heap(Layout),
    #[cfg(unix)]
    // Held purely for ownership: dropping it unmaps/unlinks the segment.
    #[allow(dead_code)]
    Shm(crate::shm::SharedMemory),
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// Region backed by a mapped POSIX shared-memory segment.
    #[cfg(unix)]
    pub fn shared(shm: crate::shm::SharedMemory) -> Self {
        let (base, len) = shm.region();
        Self {
            base,
            len,
            kind: RegionKind::Shm(shm),
        }
    }

    /// Process-local heap-backed region (64 B aligned, zeroed).
    pub fn heap(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());
        let len = region_len(capacity);
        let layout = Layout::from_size_align(len, 64).unwrap();
        let base = ptr::NonNull::new(unsafe { alloc_zeroed(layout) }).expect("alloc");
        let region = Self {
            base,
            len,
            kind: RegionKind::Heap(layout),
        };
        unsafe {
            let h = base.as_ptr() as *mut RingHeader;
            (*h).capacity = capacity;
            (*h).magic = RING_MAGIC;
        }
        region
    }

    fn header(&self) -> &RingHeader {
        unsafe { &*(self.base.as_ptr() as *const RingHeader) }
    }

    fn slot_ptr(&self, i: usize) -> *mut Frame {
        unsafe {
            self.base
                .as_ptr()
                .add(64 + i * std::mem::size_of::<Frame>()) as *mut Frame
        }
    }

    /// Number of frame slots.
    pub fn capacity(&self) -> usize {
        self.header().capacity
    }

    /// Mapped region length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the region is empty (never — kept for API completeness).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        match &self.kind {
            RegionKind::Heap(layout) => unsafe { std::alloc::dealloc(self.base.as_ptr(), *layout) },
            #[cfg(unix)]
            RegionKind::Shm(_) => {}
        }
    }
}

/// Producer endpoint of an SPSC ring.
#[derive(Clone)]
pub struct Producer {
    region: Arc<Region>,
}

/// Consumer endpoint of an SPSC ring.
#[derive(Clone)]
pub struct Consumer {
    region: Arc<Region>,
}

/// An SPSC ring pair.
pub struct SpscRing {
    region: Arc<Region>,
}

impl SpscRing {
    /// Creates a ring over an existing region.
    pub fn new(region: Region) -> Self {
        Self {
            region: Arc::new(region),
        }
    }

    /// Heap-backed ring with `capacity` (power of two) frame slots.
    pub fn heap(capacity: usize) -> Self {
        Self::new(Region::heap(capacity))
    }

    /// Shared region, e.g. to attach a second SPSC view.
    pub fn region(&self) -> &Arc<Region> {
        &self.region
    }

    /// Splits into `(producer, consumer)`.
    pub fn split(&self) -> (Producer, Consumer) {
        (
            Producer {
                region: Arc::clone(&self.region),
            },
            Consumer {
                region: Arc::clone(&self.region),
            },
        )
    }
}

impl Producer {
    /// Non-blocking push; returns `false` when the ring is full. Does not
    /// account the frame as dropped — the caller decides (see [`Self::push`]).
    pub fn try_push(&self, frame: Frame) -> bool {
        let h = self.region.header();
        let head = h.head.load(Ordering::Relaxed);
        let tail = h.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= h.capacity {
            return false;
        }
        unsafe {
            ptr::write_volatile(self.region.slot_ptr(head & (h.capacity - 1)), frame);
        }
        h.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Push with drop accounting: on a full ring the frame is discarded and
    /// `dropped` is incremented (real-time producers may not wait).
    pub fn push(&self, frame: Frame) -> bool {
        if !self.try_push(frame) {
            self.region.header().dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Spin-waiting push: retries until the ring has space — never drops.
    /// `waits` counts stall iterations for diagnostics.
    pub fn send(&self, frame: Frame) {
        while !self.try_push(frame) {
            std::hint::spin_loop();
        }
    }

    /// Frames dropped due to back-pressure.
    pub fn dropped(&self) -> usize {
        self.region.header().dropped.load(Ordering::Relaxed)
    }

    /// Unread frames currently buffered.
    pub fn pending(&self) -> usize {
        let h = self.region.header();
        h.head.load(Ordering::Acquire) - h.tail.load(Ordering::Acquire)
    }
}

impl Consumer {
    /// Pops the oldest frame, or `None` when empty.
    pub fn pop(&self) -> Option<Frame> {
        let h = self.region.header();
        let tail = h.tail.load(Ordering::Relaxed);
        if tail == h.head.load(Ordering::Acquire) {
            return None;
        }
        let frame = unsafe { ptr::read_volatile(self.region.slot_ptr(tail & (h.capacity - 1))) };
        h.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(frame)
    }

    /// Frames dropped by the producer (visible to the consumer too).
    pub fn dropped(&self) -> usize {
        self.region.header().dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::Channel;

    #[test]
    fn ring_roundtrip_and_drops() {
        let ring = SpscRing::heap(8);
        let (p, c) = ring.split();
        for i in 0..8u64 {
            p.send(Frame::scalar(
                Channel::ThermocoupleN,
                i,
                i * 100,
                300.0 + i as f64,
            ));
        }
        assert!(!p.push(Frame::scalar(Channel::ThermocoupleN, 8, 800, 0.0)));
        assert_eq!(p.dropped(), 1);
        for i in 0..8u64 {
            let f = c.pop().unwrap();
            assert_eq!(f.seq, i);
        }
        assert!(c.pop().is_none());
    }

    #[test]
    fn ring_wraps_counters() {
        let ring = SpscRing::heap(4);
        let (p, c) = ring.split();
        for i in 0..100u64 {
            while !p.try_push(Frame::scalar(Channel::FbgStrain, i, i, i as f64)) {
                while c.pop().is_none() {}
            }
            let f = c.pop().unwrap();
            assert_eq!(f.seq, i);
        }
    }
}
