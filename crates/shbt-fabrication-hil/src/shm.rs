//! POSIX shared-memory regions for the SPSC ring: `shm_open` + `ftruncate` +
//! `mmap`, with the [`crate::ring::RingHeader`] placed on the first cache line.
//!
//! This is the crate's only `unsafe` surface; the ring logic itself uses
//! volatile loads/stores plus `Acquire`/`Release` atomics on the shared region.

#![allow(unsafe_code)]

use crate::ring::{region_len, RING_MAGIC};
use std::ffi::CString;
use std::io;
use std::ptr::{self, NonNull};

/// A `mmap`ed POSIX shared-memory segment.
#[derive(Debug)]
pub struct SharedMemory {
    ptr: NonNull<u8>,
    len: usize,
    /// Name kept for unlinking; `None` for attach-only opens.
    name: Option<CString>,
    owner: bool,
}

impl SharedMemory {
    /// Creates (or replaces) the segment `name` (`/xyz` naming) and sizes it
    /// for `capacity` frames.
    pub fn create(name: &str, capacity: usize) -> io::Result<Self> {
        let len = region_len(capacity);
        let cname = c_name(name)?;
        unsafe {
            // O_RDWR | O_CREAT | O_EXCL would fail on stale segments; unlink
            // first so restarts recover cleanly.
            libc::shm_unlink(cname.as_ptr());
            let fd = libc::shm_open(cname.as_ptr(), libc::O_RDWR | libc::O_CREAT, 0o600);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::ftruncate(fd, len as libc::off_t) != 0 {
                let e = io::Error::last_os_error();
                libc::close(fd);
                return Err(e);
            }
            let p = libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            if p == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            ptr::write_bytes(p, 0, len);
            let hdr = p as *mut crate::ring::RingHeader;
            (*hdr).capacity = capacity;
            (*hdr).magic = RING_MAGIC;
            Ok(Self {
                ptr: NonNull::new(p as *mut u8).unwrap(),
                len,
                name: Some(cname),
                owner: true,
            })
        }
    }

    /// Attaches to an existing segment created by [`Self::create`].
    ///
    /// Polls until the peer's `capacity` field is non-zero (creator finished
    /// `ftruncate` + header init) or ~1 s elapses.
    pub fn attach(name: &str) -> io::Result<Self> {
        let cname = c_name(name)?;
        unsafe {
            let fd = libc::shm_open(cname.as_ptr(), libc::O_RDWR, 0o600);
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            let mut len = 0usize;
            for _ in 0..1000 {
                let mut st = std::mem::zeroed::<libc::stat>();
                if libc::fstat(fd, &mut st) == 0 && st.st_size >= 64 {
                    len = st.st_size as usize;
                    break;
                }
                std::thread::yield_now();
            }
            if len == 0 {
                libc::close(fd);
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "shm segment not initialised",
                ));
            }
            let p = libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            if p == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                ptr: NonNull::new(p as *mut u8).unwrap(),
                len,
                name: Some(cname),
                owner: false,
            })
        }
    }

    /// Base pointer and length of the mapped region.
    pub fn region(&self) -> (NonNull<u8>, usize) {
        (self.ptr, self.len)
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
            if self.owner {
                if let Some(name) = &self.name {
                    libc::shm_unlink(name.as_ptr());
                }
            }
        }
    }
}

fn c_name(name: &str) -> io::Result<CString> {
    let n = if name.starts_with('/') {
        name.to_string()
    } else {
        format!("/{name}")
    };
    CString::new(n).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::{Region, SpscRing};
    use crate::telemetry::{Channel, Frame};

    #[test]
    fn shm_ring_between_two_mappings() {
        let name = format!("shbt-test-{}", std::process::id());
        let a = SharedMemory::create(&name, 16).unwrap();
        let b = SharedMemory::attach(&name).unwrap();
        let ring_a = SpscRing::new(Region::shared(a));
        let ring_b = SpscRing::new(Region::shared(b));
        let (prod, _) = ring_a.split();
        let (_, cons) = ring_b.split();
        prod.push(Frame::scalar(Channel::CcdSpectrometer, 1, 10, 42.0));
        let f = cons.pop().unwrap();
        assert_eq!(f.seq, 1);
        assert_eq!(f.values[0], 42.0);
        assert_eq!(f.channel, Channel::CcdSpectrometer);
    }
}
