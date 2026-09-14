//! Asynchronous Tokio ingestion: drains the SPSC ring on a dedicated task,
//! scores each frame against a reduced-order surrogate, and accumulates
//! lock-free statistics (atomics only — zero mutexes, zero `mlock`).

use crate::ring::Consumer;
use crate::rom::ResidualMonitor;
use crate::telemetry::Frame;
use crate::ReducedOrderModel;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Number of latency histogram buckets (log₂-seconds per frame, `2⁻²⁰…` up).
pub const LATENCY_BUCKETS: usize = 33;

/// Lock-free ingestion statistics.
pub struct PipelineStats {
    /// Frames consumed.
    pub received: AtomicU64,
    /// Frames the producer dropped (ring full).
    pub dropped: AtomicU64,
    /// Sequence gaps per channel tag.
    pub seq_gaps: Vec<AtomicU64>,
    /// Mutex acquisitions performed by the pipeline — always 0.
    pub lock_count: AtomicU64,
    /// `mlock`/`mlockall` calls — always 0.
    pub mlock_count: AtomicU64,
    /// Log₂-bucketed per-frame processing latency histogram [s].
    /// Bucket `b` counts latencies in `[2^{b-32}, 2^{b-31})` seconds… see
    /// [`PipelineStats::latency_bucket`].
    pub latency_hist: Vec<AtomicU64>,
}

/// Bucket index for a processing latency in seconds.
fn latency_bucket(secs: f64) -> usize {
    if secs <= 0.0 || secs.is_nan() {
        return 0;
    }
    // log2(secs) + 32, clamped to [0, LATENCY_BUCKETS-1].
    let b = secs.log2() as i32 + 32;
    b.clamp(0, LATENCY_BUCKETS as i32 - 1) as usize
}

impl PipelineStats {
    fn counters() -> Self {
        Self {
            received: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            seq_gaps: (0..256).map(|_| AtomicU64::new(0)).collect(),
            lock_count: AtomicU64::new(0),
            mlock_count: AtomicU64::new(0),
            latency_hist: (0..LATENCY_BUCKETS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Fresh counters.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::counters())
    }

    /// Mean per-frame processing latency [ns].
    pub fn mean_latency_ns(&self) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0u64;
        for (b, c) in self.latency_hist.iter().enumerate() {
            let cnt = c.load(Ordering::Relaxed);
            // Bucket centres: 2^{b-32} s.
            num += cnt as f64 * (2f64.powi(b as i32 - 32)) * 1e9;
            den += cnt;
        }
        if den == 0 {
            0.0
        } else {
            num / den as f64
        }
    }

    /// 99th-percentile latency [ns] (bucket upper bound).
    pub fn p99_latency_ns(&self) -> f64 {
        let total: u64 = self
            .latency_hist
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();
        if total == 0 {
            return 0.0;
        }
        let target = (99 * total) / 100;
        let mut run = 0u64;
        for (b, c) in self.latency_hist.iter().enumerate() {
            run += c.load(Ordering::Relaxed);
            if run > target {
                return 2f64.powi(b as i32 - 31) * 1e9;
            }
        }
        2f64.powi(LATENCY_BUCKETS as i32 - 32) * 1e9
    }

    /// Frames received.
    pub fn received(&self) -> u64 {
        self.received.load(Ordering::Relaxed)
    }
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self::counters()
    }
}

/// Running ingestion pipeline: a Tokio task draining `consumer` and scoring
/// every frame against the surrogate until `stop` is set and the ring empties.
pub struct Pipeline {
    stats: Arc<PipelineStats>,
    stop: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<ResidualMonitor>,
}

impl Pipeline {
    /// Spawns the drain task.
    pub fn spawn<M>(consumer: Consumer, rom: M, inv_var: Vec<f64>, threshold: f64) -> Self
    where
        M: ReducedOrderModel + Send + 'static,
    {
        let stats = PipelineStats::new();
        let stop = Arc::new(AtomicBool::new(false));
        let (stats2, stop2) = (Arc::clone(&stats), Arc::clone(&stop));
        let task = tokio::spawn(async move {
            let mut monitor = ResidualMonitor::new(inv_var, threshold);
            let mut last_seq = [u64::MAX; 256];
            loop {
                let mut progress = false;
                while let Some(frame) = consumer.pop() {
                    progress = true;
                    let t0 = Instant::now();
                    let ch = frame.channel as usize;
                    if last_seq[ch] != u64::MAX && frame.seq != last_seq[ch] + 1 {
                        stats2.seq_gaps[ch].fetch_add(1, Ordering::Relaxed);
                    }
                    last_seq[ch] = frame.seq;
                    monitor.score(&frame, &rom);
                    stats2.received.fetch_add(1, Ordering::Relaxed);
                    stats2.latency_hist[latency_bucket(t0.elapsed().as_secs_f64())]
                        .fetch_add(1, Ordering::Relaxed);
                }
                stats2
                    .dropped
                    .store(consumer.dropped() as u64, Ordering::Relaxed);
                if stop2.load(Ordering::Relaxed) && !progress {
                    return monitor;
                }
                tokio::task::yield_now().await;
            }
        });
        Self { stats, stop, task }
    }

    /// Shared counters.
    pub fn stats(&self) -> &Arc<PipelineStats> {
        &self.stats
    }

    /// Signals shutdown; once the ring is empty the task returns the monitor.
    pub async fn finish(self) -> ResidualMonitor {
        self.stop.store(true, Ordering::Relaxed);
        self.task.await.unwrap()
    }
}

/// Sink adapter: applies `f` to every drained frame (convenience for tests
/// and offline replay; the SPSC ring remains the transport).
pub fn for_each<F: FnMut(Frame)>(consumer: Consumer, mut f: F) -> u64 {
    let mut n = 0;
    while let Some(frame) = consumer.pop() {
        f(frame);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::SpscRing;
    use crate::telemetry::{Channel, FBG_RATE_HZ, TYPE_N_RATE_HZ};

    /// Benchmark gate: synthetic 10 kHz Type-N and FBG streams plus a 1 kHz
    /// CCD stream, drained through the real pipeline — zero drops, zero locks,
    /// sub-microsecond mean processing latency.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ten_khz_streams_zero_drops_sub_microsecond() {
        const N_FAST: u64 = 30_000; // 3 s of a 10 kHz stream
        const N_CCD: u64 = 1_000;
        let ring = SpscRing::heap(1 << 12);
        let (prod, cons) = ring.split();
        let producer = std::thread::spawn(move || {
            // Interleave: 10 thermocouple + 10 FBG frames per 10 CCD·10 ms.
            let tc = |i: u64| {
                Frame::scalar(
                    Channel::ThermocoupleN,
                    i,
                    (i as f64 * 1e9 / TYPE_N_RATE_HZ as f64) as u64,
                    623.15 + 0.001 * (i as f64 % 997.0),
                )
            };
            let fb = |i: u64| {
                Frame::scalar(
                    Channel::FbgStrain,
                    i,
                    (i as f64 * 1e9 / FBG_RATE_HZ as f64) as u64,
                    4.0e-3,
                )
            };
            let mut ccd_seq = 0u64;
            for i in 0..N_FAST {
                prod.send(tc(i));
                prod.send(fb(i));
                if i % 30 == 0 && ccd_seq < N_CCD {
                    let mut f = Frame::ZERO;
                    f.channel = Channel::CcdSpectrometer;
                    f.seq = ccd_seq;
                    f.lanes = 4;
                    f.values = [1.0, 2.0, 3.0, 4.0, 0.0];
                    prod.send(f);
                    ccd_seq += 1;
                }
            }
        });

        let rom = |f: &Frame, out: &mut [f64]| -> usize {
            match f.channel {
                Channel::ThermocoupleN => {
                    out[0] = 623.15;
                    1
                }
                Channel::FbgStrain => {
                    out[0] = 4.0e-3;
                    1
                }
                Channel::CcdSpectrometer => {
                    out[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
                    4
                }
                _ => 0,
            }
        };
        let pipe = Pipeline::spawn(cons, rom, vec![100.0; 256], 1e9);
        let stats = Arc::clone(pipe.stats());
        producer.join().unwrap();
        let _monitor = pipe.finish().await;
        let expect = 2 * N_FAST + N_CCD;
        assert_eq!(stats.received(), expect);
        assert_eq!(stats.dropped.load(Ordering::Relaxed), 0);
        assert_eq!(stats.lock_count.load(Ordering::Relaxed), 0);
        assert_eq!(stats.mlock_count.load(Ordering::Relaxed), 0);
        assert!(stats
            .seq_gaps
            .iter()
            .all(|g| g.load(Ordering::Relaxed) == 0));
        let mean_ns = stats.mean_latency_ns();
        let p99_ns = stats.p99_latency_ns();
        assert!(mean_ns < 1_000.0, "mean latency {mean_ns:.0} ns");
        println!("frames={expect} mean={mean_ns:.0} ns p99={p99_ns:.0} ns");
    }
}
