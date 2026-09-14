//! Telemetry frame format shared between producers and the ingestion
//! pipeline. Frames are `#[repr(C)]`, one cache line wide, and `Copy` so they
//! can live in shared memory without serde.

/// Sensor channel identifiers (simulator_spec.pdf §4 instrumentation).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// Type-N thermocouple, 10 kHz.
    ThermocoupleN = 1,
    /// Fiber-Bragg-grating strain sensor, 10 kHz.
    FbgStrain = 2,
    /// CCD spectrometer bin block.
    CcdSpectrometer = 3,
    /// Any other scalar diagnostic.
    Other = 255,
}

/// Nominal sampling rates [Hz] for the specified instruments.
pub const TYPE_N_RATE_HZ: f64 = 10_000.0;
/// FBG nominal rate [Hz].
pub const FBG_RATE_HZ: f64 = 10_000.0;
/// CCD frame rate [Hz] (rows/blocks streamed; one frame carries `VALUES`
/// spectral bins).
pub const CCD_RATE_HZ: f64 = 1_000.0;

/// Scalar payload lanes in a [`Frame`].
pub const FRAME_VALUES: usize = 5;

/// One telemetry frame: 64 B, cache-line aligned.
///
/// * `timestamp_ns` — producer clock at sample time (monotonic ns).
/// * `seq` — per-channel sequence counter, gap-checked by the consumer.
/// * `values` — up to 5 scalar lanes (channel dependent; CCD reuses the frame
///   per spectral block). The struct is exactly one 64 B cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// Sequence number within the channel.
    pub seq: u64,
    /// Monotonic timestamp [ns].
    pub timestamp_ns: u64,
    /// Source channel.
    pub channel: Channel,
    /// Channel-specific flags.
    pub flags: u8,
    /// Lane count actually populated (≤ [`FRAME_VALUES`]).
    pub lanes: u8,
    /// Reserved, zeroed.
    pub _reserved: u8,
    /// CRC or transport tag (0 when unused).
    pub crc: u32,
    /// Scalar payload.
    pub values: [f64; FRAME_VALUES],
}

const _: () = assert!(std::mem::size_of::<Frame>() == 64);

impl Frame {
    /// Zeroed frame.
    pub const ZERO: Self = Self {
        seq: 0,
        timestamp_ns: 0,
        channel: Channel::Other,
        flags: 0,
        lanes: 0,
        _reserved: 0,
        crc: 0,
        values: [0.0; FRAME_VALUES],
    };

    /// Constructs a single-lane scalar sample.
    pub fn scalar(channel: Channel, seq: u64, timestamp_ns: u64, value: f64) -> Self {
        let mut f = Self::ZERO;
        f.channel = channel;
        f.seq = seq;
        f.timestamp_ns = timestamp_ns;
        f.lanes = 1;
        f.values[0] = value;
        f
    }
}
