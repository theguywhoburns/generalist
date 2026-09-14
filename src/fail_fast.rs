//! Fail-fast: any panic anywhere (including background runtime threads that
//! would otherwise spin forever, e.g. cubecl's memory server on OOM)
//! becomes an immediate loud process exit.
//!
//! Install FIRST in every long-running binary entry point:
//! `generalist::fail_fast::install();`
//!
//! Never install in library or test code: the test harness relies on
//! catching panics per test, and an abort hook would kill the whole suite
//! on the first failure.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TRIPPED: AtomicBool = AtomicBool::new(false);

pub fn install() {
    std::panic::set_hook(Box::new(|info| {
        // The hook fires at panic time on any thread, before unwinding or
        // any `catch_unwind` further up. A second concurrent panic aborts.
        if TRIPPED.swap(true, Ordering::SeqCst) {
            std::process::abort();
        }
        eprintln!("FATAL: uncaught panic, aborting run\n{info}");
        // exit(), not panic propagation: a compromised allocator or runtime
        // must not be trusted with destructors or retry loops. Checkpoints
        // are written synchronously at save points, so nothing is lost that
        // was already saved.
        std::process::exit(42);
    }));
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Whole-process stall watchdog. Panic hooks can't catch hangs (no panic
/// fires); this kills the entire process when no progress is reported
/// within `timeout_secs`. Install in long runners; ping every step.
pub struct Watchdog {
    last_ping: AtomicU64,
    last_step: AtomicUsize,
    timeout_secs: u64,
}

impl Watchdog {
    pub fn spawn(timeout_secs: u64) -> std::sync::Arc<Self> {
        let wd = std::sync::Arc::new(Self {
            last_ping: AtomicU64::new(now_secs()),
            last_step: AtomicUsize::new(0),
            timeout_secs,
        });
        let me = wd.clone();
        std::thread::Builder::new()
            .name("watchdog".to_string())
            .spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(
                    me.timeout_secs.max(10) / 2,
                ));
                let idle = now_secs().saturating_sub(me.last_ping.load(Ordering::SeqCst));
                if idle > me.timeout_secs {
                    eprintln!(
                        "FATAL: no progress for {idle}s (last completed step {}), killing whole process",
                        me.last_step.load(Ordering::SeqCst),
                    );
                    std::process::exit(44);
                }
            })
            .expect("watchdog thread");
        wd
    }

    pub fn ping_step(&self, step: usize) {
        self.last_step.store(step, Ordering::SeqCst);
        self.last_ping.store(now_secs(), Ordering::SeqCst);
    }
}
