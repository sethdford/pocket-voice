//! Talker latency benchmark — measures per-step inference time.
//! Target: <100ms per 12.5Hz step on Apple Silicon Metal.
//!
//! Run: cargo run --release -p sonata-talker --bin talker_latency

use candle_core::{DType, Device};
use std::time::Instant;

fn main() {
    let dev = Device::new_metal(0).unwrap_or_else(|_| {
        eprintln!("Metal not available, using CPU");
        Device::Cpu
    });
    println!("Device: {:?}", dev);

    let cfg = sonata_talker::TalkerConfig::default();
    println!("Talker Config:");
    println!("  Temporal: {}d, {} layers, {} heads ({} KV), {} FFN",
        cfg.d_model, cfg.n_temporal_layers, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff);
    println!("  Depth:    {}d, {} layers, {} heads, {} FFN",
        cfg.depth_dim, cfg.n_depth_layers, cfg.depth_heads, cfg.depth_d_ff);
    println!("  Codebooks: {} x {} vocab", cfg.n_codebooks, cfg.codebook_size);
    println!("  Frame rate: {} Hz ({:.0}ms budget)",
        cfg.frame_rate_hz, 1000.0 / cfg.frame_rate_hz as f64);

    let mut engine = sonata_talker::TalkerEngine::new_zeros(&cfg, &dev)
        .expect("Failed to create engine");

    // Warmup
    print!("Warming up...");
    for _ in 0..5 {
        let _ = engine.step(&[0u32; 8], None);
    }
    engine.reset().unwrap();
    println!(" done");

    // Benchmark
    let n_steps = 100;
    let mut latencies = Vec::with_capacity(n_steps);

    for i in 0..n_steps {
        let start = Instant::now();
        let _ = engine.step(&[0u32; 8], None).unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        if (i + 1) % 25 == 0 {
            println!("  step {}/{}", i + 1, n_steps);
        }
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = latencies.iter().sum::<f64>() / n_steps as f64;
    let p50 = latencies[n_steps / 2];
    let p95 = latencies[(n_steps as f64 * 0.95) as usize];
    let p99 = latencies[(n_steps as f64 * 0.99) as usize];
    let frame_ms = 1000.0 / cfg.frame_rate_hz as f64;

    println!("\n=== Talker Latency Benchmark ===");
    println!("  Steps:  {}", n_steps);
    println!("  Mean:   {:.1}ms", mean);
    println!("  P50:    {:.1}ms", p50);
    println!("  P95:    {:.1}ms", p95);
    println!("  P99:    {:.1}ms", p99);
    println!("  Budget: {:.0}ms ({} Hz)", frame_ms, cfg.frame_rate_hz);
    println!("  RTF:    {:.3}", mean / frame_ms);

    if mean < 100.0 {
        println!("  PASS — within 100ms latency target");
    } else {
        println!("  WARN — exceeds 100ms target (optimize Metal kernels)");
    }
}
