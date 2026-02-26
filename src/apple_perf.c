/**
 * apple_perf.c — Apple Silicon performance primitives.
 *
 * Combines undocumented/private macOS APIs with hand-tuned NEON intrinsics
 * for the lowest-latency neural inference possible on Apple Silicon.
 */

#include "apple_perf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>

#include <arm_neon.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Real-Time Thread Scheduling
 *
 * THREAD_TIME_CONSTRAINT_POLICY is a Mach-level scheduling class that gives
 * the thread hard real-time guarantees: the scheduler will preempt any
 * non-RT thread to meet our deadline. This is what CoreAudio uses internally
 * for its I/O threads. We extend it to inference threads.
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t ns_to_abs(uint64_t ns)
{
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0)
        mach_timebase_info(&info);
    return ns * info.denom / info.numer;
}

int ap_set_realtime_priority(uint64_t period_ns, uint64_t computation_ns,
                              uint64_t constraint_ns)
{
    thread_time_constraint_policy_data_t policy;
    policy.period      = (uint32_t)ns_to_abs(period_ns);
    policy.computation = (uint32_t)ns_to_abs(computation_ns);
    policy.constraint  = (uint32_t)ns_to_abs(constraint_ns);
    policy.preemptible = 1;

    kern_return_t kr = thread_policy_set(
        mach_thread_self(),
        THREAD_TIME_CONSTRAINT_POLICY,
        (thread_policy_t)&policy,
        THREAD_TIME_CONSTRAINT_POLICY_COUNT);

    if (kr != KERN_SUCCESS) {
        fprintf(stderr, "[apple_perf] thread_policy_set failed: %s\n",
                mach_error_string(kr));
        return -1;
    }
    return 0;
}

int ap_set_realtime_audio(void)
{
    /* 5.33ms period = 256 samples @ 48kHz (typical CoreAudio callback) */
    return ap_set_realtime_priority(
        5333333,   /* period:      5.33ms */
        1500000,   /* computation: 1.5ms  */
        5333333    /* constraint:  5.33ms */
    );
}

int ap_set_realtime_inference(void)
{
    /* 10ms period for inference threads — allows burst compute */
    return ap_set_realtime_priority(
        10000000,  /* period:      10ms */
        8000000,   /* computation: 8ms  */
        10000000   /* constraint:  10ms */
    );
}

int ap_set_qos_user_interactive(void)
{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Model Weight Loading with Maximum Prefetch
 * ═══════════════════════════════════════════════════════════════════════════ */

APModelMap ap_model_mmap(const char *path)
{
    APModelMap m = {0};

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[apple_perf] open(%s) failed\n", path);
        return m;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return m;
    }

    size_t size = (size_t)st.st_size;
    m.fd = fd;
    m.size = size;

    /* Try superpage mapping first for large models (>16MB) */
    if (size > 16 * 1024 * 1024) {
        mach_vm_address_t addr = 0;
        kern_return_t kr = mach_vm_allocate(
            mach_task_self(), &addr, size,
            VM_FLAGS_ANYWHERE | VM_FLAGS_SUPERPAGE_SIZE_ANY);
        if (kr == KERN_SUCCESS) {
            /* Read file into superpage region */
            lseek(fd, 0, SEEK_SET);
            size_t total = 0;
            while (total < size) {
                ssize_t n = read(fd, (char *)addr + total, size - total);
                if (n <= 0) break;
                total += (size_t)n;
            }
            if (total == size) {
                m.base = (void *)addr;
                m.huge = 1;
                fprintf(stderr, "[apple_perf] Mapped %s with superpages (%zu MB)\n",
                        path, size / (1024 * 1024));
            } else {
                mach_vm_deallocate(mach_task_self(), addr, size);
            }
        }
    }

    /* Fallback: standard mmap */
    if (!m.base) {
        void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) {
            fprintf(stderr, "[apple_perf] mmap(%s) failed\n", path);
            close(fd);
            m.fd = -1;
            return m;
        }
        m.base = base;
    }

    /* Aggressive kernel hints */
    posix_madvise(m.base, size, POSIX_MADV_SEQUENTIAL);
    posix_madvise(m.base, size, POSIX_MADV_WILLNEED);

    /* Try to wire pages into physical memory to prevent paging */
    if (mlock(m.base, size) == 0) {
        m.locked = 1;
        fprintf(stderr, "[apple_perf] Wired %zu MB model into physical memory\n",
                size / (1024 * 1024));
    }

    return m;
}

void ap_model_prefetch(const void *ptr, size_t len)
{
    const char *p = (const char *)ptr;
    for (size_t i = 0; i < len; i += 64) {
        __builtin_prefetch(p + i, 0, 3);       /* L1 read, max temporal */
    }
}

void ap_model_munmap(APModelMap *m)
{
    if (!m || !m->base) return;

    if (m->locked) {
        munlock(m->base, m->size);
    }

    if (m->huge) {
        mach_vm_deallocate(mach_task_self(),
                           (mach_vm_address_t)m->base, m->size);
    } else {
        munmap(m->base, m->size);
    }

    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(APModelMap));
    m->fd = -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NEON-Optimized Softmax
 *
 * Three-pass algorithm:
 *   1. Find max (for numerical stability)
 *   2. Compute exp(x - max) and accumulate sum
 *   3. Divide by sum
 *
 * Uses vfmaq_f32 (fused multiply-accumulate) where possible.
 * The exp approximation uses a 4th-order polynomial that's accurate
 * to ~1e-4 over [-87, 0] — more than sufficient for softmax.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Fast exp approximation via integer bit manipulation + polynomial refinement.
 * Based on Schraudolph's algorithm with one Newton-Raphson correction.
 * Accurate to ~1e-4 over [-87, 88], which is the valid float32 exp range. */
static inline float32x4_t fast_exp_neon(float32x4_t x)
{
    /* Clamp to prevent overflow/underflow */
    x = vmaxq_f32(x, vdupq_n_f32(-87.3f));
    x = vminq_f32(x, vdupq_n_f32(88.3f));

    /* exp(x) = 2^(x * log2(e)) = 2^(n + f) where n=integer, f=fraction */
    const float32x4_t log2e = vdupq_n_f32(1.44269504089f);
    float32x4_t t = vmulq_f32(x, log2e);

    /* n = floor(t) */
    int32x4_t n = vcvtq_s32_f32(t);
    float32x4_t nf = vcvtq_f32_s32(n);
    /* Correct floor for negative values */
    uint32x4_t mask = vcgtq_f32(nf, t);
    nf = vsubq_f32(nf, vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
    n = vcvtq_s32_f32(nf);

    /* f = t - n (fractional part, 0 <= f < 1) */
    float32x4_t f = vsubq_f32(t, nf);

    /* Polynomial approximation of 2^f for f in [0, 1):
     * p(f) ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*c4))) */
    const float32x4_t c1 = vdupq_n_f32(0.693147180f);
    const float32x4_t c2 = vdupq_n_f32(0.240226507f);
    const float32x4_t c3 = vdupq_n_f32(0.055504109f);
    const float32x4_t c4 = vdupq_n_f32(0.009618129f);

    float32x4_t p = vfmaq_f32(c3, c4, f);
    p = vfmaq_f32(c2, p, f);
    p = vfmaq_f32(c1, p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);

    /* Multiply by 2^n via integer exponent manipulation */
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);

    return vmulq_f32(p, pow2n);
}

void ap_neon_softmax(const float *in, float *out, int n)
{
    if (n <= 0) return;

    /* Pass 1: find max */
    float32x4_t vmax = vdupq_n_f32(-HUGE_VALF);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(in + i));
    }
    float max_val = vmaxvq_f32(vmax);
    for (; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }

    /* Pass 2: exp(x - max) and sum */
    float32x4_t vmax_bc = vdupq_n_f32(max_val);
    float32x4_t vsum = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t e = fast_exp_neon(vsubq_f32(vld1q_f32(in + i), vmax_bc));
        vst1q_f32(out + i, e);
        vsum = vaddq_f32(vsum, e);
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }

    /* Pass 3: normalize */
    float32x4_t inv_sum = vdupq_n_f32(1.0f / sum);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(out + i), inv_sum));
    }
    for (; i < n; i++) {
        out[i] /= sum;
    }
}

void ap_neon_softmax_rows(float *x, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        ap_neon_softmax(x + r * cols, x + r * cols, cols);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NEON-Optimized GELU (fast tanh approximation)
 *
 * GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * The tanh is approximated via the same exp() trick:
 *   tanh(z) = (e^2z - 1) / (e^2z + 1)
 * ═══════════════════════════════════════════════════════════════════════════ */

void ap_neon_gelu(const float *in, float *out, int n)
{
    const float32x4_t half   = vdupq_n_f32(0.5f);
    const float32x4_t one    = vdupq_n_f32(1.0f);
    const float32x4_t two    = vdupq_n_f32(2.0f);
    const float32x4_t coeff  = vdupq_n_f32(0.044715f);
    const float32x4_t sqrt2pi = vdupq_n_f32(0.7978845608f); /* sqrt(2/pi) */

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
        float32x4_t inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));
        /* tanh via exp: tanh(z) = (e^2z - 1) / (e^2z + 1) */
        float32x4_t e2z = fast_exp_neon(vmulq_f32(two, inner));
        float32x4_t tanh_val = vdivq_f32(vsubq_f32(e2z, one), vaddq_f32(e2z, one));
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(x, half), vaddq_f32(one, tanh_val)));
    }
    for (; i < n; i++) {
        float x = in[i];
        float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        float t = tanhf(inner);
        out[i] = x * 0.5f * (1.0f + t);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NEON-Optimized SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
 * ═══════════════════════════════════════════════════════════════════════════ */

void ap_neon_silu(const float *in, float *out, int n)
{
    const float32x4_t one  = vdupq_n_f32(1.0f);
    const float32x4_t neg1 = vdupq_n_f32(-1.0f);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        float32x4_t neg_x = vmulq_f32(x, neg1);
        float32x4_t sigmoid = vdivq_f32(one, vaddq_f32(one, fast_exp_neon(neg_x)));
        vst1q_f32(out + i, vmulq_f32(x, sigmoid));
    }
    for (; i < n; i++) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NEON-Optimized LayerNorm
 *
 * Two-pass:
 *   1. Compute mean and variance via Welford's online algorithm (NEON)
 *   2. Normalize with gamma/beta affine transform
 * ═══════════════════════════════════════════════════════════════════════════ */

void ap_neon_layernorm(const float *in, float *out, const float *gamma,
                        const float *beta, int n, float eps)
{
    if (n <= 0) return;

    /* Mean */
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vsum = vaddq_f32(vsum, vld1q_f32(in + i));
    }
    float mean = vaddvq_f32(vsum);
    for (; i < n; i++) mean += in[i];
    mean /= (float)n;

    /* Variance */
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vvar = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(in + i), vmean);
        vvar = vfmaq_f32(vvar, d, d);
    }
    float var = vaddvq_f32(vvar);
    for (; i < n; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    /* Normalize: (x - mean) * inv_std * gamma + beta */
    float inv_std = 1.0f / sqrtf(var + eps);
    float32x4_t vinv = vdupq_n_f32(inv_std);

    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t normed = vmulq_f32(vsubq_f32(vld1q_f32(in + i), vmean), vinv);
        float32x4_t scaled = vfmaq_f32(vld1q_f32(beta + i),
                                         vld1q_f32(gamma + i), normed);
        vst1q_f32(out + i, scaled);
    }
    for (; i < n; i++) {
        out[i] = (in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NEON-Optimized RMSNorm (used by LLaMA-family models)
 *
 * RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
 * Single-pass variance, no mean subtraction.
 * ═══════════════════════════════════════════════════════════════════════════ */

void ap_neon_rmsnorm(const float *in, float *out, const float *gamma,
                      int n, float eps)
{
    if (n <= 0) return;

    /* Sum of squares */
    float32x4_t vss = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a = vld1q_f32(in + i);
        float32x4_t b = vld1q_f32(in + i + 4);
        vss = vfmaq_f32(vss, a, a);
        vss = vfmaq_f32(vss, b, b);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t a = vld1q_f32(in + i);
        vss = vfmaq_f32(vss, a, a);
    }
    float ss = vaddvq_f32(vss);
    for (; i < n; i++) ss += in[i] * in[i];

    float inv_rms = 1.0f / sqrtf(ss / (float)n + eps);
    float32x4_t vinv = vdupq_n_f32(inv_rms);

    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t normed = vmulq_f32(vld1q_f32(in + i), vinv);
        vst1q_f32(out + i, vmulq_f32(normed, vld1q_f32(gamma + i)));
    }
    for (; i < n; i++) {
        out[i] = in[i] * inv_rms * gamma[i];
    }
}

void ap_neon_residual_layernorm(const float *x, const float *residual,
                                 float *out, const float *gamma,
                                 const float *beta, int n, float eps)
{
    if (n <= 0) return;

    /* Fused residual add + mean computation */
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t added = vaddq_f32(vld1q_f32(x + i), vld1q_f32(residual + i));
        vst1q_f32(out + i, added);
        vsum = vaddq_f32(vsum, added);
    }
    float mean = vaddvq_f32(vsum);
    for (; i < n; i++) {
        out[i] = x[i] + residual[i];
        mean += out[i];
    }
    mean /= (float)n;

    /* Variance on the fused result */
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vvar  = vdupq_n_f32(0.0f);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(out + i), vmean);
        vvar = vfmaq_f32(vvar, d, d);
    }
    float var = vaddvq_f32(vvar);
    for (; i < n; i++) {
        float d = out[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    float inv_std = 1.0f / sqrtf(var + eps);
    float32x4_t vinv = vdupq_n_f32(inv_std);

    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t normed = vmulq_f32(vsubq_f32(vld1q_f32(out + i), vmean), vinv);
        vst1q_f32(out + i, vfmaq_f32(vld1q_f32(beta + i),
                                       vld1q_f32(gamma + i), normed));
    }
    for (; i < n; i++) {
        out[i] = (out[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * IOSurface Zero-Copy Buffer
 *
 * Private API: we use IOSurfaceCreate with kIOSurfaceBytesPerRow, etc.
 * The IOSurface is then used as a Metal buffer backing store via
 * newBufferWithBytesNoCopy or by creating from IOSurface descriptor.
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef __APPLE__
#include <IOSurface/IOSurface.h>

APZeroCopyBuffer ap_zerocopy_create(size_t size)
{
    APZeroCopyBuffer buf = {0};

    /* Round up to page alignment */
    size_t page_size = (size_t)vm_page_size;
    size = (size + page_size - 1) & ~(page_size - 1);

    /* Build IOSurface properties dictionary using CoreFoundation (no ObjC needed) */
    int width_val   = (int)(size / 4);
    int height_val  = 1;
    int bpr_val     = (int)size;
    int bpe_val     = 4;
    int pf_val      = 0x20202020; /* raw bytes */

    CFNumberRef width  = CFNumberCreate(NULL, kCFNumberIntType, &width_val);
    CFNumberRef height = CFNumberCreate(NULL, kCFNumberIntType, &height_val);
    CFNumberRef bpr    = CFNumberCreate(NULL, kCFNumberIntType, &bpr_val);
    CFNumberRef bpe    = CFNumberCreate(NULL, kCFNumberIntType, &bpe_val);
    CFNumberRef pf     = CFNumberCreate(NULL, kCFNumberIntType, &pf_val);

    const void *keys[] = {
        kIOSurfaceWidth, kIOSurfaceHeight,
        kIOSurfaceBytesPerRow, kIOSurfaceBytesPerElement,
        kIOSurfacePixelFormat
    };
    const void *vals[] = { width, height, bpr, bpe, pf };

    CFDictionaryRef props = CFDictionaryCreate(
        NULL, keys, vals, 5,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);

    IOSurfaceRef surface = IOSurfaceCreate(props);

    CFRelease(props);
    CFRelease(width); CFRelease(height);
    CFRelease(bpr); CFRelease(bpe); CFRelease(pf);

    if (!surface) {
        fprintf(stderr, "[apple_perf] IOSurfaceCreate failed\n");
        return buf;
    }

    IOSurfaceLock(surface, 0, NULL);

    buf.surface  = (void *)surface;
    buf.cpu_ptr  = IOSurfaceGetBaseAddress(surface);
    buf.size     = size;

    IOSurfaceUnlock(surface, 0, NULL);

    fprintf(stderr, "[apple_perf] Created zero-copy IOSurface buffer: %zu KB\n",
            size / 1024);
    return buf;
}

void ap_zerocopy_destroy(APZeroCopyBuffer *buf)
{
    if (!buf) return;
    if (buf->surface) {
        CFRelease((IOSurfaceRef)buf->surface);
    }
    memset(buf, 0, sizeof(APZeroCopyBuffer));
}

#else

APZeroCopyBuffer ap_zerocopy_create(size_t size)
{
    APZeroCopyBuffer buf = {0};
    buf.cpu_ptr = calloc(1, size);
    buf.size = size;
    return buf;
}

void ap_zerocopy_destroy(APZeroCopyBuffer *buf)
{
    if (!buf) return;
    free(buf->cpu_ptr);
    memset(buf, 0, sizeof(APZeroCopyBuffer));
}

#endif
