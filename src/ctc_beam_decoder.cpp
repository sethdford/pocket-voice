/**
 * ctc_beam_decoder.cpp — CTC prefix beam search with optional KenLM scoring.
 *
 * Algorithm: prefix beam search (Hannun 2017) maintaining (p_blank, p_nonblank)
 * for each hypothesis prefix. LM scoring is applied at word boundaries
 * (space-separated tokens).
 *
 * Compiled as C++ because KenLM is C++. Exposes C-ABI only.
 */

#include "ctc_beam_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>

/* ── KenLM optional include ────────────────────────────────────────────── */
#ifdef USE_KENLM
#include "lm/model.hh"
#include "lm/state.hh"
#endif

static constexpr float NEG_INF = -1e30f;

static inline float log_add(float a, float b) {
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float mx = std::max(a, b);
    return mx + std::log1pf(std::expf(std::min(a, b) - mx));
}

/* ── Beam hypothesis ───────────────────────────────────────────────────── */

struct Hypothesis {
    std::vector<int> tokens;
    float p_blank;
    float p_nonblank;
    float lm_score;
#ifdef USE_KENLM
    lm::ngram::State lm_state;
#endif

    float total() const { return log_add(p_blank, p_nonblank); }
};

/* ── Decoder internals ─────────────────────────────────────────────────── */

struct CTCBeamDecoder {
    CTCBeamConfig config;
    int blank_id;
    int vocab_size;
    std::vector<std::string> vocab;

#ifdef USE_KENLM
    lm::ngram::Model *lm_model;
#else
    void *lm_model;
#endif

    bool has_lm;
};

/* Reconstruct word from token sequence for LM scoring.
 * SentencePiece tokens use '▁' (U+2581) as word-start marker. */
static std::string tokens_to_text(const std::vector<int> &tokens,
                                   const std::vector<std::string> &vocab) {
    std::string text;
    for (int id : tokens) {
        if (id >= 0 && id < (int)vocab.size())
            text += vocab[id];
    }
    return text;
}

/* Extract the last word from the token sequence for incremental LM scoring. */
static std::string last_word(const std::vector<int> &tokens,
                              const std::vector<std::string> &vocab) {
    std::string word;
    for (int i = (int)tokens.size() - 1; i >= 0; i--) {
        const std::string &tok = vocab[tokens[i]];
        /* SentencePiece word boundary: ▁ (3 bytes: 0xE2 0x96 0x81) or space */
        bool is_boundary = false;
        if (tok.size() >= 3 &&
            (unsigned char)tok[0] == 0xE2 &&
            (unsigned char)tok[1] == 0x96 &&
            (unsigned char)tok[2] == 0x81) {
            is_boundary = true;
            word = tok.substr(3) + word;
        } else if (!tok.empty() && tok[0] == ' ') {
            is_boundary = true;
            word = tok.substr(1) + word;
        } else {
            word = tok + word;
        }
        if (is_boundary && i < (int)tokens.size() - 1)
            break;
    }
    return word;
}

/* Check if a token starts a new word (SentencePiece convention). */
static bool is_word_start(const std::string &tok) {
    if (tok.size() >= 3 &&
        (unsigned char)tok[0] == 0xE2 &&
        (unsigned char)tok[1] == 0x96 &&
        (unsigned char)tok[2] == 0x81)
        return true;
    if (!tok.empty() && tok[0] == ' ')
        return true;
    return false;
}

#ifdef USE_KENLM
static float score_word(lm::ngram::Model *model,
                         const lm::ngram::State &in_state,
                         const std::string &word,
                         lm::ngram::State *out_state) {
    if (word.empty()) {
        *out_state = in_state;
        return 0.0f;
    }
    const lm::ngram::Vocabulary &lm_vocab = model->GetVocabulary();
    lm::WordIndex wid = lm_vocab.Index(word);
    return model->FullScore(in_state, wid, *out_state).prob;
}
#endif

/* ── Key for hypothesis dedup: the token sequence ──────────────────────── */

struct VecHash {
    size_t operator()(const std::vector<int> &v) const {
        size_t h = 0;
        for (int x : v) h = h * 31 + (unsigned)x;
        return h;
    }
};

/* ── C-ABI implementation ──────────────────────────────────────────────── */

extern "C" {

CTCBeamConfig ctc_beam_config_default(void) {
    CTCBeamConfig c;
    c.beam_size = 16;
    c.lm_weight = 1.5f;
    c.word_score = 0.0f;
    c.blank_skip_thresh = 0.0f;
    return c;
}

CTCBeamDecoder *ctc_beam_create(const char *lm_path,
                                 const char *const *vocab_arr,
                                 int vocab_size,
                                 int blank_id,
                                 const CTCBeamConfig *config) {
    auto *dec = new (std::nothrow) CTCBeamDecoder;
    if (!dec) return nullptr;

    dec->config = config ? *config : ctc_beam_config_default();
    dec->blank_id = blank_id;
    dec->vocab_size = vocab_size;
    dec->lm_model = nullptr;
    dec->has_lm = false;

    dec->vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        if (vocab_arr && vocab_arr[i])
            dec->vocab[i] = vocab_arr[i];
    }

#ifdef USE_KENLM
    if (lm_path && lm_path[0]) {
        try {
            lm::ngram::Config lm_config;
            lm_config.load_method = util::LAZY;
            dec->lm_model = new lm::ngram::Model(lm_path, lm_config);
            dec->has_lm = true;
            fprintf(stderr, "[beam_decoder] Loaded KenLM: %s\n", lm_path);
        } catch (const std::exception &e) {
            fprintf(stderr, "[beam_decoder] KenLM load failed: %s\n", e.what());
            dec->lm_model = nullptr;
            dec->has_lm = false;
        }
    }
#else
    if (lm_path && lm_path[0]) {
        fprintf(stderr, "[beam_decoder] KenLM not compiled in — running without LM\n");
    }
#endif

    return dec;
}

int ctc_beam_decode(CTCBeamDecoder *dec,
                    const float *log_probs,
                    int T, int vocab_size,
                    char *out, int out_cap) {
    if (!dec || !log_probs || !out || out_cap <= 0) return -1;
    if (vocab_size != dec->vocab_size) return -1;

    const int beam_size = dec->config.beam_size;
    const float lm_weight = dec->config.lm_weight;
    const float word_score = dec->config.word_score;
    const float blank_skip = dec->config.blank_skip_thresh;
    const int blank_id = dec->blank_id;

    /* Initialize with empty hypothesis */
    using BeamMap = std::unordered_map<std::vector<int>, Hypothesis, VecHash>;
    BeamMap beams;

    {
        Hypothesis h;
        h.p_blank = 0.0f; /* log(1) = 0 */
        h.p_nonblank = NEG_INF;
        h.lm_score = 0.0f;
#ifdef USE_KENLM
        if (dec->has_lm)
            dec->lm_model->BeginSentenceWrite(&h.lm_state);
#endif
        beams[h.tokens] = std::move(h);
    }

    for (int t = 0; t < T; t++) {
        const float *lp = log_probs + (size_t)t * vocab_size;

        /* Optional: skip time step if blank dominates */
        if (blank_skip > 0.0f && std::expf(lp[blank_id]) > blank_skip)
            continue;

        BeamMap next_beams;

        for (auto &[prefix, hyp] : beams) {
            float p_total = hyp.total();

            /* ── Extend with blank ────────────────────────────────────── */
            {
                auto it = next_beams.find(prefix);
                if (it == next_beams.end()) {
                    Hypothesis nh = hyp;
                    nh.p_blank = p_total + lp[blank_id];
                    nh.p_nonblank = NEG_INF;
                    next_beams[prefix] = std::move(nh);
                } else {
                    it->second.p_blank = log_add(it->second.p_blank,
                                                  p_total + lp[blank_id]);
                }
            }

            /* ── Extend with each non-blank token ─────────────────────── */
            for (int c = 0; c < vocab_size; c++) {
                if (c == blank_id) continue;

                float log_p = lp[c];

                /* Build extended prefix */
                std::vector<int> new_prefix = prefix;
                new_prefix.push_back(c);

                float p_nb;
                if (!prefix.empty() && c == prefix.back()) {
                    /* Repeated token: can only extend from blank state */
                    p_nb = hyp.p_blank + log_p;
                    /* Also keep the unextended prefix alive via nonblank */
                    auto it = next_beams.find(prefix);
                    if (it == next_beams.end()) {
                        Hypothesis nh = hyp;
                        nh.p_blank = NEG_INF;
                        nh.p_nonblank = hyp.p_nonblank + log_p;
                        next_beams[prefix] = std::move(nh);
                    } else {
                        it->second.p_nonblank = log_add(it->second.p_nonblank,
                                                         hyp.p_nonblank + log_p);
                    }
                } else {
                    p_nb = p_total + log_p;
                }

                /* LM scoring at word boundaries */
                float new_lm_score = hyp.lm_score;
#ifdef USE_KENLM
                lm::ngram::State new_lm_state = hyp.lm_state;
                if (dec->has_lm && c < (int)dec->vocab.size()) {
                    if (is_word_start(dec->vocab[c]) && !prefix.empty()) {
                        std::string w = last_word(prefix, dec->vocab);
                        if (!w.empty()) {
                            lm::ngram::State tmp;
                            new_lm_score += score_word(dec->lm_model, hyp.lm_state,
                                                        w, &tmp);
                            new_lm_state = tmp;
                        }
                    }
                }
#endif

                auto it = next_beams.find(new_prefix);
                if (it == next_beams.end()) {
                    Hypothesis nh;
                    nh.tokens = new_prefix;
                    nh.p_blank = NEG_INF;
                    nh.p_nonblank = p_nb;
                    nh.lm_score = new_lm_score;
#ifdef USE_KENLM
                    if (dec->has_lm)
                        nh.lm_state = new_lm_state;
#endif
                    next_beams[new_prefix] = std::move(nh);
                } else {
                    it->second.p_nonblank = log_add(it->second.p_nonblank, p_nb);
                }
            }
        }

        /* Prune to top beam_size */
        std::vector<std::pair<std::vector<int>, Hypothesis>> sorted(
            next_beams.begin(), next_beams.end());

        auto score_fn = [lm_weight, word_score](const Hypothesis &h) {
            float s = h.total();
            if (lm_weight > 0.0f) s += lm_weight * h.lm_score;
            int n_words = 0;
            for (int id : h.tokens) (void)id, n_words++; /* rough approximation */
            s += word_score * n_words;
            return s;
        };

        std::partial_sort(sorted.begin(),
                          sorted.begin() + std::min(beam_size, (int)sorted.size()),
                          sorted.end(),
                          [&](const auto &a, const auto &b) {
                              return score_fn(a.second) > score_fn(b.second);
                          });
        if ((int)sorted.size() > beam_size)
            sorted.resize(beam_size);

        beams.clear();
        for (auto &[k, v] : sorted)
            beams[k] = std::move(v);
    }

    /* Find best hypothesis */
    const Hypothesis *best = nullptr;
    float best_score = NEG_INF;
    auto score_final = [&](const Hypothesis &h) {
        float s = h.total();
        if (lm_weight > 0.0f) s += lm_weight * h.lm_score;
        return s;
    };
    for (auto &[prefix, hyp] : beams) {
        float s = score_final(hyp);
        if (s > best_score) {
            best_score = s;
            best = &hyp;
        }
    }

    if (!best || best->tokens.empty()) {
        out[0] = '\0';
        return 0;
    }

    /* Convert tokens to text */
    std::string text = tokens_to_text(best->tokens, dec->vocab);

    /* Replace SentencePiece '▁' with space */
    std::string clean;
    clean.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (i + 2 < text.size() &&
            (unsigned char)text[i] == 0xE2 &&
            (unsigned char)text[i+1] == 0x96 &&
            (unsigned char)text[i+2] == 0x81) {
            clean += ' ';
            i += 3;
        } else {
            clean += text[i];
            i++;
        }
    }

    /* Trim leading space */
    const char *result = clean.c_str();
    while (*result == ' ') result++;

    int len = (int)strlen(result);
    if (len >= out_cap) len = out_cap - 1;
    memcpy(out, result, len);
    out[len] = '\0';
    return len;
}

void ctc_beam_destroy(CTCBeamDecoder *dec) {
    if (!dec) return;
#ifdef USE_KENLM
    delete dec->lm_model;
#endif
    delete dec;
}

} /* extern "C" */
