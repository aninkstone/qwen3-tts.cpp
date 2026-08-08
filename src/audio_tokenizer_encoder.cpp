#include "audio_tokenizer_encoder.h"
#include "gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

// ========================================================================
// PROFILER TOOL (Must be in global scope, strictly ASCII)
// ========================================================================
#include <chrono>
#include <string>
#include <iostream>

class TimeProfiler {
public:
    TimeProfiler(const std::string& name) : mName(name) {
        mStart = std::chrono::steady_clock::now();
        std::cerr << "[Profiler] START -> [" << mName << "] Execution started..." << std::endl;
    }
    
    ~TimeProfiler() {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - mStart).count();
        std::cerr << "[Profiler] END <- [" << mName << "] Execution completed! Duration: " << duration << " ms" << std::endl;
    }
private:
    std::string mName;
    std::chrono::steady_clock::time_point mStart;
};

#define INKER_PROFILE_SCOPE(name) TimeProfiler _profiler_##__LINE__(name)
// ========================================================================


#define QWEN3_TTS_MAX_NODES 16384

namespace qwen3_tts {

// Mel filterbank computation using librosa slaney normalization
static void compute_mel_filterbank_slaney(float * filterbank, int n_mels, int n_fft, 
                                           int sample_rate, float f_min, float f_max) {
    auto hz_to_mel_slaney = [](float hz) -> float {
        const float f_sp = 200.0f / 3.0f;  
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - 0.0f) / f_sp;  
        const float logstep = logf(6.4f) / 27.0f;  
        
        if (hz < min_log_hz) {
            return (hz - 0.0f) / f_sp;
        } else {
            return min_log_mel + logf(hz / min_log_hz) / logstep;
        }
    };
    
    auto mel_to_hz_slaney = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = (min_log_hz - 0.0f) / f_sp;
        const float logstep = logf(6.4f) / 27.0f;
        
        if (mel < min_log_mel) {
            return 0.0f + f_sp * mel;
        } else {
            return min_log_hz * expf(logstep * (mel - min_log_mel));
        }
    };
    
    float mel_min = hz_to_mel_slaney(f_min);
    float mel_max = hz_to_mel_slaney(f_max);
    
    int n_fft_bins = n_fft / 2 + 1;
    
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }
    
    std::vector<float> hz_points(n_mels + 2);
    std::vector<float> fft_freqs(n_fft_bins);
    
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz_slaney(mel_points[i]);
    }
    
    for (int i = 0; i < n_fft_bins; ++i) {
        fft_freqs[i] = (float)i * sample_rate / n_fft;
    }
    
    memset(filterbank, 0, n_mels * n_fft_bins * sizeof(float));
    
    for (int m = 0; m < n_mels; ++m) {
        float f_left = hz_points[m];
        float f_center = hz_points[m + 1];
        float f_right = hz_points[m + 2];
        
        float enorm = 2.0f / (f_right - f_left);
        
        for (int k = 0; k < n_fft_bins; ++k) {
            float freq = fft_freqs[k];
            
            if (freq >= f_left && freq <= f_center) {
                if (f_center > f_left) {
                    filterbank[m * n_fft_bins + k] = enorm * (freq - f_left) / (f_center - f_left);
                }
            } else if (freq > f_center && freq <= f_right) {
                if (f_right > f_center) {
                    filterbank[m * n_fft_bins + k] = enorm * (f_right - freq) / (f_right - f_center);
                }
            }
        }
    }
}

// ========================================================================
// REPLACED: High-performance Cooley-Tukey FFT O(N log N) instead of O(N^2) DFT
// ========================================================================
static void compute_fft(const float * input, float * real, float * imag, int n) {
    for (int i = 0; i < n; ++i) {
        real[i] = input[i];
        imag[i] = 0.0f;
    }
    
    for (int i = 0, j = 0; i < n; ++i) {
        if (i < j) {
            float temp_r = real[i];
            real[i] = real[j];
            real[j] = temp_r;
            
            float temp_i = imag[i];
            imag[i] = imag[j];
            imag[j] = temp_i;
        }
        int bit = n >> 1;
        for (; (bit & j) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
    }
    
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wlen_cos = cosf(ang);
        float wlen_sin = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float w_cos = 1.0f;
            float w_sin = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                int idx1 = i + j;
                int idx2 = i + j + len / 2;
                
                float u_real = real[idx1];
                float u_imag = imag[idx1];
                
                float v_real = real[idx2] * w_cos - imag[idx2] * w_sin;
                float v_imag = real[idx2] * w_sin + imag[idx2] * w_cos;
                
                real[idx1] = u_real + v_real;
                imag[idx1] = u_imag + v_imag;
                
                real[idx2] = u_real - v_real;
                imag[idx2] = u_imag - v_imag;
                
                float next_w_cos = w_cos * wlen_cos - w_sin * wlen_sin;
                float next_w_sin = w_cos * wlen_sin + w_sin * wlen_cos;
                w_cos = next_w_cos;
                w_sin = next_w_sin;
            }
        }
    }
}

// Periodic Hann window
static void compute_hann_window(float * window, int n) {
    for (int i = 0; i < n; ++i) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / n));
    }
}

// Compute centered window for STFT
static void compute_centered_window(float * window, int n_fft, int win_length) {
    memset(window, 0, n_fft * sizeof(float));
    int offset = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; ++i) {
        window[offset + i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / win_length));
    }
}

AudioTokenizerEncoder::AudioTokenizerEncoder() = default;

AudioTokenizerEncoder::~AudioTokenizerEncoder() {
    free_speaker_encoder_model(model_);
    
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
}

bool AudioTokenizerEncoder::load_model(const std::string & model_path) {
    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }
    
    model_.config.sample_rate = loader.get_u32("qwen3-tts.speaker_encoder.sample_rate", 24000);
    model_.config.embedding_dim = loader.get_u32("qwen3-tts.speaker_encoder.embedding_length", 1024);
    
    int64_t n_tensors = loader.get_n_tensors();
    int spk_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "spk_enc.", 8) == 0) {
            spk_tensor_count++;
        }
    }
    
    if (spk_tensor_count == 0) {
        error_msg_ = "No speaker encoder tensors found in model";
        return false;
    }
    
    size_t ctx_size = ggml_tensor_overhead() * spk_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }
    
    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "spk_enc.", 8) != 0) {
            continue;
        }
        
        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }
        
        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        
        model_.tensors[name] = tensor;
        
        std::string sname(name);
        
        if (sname == "spk_enc.conv0.weight") model_.conv0_w = tensor;
        else if (sname == "spk_enc.conv0.bias") model_.conv0_b = tensor;
        else if (sname == "spk_enc.mfa.weight") model_.mfa_w = tensor;
        else if (sname == "spk_enc.mfa.bias") model_.mfa_b = tensor;
        else if (sname == "spk_enc.asp.conv.weight") model_.asp_conv_w = tensor;
        else if (sname == "spk_enc.asp.conv.bias") model_.asp_conv_b = tensor;
        else if (sname == "spk_enc.asp.tdnn.weight") model_.asp_tdnn_w = tensor;
        else if (sname == "spk_enc.asp.tdnn.bias") model_.asp_tdnn_b = tensor;
        else if (sname == "spk_enc.fc.weight") model_.fc_w = tensor;
        else if (sname == "spk_enc.fc.bias") model_.fc_b = tensor;
        else {
            int blk_idx, res_idx;
            char suffix[64];
            
            if (sscanf(name, "spk_enc.blk.%d.tdnn1.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].tdnn1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].tdnn1_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.tdnn2.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].tdnn2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].tdnn2_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.res2net.%d.%s", &blk_idx, &res_idx, suffix) == 3) {
                if (blk_idx >= 1 && blk_idx <= 3 && res_idx >= 0 && res_idx < 7) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].res2net_w[res_idx] = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].res2net_b[res_idx] = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.se.conv1.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].se_conv1_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].se_conv1_b = tensor;
                }
            }
            else if (sscanf(name, "spk_enc.blk.%d.se.conv2.%s", &blk_idx, suffix) == 2) {
                if (blk_idx >= 1 && blk_idx <= 3) {
                    if (strcmp(suffix, "weight") == 0) model_.blocks[blk_idx-1].se_conv2_w = tensor;
                    else if (strcmp(suffix, "bias") == 0) model_.blocks[blk_idx-1].se_conv2_b = tensor;
                }
            }
        }
    }
    
    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx, 
                                     model_.tensors, model_.buffer, error_msg_)) {
        return false;
    }
    
    state_.backend = init_preferred_backend("AudioTokenizerEncoder", &error_msg_);
    if (!state_.backend) {
        return false;
    }
    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * device_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioTokenizerEncoder backend: %s\n", device_name);
    
    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "Failed to initialize CPU fallback backend for AudioTokenizerEncoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) {
        backends.push_back(state_.backend_cpu);
    }
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), QWEN3_TTS_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TTS_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

bool AudioTokenizerEncoder::compute_mel_spectrogram(const float * samples, int32_t n_samples,
                                                     std::vector<float> & mel, int32_t & n_frames) {
    const auto & cfg = model_.config;
    
    int padding = (cfg.n_fft - cfg.hop_length) / 2;
    int padded_length = n_samples + 2 * padding;
    
    std::vector<float> padded(padded_length);
    for (int i = 0; i < padded_length; ++i) {
        int src_idx = 0;
        if (i < padding) {
            src_idx = padding - i;
        } 
        else {
            if (i >= padding + n_samples) {
                src_idx = 2 * n_samples - (i - padding) - 2;
            } 
            else {
                src_idx = i - padding;
            }
        }
        src_idx = std::max(0, std::min(n_samples - 1, src_idx));
        padded[i] = samples[src_idx];
    }
    
    n_frames = (padded_length - cfg.n_fft) / cfg.hop_length + 1;
    if (n_frames <= 0) {
        error_msg_ = "Audio too short for mel spectrogram";
        return false;
    }
    
    int n_fft_bins = cfg.n_fft / 2 + 1;
    
    std::vector<float> filterbank(cfg.n_mels * n_fft_bins);
    compute_mel_filterbank_slaney(filterbank.data(), cfg.n_mels, cfg.n_fft, 
                                   cfg.sample_rate, cfg.f_min, cfg.f_max);
    
    std::vector<float> window(cfg.n_fft);
    compute_centered_window(window.data(), cfg.n_fft, cfg.win_length);
    
    mel.resize(cfg.n_mels * n_frames);
    
    // OpenMP parallelized frame-by-frame loop with local thread buffers
    #pragma omp parallel for schedule(dynamic)
    for (int32_t f = 0; f < n_frames; ++f) {
        int start = f * cfg.hop_length;
        
        std::vector<float> local_frame(cfg.n_fft, 0.0f);
        std::vector<float> local_fft_real(cfg.n_fft);
        std::vector<float> local_fft_imag(cfg.n_fft);
        std::vector<float> local_magnitude(n_fft_bins);
        
        for (int i = 0; i < cfg.n_fft; ++i) {
            local_frame[i] = padded[start + i] * window[i];
        }
        
        compute_fft(local_frame.data(), local_fft_real.data(), local_fft_imag.data(), cfg.n_fft);
        
        for (int k = 0; k < n_fft_bins; ++k) {
            local_magnitude[k] = sqrtf(local_fft_real[k] * local_fft_real[k] + local_fft_imag[k] * local_fft_imag[k] + 1e-9f);
        }
        
        for (int m = 0; m < cfg.n_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_fft_bins; ++k) {
                sum += filterbank[m * n_fft_bins + k] * local_magnitude[k];
            }
            mel[m * n_frames + f] = logf(std::max(sum, 1e-5f));
        }
    }
    
    return true;
}

static struct ggml_tensor * apply_reflect_pad_1d(struct ggml_context * ctx,
                                                  struct ggml_tensor * x,
                                                  int pad) {
    if (pad == 0) {
        return x;
    }
    
    int64_t T = x->ne[0];
    int64_t C = x->ne[1];
    int64_t B = x->ne[2];
    
    struct ggml_tensor * left_slices[16];
    struct ggml_tensor * right_slices[16];
    
    for (int i = 0; i < pad && i < 16; ++i) {
        int left_src_idx = pad - i;
        left_slices[i] = ggml_view_3d(ctx, x, 1, C, B,
                                       x->nb[1], x->nb[2],
                                       left_src_idx * x->nb[0]);
        left_slices[i] = ggml_cont(ctx, left_slices[i]);
        
        int right_src_idx = T - 2 - i;
        right_slices[i] = ggml_view_3d(ctx, x, 1, C, B,
                                        x->nb[1], x->nb[2],
                                        right_src_idx * x->nb[0]);
        right_slices[i] = ggml_cont(ctx, right_slices[i]);
    }
    
    struct ggml_tensor * left_pad = left_slices[0];
    for (int i = 1; i < pad && i < 16; ++i) {
        left_pad = ggml_concat(ctx, left_pad, left_slices[i], 0);
    }
    
    struct ggml_tensor * right_pad = right_slices[0];
    for (int i = 1; i < pad && i < 16; ++i) {
        right_pad = ggml_concat(ctx, right_pad, right_slices[i], 0);
    }
    
    struct ggml_tensor * padded = ggml_concat(ctx, left_pad, x, 0);
    padded = ggml_concat(ctx, padded, right_pad, 0);
    
    return padded;
}

static struct ggml_tensor * apply_conv1d(struct ggml_context * ctx,
                                          struct ggml_tensor * w,
                                          struct ggml_tensor * b,
                                          struct ggml_tensor * x,
                                          int stride, int pad, int dilation,
                                          const char * debug_name = nullptr,
                                          bool use_reflect_pad = true) {
    struct ggml_tensor * input = x;
    int actual_pad = pad;
    
    if (use_reflect_pad && pad > 0) {
        input = apply_reflect_pad_1d(ctx, x, pad);
        actual_pad = 0;
    }
    
    struct ggml_tensor * y = ggml_conv_1d(ctx, w, input, stride, actual_pad, dilation);
    if (debug_name) {
        char name[64];
        snprintf(name, sizeof(name), "%s_conv", debug_name);
        ggml_set_name(y, name);
    }
    if (b) {
        int64_t oc = y->ne[1];
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return y;
}

struct ggml_cgraph * AudioTokenizerEncoder::build_graph(int32_t n_frames) {
    const auto & cfg = model_.config;
    const int hidden_dim = cfg.hidden_dim;  // 512
    const int scale = cfg.res2net_scale;    // 8
    const int branch_dim = hidden_dim / scale;  // 64
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TTS_MAX_NODES, false);
    
     struct ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, cfg.n_mels);
     ggml_set_name(mel, "mel");
     ggml_set_input(mel);
     
     struct ggml_tensor * cur = ggml_reshape_3d(ctx0, mel, n_frames, cfg.n_mels, 1);
     ggml_set_name(cur, "mel_3d");
     
     struct ggml_tensor * mel_padded = apply_reflect_pad_1d(ctx0, cur, 2);
     ggml_set_name(mel_padded, "mel_padded");
    
     cur = ggml_conv_1d(ctx0, model_.conv0_w, mel_padded, 1, 0, 1);
     ggml_set_name(cur, "conv0_conv");

     if (model_.conv0_b) {
         int64_t oc = cur->ne[1];
         cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.conv0_b, 1, oc, 1));
     }
     ggml_set_name(cur, "conv0_pre_relu");
     cur = ggml_relu(ctx0, cur);
     ggml_set_name(cur, "conv0_out");
    
    int64_t seq_len = cur->ne[0];
    
    struct ggml_tensor * block_outputs[4];
    block_outputs[0] = cur;  
    
    int dilations[3] = {2, 3, 4};
    
    for (int blk = 0; blk < 3; ++blk) {
        const auto & block = model_.blocks[blk];
        int dilation = dilations[blk];
        
        struct ggml_tensor * residual = cur;
        
         cur = apply_conv1d(ctx0, block.tdnn1_w, block.tdnn1_b, cur, 1, 0, 1);
         cur = ggml_relu(ctx0, cur);
         if (blk == 0) {
             ggml_set_name(cur, "blk1_tdnn1");
         }
        
        struct ggml_tensor * branches[8];
        
        for (int b = 0; b < scale; ++b) {
            branches[b] = ggml_view_3d(ctx0, cur, 
                                        seq_len, branch_dim, 1,
                                        cur->nb[1], cur->nb[2], 
                                        b * branch_dim * cur->nb[1]);
            branches[b] = ggml_cont(ctx0, branches[b]);
        }
        
        struct ggml_tensor * outputs[8];
        outputs[0] = branches[0];  
        
        for (int b = 1; b < scale; ++b) {
            struct ggml_tensor * input;
            if (b == 1) {
                input = branches[b];
            } else {
                input = ggml_add(ctx0, branches[b], outputs[b - 1]);
            }
            
            if (block.res2net_w[b - 1]) {
                outputs[b] = apply_conv1d(ctx0, block.res2net_w[b - 1], block.res2net_b[b - 1], 
                                          input, 1, dilation, dilation);
                outputs[b] = ggml_relu(ctx0, outputs[b]);
            } else {
                outputs[b] = input;  
            }
        }
        
         cur = outputs[0];
         for (int b = 1; b < scale; ++b) {
             cur = ggml_concat(ctx0, cur, outputs[b], 1);
         }
         if (blk == 0) {
             ggml_set_name(cur, "blk1_res2net");
             for (int b = 0; b < scale; ++b) {
                 char name[32];
                 snprintf(name, sizeof(name), "blk1_branch%d", b);
                 ggml_set_name(outputs[b], name);
             }
         }
        
         cur = apply_conv1d(ctx0, block.tdnn2_w, block.tdnn2_b, cur, 1, 0, 1);
         cur = ggml_relu(ctx0, cur);
         if (blk == 0) {
             ggml_set_name(cur, "blk1_tdnn2");
         }
         
         struct ggml_tensor * se = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
         se = ggml_reshape_3d(ctx0, se, 1, hidden_dim, 1);
         
         se = apply_conv1d(ctx0, block.se_conv1_w, block.se_conv1_b, se, 1, 0, 1);
         se = ggml_relu(ctx0, se);
         
         se = apply_conv1d(ctx0, block.se_conv2_w, block.se_conv2_b, se, 1, 0, 1);
         se = ggml_sigmoid(ctx0, se);
         
         cur = ggml_mul(ctx0, cur, se);
         if (blk == 0) {
             ggml_set_name(cur, "blk1_se");
         }

         cur = ggml_add(ctx0, cur, residual);
        
        char block_name[32];
        snprintf(block_name, sizeof(block_name), "block_%d", blk + 1);
        ggml_set_name(cur, block_name);
        
        block_outputs[blk + 1] = cur;
    }
    
     struct ggml_tensor * mfa_input = ggml_concat(ctx0, block_outputs[1], block_outputs[2], 1);
     mfa_input = ggml_concat(ctx0, mfa_input, block_outputs[3], 1);
     ggml_set_name(mfa_input, "mfa_input");
     
     cur = apply_conv1d(ctx0, model_.mfa_w, model_.mfa_b, mfa_input, 1, 0, 1);
     cur = ggml_relu(ctx0, cur);
     ggml_set_name(cur, "mfa_out");
    
    struct ggml_tensor * global_mean = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    global_mean = ggml_reshape_3d(ctx0, global_mean, 1, 1536, 1);
    
    struct ggml_tensor * sq = ggml_sqr(ctx0, cur);
    struct ggml_tensor * mean_sq = ggml_pool_1d(ctx0, sq, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    mean_sq = ggml_reshape_3d(ctx0, mean_sq, 1, 1536, 1);
    struct ggml_tensor * var = ggml_sub(ctx0, mean_sq, ggml_sqr(ctx0, global_mean));
    var = ggml_clamp(ctx0, var, 1e-12f, 1e10f);
    struct ggml_tensor * global_std = ggml_sqrt(ctx0, var);
    
    struct ggml_tensor * mean_expanded = ggml_repeat(ctx0, global_mean, 
                                                      ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * std_expanded = ggml_repeat(ctx0, global_std,
                                                     ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    
    struct ggml_tensor * attention = ggml_concat(ctx0, cur, mean_expanded, 1);
    attention = ggml_concat(ctx0, attention, std_expanded, 1);
    
     attention = apply_conv1d(ctx0, model_.asp_tdnn_w, model_.asp_tdnn_b, attention, 1, 0, 1);
     attention = ggml_relu(ctx0, attention);  
     ggml_set_name(attention, "asp_tdnn");
     attention = ggml_tanh(ctx0, attention);  
     
     attention = apply_conv1d(ctx0, model_.asp_conv_w, model_.asp_conv_b, attention, 1, 0, 1);
     ggml_set_name(attention, "asp_conv");
     
     attention = ggml_soft_max(ctx0, attention);
     ggml_set_name(attention, "asp_softmax");
    
    struct ggml_tensor * weighted = ggml_mul(ctx0, attention, cur);
    struct ggml_tensor * weighted_mean = ggml_pool_1d(ctx0, weighted, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    weighted_mean = ggml_scale(ctx0, weighted_mean, (float)seq_len);  
    weighted_mean = ggml_reshape_3d(ctx0, weighted_mean, 1, 1536, 1);
    
    struct ggml_tensor * mean_for_std = ggml_repeat(ctx0, weighted_mean,
                                                     ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, seq_len, 1536, 1));
    struct ggml_tensor * diff = ggml_sub(ctx0, cur, mean_for_std);
    struct ggml_tensor * diff_sq = ggml_sqr(ctx0, diff);
    struct ggml_tensor * weighted_var = ggml_mul(ctx0, attention, diff_sq);
    struct ggml_tensor * var_sum = ggml_pool_1d(ctx0, weighted_var, GGML_OP_POOL_AVG, seq_len, seq_len, 0);
    var_sum = ggml_scale(ctx0, var_sum, (float)seq_len);  
    var_sum = ggml_reshape_3d(ctx0, var_sum, 1, 1536, 1);
    var_sum = ggml_clamp(ctx0, var_sum, 1e-12f, 1e10f);
    struct ggml_tensor * weighted_std = ggml_sqrt(ctx0, var_sum);
    
     struct ggml_tensor * pooled = ggml_concat(ctx0, weighted_mean, weighted_std, 1);
     ggml_set_name(pooled, "asp_pooled");
     
     cur = apply_conv1d(ctx0, model_.fc_w, model_.fc_b, pooled, 1, 0, 1);
     ggml_set_name(cur, "fc_out");
    
    cur = ggml_reshape_1d(ctx0, cur, cfg.embedding_dim);
    
    ggml_set_name(cur, "embedding");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    
    return gf;
}

bool AudioTokenizerEncoder::encode(const float * samples, int32_t n_samples,
                                    std::vector<float> & embedding) {
    if (!model_.ctx) {
        error_msg_ = "Model not loaded";
        return false;
    }
    
    std::vector<float> mel;
    int32_t n_frames;
    
    {
        INKER_PROFILE_SCOPE("Encoder Stage 1: compute_mel_spectrogram");
        if (!compute_mel_spectrogram(samples, n_samples, mel, n_frames)) {
            return false;
        }
    }
    
    struct ggml_cgraph * gf = nullptr;
    {
        INKER_PROFILE_SCOPE("Encoder Stage 2: build_graph and alloc");
        gf = build_graph(n_frames);

        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            error_msg_ = "Failed to allocate graph";
            return false;
        }
    }
    
    struct ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf, "mel");
    if (!mel_tensor) {
        error_msg_ = "Failed to find mel tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    {
        INKER_PROFILE_SCOPE("Encoder Stage 3: Tensor Set (RAM to GPU)");
        ggml_backend_tensor_set(mel_tensor, mel.data(), 0, mel.size() * sizeof(float));
    }
    
    {
        INKER_PROFILE_SCOPE("Encoder Stage 4: graph_compute");
        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "Failed to compute graph";
            ggml_backend_sched_reset(state_.sched);
            return false;
        }
    }

    struct ggml_tensor * emb_tensor = ggml_graph_get_tensor(gf, "embedding");
    if (!emb_tensor) {
        error_msg_ = "Failed to find embedding tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    
    embedding.resize(model_.config.embedding_dim);
    {
        INKER_PROFILE_SCOPE("Encoder Stage 5: Tensor Get (GPU to RAM)");
        ggml_backend_tensor_get(emb_tensor, embedding.data(), 0, embedding.size() * sizeof(float));
    }
    
    ggml_backend_sched_reset(state_.sched);
    
    return true;
}

void free_speaker_encoder_model(speaker_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
