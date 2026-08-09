#include "ggml.h"

#if __has_include("gguf.h")
#include "gguf.h"
#elif __has_include("ggml-gguf.h")
#include "ggml-gguf.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <utility>

void copy_gguf_kv(struct gguf_context * ctx_in, struct gguf_context * ctx_out) {
    const int n_kv = gguf_get_n_kv(ctx_in);
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(ctx_in, i);
        const enum gguf_type type = gguf_get_kv_type(ctx_in, i);
        
        if (type == GGUF_TYPE_STRING) {
            gguf_set_val_str(ctx_out, key, gguf_get_val_str(ctx_in, i));
            continue;
        } 
        else if (type == GGUF_TYPE_ARRAY) {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx_in, i);
            const int arr_n = gguf_get_arr_n(ctx_in, i);
            
            if (arr_type == GGUF_TYPE_STRING) {
                std::vector<const char *> tmp_strs(arr_n);
                for (int j = 0; j < arr_n; ++j) {
                    tmp_strs[j] = gguf_get_arr_str(ctx_in, i, j);
                }
                gguf_set_arr_str(ctx_out, key, tmp_strs.data(), arr_n);
            } else {
                const void * arr_data = gguf_get_arr_data(ctx_in, i);
                gguf_set_arr_data(ctx_out, key, arr_type, arr_data, arr_n);
            }
            continue;
        }

        const void * data = gguf_get_val_data(ctx_in, i);
        if (!data) {
            continue;
        }

        if (type == GGUF_TYPE_UINT8) {
            gguf_set_val_u8(ctx_out, key, *(const uint8_t*)data);
        } else if (type == GGUF_TYPE_INT8) {
            gguf_set_val_i8(ctx_out, key, *(const int8_t*)data);
        } else if (type == GGUF_TYPE_UINT16) {
            gguf_set_val_u16(ctx_out, key, *(const uint16_t*)data);
        } else if (type == GGUF_TYPE_INT16) {
            gguf_set_val_i16(ctx_out, key, *(const int16_t*)data);
        } else if (type == GGUF_TYPE_UINT32) {
            gguf_set_val_u32(ctx_out, key, *(const uint32_t*)data);
        } else if (type == GGUF_TYPE_INT32) {
            gguf_set_val_i32(ctx_out, key, *(const int32_t*)data);
        } else if (type == GGUF_TYPE_FLOAT32) {
            gguf_set_val_f32(ctx_out, key, *(const float*)data);
        } else if (type == GGUF_TYPE_UINT64) {
            gguf_set_val_u64(ctx_out, key, *(const uint64_t*)data);
        } else if (type == GGUF_TYPE_INT64) {
            gguf_set_val_i64(ctx_out, key, *(const int64_t*)data);
        } else if (type == GGUF_TYPE_FLOAT64) {
            gguf_set_val_f64(ctx_out, key, *(const double*)data);
        } else if (type == GGUF_TYPE_BOOL) {
            gguf_set_val_bool(ctx_out, key, *(const bool*)data);
        }
    }
}
bool is_tensor_quantizable(const char * name, struct ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        return false;
    }
    
    const int n_dims = ggml_n_dims(tensor);
    if (n_dims < 2) {
        return false;
    }
    
    if (strstr(name, "_norm") != NULL) {
        return false;
    }
    if (strstr(name, ".bias") != NULL) {
        return false;
    }
    if (strstr(name, "_embd") != NULL) {
        return false;
    }
    if (strstr(name, "codebook") != NULL) {
        return false;
    }
    
    if (tensor->ne[0] % 256 != 0) {
        return false;
    }
    
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_f16.gguf> <output_q4_k.gguf>\n", argv[0]);
        return 1;
    }
    
    const char * fname_inp = argv[1];
    const char * fname_out = argv[2];
    
    struct ggml_init_params ggml_params = {};
    ggml_params.mem_size = 32 * 1024 * 1024;
    ggml_params.mem_buffer = NULL;
    ggml_params.no_alloc = true;
    struct ggml_context * ctx_ggml = ggml_init(ggml_params);
    
    struct gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = &ctx_ggml;
    
    struct gguf_context * ctx_inp = gguf_init_from_file(fname_inp, params);
    if (ctx_inp == NULL) {
        fprintf(stderr, "Failed to open input file.\n");
        return 1;
    }
    
    struct gguf_context * ctx_out = gguf_init_empty();
    if (ctx_out == NULL) {
        fprintf(stderr, "Failed to create output context.\n");
        return 1;
    }
    
    copy_gguf_kv(ctx_inp, ctx_out);
    
    FILE * f_inp = fopen(fname_inp, "rb");
    if (f_inp == NULL) {
        fprintf(stderr, "Failed to open input file for reading binary data.\n");
        return 1;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx_inp);
    
    struct ggml_init_params out_ggml_params = {};
    out_ggml_params.mem_size = 64 * 1024 * 1024;
    out_ggml_params.mem_buffer = NULL;
    out_ggml_params.no_alloc = true;
    struct ggml_context * ctx_out_ggml = ggml_init(out_ggml_params);
    
    std::vector<std::vector<uint8_t>> out_tensor_data;
    const int n_tensors = gguf_get_n_tensors(ctx_inp);
    out_tensor_data.reserve(n_tensors);
    
    struct ggml_tensor * cur = ggml_get_first_tensor(ctx_ggml);
    while (cur != NULL) {
        const char * name = ggml_get_name(cur);
        const int tensor_idx = gguf_find_tensor(ctx_inp, name);
        const size_t offset = gguf_get_tensor_offset(ctx_inp, tensor_idx);
        
        fseek(f_inp, data_offset + offset, SEEK_SET);
        
        const size_t raw_size = ggml_nbytes(cur);
        std::vector<uint8_t> raw_data(raw_size);
        fread(raw_data.data(), 1, raw_size, f_inp);
        
        if (is_tensor_quantizable(name, cur)) {
            const int64_t nelements = ggml_nelements(cur);
            std::vector<float> f32_data(nelements);
            
            if (cur->type == GGML_TYPE_F16) {
                ggml_fp16_t * f16_ptr = (ggml_fp16_t *)raw_data.data();
                for (int64_t k = 0; k < nelements; ++k) {
                    f32_data[k] = ggml_fp16_to_fp32(f16_ptr[k]);
                }
            } else {
                memcpy(f32_data.data(), raw_data.data(), raw_size);
            }
            
            const size_t q_size = ggml_row_size(GGML_TYPE_Q4_K, nelements);
            std::vector<uint8_t> q_data(q_size);
            
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f32_data.data(), q_data.data(), 0, nelements / cur->ne[0], cur->ne[0], nullptr);
            
            struct ggml_tensor * q_tensor = ggml_new_tensor(ctx_out_ggml, GGML_TYPE_Q4_K, ggml_n_dims(cur), cur->ne);
            ggml_set_name(q_tensor, name);
            
            out_tensor_data.push_back(std::move(q_data));
            q_tensor->data = out_tensor_data.back().data();
            
            gguf_add_tensor(ctx_out, q_tensor);
            printf("Quantized: %-40s | %s -> Q4_K\n", name, ggml_type_name(cur->type));
        } else {
            struct ggml_tensor * o_tensor = ggml_new_tensor(ctx_out_ggml, cur->type, ggml_n_dims(cur), cur->ne);
            ggml_set_name(o_tensor, name);
            
            out_tensor_data.push_back(std::move(raw_data));
            o_tensor->data = out_tensor_data.back().data();
            
            gguf_add_tensor(ctx_out, o_tensor);
            printf("Kept:      %-40s | %s (No change)\n", name, ggml_type_name(cur->type));
        }
        
        cur = ggml_get_next_tensor(ctx_ggml, cur);
    }
    
    fclose(f_inp);
    
    gguf_write_to_file(ctx_out, fname_out, false);
    
    gguf_free(ctx_out);
    gguf_free(ctx_inp);
    ggml_free(ctx_ggml);
    ggml_free(ctx_out_ggml);
    
    printf("Quantization complete: %s\n", fname_out);
    return 0;
}
