#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include "bitnet.h"

// Rotary embedding function (cosine and sine calculations)
void rotary_embedding(const std::vector<float> &inv_freq, Tensor2D &cos, Tensor2D &sin, size_t max_seq_len, size_t head_dim) {
    float angle;
    for (size_t s = 0; s < max_seq_len; ++s) {
        for (size_t d = 0; d < head_dim / 2; ++d) {
            angle = inv_freq[d] * s;
            cos[s][d] = std::cos(angle);
            cos[s][d + head_dim / 2] = cos[s][d];
            sin[s][d] = std::sin(angle);
            sin[s][d + head_dim / 2] = sin[s][d];
        }
    }
}

#endif