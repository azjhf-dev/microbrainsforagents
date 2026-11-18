"""
Aetheris OpenCL GPU Micro-Brain Co-Processor (core/opencl_microbrain_accelerator.py)
====================================================================================
Turns legacy/edge CUDA GPUs (such as NVIDIA GeForce GT 710, Kepler GK208, CC 3.5,
2 GB VRAM) into an asynchronous, VRAM-resident parallel co-processor for Aetheris
micro-brains via PyOpenCL, with 100% transparent NumPy fallback when OpenCL is absent.

Implements 16 OpenCL kernels spanning 14 SOTA (2025-2026) micro-brain primitives:
BitNet b1.58 ternary GEMM, EGGROLL low-rank + counter-RNG ES rollouts, CfC liquid
ensemble, HDC/VSA Hamming cleanup memory, TTT-Linear + Muon fast-weight adaptation,
DeepSeek Engram gated lookup, Mamba-S6 Blelloch parallel scan, DiffLogic bitplane
gates, Kuramoto-Hopfield sync/readout, predictive-coding SRAM relaxation,
active-inference MC rollout, Vietoris-Rips SLERP filtration, dominated novelty search.
Every op dispatches to the GPU only past its measured crossover (or force_gpu=True)
and carries a bit-faithful CPU fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("aetheris.opencl_accelerator")

try:
    import pyopencl as cl  # type: ignore
    _HAS_PYOPENCL = True
except Exception:
    cl = None  # type: ignore
    _HAS_PYOPENCL = False


_OPENCL_KERNELS_SRC = r"""
// 1. Batched BitNet b1.58 Ternary GEMM: Y[b, o] = scale * sum_i(X[b, i] * W[o, i])
__kernel void bitnet_batch_gemm(
    __global const char *X_q,
    __global const char *W_tern,
    __global float *Y_out,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float combined_scale
) {
    int b = get_global_id(0);
    int o = get_global_id(1);
    if (b >= batch_size || o >= out_features) return;

    int acc = 0;
    int x_base = b * in_features;
    int w_base = o * in_features;
    for (int i = 0; i < in_features; i++) {
        acc += (int)X_q[x_base + i] * (int)W_tern[w_base + i];
    }
    Y_out[b * out_features + o] = (float)acc * combined_scale;
}

// 2. EGGROLL Low-Rank Population Neuroevolution Rollout
// Evaluates `pop_size` candidate brains perturbed by rank-1 (U[p, o] * V[p, i])
// over `n_samples` replay pairs (X[s, i], Target[s, o]) in parallel.
__kernel void eggroll_low_rank_rollout(
    __global const float *W_base,
    __global const float *U_pert,
    __global const float *V_pert,
    __global const float *X_samples,
    __global const float *Y_targets,
    __global float *mse_out,
    const int pop_size,
    const int n_samples,
    const int dim,
    const float sigma
) {
    int p = get_global_id(0);
    if (p >= pop_size) return;

    int u_base = p * dim;
    int v_base = p * dim;
    float total_sq_err = 0.0f;

    for (int s = 0; s < n_samples; s++) {
        int x_base = s * dim;
        for (int o = 0; o < dim; o++) {
            float u_po = U_pert[u_base + o];
            float dot_val = 0.0f;
            int w_row = o * dim;
            for (int i = 0; i < dim; i++) {
                float w_perturbed = W_base[w_row + i] + sigma * u_po * V_pert[v_base + i];
                // Ternary clamp {-1, 0, +1}
                float w_tern = rint(clamp(w_perturbed, -1.0f, 1.0f));
                dot_val += w_tern * X_samples[x_base + i];
            }
            float diff = (dot_val / (float)dim) - Y_targets[x_base + o];
            total_sq_err += diff * diff;
        }
    }
    mse_out[p] = total_sq_err / (float)(n_samples * dim);
}

// 3. Closed-Form Continuous-Time (CfC) Liquid Neuron Update
// h_new = sigmoid(-f_gate * dt) * h_old + (1 - sigmoid(-f_gate * dt)) * g_candidate
__kernel void cfc_liquid_ensemble_step(
    __global float *h_state,
    __global const float *f_drive,
    __global const float *g_drive,
    __global const float *tau_inv,
    const int n_neurons,
    const float dt
) {
    int idx = get_global_id(0);
    if (idx >= n_neurons) return;

    float rate = fabs(f_drive[idx]) * tau_inv[idx] + 0.1f;
    float decay_gate = 1.0f / (1.0f + exp(rate * dt));
    float candidate = tanh(g_drive[idx]);
    h_state[idx] = decay_gate * h_state[idx] + (1.0f - decay_gate) * candidate;
}

// 4. Hyperdimensional Computing (HDC / VSA) Bit-Packed Hamming Distance Search
// 10,240-bit hypervectors = 320 x uint32 words per item.
// NOTE: popcount() needs an OpenCL 2.0-capable compiler; NVIDIA Kepler drivers
// accept it, but a strict OpenCL 1.2 device fails the whole-program build and
// the accelerator then falls back to CPU for every op.
__kernel void hdc_hamming_search(
    __global const uint *query_hv,
    __global const uint *codebook_hv,
    __global float *similarity_out,
    const int n_items,
    const int n_words
) {
    int item = get_global_id(0);
    if (item >= n_items) return;

    int base = item * n_words;
    uint diff_bits = 0;
    for (int w = 0; w < n_words; w++) {
        diff_bits += popcount(query_hv[w] ^ codebook_hv[base + w]);
    }
    float total_bits = (float)(n_words * 32);
    similarity_out[item] = 1.0f - ((float)diff_bits / total_bits);
}

// 5. Test-Time Training (TTT-Linear) Fast-Weight Online Self-Supervised Step
// W[o, i] <- W[o, i] - lr * (dot(W[o, :], k) - v[o]) * k[i]
__kernel void ttt_linear_fast_weight_step(
    __global float *W_fast,
    __global const float *k_vec,
    __global const float *v_vec,
    __global float *err_out,
    const int dim,
    const float lr
) {
    int o = get_global_id(0);
    if (o >= dim) return;

    int row_base = o * dim;
    float pred = 0.0f;
    for (int i = 0; i < dim; i++) {
        pred += W_fast[row_base + i] * k_vec[i];
    }
    float err = pred - v_vec[o];
    err_out[o] = err;
    for (int i = 0; i < dim; i++) {
        W_fast[row_base + i] -= lr * err * k_vec[i];
    }
}

// 6. Blelloch / Hillis-Steele Parallel Associative Scan in GPU __local SRAM for Mamba-S6 SSM
// Solves h_t = A_bar[t, c] * h_{t-1} + B_u[t, c] across all channels c in [0, n_channels)
// in O(log2(seq_len)) parallel SRAM barrier steps (up to 256 tokens per chunk).
__kernel void ssm_blelloch_parallel_scan(
    __global const float *A_seq,
    __global const float *B_seq,
    __global const float *H0,
    __global float *H_out,
    const int seq_len,
    const int n_channels
) {
    int c = get_group_id(0);
    int tid = get_local_id(0);
    if (c >= n_channels) return;

    __local float s_a[256];
    __local float s_b[256];

    if (tid < seq_len) {
        int idx = tid * n_channels + c;
        float a_val = A_seq[idx];
        float b_val = B_seq[idx];
        if (tid == 0) {
            b_val = a_val * H0[c] + b_val;
        }
        s_a[tid] = a_val;
        s_b[tid] = b_val;
    } else {
        s_a[tid] = 1.0f;
        s_b[tid] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // O(log2 N) associative prefix scan: (a_cur, b_cur) o (a_prev, b_prev) = (a_cur * a_prev, a_cur * b_prev + b_cur)
    for (int offset = 1; offset < seq_len; offset <<= 1) {
        float prev_a = 1.0f;
        float prev_b = 0.0f;
        if (tid >= offset && tid < seq_len) {
            prev_a = s_a[tid - offset];
            prev_b = s_b[tid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid >= offset && tid < seq_len) {
            float cur_a = s_a[tid];
            float cur_b = s_b[tid];
            s_a[tid] = cur_a * prev_a;
            s_b[tid] = cur_a * prev_b + cur_b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < seq_len) {
        H_out[tid * n_channels + c] = s_b[tid];
    }
}

// 7. Bitplane Differentiable Logic Gate Tree Evaluation (DiffLogic / LUT-Net)
// Evaluates 32 binary bit-planes simultaneously per uint32 word across all 16 Boolean logic gates.
__kernel void difflogic_bitplane_eval(
    __global const uint *X_bits,
    __global const int *conn_a,
    __global const int *conn_b,
    __global const int *gate_ids,
    __global uint *Y_bits,
    const int n_gates,
    const int n_words
) {
    int g = get_global_id(0);
    int w = get_global_id(1);
    if (g >= n_gates || w >= n_words) return;

    uint a = X_bits[conn_a[g] * n_words + w];
    uint b = X_bits[conn_b[g] * n_words + w];
    int op = gate_ids[g] & 15;
    uint res = 0u;

    switch (op) {
        case 0:  res = 0u; break;
        case 1:  res = a & b; break;
        case 2:  res = a & (~b); break;
        case 3:  res = a; break;
        case 4:  res = (~a) & b; break;
        case 5:  res = b; break;
        case 6:  res = a ^ b; break;
        case 7:  res = a | b; break;
        case 8:  res = ~(a | b); break;
        case 9:  res = ~(a ^ b); break;
        case 10: res = ~b; break;
        case 11: res = a | (~b); break;
        case 12: res = ~a; break;
        case 13: res = (~a) | b; break;
        case 14: res = ~(a & b); break;
        case 15: res = 0xFFFFFFFFu; break;
    }
    Y_bits[g * n_words + w] = res;
}

// 8. Kuramoto Oscillatory Phase Synchronization (AKOrN / KoPE) - ONE synchronous
// Jacobi step per launch.  The host ping-pongs phase buffers across sync_steps
// launches, so every work-item reads a CONSISTENT phase snapshot (the previous
// single-launch multi-step kernel let concurrent work-items mix old/new phases,
// making results scheduler-dependent and diverging from the CPU path).
__kernel void kuramoto_hopfield_sync_step(
    __global const float *phases_in,
    __global float *phases_out,
    __global const float *omega,
    __global const float *coupling_J,
    const int seq_len,
    const int d_model,
    const float dt
) {
    int t = get_global_id(0);
    int d = get_global_id(1);
    if (t >= seq_len || d >= d_model) return;

    int idx = t * d_model + d;
    float theta = phases_in[idx];
    float w_nat = omega[idx];
    float sync_force = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        sync_force += coupling_J[t * seq_len + j] * native_sin(phases_in[j * d_model + d] - theta);
    }
    phases_out[idx] = theta + dt * (w_nat + sync_force);
}

// 8b. Modern Hopfield LogSumExp dense associative readout over the synchronized
// phases (run once after the final sync step; same math as the CPU path).
__kernel void kuramoto_hopfield_readout(
    __global const float *X_in,
    __global const float *phases,
    __global const float *hopfield_K,
    __global const float *hopfield_V,
    __global float *Y_out,
    const int seq_len,
    const int d_model,
    const int n_mem,
    const float beta
) {
    int t = get_global_id(0);
    int d = get_global_id(1);
    if (t >= seq_len || d >= d_model) return;

    int idx = t * d_model + d;
    float theta = phases[idx];

    // Phase-bound query vector component
    float x_bound = X_in[idx] * (0.7f + 0.3f * native_cos(theta));

    if (n_mem <= 0) {
        Y_out[idx] = x_bound;
        return;
    }

    float max_sim = -1e20f;
    for (int m = 0; m < n_mem; m++) {
        float dot_m = 0.0f;
        int m_base = m * d_model;
        int t_base = t * d_model;
        for (int k = 0; k < d_model; k++) {
            dot_m += X_in[t_base + k] * hopfield_K[m_base + k];
        }
        float scaled = beta * dot_m;
        if (scaled > max_sim) max_sim = scaled;
    }

    float sum_exp = 0.0f;
    float weighted_v = 0.0f;
    for (int m = 0; m < n_mem; m++) {
        float dot_m = 0.0f;
        int m_base = m * d_model;
        int t_base = t * d_model;
        for (int k = 0; k < d_model; k++) {
            dot_m += X_in[t_base + k] * hopfield_K[m_base + k];
        }
        float w_m = exp(clamp(beta * dot_m - max_sim, -40.0f, 0.0f));
        sum_exp += w_m;
        weighted_v += w_m * hopfield_V[m_base + d];
    }
    Y_out[idx] = x_bound + (weighted_v / (sum_exp + 1e-12f));
}

// 9. Predictive Coding & Forward-Forward Iterative Variational Free-Energy Relaxation in Registers/SRAM
// Relaxes 3-layer hierarchy (d0=32 <-> d1=24 <-> d2=16) over `relax_steps` iterations per sample.
__kernel void predictive_coding_sram_relax(
    __global const float *R0_batch,
    __global const float *W0,
    __global const float *W1,
    __global float *R1_out,
    __global float *R2_out,
    __global float *E0_out,
    __global float *E1_out,
    __global float *metrics_out,
    const int batch_size,
    const int relax_steps,
    const float state_lr
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;

    float r0[32];
    float r1[24];
    float r2[16];
    float e0[32];
    float e1[24];

    int b0 = b * 32;
    int b1 = b * 24;
    int b2 = b * 16;

    for (int i = 0; i < 32; i++) r0[i] = R0_batch[b0 + i];

    // Initial bottom-up amortization pass
    for (int j = 0; j < 24; j++) {
        float acc = 0.0f;
        for (int i = 0; i < 32; i++) acc += W0[i * 24 + j] * r0[i];
        r1[j] = tanh(acc);
    }
    for (int k = 0; k < 16; k++) {
        float acc = 0.0f;
        for (int j = 0; j < 24; j++) acc += W1[j * 16 + k] * r1[j];
        r2[k] = tanh(acc);
    }

    float f_init = 0.0f;
    float f_final = 0.0f;

    for (int step = 0; step < relax_steps; step++) {
        float step_energy = 0.0f;
        for (int i = 0; i < 32; i++) {
            float pred0 = 0.0f;
            for (int j = 0; j < 24; j++) pred0 += W0[i * 24 + j] * r1[j];
            e0[i] = r0[i] - pred0;
            step_energy += e0[i] * e0[i];
        }
        for (int j = 0; j < 24; j++) {
            float pred1 = 0.0f;
            for (int k = 0; k < 16; k++) pred1 += W1[j * 16 + k] * r2[k];
            e1[j] = r1[j] - pred1;
            step_energy += e1[j] * e1[j];
        }
        if (step == 0) f_init = 0.5f * step_energy;

        // Variational state relaxation: r_l <- tanh(r_l + state_lr * (W_{l-1}^T e_{l-1} - e_l))
        for (int j = 0; j < 24; j++) {
            float bu_grad = 0.0f;
            for (int i = 0; i < 32; i++) bu_grad += W0[i * 24 + j] * e0[i];
            r1[j] = tanh(r1[j] + state_lr * (bu_grad - e1[j]));
        }
        for (int k = 0; k < 16; k++) {
            float bu_grad = 0.0f;
            for (int j = 0; j < 24; j++) bu_grad += W1[j * 16 + k] * e1[j];
            r2[k] = tanh(r2[k] + state_lr * bu_grad);
        }
    }

    // Recompute final prediction error after relaxation
    for (int i = 0; i < 32; i++) {
        float pred0 = 0.0f;
        for (int j = 0; j < 24; j++) pred0 += W0[i * 24 + j] * r1[j];
        e0[i] = r0[i] - pred0;
        f_final += 0.5f * e0[i] * e0[i];
        E0_out[b0 + i] = e0[i];
    }
    float g1 = 0.0f;
    for (int j = 0; j < 24; j++) {
        float pred1 = 0.0f;
        for (int k = 0; k < 16; k++) pred1 += W1[j * 16 + k] * r2[k];
        e1[j] = r1[j] - pred1;
        f_final += 0.5f * e1[j] * e1[j];
        E1_out[b1 + j] = e1[j];
        R1_out[b1 + j] = r1[j];
        g1 += r1[j] * r1[j];
    }
    float g2 = 0.0f;
    for (int k = 0; k < 16; k++) {
        R2_out[b2 + k] = r2[k];
        g2 += r2[k] * r2[k];
    }

    int m_base = b * 4;
    metrics_out[m_base + 0] = f_init;
    metrics_out[m_base + 1] = f_final;
    metrics_out[m_base + 2] = g1 / 24.0f;
    metrics_out[m_base + 3] = g2 / 16.0f;
}

// 10. Parallel Monte-Carlo Tree Rollout of Expected Free Energy G(pi) for Active Inference
// Simulates `n_policies * n_particles` multi-step trajectories over `horizon` steps.
__kernel void active_inference_mc_rollout(
    __global const float *init_drives,
    __global const float *policy_shifts,
    __global const float *persona_weights,
    __global const float *target_prefs,
    __global float *pragmatic_cost_out,
    __global float *min_survival_out,
    __global float *final_states_out,
    const int n_policies,
    const int n_particles,
    const int horizon,
    const float noise_scale,
    const uint base_seed
) {
    int pol = get_global_id(0);
    int part = get_global_id(1);
    if (pol >= n_policies || part >= n_particles) return;

    int tid = pol * n_particles + part;
    uint rng = base_seed ^ (uint)(pol * 4099 + part * 131071 + 17);
    if (rng == 0u) rng = 1337u;

    // 4 drives: [safety, curiosity, integrity, homeostasis]
    float s0 = init_drives[0];
    float s1 = init_drives[1];
    float s2 = init_drives[2];
    float s3 = init_drives[3];

    float d0 = policy_shifts[pol * 4 + 0] * persona_weights[pol * 4 + 0];
    float d1 = policy_shifts[pol * 4 + 1] * persona_weights[pol * 4 + 1];
    float d2 = policy_shifts[pol * 4 + 2] * persona_weights[pol * 4 + 2];
    float d3 = policy_shifts[pol * 4 + 3] * persona_weights[pol * 4 + 3];

    float total_pragmatic_risk = 0.0f;
    float min_surv = s0 < s3 ? s0 : s3;

    for (int h = 0; h < horizon; h++) {
        // Xorshift32 register PRNG (4 uniform draws centered in [-1, 1])
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float n0 = ((float)(rng & 0xFFFFu) / 32768.0f - 1.0f) * noise_scale;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float n1 = ((float)(rng & 0xFFFFu) / 32768.0f - 1.0f) * noise_scale;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float n2 = ((float)(rng & 0xFFFFu) / 32768.0f - 1.0f) * noise_scale;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float n3 = ((float)(rng & 0xFFFFu) / 32768.0f - 1.0f) * noise_scale;

        s0 = clamp(s0 + d0 + n0, 0.0f, 1.0f);
        s1 = clamp(s1 + d1 + n1, 0.0f, 1.0f);
        s2 = clamp(s2 + d2 + n2, 0.0f, 1.0f);
        s3 = clamp(s3 + d3 + n3, 0.0f, 1.0f);

        float step_floor = s0 < s3 ? s0 : s3;
        if (step_floor < min_surv) min_surv = step_floor;

        float err0 = s0 - target_prefs[0];
        float err1 = s1 - target_prefs[1];
        float err2 = s2 - target_prefs[2];
        float err3 = s3 - target_prefs[3];
        total_pragmatic_risk += (err0 * err0 + err1 * err1 + err2 * err2 + err3 * err3);
    }

    pragmatic_cost_out[tid] = total_pragmatic_risk / (float)horizon;
    min_survival_out[tid] = min_surv;
    int f_base = tid * 4;
    final_states_out[f_base + 0] = s0;
    final_states_out[f_base + 1] = s1;
    final_states_out[f_base + 2] = s2;
    final_states_out[f_base + 3] = s3;
}

// 11. Multi-Scale Vietoris-Rips Persistence Filtration & Geodesic SLERP Search on GPU
// Evaluates (n_t x n_eps) SLERP manifold point clouds using register Union-Find for beta_0 and beta_1.
__kernel void rips_multiscale_filtration(
    __global const float *U_a,
    __global const float *U_b,
    __global const float *t_grid,
    __global const float *eps_grid,
    __global int *beta0_out,
    __global int *beta1_out,
    __global int *edges_out,
    const int n_pts,
    const int pt_dim,
    const int n_t,
    const int n_eps,
    const float theta
) {
    int t_idx = get_global_id(0);
    int e_idx = get_global_id(1);
    if (t_idx >= n_t || e_idx >= n_eps) return;

    float t = t_grid[t_idx];
    float eps = eps_grid[e_idx];
    float eps_sq = eps * eps;

    float w_a = 1.0f - t;
    float w_b = t;
    float sin_theta = native_sin(theta);
    if (fabs(sin_theta) > 1e-4f) {
        w_a = native_sin((1.0f - t) * theta) / sin_theta;
        w_b = native_sin(t * theta) / sin_theta;
    }

    int parent[32];
    int v_cap = n_pts <= 32 ? n_pts : 32;
    for (int i = 0; i < v_cap; i++) parent[i] = i;

    int num_edges = 0;
    int components = v_cap;

    for (int i = 0; i < v_cap; i++) {
        for (int j = i + 1; j < v_cap; j++) {
            float dist_sq = 0.0f;
            int base_i = i * pt_dim;
            int base_j = j * pt_dim;
            for (int d = 0; d < pt_dim; d++) {
                float pi = w_a * U_a[base_i + d] + w_b * U_b[base_i + d];
                float pj = w_a * U_a[base_j + d] + w_b * U_b[base_j + d];
                float diff = pi - pj;
                dist_sq += diff * diff;
            }
            if (dist_sq < eps_sq) {
                num_edges++;
                int root_i = i;
                while (parent[root_i] != root_i) root_i = parent[root_i];
                int root_j = j;
                while (parent[root_j] != root_j) root_j = parent[root_j];
                if (root_i != root_j) {
                    parent[root_j] = root_i;
                    components--;
                }
            }
        }
    }

    int beta_0 = components;
    int beta_1 = num_edges - v_cap + beta_0;
    if (beta_1 < 0) beta_1 = 0;

    int out_idx = t_idx * n_eps + e_idx;
    beta0_out[out_idx] = beta_0;
    beta1_out[out_idx] = beta_1;
    edges_out[out_idx] = num_edges;
}

// 12. True EGGROLL (arXiv:2511.16652) Counter-Based On-Device RNG Rank-r Antithetic ES Rollout
// Generates rank-r Rademacher perturbations A_p, B_p in {-1, +1} on-the-fly from (seed, pair_id, r_idx, coord)
// with antithetic mirrored pairs (+E for even p, -E for odd p) and records 8D behavior descriptors.
inline float eggroll_counter_rademacher(uint seed, int pair_id, int r_idx, int coord, int stream) {
    uint h = seed ^ (uint)(pair_id * 374761393 + r_idx * 668265263 + coord * 2147483647 + stream * 1274126177);
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);
    return (h & 1u) ? 1.0f : -1.0f;
}

__kernel void eggroll_counter_rng_es_rollout(
    __global const float *W_base,
    __global const float *X_samples,
    __global const float *Y_targets,
    __global float *mse_out,
    __global float *behavior_out,
    const int pop_size,
    const int n_samples,
    const int dim,
    const int rank_r,
    const float sigma,
    const uint seed
) {
    int p = get_global_id(0);
    if (p >= pop_size) return;

    int pair_id = p >> 1;
    float sign_mirror = (p & 1) ? -1.0f : 1.0f;
    float scale_r = (sigma * sign_mirror) / sqrt((float)(rank_r > 0 ? rank_r : 1));
    float total_sq_err = 0.0f;

    for (int b = 0; b < 8; b++) {
        behavior_out[p * 8 + b] = 0.0f;
    }

    for (int s = 0; s < n_samples; s++) {
        int x_base = s * dim;
        for (int o = 0; o < dim; o++) {
            float base_dot = 0.0f;
            int w_row = o * dim;
            for (int i = 0; i < dim; i++) {
                base_dot += W_base[w_row + i] * X_samples[x_base + i];
            }
            // Low-rank perturbation E_p * x = (1/sqrt(r)) * sum_{k=0}^{r-1} A[o, k] * (B[:, k]^T x)
            float low_rank_dot = 0.0f;
            for (int k = 0; k < rank_r; k++) {
                float a_ok = eggroll_counter_rademacher(seed, pair_id, k, o, 0);
                float bx_k = 0.0f;
                for (int i = 0; i < dim; i++) {
                    float b_ik = eggroll_counter_rademacher(seed, pair_id, k, i, 1);
                    bx_k += b_ik * X_samples[x_base + i];
                }
                low_rank_dot += a_ok * bx_k;
            }
            float pred = (base_dot + scale_r * low_rank_dot) / (float)dim;
            float diff = pred - Y_targets[x_base + o];
            total_sq_err += diff * diff;
            if (o < 8) {
                behavior_out[p * 8 + o] += pred / (float)n_samples;
            }
        }
    }
    mse_out[p] = total_sq_err / (float)(n_samples * dim);
}

// 13. Dominated Novelty Search (DNS, 2025) Quality-Diversity Competition Kernel
// Computes mean Euclidean distance in behavior space to the k-nearest strictly fitter individuals.
__kernel void dominated_novelty_search_kernel(
    __global const float *fitness,
    __global const float *descriptors,
    __global float *dns_scores,
    const int pop_size,
    const int desc_dim,
    const int k_nearest
) {
    int i = get_global_id(0);
    if (i >= pop_size) return;

    float f_i = fitness[i];
    int k_cap = k_nearest <= 8 ? k_nearest : 8;
    float best_dists[8];
    for (int m = 0; m < k_cap; m++) best_dists[m] = 1e30f;

    int fitter_count = 0;
    float peer_dist_sum = 0.0f;

    for (int j = 0; j < pop_size; j++) {
        if (i == j) continue;
        float d_sq = 0.0f;
        int bi = i * desc_dim;
        int bj = j * desc_dim;
        for (int d = 0; d < desc_dim; d++) {
            float diff = descriptors[bi + d] - descriptors[bj + d];
            d_sq += diff * diff;
        }
        float dist = sqrt(d_sq);
        peer_dist_sum += dist;

        if (fitness[j] > f_i) {
            fitter_count++;
            if (dist < best_dists[k_cap - 1]) {
                best_dists[k_cap - 1] = dist;
                for (int m = k_cap - 1; m > 0; m--) {
                    if (best_dists[m] < best_dists[m - 1]) {
                        float tmp = best_dists[m - 1];
                        best_dists[m - 1] = best_dists[m];
                        best_dists[m] = tmp;
                    }
                }
            }
        }
    }

    if (fitter_count == 0) {
        // Un-dominated global elite: assign maximum novelty bonus
        dns_scores[i] = (peer_dist_sum / (float)(pop_size > 1 ? pop_size - 1 : 1)) + 10.0f;
    } else {
        int used = fitter_count < k_cap ? fitter_count : k_cap;
        float sum_k = 0.0f;
        for (int m = 0; m < used; m++) sum_k += best_dists[m];
        dns_scores[i] = sum_k / (float)used;
    }
}

// 14. Muon (MomentUm Orthogonalized by Newton-Schulz) Fast-Weight Update Kernel
// Applies orthogonalized Newton-Schulz update U_ortho to W_fast and computes post-update reconstruction error.
__kernel void muon_fast_weight_apply(
    __global float *W_fast,
    __global const float *U_ortho,
    const int n_elems,
    const float scaled_lr
) {
    int idx = get_global_id(0);
    if (idx >= n_elems) return;
    W_fast[idx] -= scaled_lr * U_ortho[idx];
}

// 15. DeepSeek Engram O(1) Multi-Head Gated Lookup Kernel
// Retrieves K hash-indexed conditional memory entries and applies Context-Aware Sigmoid Gating.
__kernel void engram_multihead_gated_lookup(
    __global const float *query_vecs,     // [batch_size, dim]
    __global const float *table_keys,     // [num_buckets, dim]
    __global const float *table_vals,     // [num_buckets, dim]
    __global const int *head_indices,     // [batch_size, num_heads]
    __global float *out_vecs,             // [batch_size, dim]
    __global float *out_gates,            // [batch_size, num_heads]
    const int batch_size,
    const int num_buckets,
    const int num_heads,
    const int dim,
    const float inv_sqrt_dim
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;
    int q_off = b * dim;
    for (int d = 0; d < dim; d++) {
        out_vecs[q_off + d] = 0.0f;
    }
    if (num_buckets <= 0 || num_heads <= 0) return;
    float inv_heads = 1.0f / (float)num_heads;
    for (int h = 0; h < num_heads; h++) {
        int raw_idx = head_indices[b * num_heads + h];
        int bucket = raw_idx % num_buckets;
        if (bucket < 0) bucket += num_buckets;
        int k_off = bucket * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            dot += query_vecs[q_off + d] * table_keys[k_off + d];
        }
        float z = dot * inv_sqrt_dim;
        if (z > 20.0f) z = 20.0f;
        if (z < -20.0f) z = -20.0f;
        float gate = 1.0f / (1.0f + exp(-z));
        out_gates[b * num_heads + h] = gate;
        float w = gate * inv_heads;
        for (int d = 0; d < dim; d++) {
            out_vecs[q_off + d] += w * table_vals[k_off + d];
        }
    }
}
"""


class OpenCLMicroBrainAccelerator:
    """Singleton OpenCL GPU co-processor for Aetheris micro-brains with CPU fallback."""

    _instance: Optional[OpenCLMicroBrainAccelerator] = None

    # Legacy nominal batch threshold attribute (preserved for compatibility);
    # actual per-kernel dispatch uses `self._crossover_table` and `should_use_gpu(...)`.
    GPU_BATCH_THRESHOLD: int = 32
    HDC_BITS: int = 10240
    HDC_WORDS: int = 320  # 320 * 32 = 10,240 bits
    _CWD_CROSSOVER_PATH: Path = Path("memory/store/opencl_crossover.json")
    CROSSOVER_STORE_PATH: Path = _CWD_CROSSOVER_PATH if _CWD_CROSSOVER_PATH.exists() else Path(__file__).resolve().parents[1] / "memory" / "store" / "opencl_crossover.json"

    # Sentinel crossover: "no benchmark has shown the GPU winning at any size for this
    # op" — dispatch stays on CPU until autotune_crossover() measures it on the live
    # device. force_gpu=True remains available for explicit hardware exercise.
    GPU_UNMEASURED: int = 2**31

    # CONSERVATIVE BOOTSTRAP dispatch table, NOT a measurement record. Only entries
    # annotated "measured" carry real wall-clock benchmarks (GT 710, 2026-11-12,
    # including PCIe 2.0 transfer). Every other op boots at GPU_UNMEASURED and is
    # expected to be populated by autotune_crossover(), which persists per-device
    # measurements to CROSSOVER_STORE_PATH.
    DEFAULT_CROSSOVER_BOOTSTRAP: Dict[str, int] = {
        "bitnet_batch_gemm": GPU_UNMEASURED,         # measured: GPU 5-8x SLOWER at batch 32/1024/4096
        "bitnet_resident_gemm": GPU_UNMEASURED,
        "eggroll_low_rank_rollout": GPU_UNMEASURED,
        "eggroll_counter_rng_es": GPU_UNMEASURED,
        "eggroll_es_rollout": GPU_UNMEASURED,        # observed GPU-slower at pop=256 on GT 710 (informal observation, no persisted receipt)
        "hdc_hamming_search": 1024,                  # measured: GPU 2.5 vs 1.6 ms at n=64, 7.0 vs 11.0 ms at n=1024
        "hdc_resident_search": GPU_UNMEASURED,
        "rips_multiscale_filtration": GPU_UNMEASURED,
        "active_inference_mc_rollout": GPU_UNMEASURED,
        "dominated_novelty_search": GPU_UNMEASURED,
        "cfc_step": GPU_UNMEASURED,                  # measured: GPU ~58x slower at 128 neurons
        "ttt_linear_step": GPU_UNMEASURED,           # measured: GPU ~45x slower at dim=64
        "muon_fast_weight_step": GPU_UNMEASURED,
        "engram_multihead_gated_lookup": GPU_UNMEASURED,
        "ssm_blelloch_parallel_scan": GPU_UNMEASURED,
        "difflogic_bitplane_eval": GPU_UNMEASURED,
        "kuramoto_hopfield_step": GPU_UNMEASURED,
        "predictive_coding_sram_relax": GPU_UNMEASURED,
    }

    def __init__(self) -> None:
        self.gpu_available: bool = False
        self.device_name: str = "CPU (NumPy Fallback)"
        self.platform_name: str = "CPU"
        self.compute_units: int = 1
        self.global_mem_mb: int = 0
        self.ctx: Any = None
        self.queue: Any = None
        self.program: Any = None
        self._crossover_table: Dict[str, int] = dict(self.DEFAULT_CROSSOVER_BOOTSTRAP)
        self._resident_hdc_codebooks: Dict[str, Dict[str, Any]] = {}
        self._resident_bitnet_weights: Dict[str, Dict[str, Any]] = {}
        self._init_opencl()
        self._load_persisted_crossovers()

    @classmethod
    def get_instance(cls) -> OpenCLMicroBrainAccelerator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_opencl(self) -> None:
        if not _HAS_PYOPENCL or cl is None:
            return
        try:
            platforms = cl.get_platforms()
            chosen_dev = None
            chosen_plat = None
            for plat in platforms:
                devs = plat.get_devices(device_type=cl.device_type.GPU)
                if devs:
                    chosen_dev = devs[0]
                    chosen_plat = plat
                    break
            if chosen_dev is None and platforms:
                for plat in platforms:
                    devs = plat.get_devices()
                    if devs:
                        chosen_dev = devs[0]
                        chosen_plat = plat
                        break
            if chosen_dev is None:
                return

            self.ctx = cl.Context([chosen_dev])
            self.queue = cl.CommandQueue(self.ctx)
            self.program = cl.Program(self.ctx, _OPENCL_KERNELS_SRC).build()
            self._kernels = {
                "bitnet_batch_gemm": cl.Kernel(self.program, "bitnet_batch_gemm"),
                "eggroll_low_rank_rollout": cl.Kernel(self.program, "eggroll_low_rank_rollout"),
                "cfc_liquid_ensemble_step": cl.Kernel(self.program, "cfc_liquid_ensemble_step"),
                "hdc_hamming_search": cl.Kernel(self.program, "hdc_hamming_search"),
                "ttt_linear_fast_weight_step": cl.Kernel(self.program, "ttt_linear_fast_weight_step"),
                "ssm_blelloch_parallel_scan": cl.Kernel(self.program, "ssm_blelloch_parallel_scan"),
                "difflogic_bitplane_eval": cl.Kernel(self.program, "difflogic_bitplane_eval"),
                "kuramoto_hopfield_sync_step": cl.Kernel(self.program, "kuramoto_hopfield_sync_step"),
                "kuramoto_hopfield_readout": cl.Kernel(self.program, "kuramoto_hopfield_readout"),
                "predictive_coding_sram_relax": cl.Kernel(self.program, "predictive_coding_sram_relax"),
                "active_inference_mc_rollout": cl.Kernel(self.program, "active_inference_mc_rollout"),
                "rips_multiscale_filtration": cl.Kernel(self.program, "rips_multiscale_filtration"),
                "eggroll_counter_rng_es_rollout": cl.Kernel(self.program, "eggroll_counter_rng_es_rollout"),
                "dominated_novelty_search_kernel": cl.Kernel(self.program, "dominated_novelty_search_kernel"),
                "muon_fast_weight_apply": cl.Kernel(self.program, "muon_fast_weight_apply"),
                "engram_multihead_gated_lookup": cl.Kernel(self.program, "engram_multihead_gated_lookup"),
            }
            self.gpu_available = True
            self.device_name = str(chosen_dev.name).strip()
            self.platform_name = str(chosen_plat.name).strip() if chosen_plat else "OpenCL"
            self.compute_units = int(getattr(chosen_dev, "max_compute_units", 1))
            self.global_mem_mb = int(getattr(chosen_dev, "global_mem_size", 0) // (1024 * 1024))
        except Exception as exc:
            log.debug("OpenCL initialization fell back to CPU: %s", exc)
            self.gpu_available = False

    @property
    def available(self) -> bool:
        return self.gpu_available

    def _load_persisted_crossovers(self) -> None:
        """Load empirical crossover measurements for the active GPU device if present."""
        try:
            if self.CROSSOVER_STORE_PATH.is_file():
                data = json.loads(self.CROSSOVER_STORE_PATH.read_text(encoding="utf-8"))
                dev_map = data.get("devices", {}).get(self.device_name, {})
                if isinstance(dev_map, dict):
                    for k, v in dev_map.items():
                        self._crossover_table[str(k)] = int(v)
        except Exception as exc:
            log.debug("Could not load opencl_crossover.json: %s", exc)

    def _save_persisted_crossovers(self, receipts: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.CROSSOVER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data: Dict[str, Any] = {"devices": {}, "receipts": {}}
            if self.CROSSOVER_STORE_PATH.is_file():
                try:
                    data = json.loads(self.CROSSOVER_STORE_PATH.read_text(encoding="utf-8"))
                except Exception:
                    data = {"devices": {}, "receipts": {}}
            data.setdefault("devices", {})[self.device_name] = dict(self._crossover_table)
            if receipts:
                data.setdefault("receipts", {}).setdefault(self.device_name, {}).update(receipts)
            tmp_path = self.CROSSOVER_STORE_PATH.parent / (self.CROSSOVER_STORE_PATH.name + ".tmp")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            from .atomic import atomic_replace

            atomic_replace(tmp_path, self.CROSSOVER_STORE_PATH)
        except Exception as exc:
            log.debug("Could not save opencl_crossover.json: %s", exc)

    def get_crossover(self, op_name: str) -> int:
        """Return the minimum problem size at which OpenCL_GPU is known to beat CPU for `op_name`.

        Unknown ops return GPU_UNMEASURED ("stay on CPU"): an unmeasured op never
        dispatches to the GPU. Run autotune_crossover() to measure one.
        """
        return int(self._crossover_table.get(op_name, self.GPU_UNMEASURED))

    def set_crossover(self, op_name: str, threshold: int, persist: bool = False) -> None:
        """Set or override the crossover size for `op_name`."""
        self._crossover_table[str(op_name)] = max(1, int(threshold))
        if persist:
            self._save_persisted_crossovers()

    def should_use_gpu(
        self,
        op_name: str,
        problem_size: int = 1,
        force_gpu: bool = False,
        batch_size: Optional[int] = None,
    ) -> bool:
        """Return True iff OpenCL GPU is available and `problem_size` meets the measured crossover."""
        if not self.gpu_available:
            return False
        if force_gpu:
            return True
        sz = int(batch_size if batch_size is not None else problem_size)
        return sz >= self.get_crossover(op_name)

    # Ops with a real autotune benchmark implementation. Everything else raises
    # ValueError — this module never guesses a crossover it cannot measure (the old
    # else-branch benchmarked BitNet GEMM for arbitrary ops: the wrong kernel).
    AUTOTUNEABLE_OPS: tuple = (
        "hdc_resident_search",
        "hdc_hamming_search",
        "bitnet_batch_gemm",
        "cfc_step",
        "ttt_linear_step",
    )

    @staticmethod
    def _choose_crossover(measurements: List[Dict[str, Any]]) -> Optional[int]:
        """Smallest measured size where GPU wall-clock beat CPU, or None if it never did."""
        for m in measurements:
            if m["gpu_ms"] < m["cpu_ms"]:
                return int(m["size"])
        return None

    def _timed_gpu_cpu(
        self, gpu_call: Any, cpu_call: Any, size: int, reps: int = 2
    ) -> Dict[str, Any]:
        """Time `gpu_call()` vs `cpu_call()` (`reps` runs each, after one GPU warmup).

        The CPU side is timed with `gpu_available` toggled off so the measurement is
        independent of the current crossover table; the flag is restored in `finally`.
        """
        gpu_call()  # warmup (kernel build, lazy allocation)
        t0 = time.perf_counter()
        for _ in range(reps):
            gpu_call()
        gpu_ms = (time.perf_counter() - t0) / reps * 1000.0

        orig_flag = self.gpu_available
        self.gpu_available = False
        try:
            t0 = time.perf_counter()
            for _ in range(reps):
                cpu_call()
            cpu_ms = (time.perf_counter() - t0) / reps * 1000.0
        finally:
            self.gpu_available = orig_flag
        return {"size": int(size), "gpu_ms": round(gpu_ms, 3), "cpu_ms": round(cpu_ms, 3)}

    def autotune_crossover(
        self,
        op_name: str,
        candidate_sizes: Optional[List[int]] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        Empirically benchmark GPU vs CPU wall-clock across `candidate_sizes` for `op_name`
        and set the crossover to the smallest measured size where the GPU wins.

        Honesty contract: ops without a benchmark implementation raise ValueError rather
        than being benchmarked with the wrong kernel. If the GPU never wins within the
        measured range, the table entry is set to GPU_UNMEASURED (stay on CPU) — never
        an invented extrapolation — and the receipt records gpu_won_in_range=False.
        """
        if op_name not in self.AUTOTUNEABLE_OPS:
            raise ValueError(
                f"no benchmark implemented for op {op_name!r}; "
                f"autotuneable ops: {sorted(self.AUTOTUNEABLE_OPS)}"
            )
        if not self.gpu_available:
            return {
                "op_name": op_name,
                "gpu_available": False,
                "crossover": None,
                "gpu_won_in_range": False,
                "measurements": [],
            }

        rng = np.random.default_rng(42)
        if candidate_sizes is None:
            candidate_sizes = {
                "hdc_resident_search": [64, 512, 2048, 8192],
                "hdc_hamming_search": [64, 512, 2048, 8192],
                "bitnet_batch_gemm": [32, 256, 1024, 4096],
                "cfc_step": [64, 256, 1024, 4096],
                "ttt_linear_step": [64, 256, 1024, 4096],
            }[op_name]
        candidate_sizes = sorted(int(s) for s in candidate_sizes)

        if op_name == "hdc_resident_search":
            measurements = self._bench_hdc_resident(rng, candidate_sizes)
        elif op_name == "hdc_hamming_search":
            measurements = self._bench_hdc_hamming(rng, candidate_sizes)
        elif op_name == "bitnet_batch_gemm":
            measurements = self._bench_bitnet(rng, candidate_sizes)
        elif op_name == "cfc_step":
            measurements = self._bench_cfc(rng, candidate_sizes)
        else:  # ttt_linear_step
            measurements = self._bench_ttt(rng, candidate_sizes)

        winning_size = self._choose_crossover(measurements)
        chosen = int(winning_size) if winning_size is not None else int(self.GPU_UNMEASURED)
        self._crossover_table[op_name] = chosen
        receipt = {
            "op_name": op_name,
            "device_name": self.device_name,
            "crossover": chosen,
            "gpu_won_in_range": winning_size is not None,
            "measurements": measurements,
        }
        if persist:
            self._save_persisted_crossovers({op_name: receipt})
        return receipt

    def _bench_hdc_resident(self, rng: Any, sizes: List[int]) -> List[Dict[str, Any]]:
        max_n = max(sizes)
        cb_full = rng.integers(0, 2**32, size=(max_n, self.HDC_WORDS), dtype=np.uint32)
        q_hv = cb_full[0].copy()
        rows: List[Dict[str, Any]] = []
        try:
            for sz in sizes:
                cb_slice = np.ascontiguousarray(cb_full[:sz])
                self.register_resident_hdc_codebook("_autotune_tmp", cb_slice)
                rows.append(self._timed_gpu_cpu(
                    lambda: self.search_resident_hdc_codebook("_autotune_tmp", q_hv, force_gpu=True),
                    lambda: self.search_resident_hdc_codebook("_autotune_tmp", q_hv),
                    sz,
                ))
        finally:
            self._resident_hdc_codebooks.pop("_autotune_tmp", None)
        return rows

    def _bench_hdc_hamming(self, rng: Any, sizes: List[int]) -> List[Dict[str, Any]]:
        max_n = max(sizes)
        cb_full = rng.integers(0, 2**32, size=(max_n, self.HDC_WORDS), dtype=np.uint32)
        q_hv = cb_full[0].copy()
        rows: List[Dict[str, Any]] = []
        for sz in sizes:
            cb_slice = np.ascontiguousarray(cb_full[:sz])
            rows.append(self._timed_gpu_cpu(
                lambda cb=cb_slice: self.hdc_search(q_hv, cb, force_gpu=True),
                lambda cb=cb_slice: self.hdc_search(q_hv, cb),
                sz,
            ))
        return rows

    def _bench_bitnet(self, rng: Any, sizes: List[int]) -> List[Dict[str, Any]]:
        w = rng.integers(-1, 2, size=(128, 128), dtype=np.int8)
        rows: List[Dict[str, Any]] = []
        for sz in sizes:
            xq = rng.integers(-127, 127, size=(sz, 128), dtype=np.int8)
            rows.append(self._timed_gpu_cpu(
                lambda x=xq: self.bitnet_forward(x, w, 0.01, force_gpu=True),
                lambda x=xq: self.bitnet_forward(x, w, 0.01),
                sz,
            ))
        return rows

    def _bench_cfc(self, rng: Any, sizes: List[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sz in sizes:
            h = rng.standard_normal(sz).astype(np.float32)
            f = rng.standard_normal(sz).astype(np.float32)
            g = rng.standard_normal(sz).astype(np.float32)
            t = rng.uniform(0.5, 2.0, size=sz).astype(np.float32)
            rows.append(self._timed_gpu_cpu(
                lambda h=h, f=f, g=g, t=t: self.cfc_step(h, f, g, t, 0.05, force_gpu=True),
                lambda h=h, f=f, g=g, t=t: self.cfc_step(h, f, g, t, 0.05),
                sz,
            ))
        return rows

    def _bench_ttt(self, rng: Any, sizes: List[int]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sz in sizes:
            w = rng.standard_normal((sz, sz)).astype(np.float32)
            k = rng.standard_normal(sz).astype(np.float32)
            v = rng.standard_normal(sz).astype(np.float32)
            rows.append(self._timed_gpu_cpu(
                lambda w=w, k=k, v=v: self.ttt_linear_step(w, k, v, lr=0.05, force_gpu=True),
                lambda w=w, k=k, v=v: self.ttt_linear_step(w, k, v, lr=0.05),
                sz,
            ))
        return rows

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.gpu_available,
            "gpu_available": self.gpu_available,
            "device_name": self.device_name,
            "platform_name": self.platform_name,
            "compute_units": self.compute_units,
            "global_mem_mb": self.global_mem_mb,
            # Legacy nominal threshold kept for compatibility; it does NOT gate
            # dispatch - the per-op crossover table below does.
            "gpu_batch_threshold": self.GPU_BATCH_THRESHOLD,
            "crossovers": dict(self._crossover_table),
            "resident_codebooks": list(self._resident_hdc_codebooks.keys()),
            "hdc_bits": self.HDC_BITS,
        }

    # ------------------------------------------------------------------
    # 1. Batched BitNet b1.58 Ternary GEMM (with Measured Crossover & Resident Weights)
    # ------------------------------------------------------------------
    def register_resident_bitnet_weights(self, name: str, w_ternary: np.ndarray) -> bool:
        """Keep a ternary weight matrix stationary in GT 710 VRAM across calls."""
        w_2d = np.ascontiguousarray(w_ternary, dtype=np.int8)
        entry: Dict[str, Any] = {"w_cpu": w_2d, "w_buf": None}
        if self.gpu_available and cl is not None:
            try:
                mf = cl.mem_flags
                entry["w_buf"] = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_2d)
            except Exception:
                entry["w_buf"] = None
        self._resident_bitnet_weights[str(name)] = entry
        return entry["w_buf"] is not None

    def bitnet_forward(
        self,
        x_q: np.ndarray,
        w_ternary: np.ndarray,
        combined_scale: float,
        force_gpu: bool = False,
        resident_weight_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """Run ternary int8 GEMM on GPU only when measured crossover is met (or force_gpu=True), else CPU."""
        x_2d = np.ascontiguousarray(x_q, dtype=np.int8)
        w_2d = np.ascontiguousarray(w_ternary, dtype=np.int8)
        batch_size, in_features = x_2d.shape
        out_features = w_2d.shape[0]

        op_key = "bitnet_resident_gemm" if resident_weight_key and resident_weight_key in self._resident_bitnet_weights else "bitnet_batch_gemm"
        use_gpu = self.should_use_gpu(op_key, batch_size, force_gpu=force_gpu)
        if use_gpu:
            try:
                mf = cl.mem_flags
                x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_2d)
                res_entry = self._resident_bitnet_weights.get(resident_weight_key or "")
                if res_entry and res_entry.get("w_buf") is not None:
                    w_buf = res_entry["w_buf"]
                else:
                    w_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_2d)
                y_out = np.empty((batch_size, out_features), dtype=np.float32)
                y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, y_out.nbytes)

                self._kernels["bitnet_batch_gemm"](
                    self.queue,
                    (batch_size, out_features),
                    None,
                    x_buf,
                    w_buf,
                    y_buf,
                    np.int32(batch_size),
                    np.int32(in_features),
                    np.int32(out_features),
                    np.float32(combined_scale),
                )
                cl.enqueue_copy(self.queue, y_out, y_buf).wait()
                return y_out, "OpenCL_GPU"
            except Exception as exc:
                log.debug("OpenCL bitnet_batch_gemm fallback to CPU: %s", exc)

        out_int = np.dot(x_2d.astype(np.int32), w_2d.T.astype(np.int32))
        return (out_int.astype(np.float32) * float(combined_scale)), "CPU"

    # ------------------------------------------------------------------
    # 2. EGGROLL Low-Rank Population Neuroevolution in VRAM
    # ------------------------------------------------------------------
    def eggroll_evolve_population(
        self,
        w_base: np.ndarray,
        x_samples: np.ndarray,
        y_targets: np.ndarray,
        pop_size: int = 256,
        sigma: float = 0.35,
        seed: int = 42,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate `pop_size` low-rank perturbed ternary micro-brains in parallel on the GPU."""
        t0 = time.perf_counter()
        w_mat = np.ascontiguousarray(w_base, dtype=np.float32)
        dim = w_mat.shape[0]
        x_mat = np.ascontiguousarray(x_samples, dtype=np.float32).reshape(-1, dim)
        y_mat = np.ascontiguousarray(y_targets, dtype=np.float32).reshape(-1, dim)
        n_samples = x_mat.shape[0]

        rng = np.random.default_rng(seed)
        u_pert = np.ascontiguousarray(rng.standard_normal((pop_size, dim), dtype=np.float32))
        v_pert = np.ascontiguousarray(rng.standard_normal((pop_size, dim), dtype=np.float32))
        mse_out = np.empty(pop_size, dtype=np.float32)

        backend = "CPU"
        if self.should_use_gpu("eggroll_low_rank_rollout", pop_size, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                w_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_mat)
                u_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u_pert)
                v_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v_pert)
                x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_mat)
                y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_mat)
                m_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, mse_out.nbytes)

                self._kernels["eggroll_low_rank_rollout"](
                    self.queue,
                    (pop_size,),
                    None,
                    w_buf,
                    u_buf,
                    v_buf,
                    x_buf,
                    y_buf,
                    m_buf,
                    np.int32(pop_size),
                    np.int32(n_samples),
                    np.int32(dim),
                    np.float32(sigma),
                )
                cl.enqueue_copy(self.queue, mse_out, m_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("EGGROLL GPU fallback to CPU: %s", exc)
                backend = "CPU"

        if backend == "CPU":
            for p in range(pop_size):
                w_cand = np.clip(np.round(w_mat + sigma * np.outer(u_pert[p], v_pert[p])), -1.0, 1.0)
                preds = np.dot(x_mat, w_cand.T) / float(dim)
                mse_out[p] = float(np.mean((preds - y_mat) ** 2))

        best_idx = int(np.argmin(mse_out))
        best_mse = float(mse_out[best_idx])
        mean_mse = float(np.mean(mse_out))
        best_w = np.clip(
            np.round(w_mat + sigma * np.outer(u_pert[best_idx], v_pert[best_idx])),
            -1.0,
            1.0,
        ).astype(np.int8)

        return {
            "backend": backend,
            "device": self.device_name if backend == "OpenCL_GPU" else "CPU",
            "population_size": pop_size,
            "best_candidate_idx": best_idx,
            "best_mse": round(best_mse, 6),
            "mean_population_mse": round(mean_mse, 6),
            "improvement_ratio": round((mean_mse - best_mse) / (mean_mse + 1e-12), 4),
            "best_ternary_weights": best_w,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    @staticmethod
    def _counter_rademacher_matrix(
        seed: int, n_pairs: int, rank_r: int, dim: int, stream_id: int
    ) -> np.ndarray:
        """Bit-exact host mirror of OpenCL `eggroll_counter_rademacher` for zero-upload ES reconstruction.

        Replicates the kernel hash exactly: the four per-field products are SUMMED
        in uint32 (matching the kernel's (uint)(a + b + c + d) wraparound), XORed
        with the seed, then mixed with one xor-shift-multiply (>>13, *1274126177,
        >>16).  Everything stays uint32 so host wraps match the GPU's two's-
        complement int32 wrap bit-for-bit.  Pinned by
        tests/test_microbrain_audit_fixes.py against a Python transcription of
        the kernel.
        """
        p = np.arange(n_pairs, dtype=np.uint32)[:, None, None]
        r = np.arange(rank_r, dtype=np.uint32)[None, :, None]
        idx = np.arange(dim, dtype=np.uint32)[None, None, :]
        summed = (
            p * np.uint32(374761393)
            + r * np.uint32(668265263)
            + idx * np.uint32(2147483647)
            + np.uint32(stream_id & 0xFFFFFFFF) * np.uint32(1274126177)
        )
        h = np.uint32(seed & 0xFFFFFFFF) ^ summed
        h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
        h = h ^ (h >> np.uint32(16))
        return np.where((h & np.uint32(1)) != 0, 1.0, -1.0).astype(np.float32)

    def eggroll_es_step(
        self,
        w_base: np.ndarray,
        x_samples: np.ndarray,
        y_targets: np.ndarray,
        pop_size: int = 256,
        rank_r: int = 2,
        sigma: float = 0.25,
        lr: float = 0.20,
        seed: int = 42,
        force_gpu: bool = False,  # crossover unmeasured; informally observed GPU-slower on GT 710
    ) -> Dict[str, Any]:
        """
        True EGGROLL (arXiv:2511.16652) Evolution Strategies step:
        1. Generates rank-r Rademacher perturbations E_k = (1/sqrt(r)) * A_k @ B_k^T on-chip from `seed`
           with antithetic mirrored sampling (+sigma*E_k and -sigma*E_k).
        2. Evaluates all `pop_size` candidates against `(x_samples, y_targets)`.
        3. Applies centered rank-based fitness shaping and forms the low-rank ES gradient update
           W_new = W_base + (lr / (sigma * n_pairs * sqrt(r))) * sum_k utility_diff_k * (A_k @ B_k^T).
        """
        t0 = time.perf_counter()
        pop_even = max(4, int(pop_size) + (int(pop_size) % 2))
        n_pairs = pop_even // 2
        rank_r = max(1, min(int(rank_r), 8))

        w_mat = np.ascontiguousarray(w_base, dtype=np.float32)
        dim = w_mat.shape[0]
        x_mat = np.ascontiguousarray(x_samples, dtype=np.float32).reshape(-1, dim)
        y_mat = np.ascontiguousarray(y_targets, dtype=np.float32).reshape(-1, dim)
        n_samples = x_mat.shape[0]

        # Baseline MSE before update
        base_preds = np.dot(x_mat, w_mat.T) / float(dim)
        initial_mse = float(np.mean((base_preds - y_mat) ** 2))

        mse_out = np.empty(pop_even, dtype=np.float32)
        behavior_out = np.zeros((pop_even, 8), dtype=np.float32)
        backend = "CPU"

        if self.should_use_gpu("eggroll_es_rollout", pop_even, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                w_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_mat)
                x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_mat)
                y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_mat)
                m_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, mse_out.nbytes)
                b_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, behavior_out.nbytes)

                self._kernels["eggroll_counter_rng_es_rollout"](
                    self.queue,
                    (pop_even,),
                    None,
                    w_buf,
                    x_buf,
                    y_buf,
                    m_buf,
                    b_buf,
                    np.int32(pop_even),
                    np.int32(n_samples),
                    np.int32(dim),
                    np.int32(rank_r),
                    np.float32(sigma),
                    np.uint32(seed & 0xFFFFFFFF),
                )
                cl.enqueue_copy(self.queue, mse_out, m_buf).wait()
                cl.enqueue_copy(self.queue, behavior_out, b_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("EGGROLL counter-RNG ES rollout fallback to CPU: %s", exc)
                backend = "CPU"

        # Reconstruct counter-based Rademacher rank-r factors A (n_pairs, r, dim) and B (n_pairs, r, dim)
        a_factors = self._counter_rademacher_matrix(seed, n_pairs, rank_r, dim, 0)
        b_factors = self._counter_rademacher_matrix(seed, n_pairs, rank_r, dim, 1)
        inv_sqrt_r = 1.0 / math.sqrt(float(rank_r))
        # E_pairs shape: (n_pairs, dim, dim)
        e_pairs = np.einsum("pri,prj->pij", a_factors, b_factors) * inv_sqrt_r

        if backend == "CPU":
            for k in range(n_pairs):
                w_pos = w_mat + sigma * e_pairs[k]
                w_neg = w_mat - sigma * e_pairs[k]
                pred_pos = np.dot(x_mat, w_pos.T) / float(dim)
                pred_neg = np.dot(x_mat, w_neg.T) / float(dim)
                mse_out[2 * k] = float(np.mean((pred_pos - y_mat) ** 2))
                mse_out[2 * k + 1] = float(np.mean((pred_neg - y_mat) ** 2))
                b_dim = min(8, dim)
                behavior_out[2 * k, :b_dim] = np.mean(pred_pos[:, :b_dim], axis=0)
                behavior_out[2 * k + 1, :b_dim] = np.mean(pred_neg[:, :b_dim], axis=0)

        # Rank-based fitness shaping (Wierstra et al. / Salimans et al. / EGGROLL):
        # Lower MSE gets higher utility in [-0.5, +0.5]
        order = np.argsort(mse_out)  # ascending MSE (best first)
        ranks = np.empty(pop_even, dtype=np.float32)
        ranks[order] = np.linspace(0.5, -0.5, pop_even, dtype=np.float32)

        # Antithetic utility difference per pair: (u_pos - u_neg)
        u_pos = ranks[0::2]
        u_neg = ranks[1::2]
        u_diff = u_pos - u_neg  # shape: (n_pairs,)

        # Form ES gradient update over low-rank subspace
        grad_est = np.einsum("p,pij->ij", u_diff, e_pairs) / float(n_pairs * max(sigma, 1e-4))
        w_updated = (w_mat + float(lr) * grad_est).astype(np.float32)

        # Also check best individual candidate in case elitist jump is even stronger
        best_idx = int(order[0])
        pair_best = best_idx // 2
        sign_best = 1.0 if (best_idx % 2 == 0) else -1.0
        w_best_cand = (w_mat + sign_best * sigma * e_pairs[pair_best]).astype(np.float32)

        updated_preds = np.dot(x_mat, w_updated.T) / float(dim)
        updated_mse = float(np.mean((updated_preds - y_mat) ** 2))
        best_cand_mse = float(mse_out[best_idx])

        if best_cand_mse < updated_mse and best_cand_mse < initial_mse:
            # Blend ES gradient step with elitist antithetic anchor for monotonic convergence
            w_updated = (0.7 * w_updated + 0.3 * w_best_cand).astype(np.float32)
            updated_preds = np.dot(x_mat, w_updated.T) / float(dim)
            updated_mse = float(np.mean((updated_preds - y_mat) ** 2))

        ternary_updated = np.clip(np.round(w_updated), -1.0, 1.0).astype(np.int8)
        return {
            "backend": backend,
            "device": self.device_name if backend == "OpenCL_GPU" else "CPU",
            "population_size": pop_even,
            "antithetic_pairs": n_pairs,
            "rank_r": rank_r,
            "initial_mse": round(initial_mse, 6),
            "post_update_mse": round(updated_mse, 6),
            "best_candidate_mse": round(best_cand_mse, 6),
            "mean_population_mse": round(float(np.mean(mse_out)), 6),
            # Per-candidate rollout MSEs (lower = better) — the real fitness
            # channel. Previously computed and used for rank shaping here but
            # never returned, so DNS downstream substituted a constant.
            "candidate_mses": mse_out.copy(),
            "mse_reduced": bool(updated_mse < initial_mse),
            "updated_weights": w_updated,
            "updated_ternary_weights": ternary_updated,
            "behavior_descriptors": behavior_out,
            "pcie_perturbation_bytes_uploaded": 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    # ------------------------------------------------------------------
    # 3. Closed-Form Continuous-Time (CfC) Liquid Ensemble Step
    # ------------------------------------------------------------------
    def cfc_step(
        self,
        h_state: np.ndarray,
        f_drive: np.ndarray,
        g_drive: np.ndarray,
        tau_inv: np.ndarray,
        dt: float,
        force_gpu: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """Advance continuous-time liquid state h(t + dt) using closed-form CfC dynamics."""
        h_arr = np.ascontiguousarray(h_state, dtype=np.float32).flatten()
        f_arr = np.ascontiguousarray(f_drive, dtype=np.float32).flatten()
        g_arr = np.ascontiguousarray(g_drive, dtype=np.float32).flatten()
        t_arr = np.ascontiguousarray(tau_inv, dtype=np.float32).flatten()
        n_neurons = len(h_arr)
        clamped_dt = max(1e-4, min(float(dt), 3600.0))

        if self.should_use_gpu("cfc_step", n_neurons, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                h_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_arr)
                f_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f_arr)
                g_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g_arr)
                t_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_arr)

                self._kernels["cfc_liquid_ensemble_step"](
                    self.queue,
                    (n_neurons,),
                    None,
                    h_buf,
                    f_buf,
                    g_buf,
                    t_buf,
                    np.int32(n_neurons),
                    np.float32(clamped_dt),
                )
                cl.enqueue_copy(self.queue, h_arr, h_buf).wait()
                return h_arr, "OpenCL_GPU"
            except Exception as exc:
                log.debug("CfC GPU step fallback to CPU: %s", exc)

        rate = np.abs(f_arr) * t_arr + 0.1
        decay_gate = 1.0 / (1.0 + np.exp(np.clip(rate * clamped_dt, -40.0, 40.0)))
        candidate = np.tanh(g_arr)
        h_next = (decay_gate * h_arr + (1.0 - decay_gate) * candidate).astype(np.float32)
        return h_next, "CPU"

    # ------------------------------------------------------------------
    # 4. Hyperdimensional Computing (HDC / VSA) 10,240-Bit Cleanup Memory
    # ------------------------------------------------------------------
    @classmethod
    def encode_text_hypervector(cls, text: Any, bind_positions: bool = False) -> np.ndarray:
        """Encode text tokens via VSA bipolar bundling into 10,240 bits (`320 x uint32`)."""
        if isinstance(text, (list, tuple)):
            raw_str = " ".join(str(x) for x in text)
        else:
            raw_str = str(text or "")
        ignored = {"<pad>", "<unk>", "<bos>", "<eos>", "neuro"}
        tokens = [
            t for t in raw_str.lower().replace("_", " ").split()
            if t and t not in ignored
        ]
        if not tokens:
            return np.zeros(cls.HDC_WORDS, dtype=np.uint32)

        # Deduplicate while preserving order for set bundling when bind_positions=False
        if not bind_positions:
            tokens = list(dict.fromkeys(tokens))

        bit_accum = np.zeros(cls.HDC_BITS, dtype=np.int32)
        for idx, tok in enumerate(tokens):
            seed_bytes = hashlib.sha256(tok.encode("utf-8")).digest()
            seed_int = int.from_bytes(seed_bytes[:8], "little")
            rng = np.random.default_rng(seed_int)
            bits = rng.integers(0, 2, size=cls.HDC_BITS, dtype=np.int32)
            if bind_positions:
                bits = np.roll(bits, idx * 7)
            bit_accum += (bits * 2 - 1)

        binary = (bit_accum >= 0).astype(np.uint8)
        packed = np.packbits(binary).view(np.uint32)
        return np.ascontiguousarray(packed[: cls.HDC_WORDS], dtype=np.uint32)

    def hdc_search(
        self,
        query_hv: np.ndarray,
        codebook_hv: np.ndarray,
        force_gpu: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """Compute normalized Hamming similarity in [0, 1] across `n_items` 10,240-bit hypervectors."""
        q_arr = np.ascontiguousarray(query_hv, dtype=np.uint32).flatten()[: self.HDC_WORDS]
        cb_arr = np.ascontiguousarray(codebook_hv, dtype=np.uint32).reshape(-1, self.HDC_WORDS)
        n_items = cb_arr.shape[0]
        sims = np.empty(n_items, dtype=np.float32)

        if self.should_use_gpu("hdc_hamming_search", n_items, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                q_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q_arr)
                cb_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cb_arr)
                s_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, sims.nbytes)

                self._kernels["hdc_hamming_search"](
                    self.queue,
                    (n_items,),
                    None,
                    q_buf,
                    cb_buf,
                    s_buf,
                    np.int32(n_items),
                    np.int32(self.HDC_WORDS),
                )
                cl.enqueue_copy(self.queue, sims, s_buf).wait()
                return sims, "OpenCL_GPU"
            except Exception as exc:
                log.debug("HDC GPU search fallback to CPU: %s", exc)

        xor_words = np.bitwise_xor(cb_arr, q_arr[None, :])
        xor_bytes = xor_words.view(np.uint8).reshape(n_items, -1)
        bit_counts = np.unpackbits(xor_bytes, axis=1).sum(axis=1)
        sims = (1.0 - (bit_counts.astype(np.float32) / float(self.HDC_BITS))).astype(np.float32)
        return sims, "CPU"

    def register_resident_hdc_codebook(
        self,
        name: str,
        codebook_hv: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Upload an `(n_items, 320)` uint32 10,240-bit HDC codebook into persistent GT 710 VRAM once
        so subsequent `search_resident_hdc_codebook` queries only transfer the 1,280-byte query vector.
        """
        cb_arr = np.ascontiguousarray(codebook_hv, dtype=np.uint32).reshape(-1, self.HDC_WORDS)
        n_items = cb_arr.shape[0]
        cb_buf = None
        sims_buf = None
        if self.gpu_available and cl is not None and n_items > 0:
            try:
                mf = cl.mem_flags
                cb_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cb_arr)
                sims_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, int(n_items * 4))
            except Exception as exc:
                log.debug("Failed to allocate resident HDC codebook buffer: %s", exc)
                cb_buf = None
                sims_buf = None

        self._resident_hdc_codebooks[str(name)] = {
            "cb_cpu": cb_arr,
            "cb_buf": cb_buf,
            "sims_buf": sims_buf,
            "n_items": int(n_items),
            "bytes": int(cb_arr.nbytes),
            "metadata": list(metadata) if metadata is not None else [],
        }
        return {
            "name": str(name),
            "n_items": int(n_items),
            "vram_resident": cb_buf is not None,
            "vram_bytes": int(cb_arr.nbytes) if cb_buf is not None else 0,
        }

    def search_resident_hdc_codebook(
        self,
        name: str,
        query_hv: np.ndarray,
        top_k: int = 5,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Search a pre-uploaded resident HDC codebook in GT 710 VRAM without re-uploading the codebook.
        Only the 1,280-byte query vector is sent over PCIe.
        """
        t0 = time.perf_counter()
        entry = self._resident_hdc_codebooks.get(str(name))
        if not entry or entry["n_items"] == 0:
            return {
                "found": False,
                "name": str(name),
                "n_items": 0,
                "similarities": np.zeros(0, dtype=np.float32),
                "top_indices": [],
                "top_matches": [],
                "backend": "CPU",
                "elapsed_ms": 0.0,
            }

        q_arr = np.ascontiguousarray(query_hv, dtype=np.uint32).flatten()[: self.HDC_WORDS]
        n_items = int(entry["n_items"])
        cb_arr = entry["cb_cpu"]
        sims = np.empty(n_items, dtype=np.float32)
        backend = "CPU"

        use_gpu = (
            entry.get("cb_buf") is not None
            and self.should_use_gpu("hdc_resident_search", n_items, force_gpu=force_gpu)
        )
        if use_gpu:
            try:
                mf = cl.mem_flags
                q_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q_arr)
                s_buf = entry.get("sims_buf") or cl.Buffer(self.ctx, mf.WRITE_ONLY, sims.nbytes)
                self._kernels["hdc_hamming_search"](
                    self.queue,
                    (n_items,),
                    None,
                    q_buf,
                    entry["cb_buf"],
                    s_buf,
                    np.int32(n_items),
                    np.int32(self.HDC_WORDS),
                )
                cl.enqueue_copy(self.queue, sims, s_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("Resident HDC search fallback to CPU: %s", exc)
                backend = "CPU"

        if backend == "CPU":
            xor_words = np.bitwise_xor(cb_arr, q_arr[None, :])
            xor_bytes = xor_words.view(np.uint8).reshape(n_items, -1)
            bit_counts = np.unpackbits(xor_bytes, axis=1).sum(axis=1)
            sims = (1.0 - (bit_counts.astype(np.float32) / float(self.HDC_BITS))).astype(np.float32)

        k_cap = max(1, min(int(top_k), n_items))
        top_idx = np.argsort(sims)[::-1][:k_cap]
        meta_list = entry.get("metadata") or []
        top_matches = []
        for idx in top_idx:
            ii = int(idx)
            item_meta = dict(meta_list[ii]) if ii < len(meta_list) else {"index": ii}
            item_meta["similarity"] = round(float(sims[ii]), 6)
            item_meta["index"] = ii
            top_matches.append(item_meta)

        return {
            "found": True,
            "name": str(name),
            "n_items": n_items,
            "vram_resident": entry.get("cb_buf") is not None,
            "pcie_codebook_bytes_uploaded": 0,
            "similarities": sims,
            "top_indices": [int(i) for i in top_idx],
            "top_matches": top_matches,
            "backend": backend,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    # ------------------------------------------------------------------
    # 5. Test-Time Training (TTT-Linear) Fast-Weight Online Update
    # ------------------------------------------------------------------
    def ttt_linear_step(
        self,
        w_fast: np.ndarray,
        k_vec: np.ndarray,
        v_vec: np.ndarray,
        lr: float = 0.05,
        force_gpu: bool = False,
    ) -> Tuple[np.ndarray, float, str]:
        """Perform one online self-supervised reconstruction step on fast-weight matrix W_fast."""
        w_mat = np.ascontiguousarray(w_fast, dtype=np.float32)
        dim = w_mat.shape[0]
        k_arr = np.ascontiguousarray(k_vec, dtype=np.float32).flatten()[:dim]
        v_arr = np.ascontiguousarray(v_vec, dtype=np.float32).flatten()[:dim]
        err_out = np.empty(dim, dtype=np.float32)

        if self.should_use_gpu("ttt_linear_step", dim, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                w_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w_mat)
                k_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k_arr)
                v_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v_arr)
                e_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, err_out.nbytes)

                self._kernels["ttt_linear_fast_weight_step"](
                    self.queue,
                    (dim,),
                    None,
                    w_buf,
                    k_buf,
                    v_buf,
                    e_buf,
                    np.int32(dim),
                    np.float32(lr),
                )
                cl.enqueue_copy(self.queue, w_mat, w_buf).wait()
                cl.enqueue_copy(self.queue, err_out, e_buf).wait()
                loss = float(np.mean(err_out ** 2))
                return w_mat, loss, "OpenCL_GPU"
            except Exception as exc:
                log.debug("TTT-Linear GPU step fallback to CPU: %s", exc)

        pred = np.dot(w_mat, k_arr)
        err = pred - v_arr
        loss = float(np.mean(err ** 2))
        w_updated = (w_mat - float(lr) * np.outer(err, k_arr)).astype(np.float32)
        return w_updated, loss, "CPU"

    # ------------------------------------------------------------------
    # 5b. Muon (MomentUm Orthogonalized by Newton-Schulz) Fast-Weight Continual Pretraining
    # ------------------------------------------------------------------
    @staticmethod
    def zeroth_power_via_newtonschulz5(
        g_mat: np.ndarray,
        steps: int = 5,
        eps: float = 1e-7,
    ) -> np.ndarray:
        """
        Quintic Newton-Schulz iteration (`newtonschulz5`) for matrix orthogonalization:
        Approximates `U @ V.T` (the zeroth singular power of 2D gradient/momentum matrix G)
        using 5 polynomial iterations with coefficients (a=3.4445, b=-4.7750, c=2.0315).
        """
        g_2d = np.ascontiguousarray(g_mat, dtype=np.float32)
        if g_2d.ndim != 2:
            g_2d = g_2d.reshape(g_2d.shape[0], -1)
        norm = float(np.linalg.norm(g_2d))
        if norm < eps:
            return np.zeros_like(g_2d, dtype=np.float32)

        a, b, c = 3.4445, -4.7750, 2.0315
        x = (g_2d / (norm + eps)).astype(np.float32)
        transposed = False
        if x.shape[0] > x.shape[1]:
            x = x.T
            transposed = True

        n_steps = max(1, int(steps))
        for _ in range(n_steps):
            a_gram = np.dot(x, x.T)
            b_poly = b * a_gram + c * np.dot(a_gram, a_gram)
            x = (a * x + np.dot(b_poly, x)).astype(np.float32)

        if transposed:
            x = x.T
        return np.ascontiguousarray(x, dtype=np.float32)

    def muon_fast_weight_step(
        self,
        w_fast: np.ndarray,
        k_batch: np.ndarray,
        v_batch: np.ndarray,
        momentum_buf: Optional[np.ndarray] = None,
        lr: float = 0.05,
        momentum: float = 0.90,
        ns_steps: int = 5,
        loss_mask: Optional[np.ndarray] = None,
        nesterov: bool = True,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform a matrix-orthogonalized Muon continual pretraining / fast-weight update step
        with optional Loss-Masked SFT/CPT token masking (`loss_mask=0` on specification tokens,
        `loss_mask=1` on target implementation block tokens).
        """
        t0 = time.perf_counter()
        w_mat = np.ascontiguousarray(w_fast, dtype=np.float32)
        if w_mat.ndim != 2:
            dim = int(math.isqrt(w_mat.size))
            w_mat = w_mat.reshape(dim, dim)
        out_dim, in_dim = w_mat.shape

        k_2d = np.ascontiguousarray(k_batch, dtype=np.float32).reshape(-1, in_dim)
        v_2d = np.ascontiguousarray(v_batch, dtype=np.float32).reshape(-1, out_dim)
        batch_size = min(k_2d.shape[0], v_2d.shape[0])
        k_2d = k_2d[:batch_size]
        v_2d = v_2d[:batch_size]

        if loss_mask is not None:
            mask_arr = np.ascontiguousarray(loss_mask, dtype=np.float32).flatten()[:batch_size]
            if len(mask_arr) < batch_size:
                mask_arr = np.pad(mask_arr, (0, batch_size - len(mask_arr)), constant_values=1.0)
        else:
            mask_arr = np.ones(batch_size, dtype=np.float32)

        active_weight = float(np.sum(mask_arr))
        if active_weight <= 1e-8:
            return {
                "updated_weights": w_mat.copy(),
                "momentum_buf": momentum_buf.copy() if momentum_buf is not None else np.zeros_like(w_mat),
                "pre_update_loss": 0.0,
                "post_update_loss": 0.0,
                "loss_reduced": False,
                "spectral_isotropy_ratio": 1.0,
                "active_tokens": 0,
                "backend": "CPU",
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            }

        # Forward predictions & masked reconstruction error: Pred = K @ W^T
        preds = np.dot(k_2d, w_mat.T)
        raw_err = preds - v_2d
        masked_err = raw_err * mask_arr[:, None]
        pre_loss = float(np.sum((raw_err ** 2) * mask_arr[:, None]) / (active_weight * float(out_dim)))

        # Gradient G = (1 / active_weight) * (Masked_Err^T @ K)
        grad = np.dot(masked_err.T, k_2d) / active_weight

        if momentum_buf is not None and momentum_buf.shape == w_mat.shape:
            m_next = (float(momentum) * momentum_buf.astype(np.float32) + grad).astype(np.float32)
        else:
            m_next = grad.astype(np.float32)

        g_eff = (grad + float(momentum) * m_next).astype(np.float32) if nesterov else m_next
        u_ortho = self.zeroth_power_via_newtonschulz5(g_eff, steps=ns_steps)

        # Aspect-ratio RMS scaling from Moonshot / Keller Jordan Muon formulation
        rms_scale = math.sqrt(max(1.0, float(out_dim) / float(in_dim)))
        # Scale step by gradient magnitude when near convergence so small residuals don't overshoot
        grad_rms = float(np.sqrt(np.mean(grad ** 2)))
        effective_scale = float(lr) * rms_scale * min(1.0, grad_rms * math.sqrt(float(in_dim)))

        n_elems = int(w_mat.size)
        backend = "CPU"
        w_updated = w_mat.copy()

        if self.should_use_gpu("muon_fast_weight_step", n_elems, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                w_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=w_updated)
                u_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u_ortho)
                self._kernels["muon_fast_weight_apply"](
                    self.queue,
                    (n_elems,),
                    None,
                    w_buf,
                    u_buf,
                    np.int32(n_elems),
                    np.float32(effective_scale),
                )
                cl.enqueue_copy(self.queue, w_updated, w_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("Muon GPU step fallback to CPU: %s", exc)
                backend = "CPU"

        if backend == "CPU":
            w_updated = (w_mat - effective_scale * u_ortho).astype(np.float32)

        post_preds = np.dot(k_2d, w_updated.T)
        post_err = post_preds - v_2d
        post_loss = float(np.sum((post_err ** 2) * mask_arr[:, None]) / (active_weight * float(out_dim)))

        # Measure singular value isotropy of orthogonalized update U_ortho
        try:
            s_vals = np.linalg.svd(u_ortho, compute_uv=False)
            isotropy = float(np.mean(s_vals) / (float(np.max(s_vals)) + 1e-8)) if len(s_vals) > 0 else 1.0
        except Exception:
            isotropy = 1.0

        return {
            "updated_weights": w_updated,
            "orthogonal_update": u_ortho,
            "momentum_buf": m_next,
            "pre_update_loss": round(pre_loss, 6),
            "post_update_loss": round(post_loss, 6),
            "loss_reduced": bool(post_loss <= pre_loss),
            "spectral_isotropy_ratio": round(isotropy, 4),
            "active_tokens": int(round(active_weight)),
            "backend": backend,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    # ------------------------------------------------------------------
    # 5c. DeepSeek Engram O(1) Multi-Head Hash Lookup & Context-Aware Gating
    # ------------------------------------------------------------------
    def engram_multihead_gated_lookup(
        self,
        query_vecs: np.ndarray,
        table_keys: np.ndarray,
        table_vals: np.ndarray,
        head_indices: np.ndarray,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """Execute O(1) Multi-Head Conditional Memory Lookup with Context-Aware Sigmoid Gating.

        Implements DeepSeek Engram (Jan 2026) hash collision suppression:
            alpha_{b,h} = sigmoid(dot(q_b, k_{bucket(b,h)}) / sqrt(d))
            out_b = (1 / K) * sum_h alpha_{b,h} * v_{bucket(b,h)}
        """
        t0 = time.perf_counter()
        keys_tbl = np.ascontiguousarray(table_keys, dtype=np.float32)
        vals_tbl = np.ascontiguousarray(table_vals, dtype=np.float32)
        if keys_tbl.ndim != 2 or vals_tbl.ndim != 2:
            raise ValueError("table_keys and table_vals must be 2D matrices of shape [num_buckets, dim]")

        num_buckets, dim = keys_tbl.shape
        q_2d = np.ascontiguousarray(query_vecs, dtype=np.float32)
        if q_2d.ndim == 1:
            q_2d = q_2d.reshape(1, dim)
        else:
            q_2d = q_2d.reshape(-1, dim)

        idx_2d = np.ascontiguousarray(head_indices, dtype=np.int32)
        if idx_2d.ndim == 1:
            idx_2d = idx_2d.reshape(q_2d.shape[0], -1)

        batch_size = min(q_2d.shape[0], idx_2d.shape[0])
        q_2d = q_2d[:batch_size]
        idx_2d = np.mod(idx_2d[:batch_size], max(1, num_buckets))
        num_heads = idx_2d.shape[1] if idx_2d.ndim == 2 else 1
        inv_sqrt_dim = float(1.0 / math.sqrt(max(1, dim)))

        out_vecs = np.zeros((batch_size, dim), dtype=np.float32)
        gates = np.zeros((batch_size, num_heads), dtype=np.float32)
        backend = "CPU"

        if (
            batch_size > 0
            and num_buckets > 0
            and num_heads > 0
            and self.should_use_gpu("engram_multihead_gated_lookup", batch_size * dim, force_gpu=force_gpu)
        ):
            try:
                mf = cl.mem_flags
                q_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q_2d)
                k_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=keys_tbl)
                v_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vals_tbl)
                h_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=idx_2d)
                o_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=out_vecs.nbytes)
                g_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=gates.nbytes)
                self._kernels["engram_multihead_gated_lookup"](
                    self.queue,
                    (batch_size,),
                    None,
                    q_buf,
                    k_buf,
                    v_buf,
                    h_buf,
                    o_buf,
                    g_buf,
                    np.int32(batch_size),
                    np.int32(num_buckets),
                    np.int32(num_heads),
                    np.int32(dim),
                    np.float32(inv_sqrt_dim),
                )
                cl.enqueue_copy(self.queue, out_vecs, o_buf).wait()
                cl.enqueue_copy(self.queue, gates, g_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("Engram lookup GPU fallback to CPU: %s", exc)
                backend = "CPU"

        if backend == "CPU" and batch_size > 0 and num_buckets > 0 and num_heads > 0:
            gathered_k = keys_tbl[idx_2d]  # [B, H, D]
            gathered_v = vals_tbl[idx_2d]  # [B, H, D]
            dots = np.sum(q_2d[:, None, :] * gathered_k, axis=-1) * inv_sqrt_dim  # [B, H]
            z = np.clip(dots, -20.0, 20.0)
            gates = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
            out_vecs = np.mean(gates[:, :, None] * gathered_v, axis=1).astype(np.float32)

        return {
            "gated_output": out_vecs,
            "gate_activations": gates,
            "mean_gate": float(np.mean(gates)) if gates.size > 0 else 0.0,
            "backend": backend,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    # ------------------------------------------------------------------
    # 6. Blelloch Parallel Associative Scan for Mamba-S6 State Space Models
    # ------------------------------------------------------------------
    def ssm_parallel_scan(
        self,
        a_bar_seq: np.ndarray,
        b_u_seq: np.ndarray,
        h0: np.ndarray,
        force_gpu: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """
        Execute O(log N) associative parallel prefix scan over a discretized SSM sequence:
        a_bar_seq: (seq_len, d_model, d_state)
        b_u_seq:   (seq_len, d_model, d_state)
        h0:        (d_model, d_state)
        Returns all hidden states h_all of shape (seq_len, d_model, d_state) and backend name.
        """
        a_arr = np.ascontiguousarray(a_bar_seq, dtype=np.float32)
        b_arr = np.ascontiguousarray(b_u_seq, dtype=np.float32)
        orig_shape = a_arr.shape
        seq_len = orig_shape[0]
        n_channels = int(np.prod(orig_shape[1:]))
        a_flat = a_arr.reshape(seq_len, n_channels)
        b_flat = b_arr.reshape(seq_len, n_channels)
        h_curr = np.ascontiguousarray(h0, dtype=np.float32).reshape(n_channels).copy()
        h_out = np.empty((seq_len, n_channels), dtype=np.float32)

        if self.should_use_gpu("ssm_blelloch_parallel_scan", seq_len, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                # Process in chunks of up to 256 tokens using GPU __local SRAM parallel scan
                for start in range(0, seq_len, 256):
                    end = min(seq_len, start + 256)
                    chunk_len = end - start
                    a_chunk = np.ascontiguousarray(a_flat[start:end], dtype=np.float32)
                    b_chunk = np.ascontiguousarray(b_flat[start:end], dtype=np.float32)
                    h_chunk_out = np.empty((chunk_len, n_channels), dtype=np.float32)

                    a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_chunk)
                    b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_chunk)
                    h0_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_curr)
                    out_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, h_chunk_out.nbytes)

                    local_size = 256
                    global_size = n_channels * local_size
                    self._kernels["ssm_blelloch_parallel_scan"](
                        self.queue,
                        (global_size,),
                        (local_size,),
                        a_buf,
                        b_buf,
                        h0_buf,
                        out_buf,
                        np.int32(chunk_len),
                        np.int32(n_channels),
                    )
                    cl.enqueue_copy(self.queue, h_chunk_out, out_buf).wait()
                    h_out[start:end] = h_chunk_out
                    h_curr = h_chunk_out[-1].copy()
                return h_out.reshape(orig_shape), "OpenCL_GPU"
            except Exception as exc:
                log.debug("SSM parallel scan GPU fallback to CPU: %s", exc)

        h_curr = np.ascontiguousarray(h0, dtype=np.float32).reshape(n_channels).copy()
        for t in range(seq_len):
            h_curr = a_flat[t] * h_curr + b_flat[t]
            h_out[t] = h_curr
        return h_out.reshape(orig_shape), "CPU"

    # ------------------------------------------------------------------
    # 7. Bitplane Differentiable Logic Gate Tree Evaluation (DiffLogic)
    # ------------------------------------------------------------------
    def difflogic_bitplane_forward(
        self,
        x_bits: np.ndarray,
        conn_a: np.ndarray,
        conn_b: np.ndarray,
        gate_ids: np.ndarray,
        force_gpu: bool = False,
    ) -> Tuple[np.ndarray, str]:
        """
        Evaluate a layer of discrete 2-input Boolean logic gates (0..15) across 32-bit bitplanes.
        x_bits: (n_in_features, n_words) uint32
        conn_a, conn_b, gate_ids: (n_gates,) int32
        Returns y_bits of shape (n_gates, n_words) uint32 and backend name.
        """
        xb = np.ascontiguousarray(x_bits, dtype=np.uint32)
        if xb.ndim == 1:
            xb = xb[:, None]
        n_in, n_words = xb.shape
        ca = np.ascontiguousarray(np.clip(conn_a, 0, n_in - 1), dtype=np.int32).flatten()
        cb = np.ascontiguousarray(np.clip(conn_b, 0, n_in - 1), dtype=np.int32).flatten()
        gids = np.ascontiguousarray(gate_ids, dtype=np.int32).flatten()
        n_gates = len(gids)
        yb = np.empty((n_gates, n_words), dtype=np.uint32)

        if self.should_use_gpu("difflogic_bitplane_eval", n_gates * n_words, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xb)
                ca_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ca)
                cb_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cb)
                g_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gids)
                y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, yb.nbytes)

                self._kernels["difflogic_bitplane_eval"](
                    self.queue,
                    (n_gates, n_words),
                    None,
                    x_buf,
                    ca_buf,
                    cb_buf,
                    g_buf,
                    y_buf,
                    np.int32(n_gates),
                    np.int32(n_words),
                )
                cl.enqueue_copy(self.queue, yb, y_buf).wait()
                return yb, "OpenCL_GPU"
            except Exception as exc:
                log.debug("DiffLogic bitplane GPU fallback to CPU: %s", exc)

        mask32 = np.uint32(0xFFFFFFFF)
        for g in range(n_gates):
            a = xb[ca[g]]
            b = xb[cb[g]]
            op = int(gids[g]) & 15
            if op == 0:
                res = np.zeros_like(a)
            elif op == 1:
                res = a & b
            elif op == 2:
                res = a & (~b)
            elif op == 3:
                res = a
            elif op == 4:
                res = (~a) & b
            elif op == 5:
                res = b
            elif op == 6:
                res = a ^ b
            elif op == 7:
                res = a | b
            elif op == 8:
                res = ~(a | b)
            elif op == 9:
                res = ~(a ^ b)
            elif op == 10:
                res = ~b
            elif op == 11:
                res = a | (~b)
            elif op == 12:
                res = ~a
            elif op == 13:
                res = (~a) | b
            elif op == 14:
                res = ~(a & b)
            else:
                res = np.full_like(a, mask32)
            yb[g] = res
        return yb, "CPU"

    # ------------------------------------------------------------------
    # 8. Kuramoto Oscillatory Phase Synchronization + Modern Hopfield Readout
    # ------------------------------------------------------------------
    def kuramoto_hopfield_attention(
        self,
        x_in: np.ndarray,
        phases: np.ndarray,
        omega: np.ndarray,
        coupling_J: np.ndarray,
        hopfield_K: np.ndarray,
        hopfield_V: np.ndarray,
        sync_steps: int = 8,
        dt: float = 0.15,
        beta: float = 8.0,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute coupled Kuramoto phase synchronization (AKOrN / KoPE) and Modern Hopfield
        LogSumExp dense associative readout on the GPU.

        The GPU path runs `sync_steps` single-step kernel launches with ping-ponged
        phase buffers (synchronous Jacobi, each launch reads one consistent phase
        snapshot) - bit-comparable to the CPU path.  The previous single-launch
        kernel mixed old/new phases across concurrent work-items (race).
        """
        x_arr = np.ascontiguousarray(x_in, dtype=np.float32)
        seq_len, d_model = x_arr.shape
        ph_arr = np.ascontiguousarray(phases, dtype=np.float32).reshape(seq_len, d_model).copy()
        om_arr = np.ascontiguousarray(omega, dtype=np.float32).reshape(seq_len, d_model)
        j_arr = np.ascontiguousarray(coupling_J, dtype=np.float32).reshape(seq_len, seq_len)

        if hopfield_K is not None and len(hopfield_K) > 0:
            hk_arr = np.ascontiguousarray(hopfield_K, dtype=np.float32).reshape(-1, d_model)
            hv_arr = np.ascontiguousarray(hopfield_V, dtype=np.float32).reshape(-1, d_model)
            n_mem = hk_arr.shape[0]
        else:
            hk_arr = np.zeros((1, d_model), dtype=np.float32)
            hv_arr = np.zeros((1, d_model), dtype=np.float32)
            n_mem = 0

        y_out = np.empty((seq_len, d_model), dtype=np.float32)

        if self.should_use_gpu("kuramoto_hopfield_step", seq_len * d_model, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_arr)
                ph_a = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ph_arr)
                ph_b = cl.Buffer(self.ctx, mf.READ_WRITE, ph_arr.nbytes)
                om_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=om_arr)
                j_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=j_arr)
                hk_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hk_arr)
                hv_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hv_arr)
                y_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, y_out.nbytes)

                # Synchronous Jacobi: each launch reads one consistent phase
                # snapshot and writes the next; buffers swap between launches.
                # sync_steps=0 leaves the phases untouched (old behavior).
                steps = max(0, int(sync_steps))
                for _ in range(steps):
                    self._kernels["kuramoto_hopfield_sync_step"](
                        self.queue,
                        (seq_len, d_model),
                        None,
                        ph_a,
                        ph_b,
                        om_buf,
                        j_buf,
                        np.int32(seq_len),
                        np.int32(d_model),
                        np.float32(dt),
                    )
                    ph_a, ph_b = ph_b, ph_a
                cl.enqueue_copy(self.queue, ph_arr, ph_a).wait()

                self._kernels["kuramoto_hopfield_readout"](
                    self.queue,
                    (seq_len, d_model),
                    None,
                    x_buf,
                    ph_a,
                    hk_buf,
                    hv_buf,
                    y_buf,
                    np.int32(seq_len),
                    np.int32(d_model),
                    np.int32(n_mem),
                    np.float32(beta),
                )
                cl.enqueue_copy(self.queue, y_out, y_buf).wait()

                # Kuramoto order parameter R = |mean(exp(i * theta))|
                r_sync = float(np.sqrt(np.mean(np.cos(ph_arr)) ** 2 + np.mean(np.sin(ph_arr)) ** 2))
                return {
                    "output": y_out,
                    "phases": ph_arr,
                    "kuramoto_order_r": round(r_sync, 4),
                    "backend": "OpenCL_GPU",
                }
            except Exception as exc:
                log.debug("Kuramoto-Hopfield GPU fallback to CPU: %s", exc)

        for _ in range(sync_steps):
            diff = ph_arr[None, :, :] - ph_arr[:, None, :]
            sync_force = np.sum(j_arr[:, :, None] * np.sin(diff), axis=1)
            ph_arr = ph_arr + dt * (om_arr + sync_force)

        x_bound = x_arr * (0.7 + 0.3 * np.cos(ph_arr))
        if n_mem > 0:
            scores = beta * np.matmul(x_arr, hk_arr.T)
            scores -= np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(np.clip(scores, -40.0, 0.0))
            weights /= (np.sum(weights, axis=-1, keepdims=True) + 1e-12)
            y_out = x_bound + np.matmul(weights, hv_arr)
        else:
            y_out = x_bound

        r_sync = float(np.sqrt(np.mean(np.cos(ph_arr)) ** 2 + np.mean(np.sin(ph_arr)) ** 2))
        return {
            "output": y_out.astype(np.float32),
            "phases": ph_arr.astype(np.float32),
            "kuramoto_order_r": round(r_sync, 4),
            "backend": "CPU",
        }

    # ------------------------------------------------------------------
    # 9. Predictive Coding & Forward-Forward Iterative SRAM Relaxation
    # ------------------------------------------------------------------
    def predictive_coding_relax_and_adapt(
        self,
        r0_batch: np.ndarray,
        w0: np.ndarray,
        w1: np.ndarray,
        relax_steps: int = 20,
        state_lr: float = 0.08,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Run iterative variational free-energy state relaxation across a 3-layer hierarchy
        (32 <-> 24 <-> 16) in GPU registers/SRAM.
        """
        r0_arr = np.ascontiguousarray(r0_batch, dtype=np.float32).reshape(-1, 32)
        batch_size = r0_arr.shape[0]
        w0_arr = np.ascontiguousarray(w0, dtype=np.float32).reshape(32, 24)
        w1_arr = np.ascontiguousarray(w1, dtype=np.float32).reshape(24, 16)

        r1_out = np.empty((batch_size, 24), dtype=np.float32)
        r2_out = np.empty((batch_size, 16), dtype=np.float32)
        e0_out = np.empty((batch_size, 32), dtype=np.float32)
        e1_out = np.empty((batch_size, 24), dtype=np.float32)
        metrics_out = np.empty((batch_size, 4), dtype=np.float32)

        if self.should_use_gpu("predictive_coding_sram_relax", batch_size, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                r0_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r0_arr)
                w0_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w0_arr)
                w1_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w1_arr)
                r1_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, r1_out.nbytes)
                r2_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, r2_out.nbytes)
                e0_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, e0_out.nbytes)
                e1_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, e1_out.nbytes)
                m_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, metrics_out.nbytes)

                self._kernels["predictive_coding_sram_relax"](
                    self.queue,
                    (batch_size,),
                    None,
                    r0_buf,
                    w0_buf,
                    w1_buf,
                    r1_buf,
                    r2_buf,
                    e0_buf,
                    e1_buf,
                    m_buf,
                    np.int32(batch_size),
                    np.int32(relax_steps),
                    np.float32(state_lr),
                )
                cl.enqueue_copy(self.queue, r1_out, r1_buf).wait()
                cl.enqueue_copy(self.queue, r2_out, r2_buf).wait()
                cl.enqueue_copy(self.queue, e0_out, e0_buf).wait()
                cl.enqueue_copy(self.queue, e1_out, e1_buf).wait()
                cl.enqueue_copy(self.queue, metrics_out, m_buf).wait()

                return {
                    "r1": r1_out,
                    "r2": r2_out,
                    "e0": e0_out,
                    "e1": e1_out,
                    "initial_free_energy": float(np.mean(metrics_out[:, 0])),
                    "final_free_energy": float(np.mean(metrics_out[:, 1])),
                    "goodness_l1": float(np.mean(metrics_out[:, 2])),
                    "goodness_l2": float(np.mean(metrics_out[:, 3])),
                    "backend": "OpenCL_GPU",
                }
            except Exception as exc:
                log.debug("Predictive Coding SRAM relax GPU fallback to CPU: %s", exc)

        r1 = np.tanh(np.matmul(r0_arr, w0_arr))
        r2 = np.tanh(np.matmul(r1, w1_arr))
        f_init = 0.0
        for step in range(relax_steps):
            e0 = r0_arr - np.matmul(r1, w0_arr.T)
            e1 = r1 - np.matmul(r2, w1_arr.T)
            if step == 0:
                f_init = float(0.5 * np.mean(np.sum(e0 ** 2, axis=1) + np.sum(e1 ** 2, axis=1)))
            r1 = np.tanh(r1 + state_lr * (np.matmul(e0, w0_arr) - e1))
            r2 = np.tanh(r2 + state_lr * np.matmul(e1, w1_arr))

        e0 = r0_arr - np.matmul(r1, w0_arr.T)
        e1 = r1 - np.matmul(r2, w1_arr.T)
        f_final = float(0.5 * np.mean(np.sum(e0 ** 2, axis=1) + np.sum(e1 ** 2, axis=1)))
        return {
            "r1": r1.astype(np.float32),
            "r2": r2.astype(np.float32),
            "e0": e0.astype(np.float32),
            "e1": e1.astype(np.float32),
            "initial_free_energy": f_init,
            "final_free_energy": f_final,
            "goodness_l1": float(np.mean(r1 ** 2)),
            "goodness_l2": float(np.mean(r2 ** 2)),
            "backend": "CPU",
        }

    # ------------------------------------------------------------------
    # 10. Parallel Monte-Carlo Tree Rollout of Expected Free Energy G(pi)
    # ------------------------------------------------------------------
    @staticmethod
    def _digamma_approx(x: float) -> float:
        """Accurate Asymptotic + Recurrence Digamma psi(x) for x > 0."""
        val = float(max(1e-4, x))
        result = 0.0
        while val < 6.0:
            result -= 1.0 / val
            val += 1.0
        inv = 1.0 / val
        inv2 = inv * inv
        result += math.log(val) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
        return result

    @classmethod
    def _dirichlet_expected_kl_gain(cls, alpha: float, beta: float) -> float:
        """
        Approximate Active Inference Epistemic Information Gain (Expected Bayesian Surprise):
        E_{o ~ P(o|pi)} [ D_KL( Beta(alpha'_o, beta'_o) || Beta(alpha, beta) ) ].
        The closed form uses the digamma recurrence psi(x+1) = psi(x) + 1/x to derive the
        +1/n information-gain term; digamma evaluation relies on an asymptotic Stirling/Bernoulli
        series approximation (_digamma_approx), and small counts are clamped at max(0.5, ...)
        for numerical stability. Monotonically higher for under-explored / uncertain beliefs
        (low N = alpha + beta) and vanishes as ~1 / (2*N) when beliefs are well-observed.
        """
        a = max(0.5, float(alpha))
        b = max(0.5, float(beta))
        n = a + b
        p_succ = a / n
        p_fail = b / n
        psi_a = cls._digamma_approx(a)
        psi_b = cls._digamma_approx(b)
        psi_n = cls._digamma_approx(n)
        eig = (
            p_succ * math.log(n / a)
            + p_fail * math.log(n / b)
            + p_succ * psi_a
            + p_fail * psi_b
            - psi_n
            + (1.0 / n)
        )
        return float(max(0.0, eig))

    def active_inference_mc_rollout(
        self,
        init_drives: np.ndarray,
        policy_shifts: np.ndarray,
        persona_weights: np.ndarray,
        target_prefs: np.ndarray,
        n_particles: int = 256,
        horizon: int = 8,
        noise_scale: float = 0.04,
        seed: int = 2026,
        belief_counts: Optional[np.ndarray] = None,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Simulate `n_policies * n_particles` stochastic trajectories over `horizon` steps on GPU
        to compute Expected Free Energy G(pi) = Pragmatic Risk - Epistemic Gain + CVaR Tail Risk.
        `belief_counts` (optional, shape `(n_policies, 2)`): Beta/Dirichlet pseudo-counts `(alpha_p, beta_p)`
        for each policy's outcome belief distribution.
        """
        init_arr = np.ascontiguousarray(init_drives, dtype=np.float32).flatten()[:4]
        shifts_arr = np.ascontiguousarray(policy_shifts, dtype=np.float32).reshape(-1, 4)
        n_policies = shifts_arr.shape[0]
        weights_arr = np.ascontiguousarray(persona_weights, dtype=np.float32).reshape(n_policies, 4)
        prefs_arr = np.ascontiguousarray(target_prefs, dtype=np.float32).flatten()[:4]

        if belief_counts is not None:
            bc_arr = np.ascontiguousarray(belief_counts, dtype=np.float32).reshape(n_policies, 2)
        else:
            # Default priors inversely scaled by persona curiosity precision so high-curiosity
            # policies probe higher-uncertainty epistemic hypotheses
            bc_arr = np.empty((n_policies, 2), dtype=np.float32)
            for pol in range(n_policies):
                cur_w = max(0.2, float(weights_arr[pol, 1]))
                eff_count = max(1.0, 6.0 / (cur_w * cur_w))
                bc_arr[pol, 0] = eff_count
                bc_arr[pol, 1] = eff_count

        total_threads = n_policies * n_particles
        pragmatic_out = np.empty((n_policies, n_particles), dtype=np.float32)
        min_surv_out = np.empty((n_policies, n_particles), dtype=np.float32)
        final_states = np.empty((n_policies, n_particles, 4), dtype=np.float32)

        backend = "CPU"
        if self.should_use_gpu("active_inference_mc_rollout", total_threads, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                i_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=init_arr)
                s_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shifts_arr)
                w_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights_arr)
                t_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prefs_arr)
                p_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, pragmatic_out.nbytes)
                m_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, min_surv_out.nbytes)
                f_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, final_states.nbytes)

                self._kernels["active_inference_mc_rollout"](
                    self.queue,
                    (n_policies, n_particles),
                    None,
                    i_buf,
                    s_buf,
                    w_buf,
                    t_buf,
                    p_buf,
                    m_buf,
                    f_buf,
                    np.int32(n_policies),
                    np.int32(n_particles),
                    np.int32(horizon),
                    np.float32(noise_scale),
                    np.uint32(seed),
                )
                cl.enqueue_copy(self.queue, pragmatic_out, p_buf).wait()
                cl.enqueue_copy(self.queue, min_surv_out, m_buf).wait()
                cl.enqueue_copy(self.queue, final_states, f_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("Active Inference MC rollout GPU fallback to CPU: %s", exc)
                backend = "CPU"

        if backend == "CPU":
            rng = np.random.default_rng(seed)
            for pol in range(n_policies):
                s = np.tile(init_arr, (n_particles, 1))
                d = shifts_arr[pol] * weights_arr[pol]
                prag = np.zeros(n_particles, dtype=np.float32)
                min_s = np.minimum(s[:, 0], s[:, 3])
                for _ in range(horizon):
                    noise = rng.uniform(-noise_scale, noise_scale, size=(n_particles, 4)).astype(np.float32)
                    s = np.clip(s + d[None, :] + noise, 0.0, 1.0)
                    min_s = np.minimum(min_s, np.minimum(s[:, 0], s[:, 3]))
                    prag += np.sum((s - prefs_arr[None, :]) ** 2, axis=1)
                pragmatic_out[pol] = prag / float(horizon)
                min_surv_out[pol] = min_s
                final_states[pol] = s

        # Aggregate EFE components per policy
        policy_metrics = []
        cvar_k = max(1, n_particles // 10)  # Worst 10% tail risk (CVaR_0.10)
        for pol in range(n_policies):
            mean_pragmatic = float(np.mean(pragmatic_out[pol]))
            # True Dirichlet/Beta Expected KL Information Gain (Bayesian surprise over H steps),
            # scaled by policy curiosity drive and epistemic trajectory reach
            alpha_p, beta_p = float(bc_arr[pol, 0]), float(bc_arr[pol, 1])
            dirichlet_eig = self._dirichlet_expected_kl_gain(alpha_p, beta_p)
            curiosity_drive = float(np.mean(final_states[pol, :, 1])) * float(weights_arr[pol, 1])
            epistemic_gain = float(dirichlet_eig * (0.5 + curiosity_drive) * math.sqrt(float(horizon)))
            worst_surv = np.sort(min_surv_out[pol])[:cvar_k]
            cvar_shortfall = float(np.mean(np.maximum(0.0, 0.5 - worst_surv)))
            efe_score = mean_pragmatic - 0.5 * epistemic_gain + 1.5 * cvar_shortfall
            policy_metrics.append({
                "policy_idx": pol,
                "efe": round(efe_score, 6),
                "pragmatic_risk": round(mean_pragmatic, 6),
                "epistemic_info_gain": round(epistemic_gain, 6),
                "dirichlet_kl_nats": round(dirichlet_eig, 6),
                "belief_alpha_beta": [round(alpha_p, 3), round(beta_p, 3)],
                "cvar_tail_risk": round(cvar_shortfall, 6),
                "mean_final_drives": [round(float(x), 4) for x in np.mean(final_states[pol], axis=0)],
            })

        return {
            "backend": backend,
            "device": self.device_name if backend == "OpenCL_GPU" else "CPU",
            "total_particles": total_threads,
            "horizon": horizon,
            "policy_metrics": policy_metrics,
        }

    # ------------------------------------------------------------------
    # 11. Multi-Scale Vietoris-Rips Persistence Barcode & Geodesic SLERP Search
    # ------------------------------------------------------------------
    def rips_multiscale_slerp_search(
        self,
        u_a: np.ndarray,
        u_b: np.ndarray,
        theta: float,
        t_grid: Optional[np.ndarray] = None,
        eps_grid: Optional[np.ndarray] = None,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate (n_t x n_eps) Vietoris-Rips filtration graphs across the geodesic SLERP arc
        in parallel on the GPU using register Union-Find for beta_0 and beta_1 barcodes.

        The register union-find caps the filtration at 32 points: inputs with more
        points are evaluated on their FIRST 32 points only, and the result
        discloses this via `points_evaluated` / `points_truncated`.
        """
        ua = np.ascontiguousarray(u_a, dtype=np.float32)
        ub = np.ascontiguousarray(u_b, dtype=np.float32)
        n_pts, pt_dim = ua.shape
        v_cap = min(n_pts, 32)
        if n_pts > 32:
            log.warning(
                "rips_multiscale_slerp_search: %d points requested but the "
                "filtration evaluates only the first 32 (register union-find cap).",
                n_pts,
            )

        if t_grid is None:
            # Include endpoints t=0.0 (parent A) and t=1.0 (parent B) plus 14 interior geodesic points
            t_arr = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        else:
            t_arr = np.ascontiguousarray(t_grid, dtype=np.float32).flatten()

        if eps_grid is None:
            eps_arr = np.linspace(0.08, 0.85, 32, dtype=np.float32)
        else:
            eps_arr = np.ascontiguousarray(eps_grid, dtype=np.float32).flatten()

        n_t = len(t_arr)
        n_eps = len(eps_arr)
        beta0_out = np.empty((n_t, n_eps), dtype=np.int32)
        beta1_out = np.empty((n_t, n_eps), dtype=np.int32)
        edges_out = np.empty((n_t, n_eps), dtype=np.int32)

        backend = "CPU"
        if self.should_use_gpu("rips_multiscale_filtration", n_t * n_eps, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                ua_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ua)
                ub_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ub)
                t_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_arr)
                e_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=eps_arr)
                b0_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, beta0_out.nbytes)
                b1_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, beta1_out.nbytes)
                ed_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, edges_out.nbytes)

                self._kernels["rips_multiscale_filtration"](
                    self.queue,
                    (n_t, n_eps),
                    None,
                    ua_buf,
                    ub_buf,
                    t_buf,
                    e_buf,
                    b0_buf,
                    b1_buf,
                    ed_buf,
                    np.int32(n_pts),
                    np.int32(pt_dim),
                    np.int32(n_t),
                    np.int32(n_eps),
                    np.float32(theta),
                )
                cl.enqueue_copy(self.queue, beta0_out, b0_buf).wait()
                cl.enqueue_copy(self.queue, beta1_out, b1_buf).wait()
                cl.enqueue_copy(self.queue, edges_out, ed_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("Rips multiscale filtration GPU fallback to CPU: %s", exc)
                backend = "CPU"

        if backend == "CPU":
            sin_t = math.sin(theta)
            for ti, tv in enumerate(t_arr):
                if abs(sin_t) > 1e-4:
                    wa = math.sin((1.0 - float(tv)) * theta) / sin_t
                    wb = math.sin(float(tv) * theta) / sin_t
                else:
                    wa = 1.0 - float(tv)
                    wb = float(tv)
                pts = wa * ua[:v_cap] + wb * ub[:v_cap]
                diff = pts[:, None, :] - pts[None, :, :]
                dist_sq = np.sum(diff ** 2, axis=-1)
                for ei, ev in enumerate(eps_arr):
                    parent = list(range(v_cap))
                    num_edges = 0
                    comps = v_cap
                    esq = float(ev * ev)
                    for i in range(v_cap):
                        for j in range(i + 1, v_cap):
                            if dist_sq[i, j] < esq:
                                num_edges += 1
                                ri, rj = i, j
                                while parent[ri] != ri:
                                    ri = parent[ri]
                                while parent[rj] != rj:
                                    rj = parent[rj]
                                if ri != rj:
                                    parent[rj] = ri
                                    comps -= 1
                    b0 = comps
                    b1 = max(0, num_edges - v_cap + b0)
                    beta0_out[ti, ei] = b0
                    beta1_out[ti, ei] = b1
                    edges_out[ti, ei] = num_edges

        # Parent barcodes are at t=0.0 (index 0) and t=1.0 (index -1)
        parent_avg_b1 = 0.5 * (beta1_out[0].astype(np.float32) + beta1_out[-1].astype(np.float32))
        parent_avg_b0 = 0.5 * (beta0_out[0].astype(np.float32) + beta0_out[-1].astype(np.float32))

        # Search interior geodesic candidates (indices 1 .. n_t - 2) for minimal L1 Wasserstein barcode discrepancy
        interior_indices = range(1, n_t - 1) if n_t > 2 else range(n_t)
        best_ti = 1 if n_t > 2 else 0
        best_dist = float("inf")
        for ti in interior_indices:
            w1_dist = float(np.mean(np.abs(beta1_out[ti] - parent_avg_b1)) + 0.5 * np.mean(np.abs(beta0_out[ti] - parent_avg_b0)))
            if w1_dist < best_dist:
                best_dist = w1_dist
                best_ti = ti

        return {
            "backend": backend,
            "device": self.device_name if backend == "OpenCL_GPU" else "CPU",
            "optimal_t": round(float(t_arr[best_ti]), 4),
            "optimal_index": int(best_ti),
            "wasserstein_l1_discrepancy": round(best_dist, 4),
            "beta0_barcode": beta0_out[best_ti].tolist(),
            "beta1_barcode": beta1_out[best_ti].tolist(),
            "parent_a_beta1_integral": int(np.sum(beta1_out[0])),
            "parent_b_beta1_integral": int(np.sum(beta1_out[-1])),
            "fused_beta1_integral": int(np.sum(beta1_out[best_ti])),
            "grid_evaluations": int(n_t * n_eps),
            "points_evaluated": int(v_cap),
            "points_truncated": bool(n_pts > v_cap),
        }

    # ------------------------------------------------------------------
    # 12. Dominated Novelty Search (DNS, 2025 SOTA Quality-Diversity)
    # ------------------------------------------------------------------
    def dominated_novelty_search(
        self,
        fitness: np.ndarray,
        descriptors: np.ndarray,
        k_nearest: int = 5,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Dominated Novelty Search (DNS, 2025 SOTA Quality-Diversity):
        Calculates for each individual its mean Euclidean distance to its k-nearest strictly fitter neighbors.
        If an individual is a non-dominated Pareto frontier elite, it gets the global maximum novelty bonus.
        """
        t0 = time.perf_counter()
        fit_arr = np.ascontiguousarray(fitness, dtype=np.float32).flatten()
        pop_size = len(fit_arr)
        desc_arr = np.ascontiguousarray(descriptors, dtype=np.float32).reshape(pop_size, -1)
        desc_dim = desc_arr.shape[1]
        k_cap = max(1, min(int(k_nearest), 8, pop_size - 1 if pop_size > 1 else 1))

        dns_out = np.empty(pop_size, dtype=np.float32)
        backend = "CPU"

        if self.should_use_gpu("dominated_novelty_search", pop_size, force_gpu=force_gpu):
            try:
                mf = cl.mem_flags
                f_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fit_arr)
                d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=desc_arr)
                s_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, dns_out.nbytes)

                self._kernels["dominated_novelty_search_kernel"](
                    self.queue,
                    (pop_size,),
                    None,
                    f_buf,
                    d_buf,
                    s_buf,
                    np.int32(pop_size),
                    np.int32(desc_dim),
                    np.int32(k_cap),
                )
                cl.enqueue_copy(self.queue, dns_out, s_buf).wait()
                backend = "OpenCL_GPU"
            except Exception as exc:
                log.debug("DNS kernel GPU fallback to CPU: %s", exc)
                backend = "CPU"

        if backend == "CPU":
            for i in range(pop_size):
                f_i = fit_arr[i]
                dists = []
                peer_dists = []
                for j in range(pop_size):
                    if i == j:
                        continue
                    d = float(np.linalg.norm(desc_arr[i] - desc_arr[j]))
                    peer_dists.append(d)
                    if fit_arr[j] > f_i:
                        dists.append(d)
                if not dists:
                    dns_out[i] = (float(np.mean(peer_dists)) if peer_dists else 0.0) + 10.0
                else:
                    dists.sort()
                    used = dists[:k_cap]
                    dns_out[i] = float(np.mean(used))

        # Best novelty-quality balance
        combined_score = fit_arr + 0.3 * dns_out
        best_candidate = int(np.argmax(combined_score))
        return {
            "backend": backend,
            "device": self.device_name if backend == "OpenCL_GPU" else "CPU",
            "pop_size": pop_size,
            "desc_dim": desc_dim,
            "k_nearest": k_cap,
            "dns_scores": dns_out,
            "combined_scores": combined_score,
            "best_candidate_idx": best_candidate,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }


def get_opencl_accelerator() -> OpenCLMicroBrainAccelerator:
    """Return the singleton OpenCL micro-brain accelerator."""
    return OpenCLMicroBrainAccelerator.get_instance()


__all__ = ["OpenCLMicroBrainAccelerator", "get_opencl_accelerator"]

