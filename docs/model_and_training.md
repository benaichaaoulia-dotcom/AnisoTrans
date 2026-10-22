# Model and training

This page describes the model and training procedure. Setup, data preparation, and run commands are in [README.md](../README.md).

## Inputs and output

- `theta`: 8 normalised configuration parameters for one configuration.
- `K`: the normalised permeability field, shape (N, 3), shared by all configurations.
- Output: the normalised pressure field, shape (N,), for the cells of one configuration.

## Architecture (`AnisoTransV2`)

1. **Config scaler** (`ConfigScaler`): maps `theta` to three positive scaling factors $s \in \mathbb{R}^3$. The output is bounded by `exp_clip` ([-3, 3]) and initialised from the physics pins.
2. **Transmissibility**: $K_s = K \odot s$ and $T = \tfrac{1}{3}\sum_j K_{s,j}$ per cell.
3. **Token embedding**: each cell is a 14-vector $[K;\,K_s;\,\theta]$, embedded to `dim` = 256, then added to a Fourier positional encoding of the cell coordinates (`n_fourier` = 16). A LayerNorm follows.
4. **Sparse attention blocks**: `n_layers` = 8 blocks. Each cell attends to its `knn_k` = 16 nearest neighbours (candidates selected by k-NN on normalised coordinates). With `n_heads` = 8 and `ffn_mult` = 4, dropout 0.1.
5. **Physics bias**: the attention logits receive the bias $\alpha \cdot \tfrac{1}{2}(T_i + T_j)$ for each neighbour pair. $\alpha = \alpha_{\max}\,\sigma(\alpha_{\text{raw}})$ is learned, with $\alpha_{\max} = 1$ and initial value 0.1.
6. **Readout**: LayerNorm, Linear 256 to 64, ReLU, Linear 64 to 1.

Gradient checkpointing is used for the blocks during training.

## Objective

For a batch of configurations, the loss is:

$$
\mathcal{L} = \frac{1}{P}\sum_{\text{configs}}\sum_{\text{valid fit cells}} \left(\hat{y} - y\right)^2 \;+\; \lambda_{\text{anchor}} \cdot \frac{1}{B}\sum_{b=1}^{B}\left\lVert s(\theta_b) - s_{\text{phys}}(\theta_b)\right\rVert_2^2
$$

- $P$ is the number of valid fit (config, cell) pairs.
- $\lambda_{\text{anchor}} = 0.05$.
- $s_{\text{phys}}$ is the physics-pinned scaler target.

The `lambda_smooth` term used in the source notes is not applied by the source training step, so it is not implemented here.

## Training schedule

| Setting | Value |
|---|---|
| Optimiser | AdamW (weight decay 1e-2) |
| Learning rate | 3e-4, linear warmup for 10 epochs, then cosine decay to 1e-5 |
| Max epochs | 150 |
| Early stopping | patience 20 epochs |
| Configs per optimiser step | `accum` = 8 (reduced by half on CUDA out-of-memory) |
| Gradient clipping | norm 5.0 |
| Precision | fp16 autocast with gradient scaling |
| Seed | 42 |

The best checkpoint is selected on the validation split (40 configurations). The test split (75 configurations) is evaluated once, from the frozen checkpoint, and written to `eval_frozen_test.json`.

## Parameter count

The model has 6,364,709 trainable parameters at the configuration in [configs/v2.yaml](../configs/v2.yaml).
