"""
Complete RNN-to-RNN hypernetwork model for continuous behavioral features.
Combines MMD encoder and continuous decoder.
"""
from model.mmdae_enc_continuous import MMDAEContinuous
from model.dec_continuous import DECRNNContinuous


class HYPMMDContinuous:
    """
    Complete hypernetwork model for continuous behavioral modeling.

    Architecture:
    1. Encoder: Behavioral sequences -> latent code z (via MMD autoencoder)
    2. Decoder: z -> RNN weights -> behavioral predictions (via hypernetwork)

    Args:
        enc_cells: Number of LSTM units in encoder
        dec_cells: Number of GRU units in decoder
        feature_dim: Dimensionality of continuous features
        s_size: State space size (0 if no external states)
        latent_size: Dimensionality of latent space
        n_T: Maximum sequence length
        static_loops: Whether to use static RNN
        mmd_coef: Coefficient for MMD loss
    """

    def __init__(self, enc_cells, dec_cells, feature_dim, s_size, latent_size, n_T, static_loops, mmd_coef=2):
        self.enc = MMDAEContinuous(
            enc_cells, feature_dim, s_size, latent_size, n_T, static_loops, mmd_coef
        )

        self.dec = DECRNNContinuous(
            dec_cells, feature_dim, s_size, 1, self.enc.z, n_T, static_loops
        )
