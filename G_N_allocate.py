import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NN_cff_vec_alloc(nn.Module):
    """
    Random cosine feature map with per-input-dimension feature allocation.

    Output features: [N, M, d_out], where M = sum_i m_i.

    This implements a "dimension-wise block" feature design:
      phi(x) = concat_i sqrt(2/m_i) * cos( (x_i / ell_i) * W_i + b_i )

    which corresponds most naturally to an additive/separable kernel design.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        total_features: int = 512,
        lengthscale='auto',
        sigma_in=1.0,
        sigma_out=1.0,
        trainable_kernel: bool = False,
        projection_type: str = 'Gaussian',
        device: str = 'cpu',
        # allocation controls (choose one)
        features_per_dim=None,        # e.g. [128, 32, 32, ...], must sum to total_features if given
        feature_weights=None,         # e.g. weights proportional to 1/ell or any heuristic; will be normalized
        min_features_per_dim: int = 1 # ensures no dim gets 0 features unless you explicitly allow it
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.total_features = int(total_features)
        self.device = device
        self.kernel = projection_type
        self.trainable_kernel = trainable_kernel

        # lengthscale: ARD style
        if isinstance(lengthscale, str) and lengthscale == 'auto':
            base = np.sqrt(self.d_in)
            lengthscale_vec = torch.ones(self.d_in, device=device).float() * float(base)
        else:
            # keep your original convention: multiply by sqrt(d_in)
            ls = float(lengthscale) * np.sqrt(self.d_in)
            lengthscale_vec = torch.ones(self.d_in, device=device).float() * ls

        self.lengthscale = nn.Parameter(lengthscale_vec, requires_grad=trainable_kernel)

        # sigma controls (vector or scalar) for input/output scaling
        self.sigma_in = np.array(sigma_in)
        self.sigma_out = np.array(sigma_out)

        # feature allocation
        self.m_per_dim = self._resolve_allocation(
            features_per_dim=features_per_dim,
            feature_weights=feature_weights,
            min_features_per_dim=min_features_per_dim
        )
        self.M = int(sum(self.m_per_dim))

        # parameters holding concatenated blocks
        # W_cat: [M, d_out], but conceptually split into blocks for each dim i
        # b_cat: [M, d_out]
        self.W_cat = nn.Parameter(torch.empty(self.M, self.d_out, device=device), requires_grad=False)
        self.b_cat = nn.Parameter(torch.empty(self.M, self.d_out, device=device), requires_grad=False)

        # offsets for slicing blocks quickly
        offsets = [0]
        for m in self.m_per_dim:
            offsets.append(offsets[-1] + int(m))
        self.register_buffer("offsets", torch.tensor(offsets, device=device, dtype=torch.long))

    def _resolve_allocation(self, features_per_dim, feature_weights, min_features_per_dim: int):
        if features_per_dim is not None:
            m = np.asarray(features_per_dim, dtype=int).tolist()
            assert len(m) == self.d_in, "features_per_dim must have length d_in."
            assert sum(m) == self.total_features, "features_per_dim must sum to total_features."
            return [int(x) for x in m]

        if feature_weights is None:
            # default: uniform allocation
            base = self.total_features // self.d_in
            rem = self.total_features - base * self.d_in
            m = [base] * self.d_in
            for i in range(rem):
                m[i] += 1
            return m

        w = np.asarray(feature_weights, dtype=float)
        assert w.shape == (self.d_in,), "feature_weights must have shape (d_in,)."
        w = np.maximum(w, 0.0)
        if w.sum() == 0:
            # fall back to uniform if all zeros
            return self._resolve_allocation(None, None, min_features_per_dim)

        # allocate with floor + remainder distribution, enforcing minimum
        M = self.total_features
        m = np.floor(M * (w / w.sum())).astype(int)

        # enforce minimum
        if min_features_per_dim is not None and min_features_per_dim > 0:
            m = np.maximum(m, int(min_features_per_dim))

        # fix sum to exactly M
        diff = int(M - m.sum())
        if diff > 0:
            # add remaining features to dims with largest fractional "desire"
            desire = (M * (w / w.sum())) - m
            idx = np.argsort(-desire)
            for k in range(diff):
                m[idx[k % self.d_in]] += 1
        elif diff < 0:
            # remove extra features from dims with smallest desire, but don't go below minimum
            desire = (M * (w / w.sum())) - m
            idx = np.argsort(desire)  # smallest first
            k = 0
            while diff < 0 and k < 10 * self.d_in:
                i = idx[k % self.d_in]
                if min_features_per_dim is not None and m[i] <= int(min_features_per_dim):
                    k += 1
                    continue
                m[i] -= 1
                diff += 1
                k += 1
            assert m.sum() == M, "Could not adjust allocation to match total_features under min constraint."

        return [int(x) for x in m.tolist()]

    def resample(self):
        dtype = torch.float32

        # sigma_in -> [d_in]
        if np.array(self.sigma_in).shape:
            sigma_in = torch.tensor(self.sigma_in, device=self.device, dtype=dtype)
            assert sigma_in.shape == (self.d_in,)
        else:
            sigma_in = torch.full((self.d_in,), float(self.sigma_in), device=self.device, dtype=dtype)

        # sigma_out -> [d_out]
        if np.array(self.sigma_out).shape:
            sigma_out = torch.tensor(self.sigma_out, device=self.device, dtype=dtype)
            assert sigma_out.shape == (self.d_out,)
        else:
            sigma_out = torch.full((self.d_out,), float(self.sigma_out), device=self.device, dtype=dtype)

        # sample per-dim blocks, then concatenate
        W_blocks = []
        b_blocks = []

        for i in range(self.d_in):
            m_i = int(self.m_per_dim[i])
            if self.kernel == 'Gaussian':
                # W_i: [m_i, d_out]
                Wi = torch.randn(m_i, self.d_out, device=self.device, dtype=dtype)
            elif self.kernel == 'Cauchy':
                laplace = torch.distributions.Laplace(
                    torch.tensor(0.0, dtype=dtype, device=self.device),
                    torch.tensor(1.0, dtype=dtype, device=self.device)
                )
                Wi = laplace.sample((m_i, self.d_out))
            else:
                raise ValueError(f"Unknown projection type: {self.kernel}")

            # scale: input-dim dependent * output-dim dependent
            Wi = Wi * sigma_in[i] * sigma_out[None, :]

            bi = torch.rand(m_i, self.d_out, device=self.device, dtype=dtype) * (2.0 * torch.pi)

            W_blocks.append(Wi)
            b_blocks.append(bi)

        self.W_cat.data = torch.cat(W_blocks, dim=0)  # [M, d_out]
        self.b_cat.data = torch.cat(b_blocks, dim=0)  # [M, d_out]

    def forward(self, x: torch.Tensor):
        """
        x: [N, d_in]
        returns: [N, M, d_out]
        """
        assert x.dim() == 2 and x.shape[1] == self.d_in
        x = x.to(self.device, dtype=torch.float32)

        # ARD scaling
        x_scaled = x / self.lengthscale  # [N, d_in]

        # build features blockwise to avoid huge intermediate tensor
        feats = []
        for i in range(self.d_in):
            start = int(self.offsets[i].item())
            end = int(self.offsets[i + 1].item())
            Wi = self.W_cat[start:end, :]  # [m_i, d_out]
            bi = self.b_cat[start:end, :]  # [m_i, d_out]

            # x_i: [N, 1, 1], Wi: [1, m_i, d_out] => broadcast to [N, m_i, d_out]
            xi = x_scaled[:, i].view(-1, 1, 1)
            proj = xi * Wi.view(1, -1, self.d_out) + bi.view(1, -1, self.d_out)

            # scale per block so each block has comparable variance
            m_i = max(1, end - start)
            scale = torch.sqrt(torch.tensor(2.0 / float(m_i), device=self.device))
            feats.append(torch.cos(proj) * scale)

        return torch.cat(feats, dim=1)  # [N, M, d_out]


def train_feature_model_alloc(
    model,
    input_data: torch.Tensor,
    targets: torch.Tensor,
    n_epochs: int = 1000,
    batch_size: int = 100,
    print_every: int = 100,
    lr: float = 1e-2,
    lambda_reg: float = 1.0,
    weight_given: torch.Tensor = torch.empty(0),
    model_given: bool = False,
    vectorize: bool = False,
):
    """
    This is the same training logic you already have, but robust to variable M (=sum m_i).

    model.forward(input_data) must return:
      - if vectorize=False: features [N, M, d_out], weights [M, d_out] => pred [N, d_out]
      - if vectorize=True:  features [N, M, d_out], weights [M, 1]   => pred [N, d_out] (shared across outputs)
    """
    if not model_given:
        model.resample()

    assert input_data.shape[0] == targets.shape[0]
    features = model.forward(input_data)          # [N, M, d_out]
    M = features.shape[1]
    d_out = features.shape[2]
    assert targets.shape[-1] == d_out, f"targets last dim must be d_out={d_out}"

    if vectorize:
        weights_dim2 = 1
    else:
        weights_dim2 = d_out

    if weight_given.numel() > 0:
        if weight_given.shape == (M, weights_dim2):
            weights = weight_given.detach().clone().requires_grad_(True)
        else:
            raise ValueError(f"Given weight shape {weight_given.shape} != expected {(M, weights_dim2)}")
    else:
        weights = torch.randn(M, weights_dim2, device=features.device, requires_grad=True)

    optimizer = optim.SGD([weights], lr=lr)
    loss_history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(features.shape[0], device=features.device)
        feat_shuf = features[perm]
        targ_shuf = targets[perm]

        for i in range(0, features.shape[0], batch_size):
            batch_features = feat_shuf[i:i + batch_size]  # [B, M, d_out]
            batch_targets = targ_shuf[i:i + batch_size]   # [B, d_out]

            optimizer.zero_grad()

            if vectorize:
                # weights: [M,1] -> broadcast to [B,M,d_out], sum over M
                pred = (batch_features * weights.view(1, M, 1)).sum(dim=1)  # [B, d_out]
            else:
                # einsum over feature dimension M: [B,M,d_out] x [M,d_out] -> [B,d_out]
                pred = torch.einsum('bmd,md->bd', batch_features, weights)

            mse_loss = ((pred - batch_targets) ** 2).mean()
            l2_loss = (weights ** 2).mean()
            loss = mse_loss + float(lambda_reg) * l2_loss

            loss.backward(retain_graph=getattr(model, "trainable_kernel", False))
            optimizer.step()

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}, MSE: {mse_loss.item():.6f}, L2: {l2_loss.item():.6f}")
            loss_history.append((epoch + 1, mse_loss.item(), l2_loss.item()))

    return model, weights, loss_history
