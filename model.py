"""Model definition and inference, shared by training and serving.

Deliberately imports nothing beyond torch/numpy and `features`. The serving
environment therefore needs neither scikit-learn (whose pickles are
version-fragile) nor pyvista/VTK (about 100 MB) nor the dataset: a clone with
`artifacts/model.pt` plus `artifacts/model_meta.json` is enough to run the demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

import features
from config import MODEL_META, MODEL_PT


class MLP(nn.Module):
    """Plain feed-forward net: Linear/SiLU stack, linear output."""

    def __init__(self, n_in: int, hidden: list[int], n_out: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.SiLU()]
            width = h
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU via tanh rather than exp: algebraically identical to x*sigmoid(x)
    but it cannot overflow for large-magnitude inputs."""
    return x * 0.5 * (1.0 + np.tanh(0.5 * x))


def _stack_members(state_dicts: list) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Ensemble weights stacked on a leading member axis, transposed to (in, out).

    nn.Linear stores its weight as (out, in) and computes ``x @ W.T``; we store
    (in, out) so the forward pass is a plain contraction.
    """
    keys = sorted(
        (k for k in state_dicts[0] if k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    weights, biases = [], []
    for key in keys:
        bias_key = key[: -len("weight")] + "bias"
        weights.append(np.stack([sd[key].numpy().T.astype(np.float64) for sd in state_dicts]))
        biases.append(np.stack([sd[bias_key].numpy().astype(np.float64) for sd in state_dicts]))
    return weights, biases


@dataclass
class Prediction:
    cl: np.ndarray
    cd: np.ndarray
    cl_std: np.ndarray
    cd_std: np.ndarray


class Bundle:
    """A trained ensemble plus its scalers and the linear baseline."""

    def __init__(self, state: dict, meta: dict) -> None:
        if meta["feature_version"] != features.FEATURE_VERSION:
            raise RuntimeError(
                f"artifact was trained with feature_version "
                f"{meta['feature_version']}, but features.py is "
                f"{features.FEATURE_VERSION}. Retrain before serving."
            )
        self.meta = meta
        arch = meta["arch"]
        self.members: list[MLP] = []
        for sd in state["members"]:
            net = MLP(arch["n_in"], list(arch["hidden"]), arch["n_out"])
            net.load_state_dict(sd)
            net.eval()
            self.members.append(net)

        # Inference runs in numpy, not torch. These nets are ~34k parameters on
        # a batch of one, where PyTorch's per-op dispatch overhead dwarfs the
        # arithmetic -- measured at 1.7 ms against roughly 30 us for the same
        # maths in numpy. Stacking the ensemble's weights on a leading axis also
        # turns all five members into a single einsum per layer instead of five
        # separate forward passes.
        self._w, self._b = _stack_members(state["members"])

        self.x_mean = state["x_mean"].numpy()
        self.x_std = state["x_std"].numpy()
        self.y_mean = state["y_mean"].numpy()
        self.y_std = state["y_std"].numpy()
        self.linear_coef = state["linear_coef"].numpy()
        self.linear_intercept = state["linear_intercept"].numpy()

    # -- inference --------------------------------------------------------

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(x) - self.x_mean) / self.x_std

    def _forward(self, xs: np.ndarray) -> np.ndarray:
        """All ensemble members at once: (N, n_in) -> (M, N, n_out)."""
        # matmul rather than einsum: it broadcasts (N, i) against the stacked
        # (M, i, o) weights to (M, N, o) just the same, without paying to parse
        # a subscript string on every call -- which is not negligible when the
        # arithmetic itself is only ~34k multiply-adds.
        h = xs @ self._w[0] + self._b[0][:, None, :]
        for w, b in zip(self._w[1:], self._b[1:]):
            h = _silu(h) @ w + b[:, None, :]
        return h

    def predict(self, x: np.ndarray) -> Prediction:
        """Ensemble prediction from raw (unstandardised) features.

        The point estimate is the member mean taken in *model* space and then
        inverted, so Cd is the geometric mean of the members. The reported
        spread is the standard deviation of the members in *physical* space,
        which is what a user reading "Cd = 0.0121 +/- 0.0004" expects.
        """
        raw = self._forward(self._standardise(x))
        members = raw * self.y_std + self.y_mean          # (M, N, 2) model space

        cl, cd = features.inverse_targets(members.mean(axis=0))
        cl_mem = members[:, :, 0]
        cd_mem = np.exp(members[:, :, 1])
        return Prediction(
            cl=cl, cd=cd,
            cl_std=cl_mem.std(axis=0), cd_std=cd_mem.std(axis=0),
        )

    def predict_linear(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The baseline's prediction, same features and same target transform."""
        xs = self._standardise(x)
        y = xs @ self.linear_coef.T + self.linear_intercept
        return features.inverse_targets(y * self.y_std + self.y_mean)

    # -- introspection ----------------------------------------------------

    @property
    def input_ranges(self) -> dict[str, list[float]]:
        return self.meta["input_ranges"]

    def extrapolation_report(self, x: np.ndarray) -> dict:
        """Which features sit outside the training box, and by how much.

        Returned to API callers as a warning rather than an error: FR-12 is
        explicitly about showing where the model fails, so refusing
        out-of-range input would hide the most interesting behaviour.
        """
        x = np.atleast_2d(x)[0]
        warnings, fields = [], {}
        for i, name in enumerate(features.FEATURE_NAMES):
            lo, hi = self.input_ranges[name]
            v = float(x[i])
            if v < lo or v > hi:
                over = round(v - hi, 6) if v > hi else round(v - lo, 6)
                fields[name] = {"value": v, "range": [lo, hi], "delta": over}
                warnings.append(
                    f"{name}={v:.4g} is outside the training range "
                    f"[{lo:.4g}, {hi:.4g}]"
                )
        return {"in_range": not warnings, "warnings": warnings, "fields": fields}


def load_bundle(model_pt: Path = MODEL_PT, model_meta: Path = MODEL_META) -> Bundle:
    # weights_only=True is the torch >= 2.6 default and we keep it: the artifact
    # holds only tensors, lists and dicts, so it needs no allowlisting and
    # cannot execute anything on load.
    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    meta = json.loads(Path(model_meta).read_text(encoding="utf-8"))
    return Bundle(state, meta)
