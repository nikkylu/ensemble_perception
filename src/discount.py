"""
Outlier discounting: is the outlier left out of the represented mean?

After Avci & Boduroglu (2021). A continuous mean readout is trained on the
STANDARD stimuli (same domain and set size) and applied to the DISCOUNT
stimuli, whose displays hold N-1 group elements plus one outlier. Its
prediction is compared with the local mean (group only) and the global mean
(all elements, the outlier unwrapped on its side — the paper's arithmetic mean):

    β = Σ (pred − local)(global − local) / Σ (global − local)²

    β = 0  outlier discounted (prediction tracks the group mean)
    β = 1  outlier weighted like any other element
    β > 1  outlier over-weighted

reported separately for less / more distinct outliers, per slice and layer.

Baseline β_circ: the β a representation would show if it simply averaged
the elements on the circle (circular global mean). For orientation a 90°
outlier is opposite in doubled-angle space, so circular averaging barely
moves the mean (β_circ ≈ 0.1) — β ≈ 0 for "more distinct" is then expected
without any discounting. Compare β with β_circ, not with 0. (Size is linear:
β_circ = 1.)

Readout (readout.py): ridge regression on features standardized with the
training data, onto the realized mean of each standard display — (cos, sin)
of the angle for orientation / hue, ln size for size. λ is chosen by 5-fold
CV on the standard set (error reported as "cv err": circular degrees, or ln
units for size).

Slices: cls, bar (all element tokens incl. the outlier), bg, all_patches, and
bar_group — the mean of the group tokens only (outlier token removed): if
tokens carried only their own element, bar_group should give β ≈ 0.

Usage:
    python3 discount.py --train features/ori_standard_n8_s42.pt \\
                        --test features/ori_discount_spatial_n8_s42.pt --save res.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loader import element_tokens, token_rows
from readout import decode, diff, encode, fit_readout  # noqa: F401 (re-export)

N_LAYERS = 12
SLICES = ["cls", "bar", "bg", "all_patches"]
N_BOOT = 1000


def bootstrap_beta(p_off, g_off, c_off, n_boot=N_BOOT, seed=0):
    """Trial-level bootstrap (resample displays with replacement; the readout
    stays fixed). 95% CIs of β and of β − β_circ — the latter is the paired
    test against plain averaging."""
    idx = np.random.default_rng(seed).integers(0, len(p_off), (n_boot, len(p_off)))
    G = g_off[idx]
    den = (G ** 2).sum(1)
    beta = (p_off[idx] * G).sum(1) / den
    dcirc = beta - (c_off[idx] * G).sum(1) / den
    return ([float(q) for q in np.percentile(beta, [2.5, 97.5])],
            [float(q) for q in np.percentile(dcirc, [2.5, 97.5])])


# ─────────────────────────────────────────────────────────────────
# analysis
# ─────────────────────────────────────────────────────────────────

def slices(feats, n):
    """{slice: (trials, L, D) float array}; bar_group drops the outlier's
    element vector, bar_outlier is the outlier's element vector alone
    (readout trained on standard single-element rows)."""
    out = {st: feats[st].float().numpy() for st in SLICES}
    if "bar_concat" in feats and "outlier_slot" in feats:
        tok = element_tokens(feats["bar_concat"].float(), n, feats["cls"].shape[-1])
        slot = feats["outlier_slot"].long()
        keep = torch.ones(tok.shape[0], n, dtype=torch.bool)
        keep[torch.arange(tok.shape[0]), slot] = False
        out["bar_group"] = (tok * keep[:, None, :, None]).sum(2).div(n - 1).numpy()
        out["bar_outlier"] = tok[torch.arange(tok.shape[0]), :, slot].numpy()
    return out


def realized_mean(feats):
    """Realized mean of each standard display (falls back to the nominal mean)."""
    if "realized_mean" in feats:                       # deg, or ln px for size
        return feats["realized_mean"].numpy().astype(float)
    m = np.asarray(feats["mean_values"], float)[feats["labels_mean"].numpy()]
    return m if feats.get("period", 180) else np.log(m)   # size: nominal px → ln px


def analyse(train, test):
    P = float(train.get("period", 180))
    n = int(test["n_bars"][0])
    m_train = realized_mean(train)
    local = test["local_mean"].numpy().astype(float)
    glob = test["global_mean_linear"].numpy().astype(float)
    circ = test["global_mean"].numpy().astype(float)
    distinct = test["outlier_distinct"].numpy()
    X_train = slices(train, int(train["n_bars"][0]))
    X_test = slices(test, n)
    # bar_outlier is a single token: its readout trains on standard single tokens
    X_single = m_single = g_single = None
    if "bar_outlier" in X_test and "bar_concat" in train:
        rows_tok, k = token_rows(train["bar_concat"].float(),
                                 int(train["n_bars"][0]), train["cls"].shape[-1])
        X_single = rows_tok.numpy()
        m_single = np.repeat(m_train, k)
        g_single = np.repeat(np.arange(len(m_train)), k)   # fold by display

    res = {}
    for st in X_test:
        if st == "bar_outlier" and X_single is not None:
            X_tr, m_tr, g_tr = X_single, m_single, g_single
        else:
            # bar_group has no standard counterpart: use the readout trained on `bar`
            X_tr, m_tr, g_tr = X_train[st if st in X_train else "bar"], m_train, None
        rows = []
        for layer in range(N_LAYERS):
            predict, cv_err, lam = fit_readout(X_tr[:, layer], m_tr, P, groups=g_tr)
            pred = predict(X_test[st][:, layer])
            row = {"cv_err": cv_err, "lambda": lam}
            for k, name in enumerate(["less", "more"]):
                sel = distinct == k
                p_off = diff(pred[sel], local[sel], P)
                g_off = diff(glob[sel], local[sel], P)
                c_off = diff(circ[sel], local[sel], P)
                beta_ci, dcirc_ci = bootstrap_beta(p_off, g_off, c_off)
                row[name] = {
                    "beta": float((p_off * g_off).sum() / (g_off ** 2).sum()),
                    "beta_circ": float((c_off * g_off).sum() / (g_off ** 2).sum()),
                    "beta_ci": beta_ci,
                    "beta_minus_circ_ci": dcirc_ci,
                    "err_local": float(np.abs(p_off).mean()),
                    "err_global": float(np.abs(diff(pred[sel], glob[sel], P)).mean()),
                    "local_global_gap": float(np.abs(g_off).mean())}
            rows.append(row)
        res[st] = rows
    return res


def print_table(res, unit):
    print(f"\nβ: 0 = outlier discounted, 1 = equal weight; β_circ = circular averaging"
          f"   (cv err in {unit})")
    print(f"{'':12s}{'layer':>6s}{'cv err':>9s}{'β less':>9s}{'β more':>9s}"
          f"{'|p-loc| more':>14s}{'|p-glob| more':>15s}")
    for st, rows in res.items():
        for layer, r in enumerate(rows):
            print(f"{st:12s}{layer:>6d}{r['cv_err']:>9.3g}{r['less']['beta']:>9.2f}"
                  f"{r['more']['beta']:>9.2f}{r['more']['err_local']:>14.3g}"
                  f"{r['more']['err_global']:>15.3g}")
        print()
    r0 = next(iter(res.values()))[0]
    print(f"circular-averaging baseline: β_circ less {r0['less']['beta_circ']:+.2f}, "
          f"more {r0['more']['beta_circ']:+.2f}")
    # summary at each slice's most accurate layer on the standard stimuli
    print("at the layer with the lowest standard cv err "
          "(trial-bootstrap 95% CI; * : β−β_circ CI excludes 0):")
    for st, rows in res.items():
        b = int(np.argmin([r["cv_err"] for r in rows]))
        parts = [f"  {st:12s} L{b:02d}  cv err {rows[b]['cv_err']:.3g}  "]
        for d in ["less", "more"]:
            r = rows[b][d]
            lo, hi = r.get("beta_ci", (np.nan, np.nan))
            sig = "*" if "beta_minus_circ_ci" in r and (
                r["beta_minus_circ_ci"][0] > 0 or r["beta_minus_circ_ci"][1] < 0) else " "
            parts.append(f"β {d} {r['beta']:+.2f} [{lo:+.2f},{hi:+.2f}]{sig}  ")
        print("".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="standard-stimuli features (.pt)")
    p.add_argument("--test", required=True, help="discount-stimuli features (.pt)")
    p.add_argument("--save", default=None, help="JSON output")
    a = p.parse_args()

    train = torch.load(a.train, map_location="cpu")
    test = torch.load(a.test, map_location="cpu")
    assert "local_mean" in test, f"{a.test} is not a discount feature file"
    assert int(train["n_bars"][0]) == int(test["n_bars"][0]), "set sizes differ"

    res = analyse(train, test)
    print_table(res, "ln px" if not train.get("period", 180) else "deg")
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps({"train": a.train, "test": a.test, "results": res}))
        print(f"saved → {a.save}")


if __name__ == "__main__":
    main()
