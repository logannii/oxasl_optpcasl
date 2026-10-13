"""
Scrubbing decision for multi-delay pCASL: can these label/control pairs be removed?

Self-contained (numpy only) version of check_scrub() in scrubbing_slides.ipynb, without plots.

For a protocol given as one LD/PLD per acquired label/control (L/C) pair and a list of pairs to
remove, the CBF and ATT precision (Cramer-Rao lower bound of the Buxton pCASL model) is computed
with all pairs and with the pairs removed. The SD increase is calculated at every ATT
(default 0.5-2.5 s) and CBF value (default 10-100 ml/100g/min) and averaged with equal weight.
The removal is affordable if the mean SD increase is at most the threshold (default +15%) for
both CBF and ATT. If not, nothing is removed.

Where removing pairs leaves some ATTs inestimable (every remaining sample is taken after the bolus
has fully arrived, or at most one distinct timepoint has seen labelled blood), a Gaussian ATT
prior (default SD 1.0 s) is added to the Fisher information at those ATTs, for the scrubbed and the
full protocol alike, so CBF keeps a finite cost.

Example:
    >>> plds = [0.0] * 8 + [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.2]
    >>> lds = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8] + [1.8] * 8
    >>> affordable, new_plds, new_lds = check_scrub(plds, lds, remove=[10])
"""

import numpy as np

# Physiological parameters of the kinetic model (Buxton et al. MRM 1998, as in oxasl_optpcasl)
T1B = 1.65      # T1 of blood (s)
T1T = 1.445     # T1 of tissue (s)
ALPHA = 0.85    # labelling efficiency
LAM = 0.9       # blood-brain partition coefficient
M0B = 1.0

NOISE = 0.0013                          # noise SD of one differenced pair, relative to M0b
CBF_VALUES = (10, 20, 35, 50, 65, 80, 100)
ATT_RANGE = (0.5, 2.5)
ATT_STEP = 0.02
ATT_PRIOR = (1.3, 1.0)                  # Gaussian ATT prior (mean, SD) in s; only the SD affects precision
THRESHOLD = 0.15


def _sensitivities(lds, plds, atts, cbf):
    """
    Derivatives of the Buxton pCASL difference signal with respect to CBF and ATT

    :param lds, plds: per-delay labelling duration and PLD (s) [NDelays]
    :param atts: ATTs (s) [NATTs]
    :param cbf: CBF (ml/100g/min)
    :return: dS/df, dS/dATT, each [NDelays, NATTs] (f in ml/g/s)
    """
    f = cbf / 6000.0
    t1p = 1.0 / (1.0 / T1T + f / LAM)
    ld = lds[:, None]
    t = (lds + plds)[:, None]               # sample time = LD + PLD
    att = atts[None, :]
    equal = np.isclose(t, ld + att)
    during = (t > att) & ((t < ld + att) | equal)
    after = (t > ld + att) & ~equal
    m = 2 * M0B * ALPHA * t1p * np.exp(-att / T1B)
    df_during = m * (1 - np.exp((att - t) / t1p))
    datt_during = m * f * (-1.0 / T1B - np.exp((att - t) / t1p) * (1.0 / t1p - 1.0 / T1B))
    df_after = m * np.exp((ld + att - t) / t1p) * (1 - np.exp(-ld / t1p))
    datt_after = m * f * (1 - np.exp(-ld / t1p)) * np.exp((ld + att - t) / t1p) * (1.0 / t1p - 1.0 / T1B)
    return df_during * during + df_after * after, datt_during * during + datt_after * after


def _variances(hess):
    """CBF ((ml/100g/min)^2) and ATT (s^2) variances from 2x2 Fisher matrices [..., 2, 2]; inf if singular"""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        det = hess[..., 0, 0] * hess[..., 1, 1] - hess[..., 0, 1] * hess[..., 1, 0]
        ok = det / (hess[..., 0, 0] * hess[..., 1, 1]) > 1e-8
        var_cbf = np.where(ok, hess[..., 1, 1] / det, np.inf) * 6000.0 ** 2
        var_att = np.where(ok, hess[..., 0, 0] / det, np.inf)
    return var_cbf, var_att


def _group(plds, lds):
    """Unique (LD, PLD) delays, number of pairs at each, and the delay of each pair"""
    keys = [(round(float(ld), 5), round(float(pld), 5)) for ld, pld in zip(lds, plds)]
    unique = list(dict.fromkeys(keys))
    index = {k: i for i, k in enumerate(unique)}
    delay_of_pair = np.array([index[k] for k in keys])
    u_lds = np.array([k[0] for k in unique])
    u_plds = np.array([k[1] for k in unique])
    return u_lds, u_plds, np.bincount(delay_of_pair, minlength=len(unique)), delay_of_pair


def _outside_estimable(u_lds, u_plds, counts, atts):
    """ATTs the retained delays cannot estimate: at/below the shortest PLD, at/beyond the second-longest LD+PLD"""
    kept = counts > 0
    times = np.unique(np.round(u_lds[kept] + u_plds[kept], 5))
    if len(times) < 2:
        return np.ones(len(atts), dtype=bool)
    return (atts <= u_plds[kept].min() + 1e-9) | (atts >= times[-2] - 1e-9)


def sd_increase(plds, lds, remove, cbf_values=CBF_VALUES, att_range=ATT_RANGE, att_step=ATT_STEP,
                att_prior=ATT_PRIOR, noise=NOISE):
    """
    Mean SD increase of CBF and ATT caused by removing the given pairs

    :param plds: PLD of each L/C pair in acquisition order (s)
    :param lds: LD of each pair (s), or a single value for all
    :param remove: 0-based indices of the pairs to remove
    :param att_prior: (mean, SD) of the Gaussian ATT prior used where ATT becomes inestimable, or None
    :return: dict {"CBF": increase, "ATT": increase}, e.g. 0.1 = estimates 10% noisier (inf if inestimable)
    """
    plds = np.asarray(plds, dtype=float)
    lds = np.broadcast_to(np.asarray(lds, dtype=float), plds.shape)
    remove = [int(i) for i in remove]
    if len(set(remove)) != len(remove) or any(not 0 <= i < len(plds) for i in remove):
        raise ValueError(f"remove must be distinct pair indices between 0 and {len(plds) - 1}")

    u_lds, u_plds, repeats, delay_of_pair = _group(plds, lds)
    counts = repeats - np.bincount(delay_of_pair[remove], minlength=len(repeats))

    lo, hi = att_range
    atts = np.linspace(lo, hi, int(1 + np.ceil((hi - lo) / att_step)))
    # ATTs at or below the shortest PLD of the full protocol cannot be estimated at all: no weight
    weights = (atts > u_plds.min()).astype(float)
    prior = _outside_estimable(u_lds, u_plds, counts, atts) if att_prior is not None else np.zeros(len(atts), bool)

    increases = {"CBF": [], "ATT": []}
    for cbf in cbf_values:
        df, datt = _sensitivities(u_lds, u_plds, atts, cbf)
        terms = np.stack([df * df, df * datt, df * datt, datt * datt], axis=-1).reshape(len(u_lds), len(atts), 2, 2)

        def fisher(n):
            return np.tensordot(n / noise ** 2, terms, axes=1)          # [NATTs, 2, 2]

        h_scrubbed, h_full = fisher(counts.astype(float)), fisher(repeats.astype(float))
        prior_here = prior.copy()
        if att_prior is not None:
            prior_here |= ~np.isfinite(_variances(h_scrubbed)[0])     # any other singular ATTs
            h_scrubbed[:, 1, 1] += np.where(prior_here, 1.0 / att_prior[1] ** 2, 0.0)
            h_full[:, 1, 1] += np.where(prior_here, 1.0 / att_prior[1] ** 2, 0.0)
        for name, var_scrubbed, var_full in zip(("CBF", "ATT"), _variances(h_scrubbed), _variances(h_full)):
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                ratio = var_scrubbed / var_full
                ratio = np.where(np.isnan(ratio), np.inf, ratio)        # undefined -> inestimable (keeps inf as inf)
                increases[name].append(np.sqrt(ratio) - 1)

    result = {}
    for name, by_cbf in increases.items():
        with np.errstate(invalid="ignore", over="ignore"):
            by_att = np.mean(by_cbf, axis=0)                               # equal weight over CBF values
            result[name] = float(np.where(weights > 0, by_att * weights, 0.0).sum() / weights.sum())
    return result


def check_scrub(plds, lds, remove, threshold=THRESHOLD, max_loss_fraction=None, **kwargs):
    """
    Decide whether the given pairs can be removed

    :param plds: PLD of each L/C pair in acquisition order (s)
    :param lds: LD of each pair (s), or a single value for all
    :param remove: 0-based indices of the pairs to remove
    :param threshold: maximum mean SD increase allowed for both CBF and ATT (0.15 = +15%)
    :param max_loss_fraction: optional limit on the fraction of pairs removed (e.g. 0.25); None = no limit
    :param kwargs: passed to sd_increase (cbf_values, att_range, att_step, att_prior, noise)
    :return: (affordable, resulting_plds, resulting_lds). If not affordable, nothing is removed and the
             original PLDs and LDs are returned
    """
    plds = [float(v) for v in plds]
    lds = [float(v) for v in np.broadcast_to(np.asarray(lds, dtype=float), len(plds))]
    remove = [int(i) for i in remove]
    if not remove:
        return True, plds, lds

    increase = sd_increase(plds, lds, remove, **kwargs)
    affordable = all(np.isfinite(v) and v <= threshold for v in increase.values())
    if max_loss_fraction is not None and len(remove) / len(plds) > max_loss_fraction + 1e-9:
        affordable = False
    if not affordable:
        return False, plds, lds
    keep = [i for i in range(len(plds)) if i not in set(remove)]
    return True, [plds[i] for i in keep], [lds[i] for i in keep]


if __name__ == "__main__":
    PLDS = [0.0] * 8 + [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.2]
    LDS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8] + [1.8] * 8
    for remove in ([10], [3, 10, 12], [3, 12]):
        affordable, new_plds, new_lds = check_scrub(PLDS, LDS, remove)
        inc = sd_increase(PLDS, LDS, remove)
        print(f"remove {remove}: affordable={affordable} (CBF {inc['CBF']:+.1%}, ATT {inc['ATT']:+.1%}); "
              f"{len(new_plds)} pairs left")
