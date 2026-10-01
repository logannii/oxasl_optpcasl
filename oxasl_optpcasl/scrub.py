"""
OXASL_OPTPCASL - Sensitivity analysis for motion scrubbing of multi-delay PCASL data

A multi-delay protocol is described as a set of unique (LD, PLD) 'delays', each
acquired as a number of label/control pairs. Removing (scrubbing) pairs reduces
the Fisher information contributed by the delay they belong to. The CRLB
covariance and the cost functions of this package are then used to quantify
how much sensitivity to CBF and ATT is lost.

Because all repeats of a delay carry identical information under the kinetic
model, the effect of scrubbing depends only on the number of pairs retained
per delay, not on which repeat was removed.
"""

import itertools
from math import comb

import numpy as np

from .cost import ATTCost, CBFCost, DOptimalCost
from .kinetic_model import BuxtonPcasl
from .scan import PcaslProtocol
from .structures import ATTDist, Limits, PhysParams, ScanParams

# Cost models used to summarise sensitivity, and the power relating each cost to
# an 'SD-equivalent' (L-optimal costs are variances, D-optimal is the determinant
# of a 2x2 covariance, i.e. ~variance^2)
METRICS = {
    "CBF": (CBFCost(), 2),
    "ATT": (ATTCost(), 2),
    "Joint": (DOptimalCost(), 4),
}

# ATT range the cost is averaged over by default (s). The protocol's whole estimable
# window is not used: its ends are covered by only one or two delays, which would
# make those delays infinitely costly to lose
DEFAULT_ATT_RANGE = (0.5, 2.5)

# Gaussian ATT prior (mean, SD in s) used wherever the retained pairs cannot estimate ATT, so
# that CBF can still be costed there. For a Gaussian prior the precision bound depends only on
# the SD: the prior adds 1/SD^2 to the ATT element of the Fisher information. The mean is kept
# for reference (it would centre the ATT estimate in a fit, but does not change the variance)
DEFAULT_ATT_PRIOR = (1.3, 1.0)


def _hessian_to_cov(hess):
    """
    Covariance of CBF (ml/100g/min) and ATT (s) from the Hessian, with singular (or
    near-singular) matrices given infinite covariance
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        norm_det = np.linalg.det(hess) / (hess[..., 0, 0] * hess[..., 1, 1])
        ok = norm_det > 1e-8
        cov = np.full(hess.shape, np.inf)
        cov[ok] = np.linalg.inv(hess[ok])
        cov[..., 0, 0] *= 6000 * 6000
        cov[..., 0, 1] *= 6000
        cov[..., 1, 0] *= 6000
    return cov, ~ok


class ScrubbedPcaslProtocol(PcaslProtocol):
    """
    Fixed multi-delay PCASL protocol in which individual label/control
    pairs may be discarded

    Parameters are the number of retained label/control pairs for each
    delay [NDelays] (or [NTrials, NDelays] to evaluate many scrubbing
    patterns at once)
    """

    def __init__(self, kinetic_model, scan_params, att_dist, lds, plds, repeats):
        self.lds = np.asarray(lds, dtype=float)
        self.plds = np.asarray(plds, dtype=float)
        self.repeats = np.asarray(repeats, dtype=float)
        if not (self.lds.shape == self.plds.shape == self.repeats.shape):
            raise ValueError("lds, plds and repeats must have the same length")

        pld_lims = Limits(self.plds.min(), self.plds.max(), 0.025)
        PcaslProtocol.__init__(self, kinetic_model, scan_params, att_dist, pld_lims)
        self.nld = 0

        # Sensitivities do not depend on the parameters (pair counts) so
        # calculate them once [NDelays, NSlices, NATTs]
        lds = np.repeat(self.lds[:, np.newaxis], len(self.slicedt), axis=-1)
        times = lds + self.plds[:, np.newaxis] + self.slicedt[np.newaxis, :]
        att = self.att_dist.atts
        df, datt = self.kinetic_model.sensitivity(lds.flatten(), times.flatten(), att)
        self._df = df.reshape(list(times.shape) + [len(att)])
        self._datt = datt.reshape(list(times.shape) + [len(att)])

    def __str__(self):
        return "Multi-delay PCASL protocol with per-delay label/control pair counts"

    def initial_params(self):
        return self.repeats.copy()

    def name_params(self, params):
        return {"pairs": params}

    def timings(self, params):
        shape = np.shape(params)
        return np.broadcast_to(self.lds, shape), np.broadcast_to(self.plds, shape)

    def protocol_summary(self, params):
        ret = []
        for ld, pld, count in zip(self.lds, self.plds, params):
            for _ in range(int(count)):
                ret.append(("", [ld], [1], pld, self.scan_params.readout))
                ret.append(("", [ld], [0], pld, self.scan_params.readout))
        return ret

    def cov(self, params):
        # If every retained delay sees the bolus only after it has fully arrived, the CBF and
        # ATT sensitivities are exactly proportional and the Hessian is singular, but rounding
        # leaves a tiny non-zero determinant. Detect this from the normalized determinant
        # (1 - correlation^2) and treat as inestimable.
        hessian = self.hessian(params)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = PcaslProtocol.cov(self, params)
            norm_det = np.linalg.det(hessian) / (
                hessian[..., 0, 0] * hessian[..., 1, 1]
            )
        cov[~(norm_det > 1e-8)] = np.inf
        return cov

    def hessian(self, params):
        # Each retained pair contributes one differenced measurement at its delay
        weights = np.asarray(params, dtype=float) / (self.scan_params.noise**2)
        df, datt = self._df, self._datt

        hess = np.zeros(weights.shape[:-1] + df.shape[1:] + (2, 2))
        hess[..., 0, 0] = np.tensordot(weights, df * df, axes=1)
        hess[..., 0, 1] = np.tensordot(weights, df * datt, axes=1)
        hess[..., 1, 0] = hess[..., 0, 1]
        hess[..., 1, 1] = np.tensordot(weights, datt * datt, axes=1)
        return hess


def pair_delays(repeats, order="interleaved"):
    """
    Map each label/control pair (in acquisition order) to the delay it belongs to

    :param repeats: Number of pairs acquired for each delay [NDelays]
    :param order: 'interleaved' - all delays are cycled through once per repeat
                  (delays with fewer repeats drop out once exhausted), or
                  'blocked' - all repeats of a delay are acquired consecutively
    :return: Tuple of delay index, repeat index, each [NPairs]
    """
    repeats = np.asarray(repeats, dtype=int)
    delay_idx, repeat_idx = [], []
    if order == "interleaved":
        for rpt in range(repeats.max()):
            for delay in np.where(repeats > rpt)[0]:
                delay_idx.append(delay)
                repeat_idx.append(rpt)
    elif order == "blocked":
        for delay, nrpt in enumerate(repeats):
            delay_idx += [delay] * nrpt
            repeat_idx += list(range(nrpt))
    else:
        raise ValueError("Unrecognized acquisition order: %s" % order)
    return np.array(delay_idx), np.array(repeat_idx)


def group_pairs(plds, lds):
    """
    Group a per-pair protocol description (e.g. as given to oxasl with --plds/--taus)
    into unique (LD, PLD) delays

    :param plds: PLD of each label/control pair in acquisition order [NPairs]
    :param lds: Labelling duration of each pair [NPairs], or a single value for all pairs
    :return: Tuple of unique LDs, unique PLDs, repeats [NDelays] and delay index of each
             pair [NPairs]. Delays are ordered by first appearance in the acquisition
    """
    plds = np.atleast_1d(np.asarray(plds, dtype=float))
    lds = np.broadcast_to(np.asarray(lds, dtype=float), plds.shape)
    combos = np.round(np.stack([lds, plds], axis=-1), 5)
    _, first, inverse = np.unique(
        combos, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    delay_of_pair = rank[inverse]
    unique_lds, unique_plds = combos[np.sort(first)].T
    return unique_lds, unique_plds, np.bincount(delay_of_pair), delay_of_pair


def protocol_att_dist(lds, plds, att_max=None, step=0.02, taper=0.2):
    """
    ATT distribution covering the ATTs a protocol can estimate

    The earliest estimable ATT is just above the shortest PLD: at or below that,
    all measurements see the fully-arrived bolus and the CBF and ATT sensitivities
    are proportional. The latest is just below the second-longest LD+PLD: beyond
    that, at most one distinct timepoint has seen any labelled blood, which cannot
    determine both CBF and ATT.

    :param lds: Labelling durations of the protocol (s)
    :param plds: PLDs of the protocol (s)
    :param att_max: Latest arrival time of interest (s). Default is the latest estimable ATT
    :return: ATTDist from one step above the shortest PLD to att_max
    """
    lds, plds = np.broadcast_arrays(
        np.asarray(lds, dtype=float), np.asarray(plds, dtype=float)
    )
    times = np.unique(np.round(lds + plds, 5))
    if len(times) < 2:
        raise ValueError(
            "Protocol needs at least two distinct LD+PLD timepoints to estimate ATT"
        )
    if att_max is None:
        att_max = times[-2] - step
    return ATTDist(np.min(plds) + step, att_max, step, taper)


class ScrubbingAnalysis(object):
    """
    Evaluate the loss of CBF/ATT sensitivity caused by discarding label/control pairs

    Costs are averaged over the ATT distribution (as for protocol optimization) and
    over a set of CBF values so that conclusions hold for a wide range of tissue
    """

    def __init__(
        self,
        lds,
        plds,
        repeats,
        delay_of_pair=None,
        cbf_values=(50.0,),
        att_dist=None,
        nslices=1,
        slicedt=0.0,
        noise=0.0013,
        att_prior=DEFAULT_ATT_PRIOR,
        **phys_kwargs,
    ):
        """
        :param lds: Labelling duration for each delay (s) [NDelays]
        :param plds: PLD for each delay (s) [NDelays]
        :param repeats: Number of label/control pairs acquired for each delay [NDelays]
        :param delay_of_pair: Delay index of each pair in acquisition order [NPairs].
                              Defaults to an interleaved acquisition order
        :param cbf_values: CBF values to evaluate (ml/100g/min)
        :param att_dist: ATT distribution the cost is averaged over. Default is uniform over
                         DEFAULT_ATT_RANGE. ATTs the full protocol cannot estimate (at or
                         below the shortest PLD) are given zero weight
        :param att_prior: (mean, SD) of a Gaussian ATT prior (s), used at the ATTs where a
                          scrubbing pattern leaves ATT inestimable (for that pattern and, for a
                          like-for-like comparison, for the full data). None gives the pure CRLB,
                          in which such patterns have infinite cost
        :param phys_kwargs: Other physiological parameters (see PhysParams)
        """
        self.lds = np.asarray(lds, dtype=float)
        self.plds = np.asarray(plds, dtype=float)
        self.repeats = np.asarray(repeats, dtype=int)
        self.ndelays = len(self.repeats)
        if delay_of_pair is None:
            delay_of_pair = pair_delays(self.repeats)[0]
        self.delay_of_pair = np.asarray(delay_of_pair, dtype=int)
        if not np.array_equal(
            np.bincount(self.delay_of_pair, minlength=self.ndelays), self.repeats
        ):
            raise ValueError("delay_of_pair is inconsistent with repeats")
        self.npairs = len(self.delay_of_pair)

        self.att_prior = att_prior
        self.cbf_values = np.atleast_1d(np.asarray(cbf_values, dtype=float))
        self.att_dist = (
            att_dist if att_dist is not None else ATTDist(*DEFAULT_ATT_RANGE, 0.02, 0)
        )
        self.scan_params = ScanParams(
            duration=0,
            npld=self.ndelays,
            nslices=nslices,
            slicedt=slicedt,
            readout=0.0,
            noise=noise,
        )
        self.protocols = [
            ScrubbedPcaslProtocol(
                BuxtonPcasl(PhysParams(f=cbf / 6000.0, **phys_kwargs)),
                self.scan_params,
                self.att_dist,
                self.lds,
                self.plds,
                self.repeats,
            )
            for cbf in self.cbf_values
        ]

    def counts(self, removed_pairs):
        """
        :param removed_pairs: Indices (acquisition order) of pairs to discard
        :return: Number of retained pairs per delay [NDelays]
        """
        removed_pairs = np.asarray(removed_pairs, dtype=int)
        if len(np.unique(removed_pairs)) != len(removed_pairs):
            raise ValueError("Removed pairs contain duplicates")
        removed = np.bincount(self.delay_of_pair[removed_pairs], minlength=self.ndelays)
        return self.repeats - removed

    def _outside_estimable(self, counts):
        """
        ATTs the retained pairs cannot estimate: at or below the shortest retained PLD, or at or
        beyond the second-longest distinct LD+PLD (same rule as the ATT weighting of the protocol)

        :return: [NTrials, 1, NATTs] or [1, NATTs] boolean
        """
        counts2 = np.atleast_2d(counts)
        atts = self.att_dist.atts
        outside = np.ones((len(counts2), len(atts)), dtype=bool)
        for i, row in enumerate(counts2):
            kept = row > 0
            times = np.unique(np.round(self.lds[kept] + self.plds[kept], 5))
            if len(times) >= 2:
                outside[i] = (atts <= self.plds[kept].min() + 1e-9) | (
                    atts >= times[-2] - 1e-9
                )
        if np.ndim(counts) == 1:
            outside = outside[0]
        return outside[..., np.newaxis, :]

    def _prior_mask(self, counts):
        """
        Where the ATT prior is used: ATTs the retained pairs cannot estimate

        :return: [NTrials, 1, NATTs] or [1, NATTs] boolean
        """
        if self.att_prior is None:
            return np.zeros(
                np.shape(counts)[:-1] + (1, len(self.att_dist.atts)), dtype=bool
            )
        return self._outside_estimable(counts)

    def _covariances(self, protocol, counts):
        """
        Covariance with the scrubbed and with the full data for one CBF value, using the ATT prior
        wherever the scrubbed data cannot estimate ATT (for both, so they are comparable)

        :return: Tuple of scrubbed cov, full cov [..., NSlices, NATTs, 2, 2] and the
                 prior mask [..., NSlices, NATTs]
        """
        counts = np.asarray(counts, dtype=float)
        h_scrubbed = protocol.hessian(counts)
        h_full = np.broadcast_to(
            protocol.hessian(self.repeats.astype(float)), h_scrubbed.shape
        ).copy()
        prior = np.broadcast_to(self._prior_mask(counts), h_scrubbed.shape[:-2]).copy()
        if self.att_prior is not None:
            prior |= _hessian_to_cov(h_scrubbed)[1]  # any other singular points
            precision = np.where(prior, 1.0 / self.att_prior[1] ** 2, 0.0)
            h_scrubbed[..., 1, 1] += precision
            h_full[..., 1, 1] += precision
        return _hessian_to_cov(h_scrubbed)[0], _hessian_to_cov(h_full)[0], prior

    def _point_costs(self, counts, metric):
        """Per-CBF list of (scrubbed cost, full cost, prior mask), each [..., NSlices, NATTs]"""
        cost_model = METRICS[metric][0]
        out = []
        for protocol in self.protocols:
            scrubbed, full, prior = self._covariances(protocol, counts)
            with np.errstate(invalid="ignore", over="ignore"):
                out.append((cost_model.cost(scrubbed), cost_model.cost(full), prior))
        return out

    @staticmethod
    def _weighted_mean(cost, weights):
        # As PcaslProtocol.cost: weight by the ATT distribution, 0 where the weight is 0,
        # inf where an included point is inestimable
        with np.errstate(invalid="ignore", over="ignore"):
            cost = np.where(weights > 0, cost * weights, 0.0)
        cost = np.where(np.isnan(cost), np.inf, cost)
        return np.mean(cost, axis=(-1, -2))

    def costs(self, counts, metric):
        """
        :param counts: Retained pairs per delay [NDelays] or [NTrials, NDelays]
        :param metric: 'CBF', 'ATT' or 'Joint'
        :return: Tuple of scrubbed and full-data cost, averaged over the ATT distribution and
                 CBF values [NTrials] or scalars
        """
        scrubbed, full = [], []
        for (c_scrubbed, c_full, _), protocol in zip(
            self._point_costs(counts, metric), self.protocols
        ):
            scrubbed.append(self._weighted_mean(c_scrubbed, protocol.att_weights))
            full.append(self._weighted_mean(c_full, protocol.att_weights))
        return np.mean(scrubbed, axis=0), np.mean(full, axis=0)

    def cost(self, counts, metric):
        """
        :return: Cost of the scrubbed data averaged over ATT distribution and CBF values
                 [NTrials] or scalar
        """
        return self.costs(counts, metric)[0]

    def sd_increase(self, counts, metric):
        """
        Fractional increase in SD-equivalent uncertainty relative to the full data

        e.g. 0.1 means estimates are 10% noisier than with no scrubbing

        :return: [NTrials] or scalar
        """
        power = METRICS[metric][1]
        scrubbed, full = self.costs(counts, metric)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (scrubbed / full) ** (1.0 / power) - 1

    def worst_sd_increase(self, counts, metric):
        """
        Largest SD increase at any single ATT and CBF value (rather than the
        cost averaged over the ATT distribution)

        :return: [NTrials] or scalar
        """
        power = METRICS[metric][1]
        worst = None
        for (c_scrubbed, c_full, _), protocol in zip(
            self._point_costs(counts, metric), self.protocols
        ):
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                ratio = c_scrubbed / c_full
            ratio = np.where(
                protocol.att_weights > 0, np.nan_to_num(ratio, nan=np.inf), -np.inf
            )
            ratio = ratio.max(axis=(-1, -2))
            worst = ratio if worst is None else np.maximum(worst, ratio)
        return worst ** (1.0 / power) - 1

    def sd_increase_by_att(self, counts, metric):
        """
        SD increase at each ATT: costs averaged over slices and CBF values, compared with the full
        data at the same ATT (using the ATT prior for both wherever the scrubbed data needs it)

        :return: [NTrials, NATTs] or [NATTs] (NaN at ATTs the protocol gives no weight)
        """
        power = METRICS[metric][1]
        point_costs = self._point_costs(counts, metric)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            scrubbed = np.mean([c.mean(axis=-2) for c, _, _ in point_costs], axis=0)
            full = np.mean([f.mean(axis=-2) for _, f, _ in point_costs], axis=0)
            increase = (scrubbed / full) ** (1.0 / power) - 1
        relevant = np.any(
            [p.att_weights.max(axis=0) > 0 for p in self.protocols], axis=0
        )
        return np.where(relevant, increase, np.nan)

    def prior_atts(self, counts):
        """
        ATTs (within the cost's ATT distribution) at which the ATT prior is used

        :return: [NTrials, NATTs] or [NATTs] boolean
        """
        relevant = np.any(
            [p.att_weights > 0 for p in self.protocols], axis=0
        )  # [NSlices, NATTs]
        used = np.zeros(np.shape(counts)[:-1] + relevant.shape, dtype=bool)
        for protocol in self.protocols:
            used |= self._covariances(protocol, counts)[2] & relevant
        return used.any(axis=-2)

    def prior_range(self, counts):
        """
        :return: Tuple of first, last ATT at which the ATT prior is used [NTrials] or scalars
                 (NaN if it is not used)
        """
        used = np.atleast_2d(self.prior_atts(counts))
        atts = self.att_dist.atts
        first = np.where(used.any(axis=1), atts[np.argmax(used, axis=1)], np.nan)
        last = np.where(
            used.any(axis=1),
            atts[len(atts) - 1 - np.argmax(used[:, ::-1], axis=1)],
            np.nan,
        )
        if np.ndim(counts) == 1:
            return first[0], last[0]
        return first, last

    def estimable_range(self, counts):
        """
        Range of ATTs that can be estimated from the retained pairs (see protocol_att_dist)

        :param counts: Retained pairs per delay [NDelays] or [NTrials, NDelays]
        :return: Tuple of earliest, latest estimable ATT [NTrials] or scalars (NaN if none)
        """
        counts = np.atleast_2d(counts)
        step = self.att_dist.step
        earliest, latest = np.full(len(counts), np.nan), np.full(len(counts), np.nan)
        for i, row in enumerate(counts):
            kept = row > 0
            times = np.unique(np.round(self.lds[kept] + self.plds[kept], 5))
            if len(times) >= 2:
                earliest[i], latest[i] = self.plds[kept].min() + step, times[-2] - step
        return earliest, latest

    def sd_increase_in_range(self, counts, metric, att_range=None):
        """
        SD increase averaged over the ATT range, excluding any ATTs that remain inestimable

        With the ATT prior (default), ATT-inestimable points are costed using the prior, so the
        range is only narrowed where no retained pair has seen any labelled blood at all (CBF is
        then inestimable too). Without the prior, every ATT-inestimable point is excluded. The full
        data is costed over the same points, so the result compares like with like.

        :param counts: Retained pairs per delay [NDelays] or [NTrials, NDelays]
        :param att_range: (min, max) ATT (s). Default is the range of the ATT distribution
        :return: Tuple of SD increase, lower and upper ATT actually used, each [NTrials] or scalar
                 (NaN if no ATT in the range can be estimated)
        """
        power = METRICS[metric][1]
        single = np.ndim(counts) == 1
        counts = np.atleast_2d(np.asarray(counts, dtype=float))
        atts = self.att_dist.atts
        lo, hi = att_range if att_range is not None else (atts[0], atts[-1])
        point_costs = self._point_costs(counts, metric)
        relevant = np.any(
            [p.att_weights.max(axis=0) > 0 for p in self.protocols], axis=0
        )
        # Use the same ATTs for every CBF value and slice
        use = np.all([np.isfinite(c).all(axis=-2) for c, _, _ in point_costs], axis=0)
        use &= relevant & (atts >= lo - 1e-9) & (atts <= hi + 1e-9)
        if self.att_prior is None:
            use &= ~self._outside_estimable(counts)[:, 0, :]
        num, den = 0.0, 0.0
        with np.errstate(invalid="ignore", over="ignore"):
            for (c_scrubbed, c_full, _), protocol in zip(point_costs, self.protocols):
                weights = protocol.att_weights * use[:, np.newaxis, :]
                num = num + np.where(weights > 0, c_scrubbed * weights, 0.0).sum(
                    axis=(-1, -2)
                )
                den = den + np.where(weights > 0, c_full * weights, 0.0).sum(
                    axis=(-1, -2)
                )
            any_used = use.any(axis=1)
            increase = np.where(any_used, (num / den) ** (1.0 / power) - 1, np.nan)
        lo_used = np.where(any_used, atts[np.argmax(use, axis=1)], np.nan)
        hi_used = np.where(
            any_used, atts[len(atts) - 1 - np.argmax(use[:, ::-1], axis=1)], np.nan
        )
        if single:
            return increase[0], lo_used[0], hi_used[0]
        return increase, lo_used, hi_used

    def combination_counts(self, n_remove, max_combinations=200000, rng=None):
        """
        Retained pairs per delay for every way of removing n_remove pairs

        :param max_combinations: If there are more combinations than this, a random
                                 sample of this size is returned instead
        :return: [NCombinations, NDelays]
        """
        if comb(self.npairs, n_remove) <= max_combinations:
            removed = np.array(
                list(itertools.combinations(range(self.npairs), n_remove)), dtype=int
            )
            removed = removed.reshape(-1, n_remove)
        else:
            rng = np.random.default_rng(rng)
            removed = np.array(
                [
                    rng.choice(self.npairs, n_remove, replace=False)
                    for _ in range(max_combinations)
                ]
            )
        delays = self.delay_of_pair[removed]
        lost = np.zeros((len(removed), self.ndelays), dtype=int)
        np.add.at(
            lost, (np.repeat(np.arange(len(removed)), n_remove), delays.ravel()), 1
        )
        return self.repeats[np.newaxis, :] - lost

    def sd_by_att(self, counts):
        """
        CRLB standard deviations of CBF and ATT as a function of ATT for each CBF value

        ATTs outside the range the protocol is sensitive to are returned as NaN. Where the
        retained pairs cannot estimate ATT, the ATT prior is used (see prior_atts)

        :param counts: Retained pairs per delay [NDelays]
        :return: Tuple of CBF SD (ml/100g/min), ATT SD (s), each [NCBF, NATTs]
                 (variance is averaged over slices)
        """
        counts = np.asarray(counts, dtype=float)
        cbf_sd, att_sd = [], []
        for protocol in self.protocols:
            cov = self._covariances(protocol, counts)[0]
            with np.errstate(invalid="ignore"):
                cbf_var = np.abs(cov[..., 0, 0]).mean(axis=0)
                att_var = np.abs(cov[..., 1, 1]).mean(axis=0)
            relevant = protocol.att_weights.max(axis=0) > 0
            cbf_sd.append(np.where(relevant, np.sqrt(cbf_var), np.nan))
            att_sd.append(np.where(relevant, np.sqrt(att_var), np.nan))
        return np.array(cbf_sd), np.array(att_sd)

    def single_pair_impact(self, metric):
        """
        SD increase caused by removing a single pair from each delay

        :return: [NDelays]
        """
        trials = np.tile(self.repeats, (self.ndelays, 1)) - np.eye(
            self.ndelays, dtype=int
        )
        return self.sd_increase(trials, metric)

    def progressive_counts(self, removal_order):
        """
        Retained pairs per delay as pairs are removed one at a time

        :param removal_order: Pair indices in the order they are removed [K]
        :return: [K+1, NDelays] - row k has the first k pairs removed
        """
        counts = [self.repeats.copy()]
        for pair in removal_order:
            nxt = counts[-1].copy()
            nxt[self.delay_of_pair[pair]] -= 1
            counts.append(nxt)
        return np.array(counts)

    def greedy_counts(self, n_remove, metric, worst=True):
        """
        Remove pairs one at a time choosing the pair that does the most (worst=True)
        or least (worst=False) damage to the given metric at each step

        :return: [n_remove+1, NDelays]
        """
        counts = [self.repeats.copy()]
        for _ in range(n_remove):
            current = counts[-1]
            candidates = np.where(current > 0)[0]
            trials = np.tile(current, (len(candidates), 1))
            trials[np.arange(len(candidates)), candidates] -= 1
            cost = self.sd_increase(trials, metric)
            # Treat inestimable parameters (inf/nan cost) as maximal damage
            cost = np.where(np.isfinite(cost), cost, np.inf)
            best = np.argmax(cost) if worst else np.argmin(cost)
            counts.append(trials[best])
        return np.array(counts)

    def random_counts(self, n_remove, n_samples=500, rng=None):
        """
        Retained pairs per delay for random subsets of n_remove pairs

        :return: [n_samples, NDelays]
        """
        rng = np.random.default_rng(rng)
        removed = np.array(
            [rng.choice(self.npairs, n_remove, replace=False) for _ in range(n_samples)]
        ).reshape(n_samples, n_remove)
        delays = self.delay_of_pair[removed]
        removed_counts = np.array(
            [np.bincount(row, minlength=self.ndelays) for row in delays]
        )
        return self.repeats[np.newaxis, :] - removed_counts

    def check(self, counts, max_sd_increase, max_loss_fraction=0.25, criterion="mean"):
        """
        Check a scrubbing pattern against sensitivity and data-loss thresholds

        :param counts: Retained pairs per delay [NDelays]
        :param max_sd_increase: Mapping from metric name to maximum permitted SD increase
        :param max_loss_fraction: Hard limit on the fraction of pairs removed
        :param criterion: 'mean' - SD increase from the ATT/CBF-averaged cost, or
                          'worst' - largest SD increase at any single ATT/CBF
        :return: Tuple of (passed, dict of metric -> SD increase, fraction removed)
        """
        counts = np.asarray(counts)
        loss = 1 - counts.sum() / self.repeats.sum()
        sd_increase = (
            self.worst_sd_increase if criterion == "worst" else self.sd_increase
        )
        increases = {name: float(sd_increase(counts, name)) for name in max_sd_increase}
        passed = loss <= max_loss_fraction + 1e-9 and all(
            np.isfinite(increases[name]) and increases[name] <= limit
            for name, limit in max_sd_increase.items()
        )
        return passed, increases, loss

    def select_pairs(
        self,
        flagged_pairs,
        max_sd_increase,
        max_loss_fraction=0.25,
        motion_score=None,
        criterion="mean",
    ):
        """
        Decide which motion-flagged pairs can be scrubbed without breaching thresholds

        Flagged pairs are considered from worst motion to least. Each is removed unless
        doing so would breach a threshold, in which case it is retained and the next
        pair is considered (a less critical delay may still be able to lose a pair).

        :param flagged_pairs: Pair indices flagged for motion
        :param max_sd_increase: Mapping from metric name to maximum permitted SD increase
        :param max_loss_fraction: Hard limit on the fraction of pairs removed
        :param motion_score: Motion severity for each flagged pair (higher is worse).
                             If not given, flagged pairs are considered in the order given
        :param criterion: 'mean' or 'worst', see ``check``
        :return: Tuple of (removed pair indices, retained flagged pair indices)
        """
        flagged_pairs = np.asarray(flagged_pairs, dtype=int)
        if motion_score is not None:
            flagged_pairs = flagged_pairs[
                np.argsort(-np.asarray(motion_score), kind="stable")
            ]

        removed, retained = [], []
        for pair in flagged_pairs:
            passed, _, _ = self.check(
                self.counts(removed + [pair]),
                max_sd_increase,
                max_loss_fraction,
                criterion,
            )
            if passed:
                removed.append(int(pair))
            else:
                retained.append(int(pair))
        return removed, retained
