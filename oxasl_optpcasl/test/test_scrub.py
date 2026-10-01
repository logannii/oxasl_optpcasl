"""
Test cases for motion scrubbing sensitivity analysis
"""

import numpy as np
import pytest

import oxasl_optpcasl as opt
from oxasl_optpcasl.scrub import (
    ScrubbedPcaslProtocol,
    ScrubbingAnalysis,
    group_pairs,
    pair_delays,
    protocol_att_dist,
)

LDS = np.array([0.5, 1.0, 1.5, 1.8, 1.8])
PLDS = np.array([0.3, 0.8, 1.3, 1.8, 2.3])


def test_matches_multi_ld_protocol():
    """With equal repeats per delay the Hessian matches the existing multi-LD protocol"""
    kinetic_model = opt.BuxtonPcasl(opt.ASLParams(f=50.0 / 6000))
    att_dist = opt.ATTDist(0.5, 3.0, 0.05, 0.2)
    params = np.concatenate([PLDS, LDS])
    scan = opt.ScanParams(duration=300, npld=5, readout=0.5, ld=list(LDS), nslices=3, slicedt=0.05)
    ref = opt.MultiPLDPcaslMultiLD(
        kinetic_model, scan, att_dist, opt.Limits(0.3, 3.0, 0.025), opt.Limits(0.1, 1.8, 0.025)
    )
    nrpt, _ = ref.repeats_total_tr(params)

    scrubbed = ScrubbedPcaslProtocol(kinetic_model, scan, att_dist, LDS, PLDS, [nrpt] * 5)
    assert np.allclose(scrubbed.hessian(scrubbed.initial_params()), ref.hessian(params))
    cost_model = opt.DOptimalCost()
    assert np.isclose(
        scrubbed.cost(scrubbed.initial_params(), cost_model), ref.cost(params, cost_model)
    )


def test_batched_cost():
    """Batched evaluation of scrubbing patterns matches individual evaluation"""
    analysis = ScrubbingAnalysis(LDS, PLDS, [2, 3, 4, 4, 5], cbf_values=[20, 60])
    trials = np.array([[2, 3, 4, 4, 5], [1, 3, 4, 4, 5], [2, 3, 4, 2, 5]])
    batched = analysis.cost(trials, "ATT")
    assert np.allclose(batched, [analysis.cost(t, "ATT") for t in trials])


def test_uniform_loss():
    """Halving every delay's repeats increases SD by sqrt(2) for every metric"""
    analysis = ScrubbingAnalysis(LDS, PLDS, [4] * 5, cbf_values=[20, 50, 80])
    for metric in ("CBF", "ATT", "Joint"):
        assert np.isclose(analysis.sd_increase([2] * 5, metric), np.sqrt(2) - 1)


def test_pair_delays():
    delays, repeats = pair_delays([2, 1, 3])
    assert list(delays) == [0, 1, 2, 0, 2, 2]
    assert list(repeats) == [0, 0, 0, 1, 1, 2]
    delays, _ = pair_delays([2, 1, 3], order="blocked")
    assert list(delays) == [0, 0, 1, 2, 2, 2]


def test_counts_and_limits():
    analysis = ScrubbingAnalysis(LDS, PLDS, [2, 2, 2, 2, 2])
    assert list(analysis.counts([0, 5])) == [0, 2, 2, 2, 2]
    with pytest.raises(ValueError):
        analysis.counts([0, 0])

    # Removing 3 of 10 pairs breaches the 25% hard limit regardless of sensitivity
    passed, _, loss = analysis.check(analysis.counts([1, 2, 3]), {"CBF": 10.0})
    assert not passed and np.isclose(loss, 0.3)


def test_select_pairs():
    """Selection never breaches thresholds and prefers high-motion pairs"""
    analysis = ScrubbingAnalysis(LDS, PLDS, [4] * 5, cbf_values=[20, 80])
    flagged = np.arange(10)
    motion = np.linspace(0.1, 1.0, 10)
    limits = {"CBF": 0.1, "ATT": 0.1}
    removed, retained = analysis.select_pairs(flagged, limits, 0.25, motion)
    assert sorted(removed + retained) == list(flagged)
    passed, _, _ = analysis.check(analysis.counts(removed), limits, 0.25)
    assert passed
    assert len(removed) <= 5
    assert removed[0] == 9


def test_worst_criterion():
    """Worst-ATT SD increase is at least the averaged one, and (pure CRLB) losing the shortest PLD is inestimable"""
    analysis = ScrubbingAnalysis(LDS, PLDS, [2, 3, 4, 4, 5], cbf_values=[20, 80], att_prior=None)
    counts = np.array([2, 1, 4, 3, 5])
    for metric in ("CBF", "ATT", "Joint"):
        assert analysis.worst_sd_increase(counts, metric) >= analysis.sd_increase(counts, metric) - 1e-9
    no_short = np.array([0, 3, 4, 4, 5])
    assert np.isinf(analysis.sd_increase(no_short, "ATT"))
    passed, _, _ = analysis.check(no_short, {"ATT": 10.0})
    assert not passed

    batched = analysis.worst_sd_increase(np.array([counts, [2, 3, 4, 4, 5]]), "ATT")
    assert np.isclose(batched[0], analysis.worst_sd_increase(counts, "ATT"))
    assert np.isclose(batched[1], 0)


def test_group_pairs():
    """Per-pair lists are grouped into unique delays in order of first appearance"""
    lds, plds, repeats, delay_of_pair = group_pairs([0.0, 0.5, 0.0, 0.5, 1.0], [1.8, 1.8, 1.8, 1.8, 1.8])
    assert np.allclose(plds, [0.0, 0.5, 1.0]) and np.allclose(lds, 1.8)
    assert list(repeats) == [2, 2, 1]
    assert list(delay_of_pair) == [0, 1, 0, 1, 2]

    # Same PLD with different LDs are different delays; scalar LD broadcasts
    _, _, repeats, _ = group_pairs([0.0, 0.0, 0.2], [0.8, 0.9, 1.8])
    assert list(repeats) == [1, 1, 1]
    _, _, repeats, _ = group_pairs([1.025, 1.525, 2.025, 2.525, 3.025], 1.8)
    assert list(repeats) == [1] * 5


def test_protocol_att_dist():
    """ATT range runs from just above the shortest PLD to just below the second-longest LD+PLD"""
    att_dist = protocol_att_dist(1.8, [1.025, 1.525, 2.025, 2.525, 3.025])
    assert np.isclose(att_dist.atts[0], 1.045)
    assert np.isclose(att_dist.atts[-1], 1.8 + 2.525 - 0.02)

    # Repeats of the longest delay are the same timepoint, so do not extend the range
    att_dist = protocol_att_dist([1.8, 1.8, 1.8], [1.0, 2.0, 2.0])
    assert np.isclose(att_dist.atts[-1], 2.8 - 0.02)
    assert np.isclose(protocol_att_dist(1.8, [1.0, 2.0], att_max=3.0).atts[-1], 3.0)
    with pytest.raises(ValueError):
        protocol_att_dist(1.8, [2.0, 2.0])

    # Full protocol is estimable (finite cost) over the whole default range
    analysis = ScrubbingAnalysis(LDS, PLDS, [1] * 5)
    for metric in ("CBF", "ATT", "Joint"):
        assert np.isfinite(analysis.worst_sd_increase(analysis.repeats, metric))


def test_default_att_range():
    """Default cost is averaged over ATT 0.5-2.5 s, so losing a late delay is finite"""
    analysis = ScrubbingAnalysis(LDS, PLDS, [1] * 5)
    assert np.isclose(analysis.att_dist.atts[0], 0.5) and np.isclose(analysis.att_dist.atts[-1], 2.5)
    counts = np.array([1, 1, 1, 1, 0])
    for metric in ("CBF", "ATT"):
        assert np.isfinite(analysis.sd_increase(counts, metric))


def test_sd_increase_in_range():
    """Pure CRLB: adaptive range matches the fixed-range cost when nothing becomes inestimable,
    and narrows instead of giving an infinite cost when it does"""
    analysis = ScrubbingAnalysis(LDS, PLDS, [2, 3, 4, 4, 5], cbf_values=[20, 80], att_prior=None)
    counts = np.array([[2, 1, 4, 3, 5], [2, 3, 4, 4, 5]])
    for metric in ("CBF", "ATT", "Joint"):
        inc, lo, hi = analysis.sd_increase_in_range(counts, metric)
        assert np.allclose(inc, analysis.sd_increase(counts, metric))
        assert np.allclose(lo, 0.5) and np.allclose(hi, 2.5)

    # Losing the shortest PLD (0.3s) makes ATT < 0.8s inestimable: fixed range -> inf,
    # adaptive range starts just above the next PLD instead. That delay (sampled at 0.8s)
    # carries no information about the ATTs that remain, so its loss costs nothing there
    no_short = np.array([0, 3, 4, 4, 5])
    assert np.isinf(analysis.sd_increase(no_short, "ATT"))
    inc, lo, hi = analysis.sd_increase_in_range(no_short, "ATT")
    assert np.isclose(inc, 0)
    assert np.isclose(lo, 0.8 + 0.02) and np.isclose(hi, 2.5)

    # Losing pairs as well narrows the range and costs precision within it
    inc, lo, hi = analysis.sd_increase_in_range(np.array([0, 1, 4, 4, 5]), "ATT")
    assert np.isfinite(inc) and inc > 0 and np.isclose(lo, 0.82)


def test_combination_counts():
    analysis = ScrubbingAnalysis(LDS, PLDS, [2, 1, 1, 1, 1])
    combos = analysis.combination_counts(2)
    assert combos.shape == (15, 5)            # C(6, 2)
    assert np.all(combos.sum(axis=1) == 4)
    assert len(analysis.combination_counts(3, max_combinations=5, rng=0)) == 5


def test_att_prior():
    """Where ATT becomes inestimable, the ATT prior keeps CBF (and ATT) finite"""
    counts = np.array([0, 3, 4, 4, 5])        # shortest PLD lost: ATT 0.5-0.8 s inestimable
    analysis = ScrubbingAnalysis(LDS, PLDS, [2, 3, 4, 4, 5], cbf_values=[20, 80], att_prior=(1.3, 1.0))
    for metric in ("CBF", "ATT", "Joint"):
        assert np.isfinite(analysis.sd_increase(counts, metric)) and analysis.sd_increase(counts, metric) > 0
    first, last = analysis.prior_range(counts)
    assert np.isclose(first, 0.5) and last < 0.82
    assert not analysis.prior_atts(analysis.repeats).any()

    # Nothing inestimable: prior not used, identical to the pure CRLB
    strict = ScrubbingAnalysis(LDS, PLDS, [2, 3, 4, 4, 5], cbf_values=[20, 80], att_prior=None)
    ok = np.array([2, 1, 4, 3, 5])
    assert np.isclose(analysis.sd_increase(ok, "CBF"), strict.sd_increase(ok, "CBF"))

    # The prior adds 1/SD^2 to the ATT Fisher information; its mean does not matter
    protocol = analysis.protocols[0]
    hess = protocol.hessian(counts.astype(float))[0, 5].copy()     # ATT = 0.6 s
    hess[1, 1] += 1.0
    expected = np.sqrt(np.linalg.inv(hess)[0, 0]) * 6000
    assert np.isclose(analysis.sd_by_att(counts)[0][0, 5], expected)
    other_mean = ScrubbingAnalysis(LDS, PLDS, [2, 3, 4, 4, 5], cbf_values=[20, 80], att_prior=(0.5, 1.0))
    assert np.isclose(other_mean.sd_increase(counts, "CBF"), analysis.sd_increase(counts, "CBF"))


def test_sd_increase_by_att():
    """Per-ATT SD increase is zero for the full data, and batched evaluation matches single"""
    analysis = ScrubbingAnalysis(LDS, PLDS, [2, 3, 4, 4, 5], cbf_values=[20, 80])
    assert np.allclose(np.nan_to_num(analysis.sd_increase_by_att(analysis.repeats, "CBF")), 0)
    counts = np.array([[2, 1, 4, 3, 5], [0, 3, 4, 4, 5]])
    batched = analysis.sd_increase_by_att(counts, "ATT")
    assert batched.shape == (2, len(analysis.att_dist.atts))
    assert np.allclose(batched[1], analysis.sd_increase_by_att(counts[1], "ATT"), equal_nan=True)
    assert np.all(np.isfinite(batched[1]))          # the ATT prior keeps inestimable ATTs finite
