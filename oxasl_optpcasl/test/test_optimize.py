"""
Test cases for PLD optimization

Ground truth is generally taken from the MATLAB code
"""

import numpy as np

import oxasl_optpcasl as opt


def test_scan_params_fixed_tr():
    """Test explicit fixed-TR behaviour

    TR is per-image repetition time. With nPLD=5 (2 images per PLD = 10 images total per block):
    total_tr = 2 * nPLD * tr_per_image = 2 * 5 * 0.8 = 8.0s per block
    repeats = floor(110 / 8.0) = 13
    """
    scan = opt.ScanParams(duration=110, npld=5, readout=7.3, ld=1.8, tr=0.8)
    protocol = opt.FixedLDPcaslProtocol(
        opt.BuxtonPcasl(opt.ASLParams(f=50.0 / 6000)),
        scan,
        opt.ATTDist(0.5, 3.0, 0.05, 0.2),
        opt.Limits(0.1, 3.0, 0.1, name="PLD"),
        opt.Limits(0.1, 1.8, 0.1, name="LD"),
    )
    repeats, total_tr = protocol.repeats_total_tr(np.array([0.4, 0.8, 1.2, 1.6, 2.0]))
    assert repeats == 13
    assert total_tr == 8.0
    assert total_tr == 8.0


def test_doptimal():
    """Test D-optimal optimization method"""
    params = opt.ASLParams(f=50.0 / 6000)
    att_dist = opt.ATTDist(0.2, 2.1, 0.001, 0.3)
    scan = opt.ASLScan("var_multi_pCASL", duration=300, npld=6, readout=0.5)
    lims = opt.Limits(0.1, 3.0, 0.025)
    optimizer = opt.DOptimal(params, scan, att_dist, lims)

    output = optimizer.optimize()
    assert np.allclose(output.plds, [0.2, 0.7, 0.725, 1.55, 1.875, 2.075])


def test_loptimal_cbf():
    """Test L-optimal optimization method for CBF"""
    params = opt.ASLParams(f=50.0 / 6000)
    att_dist = opt.ATTDist(0.2, 2.1, 0.001, 0.3)
    scan = opt.ASLScan("var_multi_pCASL", duration=300, npld=6, readout=0.5)
    lims = opt.Limits(0.1, 3.0, 0.025)
    optimizer = opt.LOptimal([[1, 0], [0, 0]], params, scan, att_dist, lims)

    output = optimizer.optimize()
    assert np.allclose(output.plds, [0.2, 1.175, 1.8, 2.025, 2.1, 2.1])


def test_loptimal_att():
    """Test L-optimal optimization method for ATT"""
    params = opt.ASLParams(f=50.0 / 6000)
    att_dist = opt.ATTDist(0.2, 2.1, 0.001, 0.3)
    scan = opt.ASLScan("var_multi_pCASL", duration=300, npld=6, readout=0.5)
    lims = opt.Limits(0.1, 3.0, 0.025)
    optimizer = opt.LOptimal([[0, 0], [0, 1]], params, scan, att_dist, lims)

    output = optimizer.optimize()
    assert np.allclose(output.plds, [0.1, 0.475, 0.7, 1.025, 1.725, 2.1])


def test_doptimal_multislice():
    """Test D-optimal optimization method with 2D readout"""
    params = opt.ASLParams(f=50.0 / 6000)
    att_dist = opt.ATTDist(0.2, 2.1, 0.001, 0.3)
    scan = opt.ASLScan(
        "var_multi_pCASL", duration=300, npld=6, readout=0.5, nslices=10, slicedt=0.0452
    )
    lims = opt.Limits(0.1, 3.0, 0.025)
    optimizer = opt.DOptimal(params, scan, att_dist, lims)

    output = optimizer.optimize()
    assert np.allclose(output.plds, [0.1, 0.575, 0.725, 1.4, 1.75, 2.025])


def test_loptimal_cbf_multislice():
    """Test L-optimal optimization method for CBF with 2D readout"""
    params = opt.ASLParams(f=50.0 / 6000)
    att_dist = opt.ATTDist(0.2, 2.1, 0.001, 0.3)
    scan = opt.ASLScan(
        "var_multi_pCASL", duration=300, npld=6, readout=0.5, nslices=10, slicedt=0.0452
    )
    lims = opt.Limits(0.1, 3.0, 0.025)
    optimizer = opt.LOptimal([[1, 0], [0, 0]], params, scan, att_dist, lims)

    output = optimizer.optimize()
    assert np.allclose(output.plds, [0.1, 1.025, 1.625, 1.8, 1.95, 2.1])


def test_loptimal_att_multislice():
    """Test L-optimal optimization method for ATT with 2D readout"""
    params = opt.ASLParams(f=50.0 / 6000)
    att_dist = opt.ATTDist(0.2, 2.1, 0.001, 0.3)
    scan = opt.ASLScan(
        "var_multi_pCASL", duration=300, npld=6, readout=0.5, nslices=10, slicedt=0.0452
    )
    lims = opt.Limits(0.1, 3.0, 0.025)
    optimizer = opt.LOptimal([[0, 0], [0, 1]], params, scan, att_dist, lims)

    output = optimizer.optimize()
    assert np.allclose(output.plds, [0.1, 0.375, 0.7, 1.075, 1.65, 2.000])
