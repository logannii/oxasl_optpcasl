#!/usr/bin/env python
"""
PCASL optimisation script for Logan (Xin Zhang)

This script demonstrates how the optimization functions are used and how to
plot the predicted CRLB standard deviations across ATT.
"""
import matplotlib.pyplot as plt
import numpy as np

import oxasl_optpcasl as opt


def main():
    """
    Simple example of optimizing the PLDs of a standard pCASL acquisition
    as per Buxton et al. MRM 1998
    """
    params = opt.ASLParams(f=50.0 / 6000)
    att_dist = opt.ATTDist(0.2, 2.1, 0.001, 0.3)
    scan = opt.ASLScan(
        opt.VAR_MULTI_PCASL,
        duration=300,
        npld=6,
        readout=0.5,
        nslices=3,
        slicedt=0.0452,
    )
    lims = opt.Limits(0.1, 3.0, 0.025)

    optimizer = opt.DOptimal(params, scan, att_dist, lims)
    output = optimizer.optimize()
    att = att_dist.atts
    cov = output.cov_optimized

    plt.subplot(1, 2, 1)
    plt.title("CBF SD")
    plt.ylabel("SD (ml/100g/min)")
    plt.xlabel("ATT (s)")
    plt.ylim(0, 9)
    for slice_idx in range(scan.nslices):
        plt.plot(
            att,
            np.squeeze(np.sqrt(np.abs(cov[slice_idx, :, 0, 0]))),
            label=f"Optimized protocol - slice {slice_idx}",
        )
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("ATT SD")
    plt.ylabel("SD (s)")
    plt.xlabel("ATT (s)")
    plt.ylim(0, 0.25)
    for slice_idx in range(scan.nslices):
        plt.plot(
            att,
            np.squeeze(np.sqrt(np.abs(cov[slice_idx, :, 1, 1]))),
            label=f"Optimized protocol - slice {slice_idx}",
        )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
