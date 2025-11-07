import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pytools.logging.api import BLogger
from scipy.ndimage import gaussian_filter1d

from pwlsplit.curve.peaks import construct_segmentation
from pwlsplit.plot import plot_prepped_data
from pwlsplit.segment.refine import opt_index
from pwlsplit.segment.split import adjust_segmentation
from pwlsplit.struct import FinalSegmentation, PreppedData

from .tools import create_bogoni_curve

if TYPE_CHECKING:
    from arraystubs import Arr2

parser = argparse.ArgumentParser(prog="pwlsplit")
parser.add_argument("file", type=str, nargs="+", help="Path to the input file(s).")
parser.add_argument("--plot", action="store_true", help="Generate plots for the segmented data.")


def export_bogoni_data[F: np.floating, I: np.integer](
    data: Arr2[F], segmentation: FinalSegmentation[F, I], *, fout: Path
) -> None:
    protocol = np.full_like(data[:, 0], "", dtype="<U16")
    cycle = np.full_like(data[:, 0], "", dtype="<U16")
    phase = np.full_like(data[:, 0], "", dtype="<U16")
    for prot, prot_vals in segmentation.prot.items():
        for cycle_name, segs in prot_vals.items():
            for k, v in enumerate(segs):
                protocol[segmentation.idx[v - 1] : segmentation.idx[v] + 1] = prot
                cycle[segmentation.idx[v - 1] : segmentation.idx[v] + 1] = cycle_name
                phase[segmentation.idx[v - 1] : segmentation.idx[v] + 1] = (
                    f"{k}_{segmentation.curves[v - 1]}"
                )
    dic = {
        "Protocol": protocol,
        "Cycle": cycle,
        "Phase": phase,
        "Time [s]": data[:, 0],
        "Stretch [-]": data[:, 1],
        "P [kPa]": data[:, 2],
        "Weight [-]": 1.0 / data[:, 3],
    }
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(fout, index=False)


def bogoni_process(file: Path, fout: str) -> None:
    folder = file.parent
    log = BLogger("INFO")
    raw = np.loadtxt(file, delimiter=",", skiprows=1, dtype=np.float64)
    y = gaussian_filter1d(raw[:, 1], sigma=20)
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    data = PreppedData(n=len(raw), x=raw[:, 1], y=y, dy=dy / dy.max(), ddy=ddy / ddy.max())
    plot_prepped_data(data, fout=(folder / f"{fout}_prepped.png"))
    protocol = create_bogoni_curve(0.3)
    segmentation = construct_segmentation(protocol)
    segmentation = adjust_segmentation(
        data, segmentation, fout=(folder / f"{fout}_segmentation.png")
    )
    segmentation.idx = opt_index(data.x, segmentation.idx, window=50, max_iter=100, log=log)
    export_bogoni_data(raw, segmentation, fout=(folder / f"{fout}.csv"))


def main() -> None:
    args = parser.parse_args()
    files = [Path(v) for f in args.file for v in Path().glob(f)]
    for file in files:
        with file.open("r") as f:
            specimen = json.load(f)
        for axis, tests in specimen.items():
            for rate, name in tests.items():
                fout = f"{axis}_{rate.replace('.', '-')}"
                print(f"Processing file: {file} for axis: {axis} at rate: {rate}")
                bogoni_process(file.parent / name, fout)


if __name__ == "__main__":
    main()
