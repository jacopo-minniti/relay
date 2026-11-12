from typing import Literal
import matplotlib.pyplot as plt
from tueplots import markers, cycler, bundles, axes, fontsizes

# tueplots
# axes: https://github.com/pnkraemer/tueplots/blob/main/tueplots/axes.py

latex_preamble = r"""\usepackage{bm,amsmath,amsfonts}"""
arxiv_font_family = {
    "font.family": "serif",
    "font.serif": [
        "Charter",
        "XCharter",
        "Times New Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
}
fontsize_icml_default = 10
fontsize_icml_default_smaller = 1
fontsize_offset_icml = 1


def icml_rcparams(
    font_family: Literal["sans-serif", "serif"] = "sans-serif",
    usetex: bool = True,
    colsize: Literal["half", "full"] = "full",
    fontsize: float = 10,
    fontsize_offset: float = 1,
):
    base = {
        **bundles.icml2024(
            family=font_family,
            column=colsize,
        ),
        # output settings
        **{"figure.dpi": 300},
        # latex settings
        **(
            {
                "text.usetex": True,
                "text.latex.preamble": latex_preamble,
            }
            if usetex
            else {}
        ),
        # fontsize settings
        **(
            fontsizes._from_base(
                base=fontsize - fontsize_icml_default_smaller,
                small_offset=fontsize_offset_icml,
            )
        ),
    }

    return base

def neurips_rcparams(
    font_family: Literal["sans-serif", "serif"] = "sans-serif",
    usetex: bool = True,
    colsize: Literal["half", "full"] = "full",
    fontsize: float = 10,
    fontsize_offset: float = 1,
):
    base = {
        **bundles.neurips2024(
            family=font_family,
        ),
        # output settings
        **{"figure.dpi": 300},
        # latex settings
        **(
            {
                "text.usetex": True,
                "text.latex.preamble": latex_preamble,
            }
            if usetex
            else {}
        ),
        # fontsize settings
        **(
            fontsizes._from_base(
                base=fontsize - fontsize_icml_default_smaller,
                small_offset=fontsize_offset_icml,
            )
        ),
    }

    return base
