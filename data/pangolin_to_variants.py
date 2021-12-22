# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:07:15 2021

@author: Ruibo

from pangolin to variant name
https://www.cdc.gov/coronavirus/2019-ncov/variants/variant-info.html

"""


def assign_variants(x: str) -> str:

    if "B.1.1.7" in x:
        return 'VOC Alpha'
    if "B.1.351" in x:
        return 'VOC Beta'
    elif "P.1" in x:
        return 'VOC Gamma'
    elif "B.1.617.2" in x:  # AY.* 算不算Delta?
        return 'VOC Delta'
    elif "B.1.1.529" in x:
        return 'VOC Omicron'

    elif "B.1.621" in x:
        return "VOI Mu"
    elif "C.37" in x:
        return "VOI Lambda"

    elif "B.1.427" in x or "B.1.429" in x:
        return "VOI Epsilon"
    elif "P.2" == x:
        return "VOI Zeta"
    elif "P.3" in x:
        return "VOI Theta"
    elif "B.1.617.1" == x:
        return "VOI Kappa"
    elif "B.1.526" == x:
        return "VOI Iota"
    elif "B.1.525" == x:
        return "VOI Eta"
    else:
        return "Others"

