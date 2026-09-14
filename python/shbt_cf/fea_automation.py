"""APDL generation for the SHBT multi-layer target."""

from __future__ import annotations


def generate_shbt_apdl_script() -> str:
    """Return a deterministic SOLID186 APDL model for the five-layer target."""
    return """! SHBT Reactor Multi-layer Target FEA Automation
/PREP7
ET,1,186
KEYOPT,1,2,1
MP,EX,1,124e9
MP,PRXY,1,0.39
MP,ALPX,1,11.3e-6
TB,BISO,1,1,2
TBDATA,1,210e6,2.0e9
MP,EX,2,116e9
MP,PRXY,2,0.32
MP,ALPX,2,8.6e-6
TB,BISO,2,1,2
TBDATA,1,250e6,1.5e9
MP,EX,3,1050e9
MP,PRXY,3,0.07
MP,ALPX,3,1.0e-6
MP,EX,4,85e9
MP,PRXY,4,0.34
MP,ALPX,4,18.0e-6
TB,BISO,4,1,2
TBDATA,1,90e6,0.8e9
MP,EX,5,115e9
MP,PRXY,5,0.34
MP,ALPX,5,17.7e-6
TB,BISO,5,1,2
TBDATA,1,70e6,1.0e9
W_target=10.0e-3
L_target=10.0e-3
BLOCK,0,W_target,0,L_target,0,2000e-6
BLOCK,0,W_target,0,L_target,2000e-6,2015e-6
BLOCK,0,W_target,0,L_target,2015e-6,2265e-6
BLOCK,0,W_target,0,L_target,2265e-6,2265.15e-6
BLOCK,0,W_target,0,L_target,2265.15e-6,2270.15e-6
VGLUE,ALL
LESIZE,ALL,0.5e-3
ESIZE,,5
VSEL,S,VOLU,,5
VATT,1,,1
VMESH,ALL
VSEL,S,VOLU,,4
VATT,2,,1
VMESH,ALL
VSEL,S,VOLU,,3
VATT,3,,1
VMESH,ALL
VSEL,S,VOLU,,2
VATT,4,,1
VMESH,ALL
VSEL,S,VOLU,,1
VATT,5,,1
VMESH,ALL
ALLSEL,ALL
DA,1,ALL,0.0
TUNIF,350.0
/SOLU
SOLVE
FINISH
"""