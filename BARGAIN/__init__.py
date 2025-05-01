"""
BARGAIN

Guaranteed Accurate AI for Less
"""

__version__ = "0.1.0"
__author__ = 'Sepanta Zeighami'
from BARGAIN.process.BARGAIN_A import BARGAIN_A
from BARGAIN.process.BARGAIN_P import BARGAIN_P
from BARGAIN.process.BARGAIN_R import BARGAIN_R

from BARGAIN.models.GPTModels import OpenAIOracle, OpenAIProxy
from BARGAIN.models.AbstractModels import Oracle, Proxy

__all__ = ['BARGAIN_A', 'BARGAIN_P', 'BARGAIN_R', 'OpenAIOracle', 'OpenAIProxy', 'Oracle', 'Proxy']
