# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..samples.SMAI import smai
from . import cyberbattle_env


class CyberBattleSMAI(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation on a tiny environment. (Useful for debugging purpose)"""

    def __init__(self, **kwargs):
        super().__init__(
            initial_environment=smai.new_environment(),
            **kwargs)
