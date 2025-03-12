# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..samples.c2t2f import c2t2f
from . import cyberbattle_env


class CyberBattleC2T2F(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation on a tiny environment. (Useful for debugging purpose)"""

    def __init__(self, **kwargs):
        super().__init__(
            initial_environment=tinytoy.new_environment(),
            **kwargs)
