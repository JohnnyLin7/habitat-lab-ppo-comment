#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.nav.map_reward import MapExplorationReward, MapManager

def _try_register_nav_task():
    try:
        from habitat.tasks.nav.nav import NavigationTask
        
        # 注册导航任务
        if not registry.contains_task(name="Nav-v0"):
            registry.register_task(name="Nav-v0", task_type=NavigationTask)
            
        # 注册地图奖励度量
        if not registry.contains_measure("map_exploration_reward"):
            registry.register_measure(
                "map_exploration_reward",
                measure_type=MapExplorationReward
            )
            
    except ImportError as e:
        navtask_import_error = e

        @registry.register_task(name="Nav-v0")
        class NavigationTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise navtask_import_error
