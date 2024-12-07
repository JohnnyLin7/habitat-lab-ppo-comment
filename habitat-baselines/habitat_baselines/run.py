#!/usr/bin/env python3

# 此为Habitat训练系统的入口点
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import random
import sys
from typing import TYPE_CHECKING

# Hydra用于管理配置
import hydra
import numpy as np
import torch

# 导入配置相关的函数和类
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    # 配置文件的路径,相对于当前文件所在目录
    config_path="config",
    # 默认使用的配置文件,这里使用PPO算法的PointNav任务示例配置
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg: "DictConfig"):
    # 对配置进行必要的修补和验证
    cfg = patch_config(cfg)
    # 根据配置决定是执行评估还是训练
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


def execute_exp(config: "DictConfig", run_type: str) -> None:
    """执行实验的主函数
    Args:
        config: Habitat的配置对象,包含了所有实验参数
        run_type: 运行类型,可以是'train'(训练)或'eval'(评估)
    """
    # 设置随机种子以确保实验可重现
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    
    # 如果配置要求且CUDA可用,则限制PyTorch只使用单线程
    # 这在某些情况下可以提高性能
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    # 从注册表中获取指定的训练器
    from habitat_baselines.common.baseline_registry import baseline_registry

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    
    # 初始化训练器
    trainer = trainer_init(config)

    # 根据运行类型执行相应的操作
    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    # 注册Habitat的配置插件
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    
    # 检查是否使用了旧版命令行参数,如果是则提供迁移指导
    if "--exp-config" in sys.argv or "--run-type" in sys.argv:
        raise ValueError(
            "The API of run.py has changed to be compatible with hydra.\n"
            "--exp-config is now --config-name and is a config path inside habitat-baselines/habitat_baselines/config/. \n"
            "--run-type train is replaced with habitat_baselines.evaluate=False (default) and --run-type eval is replaced with habitat_baselines.evaluate=True.\n"
            "instead of calling:\n\n"
            "python -u -m habitat_baselines.run --exp-config habitat-baselines/habitat_baselines/config/<path-to-config> --run-type train/eval\n\n"
            "You now need to do:\n\n"
            "python -u -m habitat_baselines.run --config-name=<path-to-config> habitat_baselines.evaluate=False/True\n"
        )
    main()
