#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 导入所需的库
import gzip
import json
import os
import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from habitat.config import read_write
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


# 定义常量
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="PointNav-v1")
class PointNavDatasetV1(Dataset):
    """继承自Dataset类的点导航数据集类"""

    episodes: List[NavigationEpisode]  # 导航任务的episodes列表
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"  # 场景内容文件路径模板

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        """检查配置文件中指定的路径是否存在"""
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    @classmethod
    def get_scenes_to_load(cls, config: "DictConfig") -> List[str]:
        """获取需要加载的场景ID列表"""
        dataset_dir = os.path.dirname(
            config.data_path.format(split=config.split)
        )
        if not cls.check_config_paths_exist(config):
            raise FileNotFoundError(
                f"Could not find dataset file `{dataset_dir}`"
            )

        cfg = config.copy()
        with read_write(cfg):
            cfg.content_scenes = []
            dataset = cls(cfg)
            # 检查是否有独立的场景文件
            has_individual_scene_files = os.path.exists(
                dataset.content_scenes_path.split("{scene}")[0].format(
                    data_path=dataset_dir
                )
            )
            if has_individual_scene_files:
                return cls._get_scenes_from_folder(
                    content_scenes_path=dataset.content_scenes_path,
                    dataset_dir=dataset_dir,
                )
            else:
                # 加载完整数据集
                cfg.content_scenes = [ALL_SCENES_MASK]
                dataset = cls(cfg)
                return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        """从文件夹中获取场景列表"""
        scenes: List[str] = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        # 遍历目录获取场景文件
        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def _load_from_file(self, fname: str, scenes_dir: str) -> None:
        """从文件加载数据到self.episodes"""

        if fname.endswith(".pickle"):
            # 注意:pointnav未实现pickle格式
            with open(fname, "rb") as f:
                self.from_binary(pickle.load(f), scenes_dir=scenes_dir)
        else:
            with gzip.open(fname, "rt") as f:
                self.from_json(f.read(), scenes_dir=scenes_dir)

    def __init__(self, config: Optional["DictConfig"] = None) -> None:
        """初始化数据集"""
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.data_path.format(split=config.split)

        # 加载主数据文件
        self._load_from_file(datasetfile_path, config.scenes_dir)

        # 读取每个场景的独立文件
        dataset_dir = os.path.dirname(datasetfile_path)
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            scenes = config.content_scenes
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            # 加载每个场景的数据
            for scene in scenes:
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )

                self._load_from_file(scene_filename, config.scenes_dir)

        else:
            # 过滤episodes
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

    def to_binary(self) -> Dict[str, Any]:
        """转换为二进制格式(未实现)"""
        raise NotImplementedError()

    def from_binary(
        self, data_dict: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> None:
        """从二进制格式加载(未实现)"""
        raise NotImplementedError()

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        """从JSON字符串加载数据"""
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        # 处理每个episode
        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            # 处理场景路径
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            # 处理导航目标
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            # 处理最短路径
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)
