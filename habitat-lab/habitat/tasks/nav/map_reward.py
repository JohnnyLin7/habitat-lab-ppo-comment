#!/usr/bin/env python3

from typing import Dict, Optional
import numpy as np
import cv2
import os
import torch
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationTask, NavigationReward
from habitat.core.spaces import ActionSpace
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps
class MapManager:
    def __init__(self, maps_dir: str):
        """初始化地图管理器
        
        Args:
            maps_dir: 地图文件目录路径
        """
        self.maps: Dict[str, np.ndarray] = {}
        self.maps_dir = maps_dir
        self._load_all_maps()
        
    def _load_all_maps(self):
        """加载所有场景地图"""
        if not os.path.exists(self.maps_dir):
            raise ValueError(f"地图目录不存在: {self.maps_dir}")
            
        for scene_file in os.listdir(self.maps_dir):
            if scene_file.endswith('.pgm'):  # 改为.pgm
                scene_id = scene_file.replace('.pgm', '')
                map_path = os.path.join(self.maps_dir, scene_file)
                # 读取RGB格式地图
                map_img = cv2.imread(map_path, cv2.IMREAD_COLOR)
                if map_img is None:
                    continue
                # 转换为二值图,因为我们知道地图中只有障碍物和可行区域
                gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                self.maps[scene_id] = binary
                
    def get_map_resolution(self, map_height: int) -> float:
        """根据地图高度计算分辨率
        
        Args:
            map_height: 地图高度(像素)
            
        Returns:
            地图分辨率(米/像素)
        """
        # 使用habitat提供的计算函数
        meters_per_pixel = maps.calculate_meters_per_pixel(
            map_resolution=map_height,
            sim=self._sim
        )
        return meters_per_pixel
                
    def get_map(self, scene_id: str) -> Optional[np.ndarray]:
        """获取指定场景的地图
        
        Args:
            scene_id: 场景ID
            
        Returns:
            场景地图数组,如果不存在则返回None
        """
        return self.maps.get(scene_id)
    
    def world_to_map(self, world_pos: np.ndarray, scene_map: np.ndarray) -> np.ndarray:
        """将世界坐标转换为地图坐标
        
        Args:
            world_pos: 世界坐标 [x, y, z]
            scene_map: 场景地图数组
            
        Returns:
            地图坐标 [x, y]
        """
        map_size = scene_map.shape[0]
        # 假设地图中心对应世界坐标原点
        map_pos = (world_pos[:2] / self.map_resolution + map_size/2).astype(np.int32)
        return np.clip(map_pos, 0, map_size-1)

@registry.register_measure
class MapExplorationReward(NavigationReward):
    """基于地图的探索奖励度量"""

    cls_uuid: str = "map_exploration_reward"

    def __init__(
        self, 
        sim: Simulator, 
        config, 
        *args, 
        task: NavigationTask,
        **kwargs
    ):
        self._sim = sim
        self._config = config
        self._map_manager = MapManager(config.MAPS_DIR)
        self._previous_position = None
        self._previous_rotation = None
        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    def _get_agent_state(self):
        """获取智能体状态"""
        agent_state = self._sim.get_agent_state()
        return agent_state.position, agent_state.rotation

    def compute_explored_area_reward(
        self,
        current_position: np.ndarray,
        scene_map: np.ndarray
    ) -> float:
        """计算探索区域奖励
        
        Args:
            current_position: 当前位置 [x,y,z]
            scene_map: 场景地图
            
        Returns:
            探索奖励值
        """
        if self._previous_position is None:
            self._previous_position = current_position
            return 0.0
            
        # 使用habitat的to_grid函数进行坐标转换
        curr_x, curr_y = maps.to_grid(
            current_position[2],  # z坐标
            current_position[0],  # x坐标
            scene_map.shape[0:2],
            sim=self._sim,
        )
        
        prev_x, prev_y = maps.to_grid(
            self._previous_position[2],
            self._previous_position[0], 
            scene_map.shape[0:2],
            sim=self._sim,
        )
        
        # 边界检查
        curr_x = np.clip(curr_x, 0, scene_map.shape[0]-1)
        curr_y = np.clip(curr_y, 0, scene_map.shape[1]-1)
        prev_x = np.clip(prev_x, 0, scene_map.shape[0]-1)
        prev_y = np.clip(prev_y, 0, scene_map.shape[1]-1)
        
        # 检查当前位置是否可行
        is_valid_pos = scene_map[curr_x, curr_y] > 0
        
        # 计算地图坐标系下的移动距离
        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        
        self._previous_position = current_position
        
        # 如果在有效区域内移动,给予奖励
        return distance * 0.1 if is_valid_pos else -0.1

    def compute_distance_to_walls(
        self,
        current_position: np.ndarray,
        scene_map: np.ndarray
    ) -> float:
        """计算到墙壁的距离奖励
        
        Args:
            current_position: 当前位置 [x,y,z]
            scene_map: 场景地图
            
        Returns:
            距离奖励值
        """
        # 使用habitat的to_grid函数进行坐标转换
        curr_x, curr_y = maps.to_grid(
            current_position[2],  # z坐标
            current_position[0],  # x坐标
            scene_map.shape[0:2],
            sim=self._sim,
        )
        
        # 边界检查
        curr_x = np.clip(curr_x, 0, scene_map.shape[0]-1)
        curr_y = np.clip(curr_y, 0, scene_map.shape[1]-1)
        
        # 使用距离变换计算到最近障碍物的距离(像素距离)
        dist_transform = cv2.distanceTransform(
            (scene_map > 0).astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        pixel_distance = dist_transform[curr_x, curr_y]
        
        # 将像素距离转换为实际距离(米)
        # 使用simulator的缩放因子进行转换
        real_distance = pixel_distance * self._sim.get_agent(0).agent_config.AGENT_0.HEIGHT
        
        # 根据实际距离(米)给予奖励
        if real_distance < 0.5:  # 太靠近墙壁
            return -0.2
        elif real_distance < 2.0:  # 保持适当距离
            return 0.1
        return 0.0  # 距离较远时中性

    def reset_metric(self, *args, episode, task, **kwargs):
        self._previous_position = None
        self._previous_rotation = None
        self._metric = None
        
    def update_metric(self, *args, episode, task, **kwargs):
        current_position, current_rotation = self._get_agent_state()
        scene_id = episode.scene_id
        
        scene_map = self._map_manager.get_map(scene_id)
        if scene_map is None:
            self._metric = 0.0
            return
            
        exploration_reward = self.compute_explored_area_reward(
            current_position,
            scene_map
        )
        wall_reward = self.compute_distance_to_walls(
            current_position,
            scene_map
        )
        
        # 合并奖励
        self._metric = (
            self._config.exploration_reward_scale * exploration_reward +
            self._config.wall_distance_reward_scale * wall_reward
        )