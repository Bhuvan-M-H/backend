"""
OpenEnv-compatible interface for SECIS
Provides gymnasium API for the crisis environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .env.crisis_env import CrisisEnv


class SECISEnv(gym.Env):
    """
    SECIS Environment compatible with OpenEnv/gymnasium
    """
    
    metadata = {
        'render_modes': ['human'],
        'name': 'SECIS-Crisis-v0'
    }
    
    def __init__(self, max_steps: int = 20, ambulances_per_agent: int = 2):
        super().__init__()
        self.env = CrisisEnv(max_steps=max_steps, ambulances_per_agent=ambulances_per_agent, single_agent_mode=True)
        
        # Define action and observation space
        # Action space: dispatch ambulance, dispatch fire truck, dispatch police, request backup, clear traffic
        self.action_space = spaces.Discrete(5)
        
        # Observation space: incidents, resources, system state
        # Simplified for OpenEnv compatibility
        self.observation_space = spaces.Dict({
            'incidents': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            'resources': spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32),
            'system_state': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'step': spaces.Discrete(max_steps)
        })
        
        self.current_step = 0
        self.max_steps = max_steps
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        state = self.env.reset()
        self.current_step = 0
        
        obs = self._state_to_observation(state)
        info = {'drift_flag': state.get('drift_flag', False)}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Map action index to action string
        action_map = {
            0: 'dispatch ambulance',
            1: 'dispatch fire truck',
            2: 'dispatch police',
            3: 'request backup',
            4: 'clear traffic'
        }
        
        action_str = action_map.get(action, 'dispatch ambulance')
        
        # Execute action
        state, reward, done, info = self.env.step({
            'action': action_str,
            'target': 0,  # Default target
            'reason': f'Agent action {action_str}'
        })
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        obs = self._state_to_observation(state)
        truncated = False
        
        return obs, reward, done, truncated, info
    
    def _state_to_observation(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert environment state to gymnasium observation"""
        incidents = state.get('incidents', [])
        resources = state.get('resources', {})
        system_state = state.get('system_state', {})
        
        # Normalize and convert to numpy arrays
        incident_array = np.zeros(10, dtype=np.float32)
        for i, inc in enumerate(incidents[:10]):
            incident_array[i] = inc.get('severity', 0.0)
        
        resource_array = np.array([
            resources.get('ambulances', 0),
            resources.get('fire_trucks', 0),
            resources.get('police', 0)
        ], dtype=np.float32)
        
        system_array = np.array([
            system_state.get('hospital_capacity', 1.0),
            system_state.get('traffic_level', 0.0)
        ], dtype=np.float32)
        
        return {
            'incidents': incident_array,
            'resources': resource_array,
            'system_state': system_array,
            'step': self.current_step
        }
    
    def render(self):
        """Render the environment (optional for OpenEnv)"""
        pass
