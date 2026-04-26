"""
Multi-Agent Environment Manager for SECIS-SA WA Hybrid
Manages 3 agents running simultaneously with their own environments
"""

from typing import Dict, Any, List
import random
from .crisis_env import CrisisEnv
from .cascade import apply_cascade_effects
from .schema_drift import apply_schema_drift
from ..training.reward import compute_multi_objective_reward
from ..training.adversarial import update_adversarial_tracker


class MultiAgentEnv:
    """Manages 3 agents running simultaneously with separate environments"""
    
    def __init__(self, max_steps=20, difficulty=0.5, adversarial_level=0.5):
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        
        # Control parameters
        self.difficulty = difficulty
        self.adversarial_level = adversarial_level
        self.tick_interval = 200  # ms: controls simulation speed (reduced for smoother flow)
        
        # Create 3 separate environments, one for each agent
        # Each agent gets 2 ambulances with coordinate-based system
        self.environments = {
            "greedy": CrisisEnv(
                max_steps=max_steps,
                ambulances_per_agent=2,
                difficulty=difficulty,
                adversarial_level=adversarial_level,
                agent_name="greedy",
                single_agent_mode=False
            ),
            "conservative": CrisisEnv(
                max_steps=max_steps,
                ambulances_per_agent=2,
                difficulty=difficulty,
                adversarial_level=adversarial_level,
                agent_name="conservative",
                single_agent_mode=False
            ),
            "adaptive": CrisisEnv(
                max_steps=max_steps,
                ambulances_per_agent=2,
                difficulty=difficulty,
                adversarial_level=adversarial_level,
                agent_name="adaptive",
                single_agent_mode=False
            )
        }
        
        # Reset all environments to initialize ambulances and incidents
        for env in self.environments.values():
            env.reset()
    
    def step_all(self, actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute one step for all agents with their actions
        
        Args:
            actions: Dictionary mapping agent names to their actions
            
        Returns:
            Dictionary with results for all agents
        """
        self.current_step += 1
        results = {}
        
        for agent_name, env in self.environments.items():
            action = actions.get(agent_name, {})
            
            # Use the CrisisEnv step method which handles movement and delivery
            state, reward, done, metadata = env.step(action)
            
            # Track cumulative reward
            try:
                env.total_reward += float(reward)
            except Exception:
                pass
            
            # Update environment state
            env.current_state = state
            env.done = done
            
            results[agent_name] = {
                "state": state,
                "reward": reward,
                "total_reward": env.total_reward,
                "done": done,
                "resolved_incidents": metadata.get("resolved_incidents", 0),
                "drift_flag": metadata.get("drift_flag", False),
                "stats": env.get_stats()
            }
        
        # Check if all environments are done
        self.done = all(env.done for env in self.environments.values())
        
        return {
            "step": self.current_step,
            "done": self.done,
            "agents": results
        }
    
    def reset(self):
        """Reset all environments"""
        self.current_step = 0
        self.done = False
        for env in self.environments.values():
            env.reset()
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get leaderboard with scores for all agents"""
        return [
            {
                "agent": "greedy",
                "score": self.environments["greedy"].total_reward,
                "resolved": self.environments["greedy"].resolved_incidents_count,
                "avg": self.environments["greedy"].total_reward / max(1, self.current_step)
            },
            {
                "agent": "conservative",
                "score": self.environments["conservative"].total_reward,
                "resolved": self.environments["conservative"].resolved_incidents_count,
                "avg": self.environments["conservative"].total_reward / max(1, self.current_step)
            },
            {
                "agent": "adaptive",
                "score": self.environments["adaptive"].total_reward,
                "resolved": self.environments["adaptive"].resolved_incidents_count,
                "avg": self.environments["adaptive"].total_reward / max(1, self.current_step)
            }
        ]
    
    def get_control_parameters(self) -> Dict[str, Any]:
        """Get current control parameters"""
        return {
            "difficulty": self.difficulty,
            "adversarial_level": self.adversarial_level,
            "tick_interval": self.tick_interval
        }
    
    def set_control_parameters(self, difficulty: float = None, adversarial_level: float = None, tick_interval: int = None):
        """Set control parameters and update all environments"""
        if difficulty is not None:
            self.difficulty = max(0.0, min(1.0, difficulty))
            for env in self.environments.values():
                env.set_control_parameters(difficulty=self.difficulty)
        
        if adversarial_level is not None:
            self.adversarial_level = max(0.0, min(1.0, adversarial_level))
            for env in self.environments.values():
                env.set_control_parameters(adversarial_level=self.adversarial_level)
        
        if tick_interval is not None:
            self.tick_interval = max(100, min(5000, tick_interval))
            for env in self.environments.values():
                env.set_control_parameters(tick_interval=self.tick_interval)
