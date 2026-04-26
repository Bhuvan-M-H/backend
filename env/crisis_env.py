"""
Crisis Environment for SECIS-SA WA Hybrid
Reinforcement learning environment with schema drift and cascade effects
Coordinate-based map system with hospitals and multi-ambulance coordination
"""

from typing import Dict, Any, List, Tuple
import random
import math
from .cascade import apply_cascade_effects
from .schema_drift import apply_schema_drift
from ..training.reward import compute_multi_objective_reward
from ..training.adversarial import update_adversarial_tracker


class CrisisEnv:
    """Crisis simulation environment with coordinate-based map and hospitals"""
    
    def __init__(self, max_steps=20, ambulances_per_agent=2, difficulty=0.5, adversarial_level=0.5, agent_name="agent", single_agent_mode=False):
        self.max_steps = max_steps
        self.current_state = {}
        self.current_step = 0
        self.done = False
        self.drift_flag = False
        self.single_agent_mode = single_agent_mode
        self.weakness_tracker = {
            "prioritization_failures": 0,
            "delays": 0,
            "ignored_high_severity": 0
        }
        self.resolved_incidents_count = 0
        self.total_reward = 0.0
        self.agent_name = agent_name
        
        # Control parameters
        self.difficulty = difficulty  # 0-1: affects incident spawn rate and severity
        self.adversarial_level = adversarial_level  # 0-1: affects adversarial scenario triggering
        self.tick_interval = 200  # ms: controls simulation speed (reduced for smoother flow)
        
        # Map dimensions (0-100 coordinate plane)
        self.map_size = 100
        
        # Fixed hospitals with predefined coordinates
        self.hospitals = [
            {"id": "hospital_1", "x": 20, "y": 20, "capacity": 100, "occupied": 0},
            {"id": "hospital_2", "x": 80, "y": 20, "capacity": 100, "occupied": 0},
            {"id": "hospital_3", "x": 50, "y": 80, "capacity": 100, "occupied": 0}
        ]
        
        # Initialize ambulances for this agent (2 per agent)
        self.ambulances_per_agent = ambulances_per_agent
        self.ambulances = self._initialize_ambulances()
        self.reset()
    
    def _initialize_ambulances(self) -> List[Dict[str, Any]]:
        """Initialize ambulances with coordinate-based positions and state machine"""
        ambulances = []
        for i in range(self.ambulances_per_agent):
            # Start ambulances near the center
            ambulances.append({
                "id": f"{self.agent_name}_amb_{i+1}",
                "x": 50.0,
                "y": 50.0,
                "state": "idle",  # idle, to_incident, to_hospital
                "target_incident": None,
                "target_hospital": None,
                "carrying_incident": None,
                "path": []  # Track path for visualization
            })
        return ambulances
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state"""
        self.current_step = 0
        self.done = False
        self.drift_flag = False
        self.total_reward = 0.0
        self.resolved_incidents_count = 0
        self.weakness_tracker = {
            "prioritization_failures": 0,
            "delays": 0,
            "ignored_high_severity": 0
        }
        
        # Reset ambulances to center
        self.ambulances = self._initialize_ambulances()
        
        # Reset hospital occupancy
        for hospital in self.hospitals:
            hospital["occupied"] = 0
        
        self.current_state = {
            "incidents": [],
            "ambulances": self.ambulances,
            "hospitals": self.hospitals,
            "map_size": self.map_size,
            "resources": {
                "hospital": 100,
                "police": 100,
                "fire": 100
            },
            "system_state": {
                "hospital_occupied": 0,
                "hospital_capacity": 100
            }
        }
        
        # Generate initial incidents with coordinate positions (affected by difficulty)
        initial_incidents = 3 + int(self.difficulty * 2)  # 3-5 incidents based on difficulty
        for _ in range(initial_incidents):
            self.current_state["incidents"].append({
                "id": f"inc_init_{random.randint(1000, 9999)}",
                "severity": random.uniform(0.3 + self.difficulty * 0.2, 0.9),  # Higher difficulty = higher severity
                "x": random.uniform(10, 90),  # Coordinate-based position
                "y": random.uniform(10, 90),
                "status": "waiting",  # waiting, picked, resolved
                "assigned_ambulance": None,
                "assigned_time": None
            })
        
        return self.current_state
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with coordinate-based movement
        
        Args:
            action: Dictionary with 'ambulance_id', 'incident_id', and 'reason'
            
        Returns:
            Tuple of (state, reward, done, metadata)
        """
        self.current_step += 1
        
        # 1. Apply action - assign incident to ambulance
        assigned_incidents = self._apply_action(action)
        
        # 2. Move all ambulances based on their state
        delivered_incidents = self._move_ambulances()
        self.resolved_incidents_count += delivered_incidents
        
        # 3. Cascade effects - randomly add new incidents (affected by difficulty)
        cascade_probability = 0.3 + (self.difficulty * 0.4)  # 0.3-0.7 based on difficulty
        self.current_state = apply_cascade_effects(self.current_state, probability=cascade_probability, single_agent_mode=self.single_agent_mode)
        
        # 4. Schema drift - randomly change structure (affected by adversarial level)
        drift_probability = 0.2 + (self.adversarial_level * 0.5)  # 0.2-0.7 based on adversarial level
        self.drift_flag, self.current_state = apply_schema_drift(self.current_state, probability=drift_probability)
        
        # 5. Compute multi-objective reward (based on hospital deliveries)
        reward, reward_breakdown = compute_multi_objective_reward(
            self.current_state,
            delivered_incidents,
            self.current_step,
            self.weakness_tracker
        )
        
        # 6. Adversarial update - track failures
        self.weakness_tracker = update_adversarial_tracker(
            self.weakness_tracker,
            action,
            self.current_state,
            reward
        )
        
        # Check if done
        if self.current_step >= self.max_steps:
            self.done = True
        
        # Update state with current ambulance positions
        self.current_state["ambulances"] = self.ambulances
        
        # Metadata
        metadata = {
            "reward_breakdown": reward_breakdown,
            "resolved_incidents": delivered_incidents,
            "drift_flag": self.drift_flag,
            "step": self.current_step,
            "weakness_tracker": self.weakness_tracker
        }
        
        return self.current_state, reward, self.done, metadata
    
    def _apply_action(self, action: Dict[str, Any]) -> int:
        """
        Apply action to assign incident to specific ambulance
        Returns number of incidents assigned
        """
        assigned_count = 0
        
        # Get incidents list (handle schema drift)
        incidents = self.current_state.get("incidents", self.current_state.get("incident_list", []))
        
        if not incidents:
            return 0
        
        # Extract ambulance_id and incident_id from action
        ambulance_id = action.get("ambulance_id")
        incident_id = action.get("target")
        
        if not ambulance_id or not incident_id:
            return 0
        
        # Find the ambulance
        ambulance = next((amb for amb in self.ambulances if amb["id"] == ambulance_id), None)
        if not ambulance or ambulance["state"] != "idle":
            return 0
        
        # Find the incident
        incident = next((inc for inc in incidents if inc["id"] == incident_id), None)
        if not incident or incident["status"] != "waiting":
            return 0
        
        # Assign incident to ambulance
        ambulance["state"] = "to_incident"
        ambulance["target_incident"] = incident
        ambulance["path"] = [(ambulance["x"], ambulance["y"])]  # Start tracking path
        
        # Update incident status
        incident["status"] = "picked"
        incident["assigned_ambulance"] = ambulance_id
        incident["assigned_time"] = self.current_step
        
        assigned_count += 1
        
        return assigned_count
    
    def _move_ambulances(self) -> int:
        """
        Move all ambulances based on their state
        Returns number of incidents delivered to hospitals
        """
        delivered_count = 0
        
        for ambulance in self.ambulances:
            if ambulance["state"] == "idle":
                continue
            
            elif ambulance["state"] == "to_incident":
                # Move towards incident
                incident = ambulance["target_incident"]
                if incident:
                    self._move_towards_target(ambulance, incident["x"], incident["y"])
                    
                    # Check if reached incident
                    if self._distance(ambulance["x"], ambulance["y"], incident["x"], incident["y"]) < 2.0:
                        # Pickup incident
                        ambulance["state"] = "to_hospital"
                        ambulance["carrying_incident"] = incident
                        ambulance["target_incident"] = None
                        
                        # Update incident status to show it's being carried
                        incident["status"] = "picked"
                        incident["assigned_ambulance"] = ambulance["id"]
                        
                        # Choose nearest hospital
                        nearest_hospital = self._find_nearest_hospital(ambulance["x"], ambulance["y"])
                        ambulance["target_hospital"] = nearest_hospital
                
            elif ambulance["state"] == "to_hospital":
                # Move towards hospital
                hospital = ambulance["target_hospital"]
                if hospital:
                    self._move_towards_target(ambulance, hospital["x"], hospital["y"])
                    
                    # Check if reached hospital
                    if self._distance(ambulance["x"], ambulance["y"], hospital["x"], hospital["y"]) < 2.0:
                        # Deliver incident
                        incident = ambulance["carrying_incident"]
                        if incident:
                            # Update incident status to resolved
                            incident["status"] = "resolved"
                            
                            # Remove incident from list only after delivery
                            incidents = self.current_state.get("incidents", [])
                            if incident in incidents:
                                incidents.remove(incident)
                            
                            # Update hospital occupancy
                            hospital["occupied"] = min(hospital["occupied"] + 1, hospital["capacity"])
                            
                            # Reset ambulance
                            ambulance["state"] = "idle"
                            ambulance["carrying_incident"] = None
                            ambulance["target_hospital"] = None
                            ambulance["path"] = []
                            
                            delivered_count += 1
        
        return delivered_count
    
    def _move_towards_target(self, ambulance: Dict[str, Any], target_x: float, target_y: float):
        """Move ambulance towards target with smooth interpolation"""
        # Calculate direction
        dx = target_x - ambulance["x"]
        dy = target_y - ambulance["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 0.1:
            return  # Already at target
        
        # Movement speed based on tick_interval (faster with lower interval)
        speed = 30.0  # Base speed units per tick (increased for smoother flow)
        
        # Normalize and apply speed
        move_x = (dx / distance) * min(speed, distance)
        move_y = (dy / distance) * min(speed, distance)
        
        # Update position
        ambulance["x"] += move_x
        ambulance["y"] += move_y
        
        # Track path for visualization
        ambulance["path"].append((ambulance["x"], ambulance["y"]))
        # Keep path length manageable (increased for smoother lines)
        if len(ambulance["path"]) > 100:
            ambulance["path"] = ambulance["path"][-100:]
    
    def _distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def _find_nearest_hospital(self, x: float, y: float) -> Dict[str, Any]:
        """Find the nearest hospital to the given coordinates"""
        nearest_hospital = None
        min_distance = float('inf')
        
        for hospital in self.hospitals:
            dist = self._distance(x, y, hospital["x"], hospital["y"])
            if dist < min_distance:
                min_distance = dist
                nearest_hospital = hospital
        
        return nearest_hospital
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics"""
        total_hospital_occupied = sum(h["occupied"] for h in self.hospitals)
        total_hospital_capacity = sum(h["capacity"] for h in self.hospitals)
        
        return {
            "step": self.current_step,
            "active_incidents": len(self.current_state.get("incidents", [])),
            "ambulances": len(self.ambulances),
            "idle_ambulances": len([amb for amb in self.ambulances if amb["state"] == "idle"]),
            "active_ambulances": len([amb for amb in self.ambulances if amb["state"] != "idle"]),
            "drift_flag": self.drift_flag,
            "hospital_occupied": total_hospital_occupied,
            "hospital_capacity": total_hospital_capacity,
            "resolved_incidents": self.resolved_incidents_count
        }
    
    def get_control_parameters(self) -> Dict[str, Any]:
        """Get current control parameters"""
        return {
            "difficulty": self.difficulty,
            "adversarial_level": self.adversarial_level,
            "tick_interval": self.tick_interval
        }
    
    def set_control_parameters(self, difficulty: float = None, adversarial_level: float = None, tick_interval: int = None):
        """Set control parameters"""
        if difficulty is not None:
            self.difficulty = max(0.0, min(1.0, difficulty))
        if adversarial_level is not None:
            self.adversarial_level = max(0.0, min(1.0, adversarial_level))
        if tick_interval is not None:
            self.tick_interval = max(100, min(5000, tick_interval))
