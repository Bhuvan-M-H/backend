"""
Adaptive LLM Agent for SECIS-SA WA Hybrid
Switches strategy dynamically based on system state
"""

from typing import Dict, Any, List
import random


class LLMAgent:
    """Adaptive LLM agent that switches strategy dynamically"""
    
    def __init__(self):
        self.memory = {"reflection_insights": []}
        self.current_strategy = "greedy"
    
    def reset_episode(self):
        """Reset agent for new episode"""
        self.current_strategy = "greedy"
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select action based on adaptive strategy
        
        Strategy switching:
        - if hospital_overflow → conservative
        - if high_severity_ignored → greedy
        - else → greedy
        """
        incidents = state.get("incidents", state.get("incident_list", []))
        ambulances = state.get("ambulances", [])
        hospitals = state.get("hospitals", [])
        
        # Filter for waiting incidents and idle ambulances
        waiting_incidents = [inc for inc in incidents if inc.get("status") == "waiting"]
        idle_ambulances = [amb for amb in ambulances if amb.get("state") == "idle"]
        
        # Check for hospital overflow
        total_hospital_occupied = sum(h.get("occupied", 0) for h in hospitals)
        total_hospital_capacity = sum(h.get("capacity", 100) for h in hospitals)
        hospital_overflow = total_hospital_occupied >= total_hospital_capacity * 0.8
        
        # Check for high severity ignored
        high_severity_ignored = sum(
            1 for inc in waiting_incidents 
            if inc.get("severity", 0) > 0.7
        )
        
        # Strategy switching
        if hospital_overflow:
            self.current_strategy = "conservative"
            reason = "Adaptive: Hospital overflow detected, switching to conservative strategy"
        elif high_severity_ignored > 0:
            self.current_strategy = "greedy"
            reason = "Adaptive: High severity incidents ignored, switching to greedy strategy"
        else:
            self.current_strategy = "greedy"
            reason = "Adaptive: Default greedy strategy"
        
        if not waiting_incidents or not idle_ambulances:
            return {
                "action": "wait",
                "target": None,
                "ambulance_id": None,
                "reason": f"{reason} - No incidents to respond to or no idle ambulances"
            }
        
        # Apply selected strategy
        if self.current_strategy == "greedy":
            # Greedy: Select highest severity incident
            sorted_incidents = sorted(
                waiting_incidents,
                key=lambda x: x.get("severity", 0),
                reverse=True
            )
            target = sorted_incidents[0]
        else:
            # Conservative: Select last incident
            target = waiting_incidents[-1]
        
        # Find nearest idle ambulance to the incident
        def distance(amb, inc):
            return ((amb.get("x", 0) - inc.get("x", 0)) ** 2 + (amb.get("y", 0) - inc.get("y", 0)) ** 2) ** 0.5
        
        nearest_ambulance = min(idle_ambulances, key=lambda amb: distance(amb, target))
        
        return {
            "action": "dispatch ambulance",
            "target": target.get("id"),
            "ambulance_id": nearest_ambulance.get("id"),
            "reason": f"{reason} - Target: {target.get('id')} (severity: {target.get('severity', 0):.2f})"
        }
    
    def update_with_reward(self, reward: float, state: Dict[str, Any]):
        """Update agent with reward"""
        pass
    
    def reflect_on_episode(self, episode: Dict[str, Any]):
        """Reflect on episode performance"""
        self.memory["reflection_insights"].append(f"Episode completed with reward: {sum(episode.get('rewards', [])):.2f}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "reflections": len(self.memory["reflection_insights"]),
            "current_strategy": self.current_strategy
        }
