"""
Greedy Agent for SECIS-SA WA Hybrid
Selects first/highest priority incident
"""

from typing import Dict, Any, List
import random


class GreedyAgent:
    """Greedy agent that selects first/highest priority incident"""
    
    def __init__(self):
        self.memory = []
    
    def reset_episode(self):
        """Reset agent for new episode"""
        self.memory = []
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select action based on greedy strategy
        
        Strategy: Select highest severity incident and assign to nearest idle ambulance
        """
        incidents = state.get("incidents", state.get("incident_list", []))
        ambulances = state.get("ambulances", [])
        
        # Filter for waiting incidents and idle ambulances
        waiting_incidents = [inc for inc in incidents if inc.get("status") == "waiting"]
        idle_ambulances = [amb for amb in ambulances if amb.get("state") == "idle"]
        
        if not waiting_incidents or not idle_ambulances:
            return {
                "action": "wait",
                "target": None,
                "ambulance_id": None,
                "reason": "No incidents to respond to or no idle ambulances"
            }
        
        # Greedy: Select highest severity incident
        sorted_incidents = sorted(
            waiting_incidents,
            key=lambda x: x.get("severity", 0),
            reverse=True
        )
        target = sorted_incidents[0]
        
        # Find nearest idle ambulance to the incident
        def distance(amb, inc):
            return ((amb.get("x", 0) - inc.get("x", 0)) ** 2 + (amb.get("y", 0) - inc.get("y", 0)) ** 2) ** 0.5
        
        nearest_ambulance = min(idle_ambulances, key=lambda amb: distance(amb, target))
        
        return {
            "action": "dispatch ambulance",
            "target": target.get("id"),
            "ambulance_id": nearest_ambulance.get("id"),
            "reason": f"Greedy: Responding to highest severity incident (severity: {target.get('severity', 0):.2f})"
        }
