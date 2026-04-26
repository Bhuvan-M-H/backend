"""
Reflection Logger for SECIS-SA WA Hybrid
Logs detailed information about each step including what happened and rewards
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List

REFLECTION_FILE = os.path.join(os.path.dirname(__file__), "reflection.json")


def log_step_reflection(
    step: int,
    agent_name: str,
    action: Dict[str, Any],
    state: Dict[str, Any],
    reward: float,
    reward_breakdown: Dict[str, Any],
    resolved_incidents: int,
    new_incidents: int,
    schema_drift: bool,
    metadata: Dict[str, Any]
):
    """
    Log detailed reflection for a step
    
    Args:
        step: Current step number
        agent_name: Name of the agent
        action: Action taken by the agent
        state: Current state after action
        reward: Reward received
        reward_breakdown: Breakdown of reward components
        resolved_incidents: Number of incidents resolved
        new_incidents: Number of new incidents spawned
        schema_drift: Whether schema drift occurred
        metadata: Additional metadata
    """
    try:
        reflection_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "agent": agent_name,
            "action": {
                "type": action.get("action", "N/A"),
                "target": action.get("target", "N/A"),
                "reason": action.get("reason", "N/A")
            },
            "what_happened": {
                "incidents_resolved": resolved_incidents,
                "incidents_spawned": new_incidents,
                "schema_drift": schema_drift,
                "hospital_occupied": state.get("system_state", {}).get("hospital_occupied", 0),
                "hospital_capacity": state.get("system_state", {}).get("hospital_capacity", 100)
            },
            "reward": {
                "total": reward,
                "breakdown": reward_breakdown
            },
            "state_snapshot": {
                "active_incidents": len(state.get("incidents", state.get("incident_list", []))),
                "ambulances_idle": sum(1 for amb in state.get("ambulances", []) if amb.get("status") == "idle"),
                "ambulances_responding": sum(1 for amb in state.get("ambulances", []) if amb.get("status") == "responding")
            },
            "metadata": metadata
        }
        
        # Read existing reflections
        reflections = []
        if os.path.exists(REFLECTION_FILE):
            with open(REFLECTION_FILE, 'r') as f:
                reflections = json.load(f)
        
        # Add new entry
        reflections.append(reflection_entry)
        
        # Keep only last 200 entries
        reflections = reflections[-200:]
        
        # Write back
        with open(REFLECTION_FILE, 'w') as f:
            json.dump(reflections, f, indent=2)
    except Exception as e:
        print(f"Error logging reflection: {e}")


def get_reflection_logs() -> List[Dict[str, Any]]:
    """Get all reflection logs"""
    try:
        if os.path.exists(REFLECTION_FILE):
            with open(REFLECTION_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error reading reflection logs: {e}")
        return []


def clear_reflection_logs():
    """Clear all reflection logs"""
    try:
        with open(REFLECTION_FILE, 'w') as f:
            json.dump([], f)
    except Exception as e:
        print(f"Error clearing reflection logs: {e}")
