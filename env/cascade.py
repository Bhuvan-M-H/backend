"""
Cascade effects for SECIS-SA WA Hybrid
Simulates cascade failures where new incidents appear dynamically
"""

from typing import Dict, Any
import random


def apply_cascade_effects(state: Dict[str, Any], probability: float = 0.15, single_agent_mode: bool = False) -> Dict[str, Any]:
    """
    Apply cascade effects - continuously spawn multiple incidents per step
    
    Args:
        state: Current environment state
        probability: Probability of cascade effects occurring
        single_agent_mode: If True, spawn fewer incidents (for single-agent mode)
        
    Returns:
        Updated state with new incidents
    """
    # Get incidents list (handle schema drift)
    incidents = state.get("incidents", state.get("incident_list", []))
    
    # Only spawn incidents if probability check passes
    if random.random() < probability:
        # Spawn fewer incidents in single-agent mode (0-1 instead of 0-2)
        if single_agent_mode:
            num_new_incidents = random.randint(0, 1)
        else:
            num_new_incidents = random.randint(0, 2)
    else:
        num_new_incidents = 0
    
    for _ in range(num_new_incidents):
        # Add new incident with coordinate-based position
        new_incident = {
            "id": f"inc_cascade_{random.randint(1000, 9999)}",
            "severity": random.uniform(0.4, 0.95),
            "x": random.uniform(10, 90),  # Coordinate-based position (0-100 map)
            "y": random.uniform(10, 90),
            "status": "waiting",
            "assigned_ambulance": None,
            "assigned_time": None
        }
        incidents.append(new_incident)
    
    # Set back to correct key based on schema drift
    if "incidents" in state:
        state["incidents"] = incidents
    elif "incident_list" in state:
        state["incident_list"] = incidents
    else:
        state["incidents"] = incidents
    
    # Update hospital occupancy (affect hospitals directly)
    hospitals = state.get("hospitals", [])
    if hospitals:
        # Randomly increase occupancy in one hospital
        target_hospital = random.choice(hospitals)
        if target_hospital["occupied"] < target_hospital["capacity"]:
            target_hospital["occupied"] = min(target_hospital["occupied"] + 2, target_hospital["capacity"])
    
    return state
