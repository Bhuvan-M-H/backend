"""
Schema drift for SECIS-SA WA Hybrid
Simulates dynamic structure changes in the environment
"""

from typing import Dict, Any, Tuple
import random


def apply_schema_drift(state: Dict[str, Any], probability: float = 0.3) -> Tuple[bool, Dict[str, Any]]:
    """
    Apply schema drift - randomly change structure (30% chance by default)
    
    Args:
        state: Current environment state
        probability: Probability of schema drift occurring
        
    Returns:
        Tuple of (drift_flag, updated_state)
    """
    drift_flag = False
    
    if random.random() < probability:
        drift_flag = True
        
        # Type of drift: incidents <-> incident_list
        if "incidents" in state:
            # Convert to incident_list
            incidents = state.pop("incidents")
            state["incident_list"] = incidents
        elif "incident_list" in state:
            # Convert to incidents
            incident_list = state.pop("incident_list")
            state["incidents"] = incident_list
        
        # May also add/remove fields
        if random.random() < 0.5:
            # Add a new field
            state["drift_metadata"] = {
                "drift_type": "structure_change",
                "timestamp": random.random()
            }
        else:
            # Remove drift_metadata if exists
            state.pop("drift_metadata", None)
    
    return drift_flag, state
