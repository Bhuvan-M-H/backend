"""
Safety constraints for SECIS-SA WA Hybrid
Ensures agent actions stay within safe bounds
"""

from typing import Dict, Any, Tuple


def check_safety_constraints(action: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, bool]]:
    """
    Check safety constraints for an action
    
    Args:
        action: Action to check
        state: Current environment state
        
    Returns:
        Tuple of (is_safe, reason, safety_flags)
    """
    safety_flags = {
        "no_idle_resource_abuse": True,
        "no_invalid_dispatch": True,
        "no_repeated_actions": True
    }
    
    # Allow "wait" action even if no idle ambulances or incidents
    if action.get("action") == "wait":
        return True, "Wait action is safe", safety_flags
    
    # Constraint 1: No idle resource abuse
    # Check if action uses idle ambulances
    ambulances = state.get("ambulances", [])
    idle_ambulances = [amb for amb in ambulances if amb.get("state") == "idle"]
    
    if not idle_ambulances:
        safety_flags["no_idle_resource_abuse"] = False
        return False, "No idle ambulances available", safety_flags
    
    # Constraint 2: No invalid dispatch
    # Check if target is valid
    target = action.get("target")
    incidents = state.get("incidents", state.get("incident_list", []))
    
    if not incidents:
        safety_flags["no_invalid_dispatch"] = False
        return False, "No incidents to dispatch to", safety_flags
    
    if isinstance(target, dict):
        target_id = target.get("id")
    else:
        target_id = target
    
    incident_ids = [inc.get("id") for inc in incidents]
    if target_id not in incident_ids:
        safety_flags["no_invalid_dispatch"] = False
        return False, f"Invalid target: {target_id}", safety_flags
    
    # Constraint 3: No repeated actions
    # Check if action is a repeat of the previous action (simplified check)
    # In a real implementation, you'd track previous actions
    # For now, we'll just check if the action has the same target as the last dispatched incident
    # This is a placeholder - you'd need to implement proper action history tracking
    
    return True, "Action is safe", safety_flags
