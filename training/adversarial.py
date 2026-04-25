"""
Adversarial system for SECIS-SA WA Hybrid
Tracks weaknesses and failures in agent behavior
"""

from typing import Dict, Any


def update_adversarial_tracker(
    weakness_tracker: Dict[str, Any],
    action: Dict[str, Any],
    state: Dict[str, Any],
    reward: float
) -> Dict[str, Any]:
    """
    Update adversarial tracker with weaknesses
    
    Tracks:
    - prioritization failures
    - delays
    - ignored high severity incidents
    
    Args:
        weakness_tracker: Current weakness tracking dictionary
        action: Agent action taken
        state: Current environment state
        reward: Reward received
        
    Returns:
        Updated weakness tracker
    """
    incidents = state.get("incidents", state.get("incident_list", []))
    
    # Track prioritization failures (if reward is very low)
    if reward < -1.0:
        weakness_tracker["prioritization_failures"] += 1
    
    # Track delays (if many incidents remain unresolved)
    if len(incidents) > 5:
        weakness_tracker["delays"] += 1
    
    # Track ignored high severity incidents
    high_severity_ignored = sum(
        1 for inc in incidents 
        if inc.get("severity", 0) > 0.7
    )
    if high_severity_ignored > 0:
        weakness_tracker["ignored_high_severity"] += high_severity_ignored
    
    return weakness_tracker
