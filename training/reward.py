"""
Multi-objective reward system for SECIS-SA WA Hybrid
"""

from typing import Dict, Any, Tuple
import random


def compute_multi_objective_reward(
    state: Dict[str, Any],
    delivered_incidents: int,
    current_step: int,
    weakness_tracker: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute multi-objective reward for coordinate-based map system
    
    Components:
    - hospital delivery reward (only rewarded on hospital drop-off)
    - travel efficiency penalty (distance-based)
    - resource penalty (idle ambulances)
    - fairness penalty (ignored high-severity incidents)
    - hospital load balancing penalty
    - anti-reward-hacking penalties
    - duplicate dispatch penalty
    
    Args:
        state: Current environment state
        delivered_incidents: Number of incidents delivered to hospitals this step
        current_step: Current step number
        weakness_tracker: Dictionary tracking agent weaknesses
        
    Returns:
        Tuple of (total_reward, reward_breakdown)
    """
    # Get incidents list (handle schema drift)
    incidents = state.get("incidents", state.get("incident_list", []))
    ambulances = state.get("ambulances", [])
    hospitals = state.get("hospitals", [])
    
    # 1. Hospital delivery reward (main reward - only on hospital drop-off)
    delivery_reward = delivered_incidents * 5.0  # Higher reward for hospital delivery
    
    # 2. Travel efficiency penalty (based on ambulance states)
    # Penalize ambulances in transit (longer routes = lower reward)
    active_ambulances = [amb for amb in ambulances if amb.get("state") != "idle"]
    travel_penalty = -0.01 * len(active_ambulances)  # Reduced penalty
    
    # 3. Resource penalty (idle ambulances when incidents exist)
    idle_count = sum(1 for amb in ambulances if amb.get("state") == "idle")
    waiting_incidents = [inc for inc in incidents if inc.get("status") == "waiting"]
    if waiting_incidents and idle_count > 0:
        resource_penalty = -0.05 * idle_count  # Reduced penalty
    else:
        resource_penalty = 0.0
    
    # 4. Fairness penalty (ignored high-severity incidents)
    high_severity_ignored = sum(
        1 for inc in waiting_incidents 
        if inc.get("severity", 0) > 0.7
    )
    fairness_penalty = -0.2 * high_severity_ignored  # Reduced penalty
    
    # 5. Hospital load balancing penalty
    # Penalize if one hospital is overloaded while others have capacity
    if hospitals:
        hospital_occupancies = [h.get("occupied", 0) / h.get("capacity", 100) for h in hospitals]
        max_occupancy = max(hospital_occupancies) if hospital_occupancies else 0
        min_occupancy = min(hospital_occupancies) if hospital_occupancies else 0
        load_imbalance = max_occupancy - min_occupancy
        load_balance_penalty = -0.2 * load_imbalance  # Reduced penalty
    else:
        load_balance_penalty = 0.0
    
    # 6. Anti-reward-hacking penalties
    # Penalty for too many unresolved incidents
    incident_penalty = -0.05 * len(waiting_incidents)  # Reduced penalty
    
    # 7. Duplicate dispatch penalty (track from weakness_tracker)
    duplicate_dispatch_penalty = -0.1 * weakness_tracker.get("prioritization_failures", 0)  # Reduced penalty
    
    # 8. Delay penalty (incidents waiting too long)
    delayed_incidents = sum(
        1 for inc in waiting_incidents
        if inc.get("assigned_time") and (current_step - inc.get("assigned_time", 0)) > 5
    )
    delay_penalty = -0.1 * delayed_incidents  # Reduced penalty
    
    # Base reward
    base_reward = 0.0  # Lower base reward to emphasize hospital delivery
    
    # Total reward
    total_reward = (
        base_reward +
        delivery_reward +
        travel_penalty +
        resource_penalty +
        fairness_penalty +
        load_balance_penalty +
        incident_penalty +
        duplicate_dispatch_penalty +
        delay_penalty
    )
    
    # Reward breakdown
    reward_breakdown = {
        "base": base_reward,
        "delivery_reward": delivery_reward,
        "travel_penalty": travel_penalty,
        "resource_penalty": resource_penalty,
        "fairness_penalty": fairness_penalty,
        "load_balance_penalty": load_balance_penalty,
        "incident_penalty": incident_penalty,
        "duplicate_dispatch_penalty": duplicate_dispatch_penalty,
        "delay_penalty": delay_penalty,
        "total": total_reward
    }
    
    # Calculate normalized scores for telemetry (0-100%)
    # Efficiency: based on ambulance utilization (more active = better efficiency)
    total_ambulances = len(ambulances)
    efficiency = max(0, min(100, (len(active_ambulances) / max(1, total_ambulances)) * 100))
    
    # Survival: starts at 100, decreases based on probability of incidents not being solved
    # Probability of not being solved increases with:
    # - More waiting incidents
    # - Higher severity incidents waiting
    # - Incidents waiting too long
    total_incidents = len(incidents)
    if total_incidents > 0:
        waiting_ratio = len(waiting_incidents) / total_incidents
        avg_severity_waiting = sum(inc.get("severity", 0) for inc in waiting_incidents) / max(1, len(waiting_incidents))
        # Survival decreases with more waiting incidents and higher severity
        survival = max(0, 100 - (waiting_ratio * 50) - (avg_severity_waiting * 30))
    else:
        survival = 100
    
    # Fairness: based on high severity ignored (fewer ignored is better)
    fairness = max(0, min(100, (1 - high_severity_ignored / max(1, len(waiting_incidents))) * 100))
    
    # Competition: based on load balancing (lower imbalance is better)
    competition = max(0, min(100, (1 - load_imbalance) * 100))
    
    # Add normalized scores to reward breakdown
    reward_breakdown["efficiency_score"] = efficiency
    reward_breakdown["survival_score"] = survival
    reward_breakdown["fairness_score"] = fairness
    reward_breakdown["competition_score"] = competition
    
    return total_reward, reward_breakdown
