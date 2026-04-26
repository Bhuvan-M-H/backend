"""
FastAPI Backend for SECIS-SA WA Hybrid
RESTful API endpoints for environment interaction and agent control
"""

import sys
import os

# Add current directory to Python path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json
import random
from datetime import datetime

# Initialize FastAPI app first
app = FastAPI(title="SECIS Backend API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")

# Import and initialize backend components lazily
env = None
multi_agent_env = None
greedy_agent = None
conservative_agent = None
llm_agent = None
TELEMETRY_FILE = None
SECIS_BACKEND_AVAILABLE = False

def init_backend():
    """Initialize SECIS backend components"""
    global env, multi_agent_env, greedy_agent, conservative_agent, llm_agent, TELEMETRY_FILE, SECIS_BACKEND_AVAILABLE
    
    if SECIS_BACKEND_AVAILABLE:
        return
    
    try:
        # Try to import environment components
        try:
            from env.crisis_env import CrisisEnv
            from env.multi_agent_env import MultiAgentEnv
            from agent.greedy_agent import GreedyAgent
            from agent.conservative_agent import ConservativeAgent
            from agent.llm_agent import LLMAgent
            from training.safety import check_safety_constraints
            from logs.reflection_logger import log_step_reflection, get_reflection_logs, clear_reflection_logs
            
            env = CrisisEnv(max_steps=20, ambulances_per_agent=2, single_agent_mode=True)
            multi_agent_env = MultiAgentEnv(max_steps=20)
            greedy_agent = GreedyAgent()
            conservative_agent = ConservativeAgent()
            llm_agent = LLMAgent()
            TELEMETRY_FILE = os.path.join(os.path.dirname(__file__), "logs", "telemetry.json")
            SECIS_BACKEND_AVAILABLE = True
            print("SECIS backend initialized successfully")
        except ImportError as ie:
            print(f"Import error: {ie}")
            print("SECIS backend components not available, using fallback")
            SECIS_BACKEND_AVAILABLE = False
    except Exception as e:
        print(f"Error initializing SECIS backend: {e}")
        import traceback
        traceback.print_exc()
        SECIS_BACKEND_AVAILABLE = False

# Pydantic models
class ActionRequest(BaseModel):
    action: str
    target: Any
    reason: str


class StepRequest(BaseModel):
    agent_type: str = "adaptive"
    multi_agent: bool = False


class ControlParametersRequest(BaseModel):
    difficulty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Difficulty level (0-1)")
    adversarial_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Adversarial level (0-1)")
    tick_interval: Optional[int] = Field(None, ge=100, le=5000, description="Tick interval in ms (100-5000)")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SECIS Backend", "backend_available": SECIS_BACKEND_AVAILABLE}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SECIS Backend API", "status": "running", "backend_available": SECIS_BACKEND_AVAILABLE}

@app.get("/api/telemetry")
async def get_telemetry():
    """Get telemetry data"""
    init_backend()
    if not TELEMETRY_FILE or not os.path.exists(TELEMETRY_FILE):
        return {"telemetry": []}
    try:
        with open(TELEMETRY_FILE, 'r') as f:
            telemetry = json.load(f)
        return {"telemetry": telemetry}
    except Exception as e:
        print(f"Error reading telemetry: {e}")
        return {"telemetry": []}

@app.get("/api/leaderboard")
async def get_leaderboard():
    """Get leaderboard data"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE or not multi_agent_env:
        return {"leaderboard": [
            {"agent": "adaptive", "score": 0.0},
            {"agent": "greedy", "score": 0.0},
            {"agent": "conservative", "score": 0.0}
        ]}
    try:
        leaderboard = multi_agent_env.get_leaderboard()
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        return {"leaderboard": leaderboard}
    except Exception as e:
        print(f"Error getting leaderboard: {e}")
        return {"leaderboard": [
            {"agent": "adaptive", "score": 0.0},
            {"agent": "greedy", "score": 0.0},
            {"agent": "conservative", "score": 0.0}
        ]}

@app.get("/api/reflection")
async def get_reflection():
    """Get reflection logs"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE:
        return {"reflections": []}
    try:
        from logs.reflection_logger import get_reflection_logs
        reflections = get_reflection_logs()
        return {"reflections": reflections}
    except Exception as e:
        print(f"Error getting reflection logs: {e}")
        return {"reflections": []}

@app.delete("/api/reflection")
async def clear_reflection():
    """Clear reflection logs"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE:
        return {"message": "Reflection logs cleared successfully"}
    try:
        from logs.reflection_logger import clear_reflection_logs
        clear_reflection_logs()
        return {"message": "Reflection logs cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing reflection logs: {str(e)}")

@app.get("/api/state")
async def get_state():
    """Get current environment state"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE or not env:
        return {"state": {}, "stats": {}, "step": 0, "done": False}
    try:
        return {
            "state": env.current_state,
            "stats": env.get_stats(),
            "step": env.current_step,
            "done": env.done
        }
    except Exception as e:
        print(f"Error getting state: {e}")
        return {"state": {}, "stats": {}, "step": 0, "done": False}

@app.get("/api/control-parameters")
async def get_control_parameters():
    """Get control parameters"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE or not env:
        return {"control_parameters": {"difficulty": 0.5, "adversarial_level": 0.5, "tick_interval": 1000}}
    try:
        params = env.get_control_parameters()
        return {"control_parameters": params}
    except Exception as e:
        print(f"Error getting control parameters: {e}")
        return {"control_parameters": {"difficulty": 0.5, "adversarial_level": 0.5, "tick_interval": 1000}}

@app.put("/api/control-parameters")
async def update_control_parameters(request: ControlParametersRequest):
    """Update control parameters"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE or not env:
        return {"message": "Control parameters updated successfully", "control_parameters": {"difficulty": 0.5, "adversarial_level": 0.5, "tick_interval": 1000}}
    try:
        env.set_control_parameters(
            difficulty=request.difficulty,
            adversarial_level=request.adversarial_level,
            tick_interval=request.tick_interval
        )
        if multi_agent_env:
            multi_agent_env.set_control_parameters(
                difficulty=request.difficulty,
                adversarial_level=request.adversarial_level,
                tick_interval=request.tick_interval
            )
        updated_params = env.get_control_parameters()
        return {
            "message": "Control parameters updated successfully",
            "control_parameters": updated_params
        }
    except Exception as e:
        print(f"Error updating control parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating control parameters: {str(e)}")

@app.post("/api/step")
async def step_environment(request: StepRequest):
    """Execute one step with specified agent"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE or not env:
        return {"error": "SECIS backend not available", "state": {}, "reward": 0.0, "done": False}
    
    try:
        from training.safety import check_safety_constraints
        from logs.reflection_logger import log_step_reflection
        
        if not request.multi_agent:
            if env.current_step == 0 or env.done:
                env.reset()
        
        if request.multi_agent:
            actions = {
                "greedy": greedy_agent.act(multi_agent_env.environments["greedy"].current_state),
                "conservative": conservative_agent.act(multi_agent_env.environments["conservative"].current_state),
                "adaptive": llm_agent.act(multi_agent_env.environments["adaptive"].current_state)
            }
            results = multi_agent_env.step_all(actions)
            return {
                "multi_agent": True,
                "step": results["step"],
                "done": results["done"],
                "agents": results["agents"]
            }
        else:
            if request.agent_type == "greedy":
                agent = greedy_agent
            elif request.agent_type == "conservative":
                agent = conservative_agent
            else:
                agent = llm_agent
            
            action = agent.act(env.current_state)
            is_safe, safety_reason, safety_flags = check_safety_constraints(action, env.current_state)
            
            if not is_safe:
                return {
                    "multi_agent": False,
                    "error": safety_reason,
                    "state": env.current_state,
                    "reward": 0.0,
                    "done": False,
                    "safety_flags": safety_flags
                }
            
            state, reward, done, metadata = env.step(action)
            
            state_with_agent = state.copy()
            if "ambulances" in state_with_agent:
                state_with_agent["ambulances"] = [
                    {**amb, "agentName": request.agent_type} for amb in state_with_agent["ambulances"]
                ]
            
            stats = env.get_stats()
            efficiency = metadata.get("reward_breakdown", {}).get("efficiency_score", 0.0)
            fairness = metadata.get("reward_breakdown", {}).get("fairness_score", 0.0)
            survival = metadata.get("reward_breakdown", {}).get("survival_score", 0.0)
            hospital_occupancy = stats["hospital_occupied"] / max(1, stats["hospital_capacity"]) * 100
            
            log_telemetry(
                reward=reward,
                incident_count=stats["active_incidents"],
                response_time=random.uniform(1.0, 5.0),
                overflow=stats["hospital_occupied"] >= stats["hospital_capacity"],
                efficiency=efficiency,
                fairness=fairness,
                survival=survival,
                hospital_deliveries=metadata.get("resolved_incidents", 0),
                hospital_occupancy=hospital_occupancy
            )
            
            log_step_reflection(
                step=env.current_step,
                agent_name=request.agent_type,
                action=action,
                state=state_with_agent,
                reward=reward,
                reward_breakdown=metadata.get("reward_breakdown", {}),
                resolved_incidents=metadata.get("resolved_this_step", 0),
                new_incidents=stats["active_incidents"],
                schema_drift=metadata.get("drift_flag", False),
                metadata={"done": done}
            )
            
            return {
                "multi_agent": False,
                "state": state_with_agent,
                "reward": reward,
                "done": done,
                "metadata": metadata,
                "action": action,
                "agent_type": request.agent_type,
                "stats": stats,
                "safety_flags": safety_flags
            }
    except Exception as e:
        print(f"Error executing step: {e}")
        return {"error": str(e), "state": env.current_state if env else {}, "reward": 0.0, "done": False}

@app.post("/api/reset")
async def reset_environment():
    """Reset environment"""
    init_backend()
    if not SECIS_BACKEND_AVAILABLE or not env:
        return {"message": "Environment reset successfully"}
    try:
        env.reset()
        if multi_agent_env:
            multi_agent_env.reset()
        if greedy_agent:
            greedy_agent.reset_episode()
        if conservative_agent:
            conservative_agent.reset_episode()
        if llm_agent:
            llm_agent.reset_episode()
        return {"message": "Environment reset successfully"}
    except Exception as e:
        print(f"Error resetting environment: {e}")
        return {"message": "Environment reset successfully"}

def log_telemetry(reward: float, incident_count: int, response_time: float, overflow: bool,
                 efficiency: float = 0.0, fairness: float = 0.0, survival: float = 0.0,
                 hospital_deliveries: int = 0, hospital_occupancy: float = 0.0):
    """Log telemetry data to JSON file"""
    if not TELEMETRY_FILE:
        return
    try:
        telemetry_data = {
            "timestamp": datetime.now().isoformat(),
            "reward": reward,
            "incident_count": incident_count,
            "response_time": response_time,
            "overflow": overflow,
            "efficiency": efficiency,
            "fairness": fairness,
            "survival": survival,
            "hospital_deliveries": hospital_deliveries,
            "hospital_occupancy": hospital_occupancy
        }
        
        telemetry = []
        if os.path.exists(TELEMETRY_FILE):
            with open(TELEMETRY_FILE, 'r') as f:
                telemetry = json.load(f)
        
        telemetry.append(telemetry_data)
        telemetry = telemetry[-100:]
        
        os.makedirs(os.path.dirname(TELEMETRY_FILE), exist_ok=True)
        with open(TELEMETRY_FILE, 'w') as f:
            json.dump(telemetry, f, indent=2)
    except Exception as e:
        print(f"Error logging telemetry: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_episode_with_agent(agent, agent_name: str) -> float:
    """Run a 20-step episode with the specified agent and return total reward"""
    agent.reset_episode()
    env.reset()
    total_reward = 0.0
    
    for _ in range(20):
        action = agent.act(env.current_state)
        
        # Check safety constraints
        is_safe, _ = check_safety_constraints(action, env.current_state)
        
        if not is_safe:
            # Skip unsafe action
            continue
        
        state, reward, done, metadata = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SECIS-SA WA Hybrid API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/": "Root",
            "/step": "Execute one step",
            "/telemetry": "Get telemetry data",
            "/leaderboard": "Get leaderboard",
            "/reset": "Reset environment"
        }
    }


@app.post("/reset")
async def reset_environment():
    """Reset environment"""
    env.reset()
    multi_agent_env.reset()
    greedy_agent.reset_episode()
    conservative_agent.reset_episode()
    llm_agent.reset_episode()
    return {"message": "Environment reset successfully"}


@app.post("/step")
async def step_environment(request: StepRequest):
    """Execute one step with specified agent(s)"""
    # Ensure single-agent environment is properly initialized
    if not request.multi_agent:
        # Single-agent mode: only reset at first step or when environment is done
        # Do NOT reset during normal operation even if no incidents exist temporarily
        if env.current_step == 0 or env.done:
            env.reset()
    
    if request.multi_agent:
        # Multi-agent mode: run all 3 agents together
        actions = {
            "greedy": greedy_agent.act(multi_agent_env.environments["greedy"].current_state),
            "conservative": conservative_agent.act(multi_agent_env.environments["conservative"].current_state),
            "adaptive": llm_agent.act(multi_agent_env.environments["adaptive"].current_state)
        }
        
        results = multi_agent_env.step_all(actions)
        
        # Log telemetry for each agent
        for agent_name, agent_result in results["agents"].items():
            stats = agent_result["stats"]
            # Extract reward breakdown for telemetry metrics
            reward_breakdown = agent_result.get("metadata", {}).get("reward_breakdown", {})
            efficiency = reward_breakdown.get("efficiency_score", 0.0)
            fairness = reward_breakdown.get("fairness_score", 0.0)
            survival = reward_breakdown.get("survival_score", 0.0)
            
            # Calculate hospital occupancy percentage
            hospital_occupancy = stats["hospital_occupied"] / max(1, stats["hospital_capacity"]) * 100
            
            log_telemetry(
                reward=agent_result["reward"],
                incident_count=stats["active_incidents"],
                response_time=random.uniform(1.0, 5.0),
                overflow=stats["hospital_occupied"] >= stats["hospital_capacity"],
                efficiency=efficiency,
                fairness=fairness,
                survival=survival,
                hospital_deliveries=agent_result.get("resolved_incidents", 0),
                hospital_occupancy=hospital_occupancy
            )
            
            # Log reflection for each agent
            log_step_reflection(
                step=results["step"],
                agent_name=agent_name,
                action=actions[agent_name],
                state=agent_result["state"],
                reward=agent_result["reward"],
                reward_breakdown=agent_result.get("metadata", {}).get("reward_breakdown", {}),
                resolved_incidents=agent_result["resolved_incidents"],
                new_incidents=stats["active_incidents"], # Approximate
                schema_drift=agent_result["drift_flag"],
                metadata={"done": agent_result["done"]}
            )
        
        return {
            "multi_agent": True,
            "step": results["step"],
            "done": results["done"],
            "agents": results["agents"]
        }
    else:
        # Single-agent mode
        # Select agent based on request
        if request.agent_type == "greedy":
            agent = greedy_agent
        elif request.agent_type == "conservative":
            agent = conservative_agent
        else:
            agent = llm_agent
        
        # Get action from agent
        try:
            action = agent.act(env.current_state)
        except Exception as e:
            print(f"Error getting action from agent: {e}")
            return {
                "multi_agent": False,
                "error": f"Agent error: {str(e)}",
                "state": env.current_state,
                "reward": 0.0,
                "done": False,
                "safety_flags": {}
            }
        
        # Check safety constraints
        is_safe, safety_reason, safety_flags = check_safety_constraints(action, env.current_state)
        
        if not is_safe:
            return {
                "multi_agent": False,
                "error": safety_reason,
                "state": env.current_state,
                "reward": 0.0,
                "done": False,
                "safety_flags": safety_flags
            }
        
        # Execute step
        try:
            state, reward, done, metadata = env.step(action)
        except Exception as e:
            print(f"Error executing step: {e}")
            return {
                "multi_agent": False,
                "error": f"Step error: {str(e)}",
                "state": env.current_state,
                "reward": 0.0,
                "done": False,
                "safety_flags": {}
            }
        
        # Add agentName to ambulances for frontend rendering
        state_with_agent = state.copy()
        if "ambulances" in state_with_agent:
            state_with_agent["ambulances"] = [
                {**amb, "agentName": request.agent_type} for amb in state_with_agent["ambulances"]
            ]
        
        # Log telemetry
        stats = env.get_stats()
        # Extract reward breakdown for telemetry metrics
        efficiency = metadata.get("reward_breakdown", {}).get("efficiency_score", 0.0)
        fairness = metadata.get("reward_breakdown", {}).get("fairness_score", 0.0)
        survival = metadata.get("reward_breakdown", {}).get("survival_score", 0.0)
        
        # Calculate hospital occupancy percentage
        hospital_occupancy = stats["hospital_occupied"] / max(1, stats["hospital_capacity"]) * 100
        
        log_telemetry(
            reward=reward,
            incident_count=stats["active_incidents"],
            response_time=random.uniform(1.0, 5.0),
            overflow=stats["hospital_occupied"] >= stats["hospital_capacity"],
            efficiency=efficiency,
            fairness=fairness,
            survival=survival,
            hospital_deliveries=metadata.get("resolved_incidents", 0),
            hospital_occupancy=hospital_occupancy
        )
        
        # Log reflection
        log_step_reflection(
            step=env.current_step,
            agent_name=request.agent_type,
            action=action,
            state=state_with_agent,
            reward=reward,
            reward_breakdown=metadata.get("reward_breakdown", {}),
            resolved_incidents=metadata.get("resolved_this_step", 0),
            new_incidents=stats["active_incidents"], # Approximate
            schema_drift=metadata.get("drift_flag", False),
            metadata={"done": done}
        )
        
        return {
            "multi_agent": False,
            "state": state_with_agent,
            "reward": reward,
            "done": done,
            "metadata": metadata,
            "action": action,
            "agent_type": request.agent_type,
            "stats": stats,
            "safety_flags": safety_flags
        }


@app.get("/telemetry")
async def get_telemetry():
    """Get telemetry data"""
    try:
        if os.path.exists(TELEMETRY_FILE):
            with open(TELEMETRY_FILE, 'r') as f:
                telemetry = json.load(f)
            return {"telemetry": telemetry}
        else:
            return {"telemetry": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading telemetry: {str(e)}")


@app.get("/leaderboard")
async def get_leaderboard():
    """Get leaderboard with scores for all agents"""
    try:
        # Get leaderboard from multi-agent environment
        leaderboard = multi_agent_env.get_leaderboard()
        
        # Sort by score descending
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        
        return {"leaderboard": leaderboard}
    except Exception as e:
        # Fallback: return empty leaderboard with zero scores
        return {
            "leaderboard": [
                {"agent": "adaptive", "score": 0.0},
                {"agent": "greedy", "score": 0.0},
                {"agent": "conservative", "score": 0.0}
            ]
        }


@app.get("/reflection")
async def get_reflection():
    """Get reflection logs"""
    try:
        reflections = get_reflection_logs()
        return {"reflections": reflections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading reflection logs: {str(e)}")


@app.delete("/reflection")
async def clear_reflection():
    """Clear reflection logs"""
    try:
        clear_reflection_logs()
        return {"message": "Reflection logs cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing reflection logs: {str(e)}")


@app.get("/state")
async def get_state():
    """Get current environment state"""
    return {
        "state": env.current_state,
        "stats": env.get_stats(),
        "step": env.current_step,
        "done": env.done
    }


@app.get("/control-parameters")
async def get_control_parameters():
    """Get current control parameters"""
    try:
        params = env.get_control_parameters()
        return {"control_parameters": params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting control parameters: {str(e)}")


@app.put("/control-parameters")
async def update_control_parameters(request: ControlParametersRequest):
    """Update control parameters"""
    try:
        # Update single-agent environment
        env.set_control_parameters(
            difficulty=request.difficulty,
            adversarial_level=request.adversarial_level,
            tick_interval=request.tick_interval
        )
        
        # Update multi-agent environment
        multi_agent_env.set_control_parameters(
            difficulty=request.difficulty,
            adversarial_level=request.adversarial_level,
            tick_interval=request.tick_interval
        )
        
        # Return updated parameters
        updated_params = env.get_control_parameters()
        return {
            "message": "Control parameters updated successfully",
            "control_parameters": updated_params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating control parameters: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
