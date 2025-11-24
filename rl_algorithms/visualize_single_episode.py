import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import time

# Establish path to SUMO
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration for GUI visualization
Sumo_config = [
    'sumo',  # Use GUI instead of headless
    '-c', 'config/ideal.sumocfg',
    '--start',  # Auto-start simulation
    '--delay', '100',  # Delay in ms between steps (100 = 10 steps/sec)
    '--lateral-resolution', '0',
    '--quit-on-end'  # Close GUI when done
]

# Define Custom SUMO Environment (same as training)
class SumoEnv(gym.Env):
    def __init__(self, config):
        super(SumoEnv, self).__init__()
        self.config = config
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)
        self.min_green_steps = 10
        self.step_count = 0
        self.max_steps = 1800
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0

    def reset(self, seed=None, **kwargs):
        if traci.isLoaded():
            traci.close()
            time.sleep(1)  # Give GUI time to close
        traci.start(self.config)
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.total_queue = 0.0
        self.last_switch_step = -self.min_green_steps
        self.current_simulation_step = 0
        state = self._get_state()
        info = {}
        return state, info

    def step(self, action):
        self.current_simulation_step = self.step_count
        self._apply_action(action)
        traci.simulationStep()
        
        new_state = self._get_state()
        reward = self._get_reward(new_state)
        self.cumulative_reward += reward
        self.total_queue += sum(new_state[:-1])
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        info = {}
        if done:
            avg_queue = self.total_queue / self.step_count if self.step_count > 0 else 0
            info = {
                "cumulative_reward": self.cumulative_reward,
                "avg_queue_length": avg_queue
            }
            print(f"\n=== Episode Summary ===")
            print(f"Cumulative Reward: {info['cumulative_reward']:.2f}")
            print(f"Average Queue Length: {info['avg_queue_length']:.2f}")
            print(f"Total Steps: {self.step_count}")

        # Print live stats every 100 steps
        if self.step_count % 100 == 0:
            current_avg_queue = self.total_queue / self.step_count
            print(f"Step {self.step_count}/{self.max_steps} | "
                  f"Reward: {self.cumulative_reward:.1f} | "
                  f"Avg Queue: {current_avg_queue:.2f} | "
                  f"Phase: {int(new_state[-1])}")

        return new_state, reward, terminated, truncated, info

    def _get_state(self):
        detector_EB_0 = "e2_2"
        detector_SB_0 = "e2_3"
        detector_SB_1 = "e2_4"
        detector_WB_0 = "e2_6"
        detector_NB_0 = "e2_11"
        detector_NB_1 = "e2_9"
        traffic_light_id = "41896158"

        q_EB_0 = self._get_queue_length(detector_EB_0)
        q_SB_0 = self._get_queue_length(detector_SB_0)
        q_SB_1 = self._get_queue_length(detector_SB_1)
        q_WB_0 = self._get_queue_length(detector_WB_0)
        q_NB_0 = self._get_queue_length(detector_NB_0)
        q_NB_1 = self._get_queue_length(detector_NB_1)
        current_phase = self._get_current_phase(traffic_light_id)

        return np.array([q_EB_0, q_SB_0, q_SB_1, q_WB_0, q_NB_0, q_NB_1, current_phase], dtype=np.float32)

    def _apply_action(self, action, tls_id="41896158"):
        if action == 0:
            return
        elif action == 1:
            if self.current_simulation_step - self.last_switch_step >= self.min_green_steps:
                current_phase = self._get_current_phase(tls_id)
                try:
                    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                    num_phases = len(program.phases)
                    if num_phases == 0:
                        return
                    next_phase = (current_phase + 1) % num_phases
                    traci.trafficlight.setPhase(tls_id, next_phase)
                    self.last_switch_step = self.current_simulation_step
                    print(f"  → Traffic light switched to phase {next_phase}")
                except traci.exceptions.TraCIException:
                    pass

    def _get_reward(self, state):
        total_queue = sum(state[:-1])
        return -float(total_queue)

    def _get_queue_length(self, detector_id):
        try:
            return traci.lanearea.getLastStepVehicleNumber(detector_id)
        except:
            return 0.0

    def _get_current_phase(self, tls_id):
        try:
            return traci.trafficlight.getPhase(tls_id)
        except:
            return 0

    def close(self):
        if traci.isLoaded():
            traci.close()

    def render(self, mode="human"):
        pass

# Main visualization
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚦 SUMO-GUI Visualization - Single Episode")
    print("="*60)
    
    # Choose which algorithm to visualize
    print("\nSelect algorithm to visualize:")
    print("1. Fixed-Time (Baseline)")
    print("2. DQN (Deep Q-Network)")
    print("3. PPO (Proximal Policy Optimization)")
    print("4. Q-Learning")
    print("5. Random Actions (for testing)")
    
    choice = input("\nEnter choice (1-5) [default=2]: ").strip() or "2"
    
    env = SumoEnv(Sumo_config)
    
    # Load trained model if applicable
    model = None
    if choice == "2":
        try:
            from stable_baselines3 import DQN
            model = DQN.load("dqn_sumo")
            print("✓ Loaded trained DQN model")
        except:
            print("✗ Could not load DQN model, using random actions")
    elif choice == "3":
        try:
            from stable_baselines3 import PPO
            model = PPO.load("ppo_sumo")
            print("✓ Loaded trained PPO model")
        except:
            print("✗ Could not load PPO model, using random actions")
    elif choice == "4":
        try:
            import pickle
            with open("q_table.pkl", "rb") as f:
                Q_table = pickle.load(f)
            print("✓ Loaded trained Q-table")
            
            def discretize_state(state):
                bins = [0, 5, 10, np.inf]
                digitized = []
                for i in range(6):
                    digitized.append(np.digitize(state[i], bins) - 1)
                digitized.append(int(state[6]))
                return tuple(digitized)
        except:
            print("✗ Could not load Q-table, using random actions")
            Q_table = None
    
    print("\n" + "="*60)
    print("Starting visualization... Watch SUMO-GUI window!")
    print("="*60 + "\n")
    
    # Run single episode
    state, _ = env.reset()
    done = False
    
    while not done:
        # Select action based on algorithm
        if choice == "1":
            # Fixed-time: just let SUMO's built-in timing handle it
            action = 0
        elif choice in ["2", "3"] and model is not None:
            # Use trained RL model
            action, _ = model.predict(state, deterministic=True)
        elif choice == "4" and Q_table is not None:
            # Use Q-table
            state_discrete = discretize_state(state)
            if state_discrete in Q_table:
                action = int(np.argmax(Q_table[state_discrete]))
            else:
                action = 0
        else:
            # Random actions
            action = env.action_space.sample()
        
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    
    env.close()
    input("\nPress Enter to exit...")