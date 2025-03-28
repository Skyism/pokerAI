from agents.agent import Agent
from gym_env import PokerEnv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import os
import time
from gym_env import WrappedEval

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Deep Q-Network model with fixed dimension handling
class PokerDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PokerDQN, self).__init__()
        
        # Feature extraction layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Dueling DQN architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Split into value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Check if input is a single sample or a batch
        if len(advantage.shape) == 1:
            # Single sample case - no batch dimension
            q_values = value + (advantage - advantage.mean())
        else:
            # Batch case - take mean along action dimension (dim=1)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class PlayerAgent(Agent):
    def __name__(self):
        return "RLAgent"

    def __init__(self, stream: bool = False, learning_rate=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        super().__init__(stream)
        self.evaluator = WrappedEval()
        
        # Initialize tracking variables
        self.hand_number = 0
        self.previous_street = -1
        self.opponent_street_actions = {0: [], 1: [], 2: [], 3: []}
        self.opp_acts = []
        self.current_bluff_amount = 0
        self.hands_played = 0
        self.bluffs_attempted = 0
        self.successful_bluffs = 0
        
        # State representation size (features we'll extract from the observation)
        self.state_size = 46  # Adjusted based on actual features
        
        # Action mapping
        # For simplicity, we'll discretize the action space:
        # - fold (1 action)
        # - check/call (1 action)
        # - raise small, medium, large (3 actions)
        # - discard card 0, 1, or none (-1) (3 actions)
        self.action_size = 8
        
        # RL hyperparameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = 64
        self.update_target_every = 100
        self.memory_size = 10000
        
        # Initialize DQN networks
        self.policy_net = PokerDQN(self.state_size, self.action_size)
        self.target_net = PokerDQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayMemory(self.memory_size)
        
        # Tracking variables for training
        self.steps_done = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.last_state = None
        self.last_action = None
        
        # Create model directory for saving
        self.model_dir = "rl_poker_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load model if exists
        self.model_path = os.path.join(self.model_dir, "poker_dqn.pt")
        if os.path.exists(self.model_path):
            self.load_model()
            self.logger.info("Loaded existing model")
    
    def extract_features(self, obs):
        """Extract relevant features from the observation to create a state representation"""
        
        # Extract basic information
        street = obs["street"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        continue_cost = obs["opp_bet"] - obs["my_bet"]
        pot_odds = continue_cost / (pot_size + continue_cost) if continue_cost > 0 else 0
        
        # Extract card information
        my_cards = [int(card) for card in obs["my_cards"]]
        community_cards = [card for card in obs["community_cards"] if card != -1]
        
        # Calculate equity through Monte Carlo simulation
        equity = self.calculate_equity(obs)
        
        # Encode cards (one-hot encoding for ranks and suits)
        # 9 possible ranks (2-9, A) * 3 suits = 27 possible cards
        my_cards_encoding = np.zeros(27)
        for card in my_cards:
            if 0 <= card < 27:  # Sanity check
                my_cards_encoding[card] = 1
                
        community_encoding = np.zeros(27)
        for card in community_cards:
            if 0 <= card < 27:  # Sanity check
                community_encoding[card] = 1
        
        # Encode opponent's recent actions (last 3)
        opp_action_encoding = np.zeros(4)  # FOLD, CHECK, CALL, RAISE
        recent_opp_acts = self.opp_acts[-3:] if len(self.opp_acts) >= 3 else self.opp_acts
        for action in recent_opp_acts:
            if action == "FOLD":
                opp_action_encoding[0] += 1
            elif action == "CHECK":
                opp_action_encoding[1] += 1
            elif action == "CALL":
                opp_action_encoding[2] += 1
            elif action.startswith("RAISE"):
                opp_action_encoding[3] += 1
        
        # Normalize opp action encoding
        if len(recent_opp_acts) > 0:
            opp_action_encoding = opp_action_encoding / len(recent_opp_acts)
            
        # Features about current game state
        game_state = np.array([
            street / 3.0,  # Normalize street
            pot_size / 100.0,  # Normalize pot size
            continue_cost / 50.0 if continue_cost > 0 else 0,  # Cost to continue
            pot_odds if not np.isnan(pot_odds) else 0,  # Pot odds
            equity,  # Hand equity
            len(community_cards) / 5.0,  # Number of community cards (normalized)
            1.0 if obs["opp_last_action"] == "FOLD" else 0.0,
            1.0 if obs["opp_last_action"] == "CHECK" else 0.0,
            1.0 if obs["opp_last_action"] == "CALL" else 0.0,
            1.0 if obs["opp_last_action"] and obs["opp_last_action"].startswith("RAISE") else 0.0,
            obs["min_raise"] / 50.0 if obs["min_raise"] > 0 else 0,  # Normalized min raise
            obs["max_raise"] / 100.0 if obs["max_raise"] > 0 else 0,  # Normalized max raise
        ])
        
        # Combine all features
        features = np.concatenate([
            my_cards_encoding,
            community_encoding,
            opp_action_encoding,
            game_state
        ])
        
        # Ensure we have the expected state size
        if len(features) < self.state_size:
            # Pad with zeros if needed
            features = np.pad(features, (0, self.state_size - len(features)))
        elif len(features) > self.state_size:
            # Truncate if needed
            features = features[:self.state_size]
            
        return torch.FloatTensor(features)
    
    def calculate_equity(self, observation):
        """Calculate hand equity using Monte Carlo simulation"""
        my_cards = [int(card) for card in observation["my_cards"]]
        community_cards = [card for card in observation["community_cards"] if card != -1]
        opp_discarded_card = [observation["opp_discarded_card"]] if observation["opp_discarded_card"] != -1 else []
        opp_drawn_card = [observation["opp_drawn_card"]] if observation["opp_drawn_card"] != -1 else []

        # Calculate equity through Monte Carlo simulation
        shown_cards = my_cards + community_cards + opp_discarded_card + opp_drawn_card
        non_shown_cards = [i for i in range(27) if i not in shown_cards]

        def evaluate_hand(cards):
            my_cards, opp_cards, community_cards = cards
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = self.evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = self.evaluator.evaluate(opp_cards, community_cards)
            return my_hand_rank < opp_hand_rank

        # Run Monte Carlo simulation
        simulations = [500, 500, 500, 500]  # Reduced for RL training speed
        num_simulations = simulations[observation["street"]]
        
        # Skip if not enough cards available
        if len(non_shown_cards) < (7 - len(community_cards) - len(opp_drawn_card)):
            return 0.5  # Default to 50% equity
            
        wins = sum(
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
        )
        equity = wins / num_simulations
        return equity
    
    def map_action_to_env_action(self, action_idx, obs):
        """Map the RL action index to a valid poker environment action"""
        valid_actions = obs["valid_actions"]
        
        # Default values
        action_type = None
        raise_amount = 0
        card_to_discard = -1
        
        # Fold (if valid)
        if action_idx == 0 and valid_actions[action_types.FOLD.value]:
            action_type = action_types.FOLD.value
            
        # Check/Call (if valid)
        elif action_idx == 1 and (valid_actions[action_types.CHECK.value] or valid_actions[action_types.CALL.value]):
            if valid_actions[action_types.CHECK.value]:
                action_type = action_types.CHECK.value
            else:
                action_type = action_types.CALL.value
                
        # Raise small (25% of pot)
        elif action_idx == 2 and valid_actions[action_types.RAISE.value]:
            action_type = action_types.RAISE.value
            pot_size = obs["my_bet"] + obs["opp_bet"]
            raise_amount = min(int(pot_size * 0.25), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            
        # Raise medium (50% of pot)
        elif action_idx == 3 and valid_actions[action_types.RAISE.value]:
            action_type = action_types.RAISE.value
            pot_size = obs["my_bet"] + obs["opp_bet"]
            raise_amount = min(int(pot_size * 0.5), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            
        # Raise large (100% of pot)
        elif action_idx == 4 and valid_actions[action_types.RAISE.value]:
            action_type = action_types.RAISE.value
            pot_size = obs["my_bet"] + obs["opp_bet"]
            raise_amount = min(int(pot_size * 1.0), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            
        # Discard card 0
        elif action_idx == 5 and valid_actions[action_types.DISCARD.value]:
            action_type = action_types.DISCARD.value
            card_to_discard = 0
            
        # Discard card 1
        elif action_idx == 6 and valid_actions[action_types.DISCARD.value]:
            action_type = action_types.DISCARD.value
            card_to_discard = 1
            
        # Don't discard (but valid to discard)
        elif action_idx == 7 and valid_actions[action_types.DISCARD.value]:
            # Find the best action other than discard
            if valid_actions[action_types.CHECK.value]:
                action_type = action_types.CHECK.value
            elif valid_actions[action_types.CALL.value]:
                action_type = action_types.CALL.value
            elif valid_actions[action_types.RAISE.value]:
                action_type = action_types.RAISE.value
                pot_size = obs["my_bet"] + obs["opp_bet"]
                raise_amount = min(int(pot_size * 0.5), obs["max_raise"])
                raise_amount = max(raise_amount, obs["min_raise"])
            elif valid_actions[action_types.FOLD.value]:
                action_type = action_types.FOLD.value
        
        # If the chosen action is not valid, find a fallback
        if action_type is None:
            if valid_actions[action_types.CHECK.value]:
                action_type = action_types.CHECK.value
            elif valid_actions[action_types.CALL.value]:
                action_type = action_types.CALL.value
            elif valid_actions[action_types.FOLD.value]:
                action_type = action_types.FOLD.value
            elif valid_actions[action_types.RAISE.value]:
                action_type = action_types.RAISE.value
                raise_amount = obs["min_raise"]
            elif valid_actions[action_types.DISCARD.value]:
                # If we must discard, choose card with lower rank
                action_type = action_types.DISCARD.value
                my_cards = [int(card) for card in obs["my_cards"]]
                card_0_rank = my_cards[0] % 9
                card_1_rank = my_cards[1] % 9
                card_to_discard = 0 if card_0_rank < card_1_rank else 1
        
        return action_type, raise_amount, card_to_discard
    
    def select_action(self, state, obs):
        """Select action using epsilon-greedy policy"""
        # Determine if we're exploring or exploiting
        if random.random() < self.epsilon:
            # Random action (exploration)
            action_idx = random.randint(0, self.action_size - 1)
            self.logger.info(f"[HAND #{self.hand_number}] [RL] Exploring with random action {action_idx}")
        else:
            # Use policy network (exploitation)
            with torch.no_grad():
                # Always add batch dimension for consistent processing
                batched_state = state.unsqueeze(0)
                q_values = self.policy_net(batched_state)
                action_idx = q_values.squeeze(0).max(0)[1].item()
                q_value = q_values.squeeze(0).max(0)[0].item()
                self.logger.info(f"[HAND #{self.hand_number}] [RL] Exploiting with policy action {action_idx}, Q-value: {q_value:.4f}")
        
        # Map to environment action
        return action_idx, self.map_action_to_env_action(action_idx, obs)
    
    def optimize_model(self):
        """Train the model using a batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create mask for non-final states
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        
        # Handle case where all next_states are None
        if non_final_mask.sum() == 0:
            return
            
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        
        # Convert other batch data to tensors
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float)
        done_batch = torch.tensor(batch.done, dtype=torch.bool)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size)
        
        # Special handling for Double DQN to ensure dimensions are correct
        if non_final_mask.sum() > 0:  # Only if we have at least one non-final state
            with torch.no_grad():
                # Double DQN: use policy_net to select action, target_net to evaluate it
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
        
        # Don't include future reward for terminal states
        next_state_values[done_batch] = 0.0
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network by copying weights from policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def save_model(self):
        """Save the current model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, self.model_path)
        self.logger.info(f"Model saved to {self.model_path}")
        
    def load_model(self):
        """Load a saved model"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            return True
        return False
    
    def act(self, observation, reward, terminated, truncated, info):
        """Choose an action based on the current observation"""
        # Track new hand when we're at preflop
        if observation["street"] == 0 and (not hasattr(self, 'last_street') or self.last_street != 0):
            self.hand_number += 1
            self.opp_acts = []
            self.logger.info(f"[HAND #{self.hand_number}] Starting new hand")
            
            # Save episode reward from previous hand if applicable
            if hasattr(self, 'current_episode_reward') and self.current_episode_reward != 0:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
        
        # Save current street for next time
        self.last_street = observation["street"]
        
        # Track opponent actions
        if observation["opp_last_action"] is not None:
            self.opp_acts.append(observation["opp_last_action"])
        
        # Extract state features
        state = self.extract_features(observation)
        
        # Select action using policy
        action_idx, action = self.select_action(state, observation)
        
        # Store transition in memory if we have a previous state
        if self.last_state is not None:
            # Store the transition
            self.memory.push(
                self.last_state,
                self.last_action,
                state,  # Next state
                reward,  # Reward
                terminated or truncated  # Done flag
            )
            
            # Update current episode reward
            self.current_episode_reward += reward
            
            # Train the model
            if self.steps_done % 4 == 0:  # Train every few steps
                loss = self.optimize_model()
                if loss is not None:
                    self.logger.debug(f"[HAND #{self.hand_number}] [RL] Training loss: {loss:.4f}")
            
            # Update target network periodically
            if self.steps_done % self.update_target_every == 0:
                self.update_target_network()
                self.logger.info(f"[HAND #{self.hand_number}] [RL] Target network updated")
            
            # Decay exploration rate
            if self.steps_done % 100 == 0:
                old_epsilon = self.epsilon
                self.decay_epsilon()
                self.logger.info(f"[HAND #{self.hand_number}] [RL] Epsilon decayed from {old_epsilon:.4f} to {self.epsilon:.4f}")
            
            # Save model periodically
            if self.steps_done % 1000 == 0:
                self.save_model()
        
        # Update step counter
        self.steps_done += 1
        
        # Store current state and action for next transition
        self.last_state = state
        self.last_action = action_idx
        
        # Log the chosen action
        action_type, raise_amount, card_to_discard = action
        action_names = ["FOLD", "CHECK/CALL", "RAISE-SMALL", "RAISE-MEDIUM", "RAISE-LARGE", "DISCARD-0", "DISCARD-1", "NO-DISCARD"]
        self.logger.info(f"[HAND #{self.hand_number}] [RL] Selected action: {action_names[action_idx]} -> ({action_type}, {raise_amount}, {card_to_discard})")
        
        return action
    
    def observe(self, observation, reward, terminated, truncated, info):
        """Process observations after an action is taken"""
        # Track bluff success/failure when hand completes
        if terminated:
            # Update hands played counter
            self.hands_played += 1
            
            # If this was the last transition of an episode, reset last_state
            if self.last_state is not None:
                # Add the final transition
                self.memory.push(
                    self.last_state,
                    self.last_action,
                    None,  # No next state at terminal
                    reward,
                    True  # Terminal state
                )
                
                # Update episode reward
                self.current_episode_reward += reward
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                
                # Reset for next episode
                self.last_state = None
                self.last_action = None
            
            # Log end of episode stats
            self.logger.info(f"[HAND #{self.hand_number}] [RL] Episode completed, reward: {reward}, total steps: {self.steps_done}")
            
            # Log average reward over last 100 episodes if available
            if len(self.episode_rewards) > 0:
                avg_reward = sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))
                self.logger.info(f"[HAND #{self.hand_number}] [RL] Avg reward (last 100 episodes): {avg_reward:.2f}")
            
            # Advanced periodic logging
            if self.hands_played % 100 == 0:
                self.logger.info(f"[RL] Performance after {self.hands_played} hands:")
                self.logger.info(f"[RL] - Epsilon: {self.epsilon:.4f}")
                self.logger.info(f"[RL] - Memory size: {len(self.memory)}")
                if len(self.episode_rewards) >= 100:
                    self.logger.info(f"[RL] - Avg reward (last 100): {sum(self.episode_rewards[-100:]) / 100:.2f}")
                    self.logger.info(f"[RL] - Max reward (last 100): {max(self.episode_rewards[-100:]):.2f}")
                self.save_model()