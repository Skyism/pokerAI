import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our environment and agent classes
from gym_env import PokerEnv
from agents.agent import Agent
from agents.test_agents import AllInAgent, CallingStationAgent, FoldAgent, RandomAgent

# Import our PlayerAgent with the PolicyNetwork
# Assumed to be in the same directory as this training script
from player import PlayerAgent, PolicyNetwork

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("poker_training")

class TrainingAgent(PlayerAgent):
    """
    Extension of PlayerAgent for training purposes.
    Adds methods to store experiences and update policy.
    """
    def __init__(self, learning_rate=1e-4, entropy_coef=0.01):
        super().__init__(stream=False)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.entropy_coef = entropy_coef
        self.gamma = 0.99  # Discount factor
        self.trajectory = []  # To store (state, action, log_prob, reward)
        self.policy_net.train()  # Set to training mode
        
    def select_action_for_training(self, state, valid_actions, min_raise, max_raise):
        """
        Select action during training, returning additional info for learning.
        """
        # Get logits from policy network
        action_type_logits, raise_logits, discard_logits = self.policy_net(state)
        
        # Apply mask to filter out invalid actions
        mask = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
        masked_logits = action_type_logits.clone()
        masked_logits[0, ~mask] = float('-inf')
        
        # Sample actions using softmax
        action_probs = torch.nn.functional.softmax(masked_logits, dim=1)
        raise_probs = torch.nn.functional.softmax(raise_logits, dim=1)
        discard_probs = torch.nn.functional.softmax(discard_logits, dim=1)
        
        # Sample from distributions
        action_dist = torch.distributions.Categorical(probs=action_probs)
        raise_dist = torch.distributions.Categorical(probs=raise_probs)
        discard_dist = torch.distributions.Categorical(probs=discard_probs)
        
        action_type = action_dist.sample().item()
        raise_amount = raise_dist.sample().item() + 1
        discard_action = discard_dist.sample().item() - 1
        
        # Calculate log probabilities
        log_prob_action = action_dist.log_prob(torch.tensor([action_type]).to(self.device))
        log_prob_raise = raise_dist.log_prob(torch.tensor([raise_amount - 1]).to(self.device))
        log_prob_discard = discard_dist.log_prob(torch.tensor([discard_action + 1]).to(self.device))
        
        # Total log probability
        log_prob = log_prob_action + log_prob_raise + log_prob_discard
        
        # Calculate entropy (for exploration)
        entropy = action_dist.entropy().mean() + raise_dist.entropy().mean() + discard_dist.entropy().mean()
        
        # Process and validate the actions
        if action_type == PokerEnv.ActionType.RAISE.value:
            raise_amount = max(min(raise_amount, max_raise), min_raise)
        else:
            raise_amount = 0
            
        if action_type == PokerEnv.ActionType.DISCARD.value:
            if discard_action < 0:
                discard_action = 0
        else:
            discard_action = -1
            
        return (action_type, raise_amount, discard_action), log_prob, entropy

    def store_transition(self, state, action, log_prob, reward):
        """Store transition in the trajectory."""
        self.trajectory.append((state, action, log_prob, reward))
    
    def update_policy(self):
        """Update policy using REINFORCE with baseline algorithm."""
        if len(self.trajectory) == 0:
            return 0.0
        
        # Collect rewards and calculate returns
        rewards = [item[3] for item in self.trajectory]
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        policy_loss = 0
        for (state, _, log_prob, _), R in zip(self.trajectory, returns):
            policy_loss += -log_prob * R  # REINFORCE loss
        
        # Add entropy term to encourage exploration
        entropy_loss = -self.entropy_coef * sum(item[2] for item in zip(self.trajectory))[0]
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Clear trajectory
        self.trajectory = []
        
        return loss.item()

    def act(self, observation, reward, terminated, truncated, info):
        """Override act method to use training-specific action selection."""
        equity = self.compute_equity(observation)
        state = self.preprocess_observation(observation, equity).to(self.device)
        valid_actions_tensor = torch.tensor(observation["valid_actions"], dtype=torch.float32).to(self.device)
        min_raise_val = observation["min_raise"]
        max_raise_val = observation["max_raise"]
        
        action, log_prob, entropy = self.select_action_for_training(
            state, valid_actions_tensor, min_raise_val, max_raise_val
        )
        
        # Store the transition for learning
        if reward != 0 or terminated:  # Only store meaningful rewards
            self.store_transition(state, action, log_prob, reward)
        
        return action

def create_opponent_pool():
    """Create a pool of opponent agents with different strategies."""
    return [
        AllInAgent,        # Very aggressive, always goes all-in when possible
        CallingStationAgent,  # Passive, always calls but never raises
        FoldAgent,         # Very tight, always folds
        RandomAgent,       # Makes random decisions
        # Add more agents here as needed
    ]

def evaluate_agent(agent, opponents, num_hands=100):
    """Evaluate agent performance against a set of opponents."""
    results = {}
    
    for opponent_class in opponents:
        opponent_name = opponent_class.__name__
        total_reward = 0
        
        for i in range(num_hands):
            env = PokerEnv()
            opponent = opponent_class()
            
            # Determine who goes first (alternate)
            small_blind_player = i % 2
            
            obs, _ = env.reset(options={"small_blind_player": small_blind_player})
            terminated = False
            
            # Main game loop
            while not terminated:
                acting_agent = obs[0]["acting_agent"]
                
                if acting_agent == 0:  # Our agent's turn
                    action = agent.act(obs[0], 0, False, False, {})
                else:  # Opponent's turn
                    action = opponent.act(obs[1], 0, False, False, {})
                
                # Step environment
                obs, reward, terminated, _, _ = env.step(action)
            
            # Track results
            total_reward += reward[0]  # Our agent's reward
        
        # Calculate average reward against this opponent
        avg_reward = total_reward / num_hands
        results[opponent_name] = avg_reward
        
    return results

def train_agent(
    num_epochs=50,
    hands_per_epoch=200,
    learning_rate=1e-4,
    entropy_coef=0.01,
    eval_frequency=5,
    save_dir="models",
    checkpoint_frequency=10
):
    """
    Train the poker agent using reinforcement learning.
    
    Args:
        num_epochs: Number of training epochs
        hands_per_epoch: Number of hands to play per epoch
        learning_rate: Learning rate for optimizer
        entropy_coef: Coefficient for entropy bonus (encourages exploration)
        eval_frequency: How often to evaluate the agent
        save_dir: Directory to save model checkpoints
        checkpoint_frequency: How often to save model checkpoints
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create our training agent
    agent = TrainingAgent(learning_rate=learning_rate, entropy_coef=entropy_coef)
    
    # Create opponent pool
    opponent_pool = create_opponent_pool()
    
    # Tracking metrics
    reward_history = []
    loss_history = []
    eval_results = {}
    
    # Create progress bar
    progress_bar = tqdm(total=num_epochs, desc="Training Progress")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []
        
        # Play hands against different opponents
        for _ in range(hands_per_epoch):
            # Randomly select an opponent
            opponent_class = random.choice(opponent_pool)
            opponent = opponent_class()
            
            # Initialize environment
            env = PokerEnv()
            obs, _ = env.reset()
            terminated = False
            hand_reward = 0
            
            # Single hand
            while not terminated:
                acting_agent = obs[0]["acting_agent"]
                
                if acting_agent == 0:  # Our agent's turn
                    action = agent.act(obs[0], 0, False, False, {})
                else:  # Opponent's turn
                    action = opponent.act(obs[1], 0, False, False, {})
                
                # Step environment
                obs, reward, terminated, _, _ = env.step(action)
                
                # Track rewards
                if terminated:
                    hand_reward = reward[0]  # Our agent's reward
            
            # Update policy after each hand
            loss = agent.update_policy()
            
            epoch_rewards.append(hand_reward)
            if loss is not None:
                epoch_losses.append(loss)
        
        # Record metrics
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        reward_history.append(avg_reward)
        loss_history.append(avg_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
        
        # Evaluate agent
        if (epoch + 1) % eval_frequency == 0:
            logger.info(f"Evaluating agent after epoch {epoch+1}...")
            results = evaluate_agent(agent, opponent_pool)
            eval_results[epoch + 1] = results
            
            # Log evaluation results
            for opponent, reward in results.items():
                logger.info(f"  vs {opponent}: {reward:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"poker_model_epoch_{epoch+1}.pth")
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Update progress bar
        progress_bar.update(1)
    
    # Save final model
    final_model_path = os.path.join(save_dir, "model_weights.pth")
    torch.save(agent.policy_net.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Close progress bar
    progress_bar.close()
    
    # Plot training metrics
    plot_training_metrics(reward_history, loss_history, eval_results, save_dir)
    
    return agent

def plot_training_metrics(rewards, losses, eval_results, save_dir):
    """Plot and save training metrics."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot rewards
    ax1.plot(rewards, 'b-')
    ax1.set_title('Average Reward per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot losses
    ax2.plot(losses, 'r-')
    ax2.set_title('Average Loss per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    
    # If we have evaluation results, plot them
    if eval_results:
        plt.figure(figsize=(10, 6))
        
        # Extract epoch numbers and opponent names
        epochs = sorted(eval_results.keys())
        opponents = list(eval_results[epochs[0]].keys())
        
        # Plot evaluation results for each opponent
        for opponent in opponents:
            rewards = [eval_results[epoch][opponent] for epoch in epochs]
            plt.plot(epochs, rewards, marker='o', label=opponent)
        
        plt.title('Evaluation Results Against Different Opponents')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'))

def self_play_training(
    num_epochs=50,
    hands_per_epoch=200,
    learning_rate=1e-4,
    save_dir="models",
    checkpoint_frequency=10
):
    """
    Train the agent against itself (self-play).
    This can be more effective for learning advanced strategies.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create our training agent
    agent = TrainingAgent(learning_rate=learning_rate)
    
    # Clone the agent for self-play (with shared weights)
    opponent_agent = TrainingAgent()
    opponent_agent.policy_net = agent.policy_net  # Share the same network
    
    # Tracking metrics
    reward_history = []
    loss_history = []
    
    # Create progress bar
    progress_bar = tqdm(total=num_epochs, desc="Self-Play Training")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []
        
        # Play hands against self
        for _ in range(hands_per_epoch):
            # Initialize environment
            env = PokerEnv()
            obs, _ = env.reset()
            terminated = False
            
            # Single hand
            while not terminated:
                acting_agent = obs[0]["acting_agent"]
                
                if acting_agent == 0:  # Our agent's turn
                    action = agent.act(obs[0], 0, False, False, {})
                else:  # Opponent's (clone) turn
                    action = opponent_agent.act(obs[1], 0, False, False, {})
                
                # Step environment
                obs, reward, terminated, _, _ = env.step(action)
                
                # Track rewards and update at end of hand
                if terminated:
                    epoch_rewards.append(reward[0])  # Our agent's reward
            
            # Update policy after each hand
            loss = agent.update_policy()
            
            if loss is not None:
                epoch_losses.append(loss)
        
        # Record metrics
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        
        reward_history.append(avg_reward)
        loss_history.append(avg_loss)
        
        logger.info(f"Self-Play Epoch {epoch+1}/{num_epochs} - Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"self_play_model_epoch_{epoch+1}.pth")
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Update progress bar
        progress_bar.update(1)
    
    # Save final model
    final_model_path = os.path.join(save_dir, "model_weights.pth")
    torch.save(agent.policy_net.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Close progress bar
    progress_bar.close()
    
    # Plot training metrics
    plt.figure(figsize=(10, 12))
    
    plt.subplot(2, 1, 1)
    plt.plot(reward_history, 'b-')
    plt.title('Average Reward per Epoch (Self-Play)')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(loss_history, 'r-')
    plt.title('Average Loss per Epoch (Self-Play)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'self_play_metrics.png'))
    
    return agent

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a poker AI using reinforcement learning')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'self_play'], 
                        help='Training mode: standard or self_play')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--hands', type=int, default=200, help='Hands per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--entropy', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Start training
    if args.mode == 'standard':
        agent = train_agent(
            num_epochs=args.epochs, 
            hands_per_epoch=args.hands,
            learning_rate=args.lr,
            entropy_coef=args.entropy,
            save_dir=args.save_dir
        )
    else:
        agent = self_play_training(
            num_epochs=args.epochs,
            hands_per_epoch=args.hands,
            learning_rate=args.lr,
            save_dir=args.save_dir
        )
    
    print(f"Training complete! Final model saved to {os.path.join(args.save_dir, 'model_weights.pth')}")