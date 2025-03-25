from agents.agent import Agent
from gym_env import PokerEnv
import random
import torch
import numpy as np
from treys import Evaluator
import os
import time

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_action_types=5, num_raise_classes=100, num_discard_classes=3):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Action type head
        self.action_type_head = torch.nn.Linear(hidden_dim // 2, num_action_types)
        
        # Raise head
        self.raise_head = torch.nn.Linear(hidden_dim // 2, num_raise_classes)
        
        # Discard head
        self.discard_head = torch.nn.Linear(hidden_dim // 2, num_discard_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        
        return action_type_logits, raise_logits, discard_logits

class PlayerAgent(Agent):
    def __name__(self):
        return "MLPokerAgent"

    def __init__(self, stream: bool = False, player_id: str = None):
        super().__init__(stream, player_id)
        self.evaluator = Evaluator()
        self.hand_counter = 0
        self.time_budget_per_hand = 1.5  # seconds
        
        # Initialize ML model
        try:
            # Try to load a pre-trained model if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            
            # Enhanced input features
            self.input_dim = 21  # Updated from 19 to match actual feature count
            self.policy_net = PolicyNetwork(input_dim=self.input_dim).to(self.device)
            
            # Initialize weights or load pre-trained weights if available
            model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
            if os.path.exists(model_path):
                self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info("Loaded pre-trained model")
            else:
                self.logger.info("Initializing new model")
                # Initialize weights strategically (not random)
                for name, param in self.policy_net.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        torch.nn.init.zeros_(param)
            
            self.use_ml_model = True
        except Exception as e:
            self.logger.warning(f"Could not initialize ML model: {e}")
            self.use_ml_model = False
        
        if self.use_ml_model:
            self.policy_net.eval()  # Set to evaluation mode
        
        # Statistics for opponent modeling
        self.hand_stats = {
            "opponent_fold_frequency": 0.0,
            "opponent_call_frequency": 0.0,
            "opponent_raise_frequency": 0.0,
            "hands_played": 0,
            "last_action": None,
            "last_street": None,
        }
        
        # Cache for Monte Carlo simulations to avoid recomputation
        self.equity_cache = {}

    def compute_equity(self, obs, num_simulations=300):
        """
        Calculate equity (win probability) using Monte Carlo simulation.
        Uses caching for efficiency.
        """
        # Create a cache key from relevant observation parts
        my_cards = tuple(sorted([int(card) for card in obs["my_cards"]]))
        community_cards = tuple(sorted([card for card in obs["community_cards"] if card != -1]))
        opp_discarded = obs["opp_discarded_card"]
        opp_drawn = obs["opp_drawn_card"]
        
        cache_key = (my_cards, community_cards, opp_discarded, opp_drawn)
        
        # Return cached value if available
        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]
        
        # Adaptive simulation count based on street
        street = obs["street"]
        if street == 0:  # Preflop
            num_simulations = min(100, num_simulations)  # Fewer sims on preflop for speed
        elif street == 3:  # River
            num_simulations = min(50, num_simulations)  # Even fewer on river as equity is more certain
        
        # Reduce simulation count if we're running low on time
        # Note: we'd need to track actual time spent per hand for this
        
        opp_discarded_card = [opp_discarded] if opp_discarded != -1 else []
        opp_drawn_card = [opp_drawn] if opp_drawn != -1 else []
        
        # Determine cards that have been shown
        shown_cards = list(my_cards) + list(community_cards) + opp_discarded_card + opp_drawn_card
        non_shown_cards = [i for i in range(27) if i not in shown_cards]
        
        # Check if we have enough cards for simulation
        if len(non_shown_cards) < (7 - len(community_cards) - len(opp_drawn_card)):
            return 0.5  # Not enough cards to run simulation
        
        def evaluate_hand(cards):
            my_cards, opp_cards, community_cards = cards
            my_cards = list(map(int_to_card, my_cards))
            opp_cards = list(map(int_to_card, opp_cards))
            community_cards = list(map(int_to_card, community_cards))
            my_hand_rank = self.evaluator.evaluate(my_cards, community_cards)
            opp_hand_rank = self.evaluator.evaluate(opp_cards, community_cards)
            return my_hand_rank < opp_hand_rank  # Lower is better in treys evaluator
        
        # Run Monte Carlo simulation
        wins = 0
        for _ in range(num_simulations):
            try:
                # Sample remaining cards
                drawn_cards = random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card))
                # Assign some to opponent and some to community cards
                opp_cards = opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)]
                remaining_community = list(community_cards) + drawn_cards[2 - len(opp_drawn_card):]
                
                if evaluate_hand((my_cards, opp_cards, remaining_community)):
                    wins += 1
            except Exception as e:
                self.logger.error(f"Error in Monte Carlo simulation: {e}")
                continue
        
        equity = wins / num_simulations if num_simulations > 0 else 0.5
        
        # Cache the result
        self.equity_cache[cache_key] = equity
        
        return equity

    def preprocess_observation(self, obs, equity=None):
        """
        Convert observation dictionary into feature tensor for ML model.
        """
        # Calculate equity if not provided
        if equity is None:
            equity = self.compute_equity(obs)
        
        # Basic game state features
        street = np.array([obs["street"] / 3.0])  
        my_cards = np.array([(card + 1) / 28.0 for card in obs["my_cards"]])
        community_cards = np.array([(card + 1) / 28.0 for card in obs["community_cards"]])
        
        # Pad community cards array to fixed length of 5
        if len(community_cards) < 5:
            community_cards = np.pad(community_cards, (0, 5 - len(community_cards)), 'constant', constant_values=0)
        
        # Betting features
        my_bet = np.array([obs["my_bet"] / 100.0])
        opp_bet = np.array([obs["opp_bet"] / 100.0])
        min_raise = np.array([obs["min_raise"] / 100.0])
        max_raise = np.array([obs["max_raise"] / 100.0])
        
        # Calculate pot odds
        continue_cost = obs["opp_bet"] - obs["my_bet"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        pot_odds = np.array([continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0])
        
        # Position and discard information
        position = np.array([1.0 if obs["acting_agent"] == 1 else 0.0])
        has_discarded = np.array([1.0 if "my_discarded_card" in obs and obs["my_discarded_card"] != -1 else 0.0])
        discarded_opp_card = np.array([(obs["opp_discarded_card"] + 1) / 28.0])
        drawn_opp_card = np.array([(obs["opp_drawn_card"] + 1) / 28.0])
        
        # Current equity
        equity_feature = np.array([equity])
        
        # Opponent modeling features
        opp_fold_freq = np.array([self.hand_stats["opponent_fold_frequency"]])
        opp_call_freq = np.array([self.hand_stats["opponent_call_frequency"]])
        opp_raise_freq = np.array([self.hand_stats["opponent_raise_frequency"]])
        
        # Combine all features
        features = np.concatenate([
            street, my_cards, community_cards, my_bet, opp_bet, 
            min_raise, max_raise, equity_feature, pot_odds, position,
            has_discarded, discarded_opp_card, drawn_opp_card,
            opp_fold_freq, opp_call_freq, opp_raise_freq
        ])
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    def select_action_ml(self, state, valid_actions, min_raise, max_raise):
        """
        Use the ML model to select an action.
        """
        with torch.no_grad():
            action_type_logits, raise_logits, discard_logits = self.policy_net(state)
            
            # Apply mask to filter out invalid actions
            mask = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            masked_logits = action_type_logits.clone()
            masked_logits[0, ~mask] = float('-inf')
            
            # Sample action using softmax
            action_probs = torch.nn.functional.softmax(masked_logits, dim=1)
            action_type = torch.multinomial(action_probs, 1).item()
            
            # Sample raise amount
            raise_probs = torch.nn.functional.softmax(raise_logits, dim=1)
            raise_amount = torch.multinomial(raise_probs, 1).item() + 1
            
            # Sample discard action
            discard_probs = torch.nn.functional.softmax(discard_logits, dim=1)
            discard_action = torch.multinomial(discard_probs, 1).item() - 1
            
            # Process and validate the actions
            if action_type == action_types.RAISE.value:
                raise_amount = max(min(raise_amount, max_raise), min_raise)
            else:
                raise_amount = 0
                
            if action_type == action_types.DISCARD.value:
                if discard_action < 0:
                    discard_action = 0
            else:
                discard_action = -1
                
            return action_type, raise_amount, discard_action

    def select_action_heuristic(self, obs, equity):
        """
        Use heuristic approach to select an action when ML model isn't available.
        Enhanced version of Challenge_Bot_1 strategy.
        """
        # Calculate pot odds for decision making
        continue_cost = obs["opp_bet"] - obs["my_bet"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0
        
        valid_actions = obs["valid_actions"]
        my_cards = [int(card) for card in obs["my_cards"]]
        street = obs["street"]
        
        raise_amount = 0
        card_to_discard = -1
        
        # Discard strategy with hand strength evaluation
        if valid_actions[action_types.DISCARD.value]:
            card_0_rank = my_cards[0] % 9  # Extract rank (0-8) for 2-9, A
            card_1_rank = my_cards[1] % 9
            card_0_suit = my_cards[0] // 9  # Extract suit (0-2) for diamonds, hearts, spades
            card_1_suit = my_cards[1] // 9
            
            # Check for pairs (same rank)
            if card_0_rank == card_1_rank:
                # Don't discard when holding a pair
                card_to_discard = -1
            
            # Check for suited cards (same suit)
            elif card_0_suit == card_1_suit:
                # Discard lower card unless both are high (7+)
                if card_0_rank >= 5 and card_1_rank >= 5:  # Both 7+
                    card_to_discard = -1
                else:
                    card_to_discard = 0 if card_0_rank < card_1_rank else 1
            
            # Handle Ace special cases
            elif card_0_rank == 8 or card_1_rank == 8:  # One card is an Ace
                other_rank = card_1_rank if card_0_rank == 8 else card_0_rank
                
                # Keep Ace with 7+ card, otherwise keep the Ace
                if other_rank >= 5:  # 7+ card with Ace
                    card_to_discard = -1
                else:
                    card_to_discard = 1 if card_0_rank == 8 else 0  # Discard non-Ace
            
            # Otherwise discard lower card
            else:
                card_to_discard = 0 if card_0_rank < card_1_rank else 1
            
            # Override if equity is already high
            if equity > 0.65:
                card_to_discard = -1
                
            if card_to_discard != -1:
                self.logger.debug(f"Discarding card {card_to_discard}: {int_to_card(my_cards[card_to_discard])}")
                return action_types.DISCARD.value, raise_amount, card_to_discard
        
        # Adaptive betting strategy based on equity and street
        equity_thresholds = {
            0: [0.85, 0.70, 0.60],  # Preflop thresholds
            1: [0.80, 0.65, 0.55],  # Flop thresholds
            2: [0.75, 0.60, 0.52],  # Turn thresholds
            3: [0.70, 0.55, 0.50],  # River thresholds
        }
        
        raise_sizes = [0.75, 0.5, 0.3]  # Percentage of pot for different raise sizes
        
        # Get thresholds for current street
        thresholds = equity_thresholds.get(street, equity_thresholds[0])
        
        # Dynamic pot odds adjustment based on opponent behavior and position
        position_adjustment = 0.05 if obs["acting_agent"] == 1 else 0  # In position bonus
        
        # Adjust pot odds requirement based on opponent modeling
        opp_adjustment = 0
        if self.hand_stats["opponent_fold_frequency"] > 0.6:
            opp_adjustment = 0.05  # Opponent folds often, can bluff more
        
        adjusted_pot_odds = pot_odds - position_adjustment - opp_adjustment
        
        # Decision making based on equity thresholds
        if equity >= thresholds[0] and valid_actions[action_types.RAISE.value]:
            # Very strong hand - raise big
            raise_amount = min(int(pot_size * raise_sizes[0]), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            action_type = action_types.RAISE.value
            if raise_amount > 20:
                self.logger.info(f"Large raise to {raise_amount} with equity {equity:.2f}")
        elif equity >= thresholds[1] and valid_actions[action_types.RAISE.value]:
            # Strong hand - medium raise
            raise_amount = min(int(pot_size * raise_sizes[1]), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            action_type = action_types.RAISE.value
        elif equity >= thresholds[2] and valid_actions[action_types.RAISE.value]:
            # Decent hand - small raise
            raise_amount = min(int(pot_size * raise_sizes[2]), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            action_type = action_types.RAISE.value
        elif equity >= adjusted_pot_odds and valid_actions[action_types.CALL.value]:
            # Enough equity to call
            action_type = action_types.CALL.value
        elif valid_actions[action_types.CHECK.value]:
            # Not enough equity, check if possible
            action_type = action_types.CHECK.value
        else:
            # Have to fold
            action_type = action_types.FOLD.value
            if obs["opp_bet"] > 20:
                self.logger.info(f"Folding to large bet of {obs['opp_bet']} with equity {equity:.2f}")
        
        return action_type, raise_amount, card_to_discard

    def act(self, observation, reward, terminated, truncated, info):
        """
        Main method to select an action based on current game state.
        """
        start_time = time.time()
        
        # Log important game state info
        if observation["street"] == 0:  # Preflop
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")
        
        # Calculate equity for decision making (caching is used internally)
        equity = self.compute_equity(observation)
        self.logger.debug(f"Calculated equity: {equity:.2f}")
        
        try:
            if self.use_ml_model:
                # Use ML model for decision
                state = self.preprocess_observation(observation, equity).to(self.device)
                valid_actions_tensor = torch.tensor(observation["valid_actions"], dtype=torch.float32).to(self.device)
                min_raise_val = observation["min_raise"]
                max_raise_val = observation["max_raise"]
                
                action = self.select_action_ml(state, valid_actions_tensor, min_raise_val, max_raise_val)
                self.logger.debug(f"ML model selected action: {action}")
            else:
                # Use heuristic approach
                action = self.select_action_heuristic(observation, equity)
                self.logger.debug(f"Heuristic selected action: {action}")
        except Exception as e:
            self.logger.error(f"Error selecting action: {e}, using fallback strategy")
            
            # Simple fallback strategy if all else fails
            valid_actions = observation["valid_actions"]
            if valid_actions[action_types.CHECK.value]:
                action = (action_types.CHECK.value, 0, -1)
            elif valid_actions[action_types.CALL.value]:
                action = (action_types.CALL.value, 0, -1)
            else:
                action = (action_types.FOLD.value, 0, -1)
        
        elapsed_time = time.time() - start_time
        self.logger.debug(f"Action decision took {elapsed_time:.3f} seconds")
        
        return action

    def observe(self, observation, reward, terminated, truncated, info):
        """
        Process results after each action to update statistics and opponent model.
        """
        # Clear cache periodically to avoid memory bloat
        if len(self.equity_cache) > 1000:
            self.equity_cache.clear()
        
        if terminated:
            if abs(reward) > 20:
                self.logger.info(f"Significant hand completed with reward: {reward}")
            
            # Update hand counter and statistics
            self.hand_counter += 1
            self.hand_stats["hands_played"] += 1
            
            # Reset per-hand tracking variables
            self.hand_stats["last_action"] = None
            self.hand_stats["last_street"] = None
        
        # Update opponent modeling based on observed actions
        # Note: This would need to track previous states and actions
        # which would require additional data structures and logic
        
        # For now, a simple heuristic approach is used
        # More sophisticated opponent modeling could be implemented