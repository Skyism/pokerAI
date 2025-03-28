from gym_env import PokerEnv
import random
import numpy as np
from gym_env import WrappedEval
from collections import defaultdict
from agents.agent import Agent

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card


class PlayerAgent(Agent):
    def __name__(self):
        return "CFRAgent"

    def __init__(self, stream: bool = False, iterations=1000):
        super().__init__(stream)
        self.evaluator = WrappedEval()
        self.iterations = iterations
        
        # Strategy and regret tables
        self.regret_sum = defaultdict(lambda: np.zeros(4))  # 4 actions: FOLD, CALL, CHECK, RAISE
        self.strategy_sum = defaultdict(lambda: np.zeros(4))
        self.node_visits = defaultdict(int)
        
        # Action mapping
        self.action_map = {
            0: action_types.FOLD.value,
            1: action_types.CALL.value,
            2: action_types.CHECK.value,
            3: action_types.RAISE.value
        }
        
        # For tracking opponent tendencies
        self.opponent_action_frequency = defaultdict(lambda: np.zeros(4))
        self.opponent_actions_count = 0
        
        # For tracking game state
        self.current_hand_history = []
        self.opp_acts = []
        self.previous_equity = 0

    def get_info_set_key(self, obs):
        """Create a unique key for the current information set"""
        my_cards = tuple(sorted(obs["my_cards"]))
        community_cards = tuple(sorted([c for c in obs["community_cards"] if c != -1]))
        street = obs["street"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        
        # Include recent betting sequence (limited to last 3 actions for practical reasons)
        betting_sequence = tuple(self.opp_acts[-3:]) if self.opp_acts else tuple()
        
        # Include my position (approximated by who acted first)
        in_position = 1 if len(self.opp_acts) > 0 and self.opp_acts[0] == "CHECK" else 0
        
        return (my_cards, community_cards, street, pot_size, betting_sequence, in_position)
    
    def calculate_strategy(self, info_set_key, valid_actions):
        """Calculate the current strategy for this information set"""
        regrets = self.regret_sum[info_set_key].copy()
        
        # Apply valid actions mask
        action_mask = np.zeros(4)
        for i in range(min(4, len(valid_actions))):
            if valid_actions[i]:
                action_mask[i] = 1
        
        # Zero out invalid actions
        regrets *= action_mask
        
        # Get positive regrets
        regrets = np.maximum(regrets, 0)
        
        # Normalize strategy
        regret_sum = np.sum(regrets)
        if regret_sum > 0:
            strategy = regrets / regret_sum
        else:
            # If all regrets are zero or negative, use a uniform random strategy over valid actions
            valid_count = np.sum(action_mask)
            if valid_count > 0:
                strategy = action_mask / valid_count
            else:
                strategy = np.zeros(4)
        
        return strategy
    
    def update_regrets(self, info_set_key, action_index, reward, counterfactual_value):
        """Update regrets for the given information set and action"""
        # Calculate regret
        regret = counterfactual_value - reward
        
        # Update regret sum
        self.regret_sum[info_set_key][action_index] += regret
        
        # Update strategy sum (for averaging)
        strategy = self.calculate_strategy(info_set_key, [True, True, True, True])  # Placeholder valid actions
        self.strategy_sum[info_set_key] += strategy
        self.node_visits[info_set_key] += 1
    
    def get_average_strategy(self, info_set_key):
        """Get the average strategy for this information set"""
        if info_set_key not in self.strategy_sum:
            return np.zeros(4)
        
        avg_strategy = self.strategy_sum[info_set_key].copy()
        avg_strategy_sum = np.sum(avg_strategy)
        
        if avg_strategy_sum > 0:
            avg_strategy /= avg_strategy_sum
        else:
            # Default to uniform random strategy
            avg_strategy = np.ones(4) / 4
        
        return avg_strategy
    
    def select_action_cfr(self, obs, equity, info):
        """Select an action based on the CFR strategy"""
        info_set_key = self.get_info_set_key(obs)
        valid_actions = obs["valid_actions"]
        
        # Get current strategy
        strategy = self.calculate_strategy(info_set_key, valid_actions)
        
        # For exploration, add some randomness early in training
        if self.node_visits[info_set_key] < 50:
            epsilon = 0.2
            
            # Create mask for valid actions (ensuring we stay within bounds)
            valid_mask = np.zeros(4)
            valid_count = 0
            
            for i in range(min(4, len(valid_actions))):
                if valid_actions[i]:
                    valid_mask[i] = 1
                    valid_count += 1
            
            # Add epsilon exploration
            if valid_count > 0:
                uniform = np.zeros(4)
                for i in range(4):
                    if i < len(valid_actions) and valid_actions[i]:
                        uniform[i] = 1 / valid_count
                strategy = (1 - epsilon) * strategy + epsilon * uniform
        
        # Select action based on strategy
        action_probs = strategy.copy()
        
        # Zero out invalid actions
        for i in range(4):
            if i >= len(valid_actions) or not valid_actions[i]:
                action_probs[i] = 0
        
        # Normalize
        action_prob_sum = sum(action_probs)
        if action_prob_sum > 0:
            action_probs = action_probs / action_prob_sum
        else:
            # If no valid actions have probability, default to first valid action
            for i in range(min(4, len(valid_actions))):
                if valid_actions[i]:
                    action_probs[i] = 1.0
                    break
        
        # If we still don't have valid probabilities, use a fallback method
        if sum(action_probs) == 0:
            # Find the first valid action
            for i in range(min(4, len(valid_actions))):
                if valid_actions[i]:
                    action_index = i
                    break
            else:
                # If no valid action found in the first 4, set a safe default
                self.logger.warning("No valid actions found in first 4 actions")
                action_index = 0  # Default to FOLD if nothing else works
        else:
            # Select action according to probability distribution
            try:
                action_index = np.random.choice(4, p=action_probs)
            except ValueError:
                # Fallback to most common valid action if there's an issue
                self.logger.warning(f"Invalid probability distribution: {action_probs}")
                for i in range(min(4, len(valid_actions))):
                    if valid_actions[i]:
                        action_index = i
                        break
                else:
                    action_index = 0  # Default to FOLD if nothing else works
        
        # Verify the selected action is valid
        if action_index >= len(valid_actions) or not valid_actions[action_index]:
            # If invalid, find a valid fallback action
            self.logger.warning(f"Selected invalid action index {action_index}, finding fallback")
            for i in range(len(valid_actions)):
                if valid_actions[i]:
                    action_index = i
                    break
        
        # Map to actual action type
        if action_index in self.action_map:
            action_type = self.action_map[action_index]
        else:
            # Fallback to a valid action if action_index is out of bounds for action_map
            for i in range(len(valid_actions)):
                if valid_actions[i] and i in self.action_map:
                    action_type = self.action_map[i]
                    action_index = i
                    break
            else:
                # Ultimate fallback
                self.logger.error("No valid mapping for any valid action, defaulting to FOLD")
                action_type = action_types.FOLD.value
                action_index = 0
        
        # Determine raise amount if RAISE is selected
        raise_amount = 0
        if action_type == action_types.RAISE.value and valid_actions[action_types.RAISE.value]:
            pot_size = obs["my_bet"] + obs["opp_bet"]
            
            # Scale raise based on equity - stronger hands raise larger
            if equity > 0.8:  # Very strong hand
                raise_pct = 0.8
            elif equity > 0.7:  # Strong hand
                raise_pct = 0.6
            elif equity > 0.6:  # Good hand
                raise_pct = 0.5
            else:  # Decent hand or bluff
                raise_pct = 0.4
            
            # Adjust in later streets
            if obs["street"] >= 2:  # Turn or River
                raise_pct *= 1.2
            
            # Calculate raise amount
            raise_amount = min(int(pot_size * raise_pct), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
        
        # Determine card to discard (if applicable)
        card_to_discard = -1
        if action_type == action_types.DISCARD.value or (
                "DISCARD" in dir(action_types) and 
                action_type == action_types.DISCARD.value and 
                len(valid_actions) > action_types.DISCARD.value and 
                valid_actions[action_types.DISCARD.value]):
            card_to_discard = self.select_card_to_discard(obs, equity)
        
        # Store the action and strategy for learning
        self.current_action = (action_type, raise_amount, card_to_discard)
        self.current_info_set = info_set_key
        self.current_strategy = strategy
        self.current_reward = None
        
        # Debug logging
        self.logger.debug(f"CFR selected action: {action_type}, raise: {raise_amount}, discard: {card_to_discard}")
        self.logger.debug(f"Strategy probs: {strategy}")
        
        return action_type, raise_amount, card_to_discard
    
    def select_card_to_discard(self, obs, equity):
        """Select which card to discard based on current hand strength"""
        my_cards = [int(card) for card in obs["my_cards"]]
        community_cards = [card for card in obs["community_cards"] if card != -1]
        
        # If hand equity is already strong, don't discard
        if equity > 0.65:
            return -1
            
        card_0_rank = my_cards[0] % 9  # Extract rank (0-8) for 2-9, A
        card_1_rank = my_cards[1] % 9
        card_0_suit = my_cards[0] // 9  # Extract suit (0-2) for diamonds, hearts, spades
        card_1_suit = my_cards[1] // 9
        
        # Check if we already have a pair
        if card_0_rank == card_1_rank:
            return -1  # Don't discard when holding a pair
        
        # Check for cards matching community
        comm_ranks = [c % 9 for c in community_cards]
        comm_suits = [c // 9 for c in community_cards]
        
        # If one card matches a community card rank, keep it (discard the other)
        if card_0_rank in comm_ranks and card_1_rank not in comm_ranks:
            return 1
        elif card_1_rank in comm_ranks and card_0_rank not in comm_ranks:
            return 0
        
        # Check for flush potential
        if card_0_suit == card_1_suit and comm_suits.count(card_0_suit) >= 2:
            return -1  # Keep both for flush potential
        
        # Handle high cards (7+ and Ace)
        if card_0_rank >= 5 and card_1_rank >= 5:  # Both high cards
            return -1  # Keep both high cards
        elif card_0_rank == 8 or card_1_rank == 8:  # One card is an Ace
            return 1 if card_0_rank == 8 else 0  # Keep the Ace
        
        # Discard lower card by default
        return 0 if card_0_rank < card_1_rank else 1
    
    def calculate_counterfactual_values(self, obs, equity):
        """Estimate counterfactual values for actions not taken"""
        info_set_key = self.get_info_set_key(obs)
        valid_actions = obs["valid_actions"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        continue_cost = obs["opp_bet"] - obs["my_bet"]
        
        # Estimate value of each action based on equity and pot size
        action_values = np.zeros(4)
        
        # FOLD value
        action_values[0] = -obs["my_bet"]  # Lose what we've put in
        
        # CALL value
        expected_value_call = (2 * equity - 1) * (pot_size + continue_cost)
        action_values[1] = expected_value_call - continue_cost
        
        # CHECK value
        expected_value_check = (2 * equity - 1) * pot_size
        action_values[2] = expected_value_check
        
        # RAISE value (assuming half pot raise)
        raise_amount = min(pot_size // 2, obs["max_raise"]) 
        if raise_amount < obs["min_raise"]:
            raise_amount = obs["min_raise"]
            
        # Assume call probability decreases as raise increases
        call_prob = max(0.1, 1 - raise_amount / (raise_amount + pot_size) - 0.2)
        
        # EV if called
        ev_if_called = (2 * equity - 1) * (pot_size + continue_cost + raise_amount)
        
        # EV if opponent folds
        ev_if_fold = pot_size
        
        # Combined expected value for raise
        expected_value_raise = call_prob * ev_if_called + (1 - call_prob) * ev_if_fold - raise_amount
        action_values[3] = expected_value_raise
        
        # Zero out invalid actions
        for i in range(4):
            if i >= len(valid_actions) or not valid_actions[i]:
                action_values[i] = float('-inf')
        
        return action_values
    
    def update_strategy(self, obs, action_taken, reward, equity):
        """Update the strategy based on the outcome of an action"""
        if hasattr(self, 'current_info_set') and self.current_info_set is not None:
            info_set_key = self.current_info_set
            
            action_map_rev = {v: k for k, v in self.action_map.items()}
            action_index = action_map_rev.get(action_taken[0], 0)  # Default to FOLD if not found
            
            # Calculate counterfactual values for all actions
            counterfactual_values = self.calculate_counterfactual_values(obs, equity)
            
            # Update regrets
            self.update_regrets(info_set_key, action_index, reward, counterfactual_values[action_index])
            
            # Clear current action tracking
            self.current_info_set = None
            self.current_action = None
    
    def learn_from_opponent(self, obs):
        """Track and learn from opponent actions"""
        if obs["opp_last_action"] is None:
            return
            
        # Map opponent action to index
        action_index = -1
        if obs["opp_last_action"] == "FOLD":
            action_index = 0
        elif obs["opp_last_action"] == "CALL":
            action_index = 1
        elif obs["opp_last_action"] == "CHECK":
            action_index = 2
        elif obs["opp_last_action"].startswith("RAISE"):
            action_index = 3
            
        if action_index >= 0:
            # Create a simplified key for opponent state
            street = obs["street"]
            pot_size_bucket = obs["my_bet"] + obs["opp_bet"] // 10  # Bucket pot sizes
            community_texture = self.categorize_board_texture(obs)
            opp_key = (street, pot_size_bucket, community_texture)
            
            # Update opponent action frequency
            self.opponent_action_frequency[opp_key][action_index] += 1
            self.opponent_actions_count += 1
            
            # Adjust our strategy based on opponent tendencies (exploitation)
            if self.opponent_actions_count > 50:  # Once we have enough data
                self.adjust_strategy_for_exploitation(opp_key)
    
    def categorize_board_texture(self, obs):
        """Categorize the board texture for simplified state representation"""
        community_cards = [card for card in obs["community_cards"] if card != -1]
        if not community_cards:
            return 0  # No community cards
            
        ranks = [c % 9 for c in community_cards]
        suits = [c // 9 for c in community_cards]
        
        # Check for paired board
        paired = len(ranks) != len(set(ranks))
        
        # Check for flush potential
        flush_potential = max([suits.count(0), suits.count(1), suits.count(2)]) if suits else 0
        has_flush_draw = flush_potential >= 3
        
        # Check for straight potential
        sorted_ranks = sorted(ranks)
        has_straight_draw = False
        for i in range(len(sorted_ranks)-1):
            if sorted_ranks[i+1] - sorted_ranks[i] <= 2:
                has_straight_draw = True
                break
                
        # Check for high cards
        high_cards = sum(1 for r in ranks if r >= 5)  # 7 or higher
        high_card_heavy = high_cards >= 2
        
        # Combine characteristics into a texture category
        if paired and has_flush_draw:
            return 1  # Paired with flush draw
        elif paired:
            return 2  # Just paired
        elif has_flush_draw and has_straight_draw:
            return 3  # Connected and suited
        elif has_flush_draw:
            return 4  # Just flush draw
        elif has_straight_draw:
            return 5  # Just straight draw
        elif high_card_heavy:
            return 6  # High card heavy
        else:
            return 7  # Uncoordinated low cards
    
    def adjust_strategy_for_exploitation(self, opp_key):
        """Adjust strategy weights based on opponent tendencies"""
        # Get opponent action frequencies for this state
        if opp_key not in self.opponent_action_frequency:
            return
            
        opp_freqs = self.opponent_action_frequency[opp_key]
        total_actions = sum(opp_freqs)
        
        if total_actions < 5:  # Need minimum sample size
            return
            
        # Calculate percentages
        fold_pct = opp_freqs[0] / total_actions
        call_pct = opp_freqs[1] / total_actions
        check_pct = opp_freqs[2] / total_actions
        raise_pct = opp_freqs[3] / total_actions
        
        # Strategy adjustments based on opponent tendencies
        # These will be applied when calculating regrets
        
        # If opponent folds too much, increase our raising frequency
        if fold_pct > 0.5:
            # Apply this adjustment to all info sets for this street/pot size
            for key in list(self.regret_sum.keys()):
                if key[2] == opp_key[0] and key[3] // 10 == opp_key[1]:
                    # Boost raise regrets
                    self.regret_sum[key][3] *= 1.2
                    
        # If opponent calls too much, adjust value betting thresholds
        if call_pct > 0.7:
            # Make value bets thinner but larger
            for key in list(self.regret_sum.keys()):
                if key[2] == opp_key[0] and key[3] // 10 == opp_key[1]:
                    # Reduce raise frequency but increase call frequency
                    self.regret_sum[key][3] *= 0.9
                    self.regret_sum[key][1] *= 1.1
                    
        # If opponent raises too much, tighten up calling range
        if raise_pct > 0.4:
            # Be more selective with calls
            for key in list(self.regret_sum.keys()):
                if key[2] == opp_key[0] and key[3] // 10 == opp_key[1]:
                    # Reduce call regrets
                    self.regret_sum[key][1] *= 0.9
                    # Increase fold regrets
                    self.regret_sum[key][0] *= 1.1
    
    def calculate_equity(self, obs):
        """Calculate equity of current hand through Monte Carlo simulation"""
        my_cards = [int(card) for card in obs["my_cards"]]
        community_cards = [card for card in obs["community_cards"] if card != -1]
        opp_discarded_card = [obs["opp_discarded_card"]] if obs["opp_discarded_card"] != -1 else []
        opp_drawn_card = [obs["opp_drawn_card"]] if obs["opp_drawn_card"] != -1 else []

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
        simulations = [2000, 1000, 1000, 1000]
        num_simulations = simulations[obs["street"]]
        wins = sum(
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
        )
        equity = wins / num_simulations
        return equity
    
    def update_cfr_strategy(self, iterations=100):
        """Run CFR iterations to improve strategy"""
        # This would normally be called between hands, but for simplicity
        # we'll just run a small number of iterations each time
        for _ in range(iterations):
            # Update regret matching for each info set
            for info_set_key in list(self.regret_sum.keys()):
                # Skip info sets with too few visits
                if self.node_visits[info_set_key] < 2:
                    continue
                    
                # Get current strategy
                strategy = self.calculate_strategy(info_set_key, [True, True, True, True])
                
                # Update strategy sum for averaging
                self.strategy_sum[info_set_key] += strategy
        
        self.logger.debug(f"Updated CFR strategy over {iterations} iterations")
    
    def act(self, observation, reward, terminated, truncated, info):
        """Main action selection function"""
        # Update information from previous actions
        if hasattr(self, 'current_action') and self.current_action is not None:
            self.update_strategy(observation, self.current_action, reward, self.previous_equity)
        
        # Log new street starts with important info
        if observation["street"] == 0:  # Preflop
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")
            self.opp_acts = []
            # Run CFR iterations between hands
            if hasattr(self, 'hands_played'):
                self.update_cfr_strategy(max(5, min(100, self.hands_played // 10)))
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")
        
        # Track opponent actions
        if observation["opp_last_action"] != None:
            self.opp_acts.append(observation["opp_last_action"])
            self.learn_from_opponent(observation)

        # Calculate equity
        equity = self.calculate_equity(observation)
        self.previous_equity = equity
        
        # Select action using CFR strategy
        action = self.select_action_cfr(observation, equity, info)
        
        return action

    def observe(self, observation, reward, terminated, truncated, info):
        """Process observation and reward after action is taken"""
        # Track whether our bluffs are working
        if terminated:
            # Initialize tracking variables if not exists
            if not hasattr(self, 'hands_played'):
                self.hands_played = 0
                self.bluffs_attempted = 0
                self.successful_bluffs = 0
            
            self.hands_played += 1
            
            # Update strategy based on final reward
            if hasattr(self, 'current_action') and self.current_action is not None:
                self.update_strategy(observation, self.current_action, reward, self.previous_equity)
                
            # Log significant hands
            if abs(reward) > 20:
                self.logger.info(f"Hand #{self.hands_played}: Significant hand completed with reward: {reward}")