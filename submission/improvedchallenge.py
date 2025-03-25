from agents.agent import Agent
from gym_env import PokerEnv
import random
from gym_env import WrappedEval

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = False):
        super().__init__(stream)
        self.evaluator = WrappedEval()

        self.hand_stats = {
            "opponent_fold_frequency": 0.0,
            "opponent_call_frequency": 0.0,
            "opponent_raise_frequency": 0.0,
            "opponent_actions": [],
            "hands_played": 0,
        }
    def select_action_heuristic(self, obs, equity):
        # Calculate pot odds for decision making
        continue_cost = obs["opp_bet"] - obs["my_bet"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0
        
        valid_actions = obs["valid_actions"]
        my_cards = [int(card) for card in obs["my_cards"]]
        street = obs["street"]
        community_cards = [card for card in obs["community_cards"] if card != -1]
        length_community = len(community_cards)
        
        raise_amount = 0
        card_to_discard = -1
        
        # Discard strategy with hand strength evaluation
        if obs["street"] == 1 and valid_actions[action_types.DISCARD.value]:
            card_0_rank = my_cards[0] % 9  # Extract rank (0-8) for 2-9, A
            card_1_rank = my_cards[1] % 9
            card_0_suit = my_cards[0] // 9  # Extract suit (0-2) for diamonds, hearts, spades
            card_1_suit = my_cards[1] // 9
            comm_card_1_rank = community_cards[0] % 9
            comm_card_2_rank = community_cards[1] % 9
            comm_card_3_rank = community_cards[2] % 9
            comm_card_1_suit = community_cards[0] // 9
            comm_card_2_suit = community_cards[1] // 9
            comm_card_3_suit = community_cards[2] // 9
            comm_ranks = [comm_card_1_rank, comm_card_2_rank, comm_card_3_rank]
            comm_suit = [comm_card_1_suit, comm_card_2_suit, comm_card_3_suit]


            
            # Check for pairs (same rank)
            if card_0_rank == card_1_rank:
                # Don't discard when holding a pair
                card_to_discard = -1
            
            elif(card_0_rank in comm_ranks):
                card_to_discard = 1
            
            elif(card_1_rank in comm_ranks):
                card_to_discard = 0
            
            # Check for suited cards (same suit)
            elif card_0_suit == card_1_suit:
                # Discard lower card unless both are high (7+)
                if comm_suit.count(card_0_suit) >= 2:
                    card_to_discard = -1
                elif card_0_rank >= 5 and card_1_rank >= 5:  # Both 7+
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
        
        adjusted_pot_odds = pot_odds - position_adjustment
        
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
            self.logger.info(f"Folding to bet of {obs['opp_bet']} with equity {equity:.2f}")
        
        return action_type, raise_amount, card_to_discard

    def act(self, observation, reward, terminated, truncated, info):
        # Log new street starts with important info
        if observation["street"] == 0:  # Preflop
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")

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
        simulations = [5000, 3000, 2000, 1000]
        num_simulations = simulations[observation["street"]]
        wins = sum(
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
        )
        equity = wins / num_simulations
        

        action = self.select_action_heuristic(observation, equity)
        self.logger.debug(f"Heuristic selected action: {action}")
        

        return action

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated and abs(reward) > 20:  # Only log significant hand results
            self.logger.info(f"Significant hand completed with reward: {reward}")
