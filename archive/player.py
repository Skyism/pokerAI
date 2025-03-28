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
            "opponent_check_tendency": 0.0,  # How often opponent checks
            "opponent_betting_pattern": [],  # Recent betting actions
            "opponent_last_action_street": -1,  # Last street opponent took action
            "opponents_board_texture_folds": 0,  # Folds on certain board textures
            "total_board_texture_actions": 0,  # Total actions on these textures
        }

        # For detecting opponent weakness
        self.previous_street = -1
        self.opponent_street_actions = {0: [], 1: [], 2: [], 3: []}

        self.opp_acts = []


    def estimate_opponent_hand_strength(self, obs, our_equity):
        """
        Enhanced system to estimate opponent's hand strength and detect weakness patterns
        """
        # Extract relevant information
        street = obs["street"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        community_cards = [card for card in obs["community_cards"] if card != -1]
        opponent_actions = self.opponent_street_actions[street]
        my_cards = [int(card) for card in obs["my_cards"]]
        
        # Log opponent actions for debugging
        for item in self.opp_acts:
            self.logger.info(item)
        
        # 1. Action-based weakness indicators
        opponent_checked = obs["opp_last_action"] == "CHECK"
        opponent_called = obs["opp_last_action"] == "CALL"
        opponent_min_raised = obs["opp_last_action"] == "RAISE" and obs["opp_bet"] <= obs["min_raise"] * 1.5
        opponent_passive_actions = self.opp_acts.count("CHECK") + self.opp_acts.count("CALL")
        opponent_aggressive_actions = sum(1 for a in self.opp_acts if a.startswith("RAISE"))
        
        if len(self.opp_acts) >= 3:
            passive_aggressive_ratio = opponent_passive_actions / max(1, opponent_aggressive_actions)
        else:
            passive_aggressive_ratio = 1.0  # Neutral if not enough data
        
        # 2. Enhanced board texture analysis
        board_ranks = [c % 9 for c in community_cards]
        board_suits = [c // 9 for c in community_cards]
        
        # High cards on board
        high_cards = sum(1 for r in board_ranks if r >= 5)  # 7 or higher
        
        # Paired board
        paired_board = len(board_ranks) != len(set(board_ranks))
        
        # Flush draw potential
        flush_potential = max([board_suits.count(0), board_suits.count(1), board_suits.count(2)]) if board_suits else 0
        
        # Straight draw potential
        straight_potential = 0
        if len(board_ranks) >= 3:
            sorted_ranks = sorted(board_ranks)
            for i in range(len(sorted_ranks)-1):
                if sorted_ranks[i+1] - sorted_ranks[i] <= 2:
                    straight_potential += 1
        
        # 3. we have there outs?
        blockers_strength = 0
        if len(community_cards) >= 3:
            my_ranks = [c % 9 for c in my_cards]
            my_suits = [c // 9 for c in my_cards]
            
            # flush outs
            if any(board_suits.count(suit) >= 2 and my_suits.count(suit) >= 1 for suit in set(board_suits)):
                blockers_strength += 0.5
                
            # straight outs
            if any(rank in my_ranks for rank in range(min(board_ranks) - 2, max(board_ranks) + 3)):
                blockers_strength += 0.4
        
        # 4. street based
        street_weakness = 0
        if street >= 2:  # Turn or river
            # Opponent checked on dangerous street
            if opponent_checked and (high_cards >= 2 or flush_potential >= 3 or straight_potential >= 2):
                street_weakness += 1.0
                
            # Opponent slowed down after previous aggression
            if opponent_checked and any(a.startswith("RAISE") for a in self.opp_acts[:-1]):
                street_weakness += 1.5
        
        # 5. betting pattern inconsistency
        betting_inconsistency = 0
        if len(self.opp_acts) >= 3:
            # Opponent was aggressive but now passive
            if any(a.startswith("RAISE") for a in self.opp_acts[:-1]) and (opponent_checked or opponent_called):
                betting_inconsistency += 1.0
                
            # Opponent min-raised when the board got scarier
            if opponent_min_raised and (high_cards >= 2 or flush_potential >= 3 or straight_potential >= 2):
                betting_inconsistency += 0.8
        
        # Calculate weakness score
        weakness_score = 0
        
        # Base weakness from passive actions
        if len(self.opp_acts) >= 3:
            weakness_score += min(2.0, passive_aggressive_ratio / 3.0)
        
        # Add checking weakness
        if opponent_checked:
            weakness_score += 1.0
            self.logger.info("Opponent checked - potential weakness")
        
        # Add board texture factors
        board_danger = 0
        if len(community_cards) >= 3:
            board_danger = min(2.0, (high_cards * 0.4 + flush_potential * 0.5 + straight_potential * 0.4))
            
            # If board is dangerous but opponent isn't betting strongly
            if board_danger >= 1.0 and (opponent_checked or opponent_called or opponent_min_raised):
                weakness_score += 1.2
                self.logger.info(f"Opponent weak action on dangerous board (danger: {board_danger:.1f})")
        
        # Add blocker effects
        weakness_score += blockers_strength
        
        # Add street-based weakness
        weakness_score += street_weakness
        
        # Add betting pattern inconsistency
        weakness_score += betting_inconsistency
        
        # Adjust by our equity - if we have decent drawing equity, opponent might be weaker
        if 0.35 <= our_equity <= 0.45:
            weakness_score += 0.5
        
        # Normalize to a 0-1 scale with a more nuanced distribution
        max_possible_score = 7.0
        weakness_likelihood = min(weakness_score / max_possible_score, 1.0)
        
        # better bluffs later
        if street == 2:  # Turn
            weakness_likelihood *= 1.1
        elif street == 3:  # River
            weakness_likelihood *= 1.2
            
        weakness_likelihood = min(weakness_likelihood, 1.0)
        
        self.logger.info(f"Opponent weakness analysis: score={weakness_score:.1f}/{max_possible_score}, likelihood={weakness_likelihood:.2f}")
        return weakness_likelihood

    def should_bluff(self, obs, equity, weakness_likelihood):
        street = obs["street"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        continue_cost = obs["opp_bet"] - obs["my_bet"]
        my_cards = [int(card) for card in obs["my_cards"]]
        community_cards = [card for card in obs["community_cards"] if card != -1]
        
        # 1. Base bluffing frequency adjusted by street
        # Preflop - less bluffing
        # Flop - moderate bluffing opportunities 
        # Turn/River - more strategic bluffing spots
        street_bluff_factor = {
            0: 0.6,   # Preflop
            1: 0.8,   # Flop
            2: 1.1,   # Turn
            3: 1.2    # River
        }.get(street, 0.8)
        
        # 2. Position-based adjustments (approximate position from betting sequence)
        position_factor = 1.0
        if len(self.opp_acts) > 0 and self.opp_acts[0] == "CHECK":
            position_factor = 1.2  # We're in position (opponent acted first)
        
        # 3. Pot size considerations - bluff more in medium pots, less in tiny or huge pots
        pot_factor = 1.0
        if pot_size < 10:
            pot_factor = 0.7  # Too small to bluff effectively
        elif pot_size > 50:
            pot_factor = 0.8  # Too large to bluff without strong reads
        else:
            pot_factor = 1.2  # Good pot size for bluffing
        
        # 4. Semi-bluff potential - check if we have draws
        semi_bluff_factor = 1.0
        if 0.3 <= equity <= 0.45:  # We have some equity but not enough to call
            semi_bluff_factor = 1.5
            self.logger.info(f"Semi-bluff potential with equity: {equity:.2f}")
        
        # 5. Board texture considerations for bluffing
        board_factor = 1.0
        if len(community_cards) >= 3:
            board_ranks = [c % 9 for c in community_cards]
            board_suits = [c // 9 for c in community_cards]
            
            # Better to bluff on boards with high cards
            high_cards = sum(1 for r in board_ranks if r >= 5)  # 7 or higher
            if high_cards >= 2:
                board_factor *= 1.2
                
            # Better to bluff on boards with potential draws
            flush_potential = max([board_suits.count(0), board_suits.count(1), board_suits.count(2)])
            if flush_potential >= 3:
                board_factor *= 1.1
                
            # Hard to bluff on paired boards
            if len(board_ranks) != len(set(board_ranks)):
                board_factor *= 0.7
        
        # 6. Adjust based on opponent's recent action
        recent_action_factor = 1.0
        opponent_checked = obs["opp_last_action"] == "CHECK"
        opponent_called = obs["opp_last_action"] == "CALL"
        opponent_raised = obs["opp_last_action"].startswith("RAISE") if obs["opp_last_action"] else False
        
        if opponent_raised:
            recent_action_factor = 0.3  # Rarely bluff into raises
            self.logger.info("Opponent raised - reducing bluff frequency")
        elif opponent_called:
            recent_action_factor = 0.5  # Call indicates some strength
            self.logger.info("Opponent called - reducing bluff frequency")
        elif opponent_checked:
            recent_action_factor = 1.4  # Check indicates potential weakness
            self.logger.info("Opponent checked - increasing bluff frequency")
        
        # 7. Bluffing frequency adjustment based on hand history
        history_factor = 1.0
        if hasattr(self, 'bluffs_attempted'):
            if not hasattr(self, 'hands_played'):
                self.hands_played = 0
                self.bluffs_attempted = 0
                self.successful_bluffs = 0
            
            # Reduce bluffing if we've been bluffing too often
            if self.hands_played > 10 and self.bluffs_attempted / self.hands_played > 0.3:
                history_factor = 0.7
                self.logger.info("Reducing bluff frequency due to high recent bluffing rate")
            
            # Increase bluffing if previous bluffs were successful
            if self.bluffs_attempted >= 3 and self.successful_bluffs / self.bluffs_attempted > 0.6:
                history_factor = 1.3
                self.logger.info("Increasing bluff frequency due to successful previous bluffs")
        else:
            # Initialize tracking variables
            self.hands_played = 0
            self.bluffs_attempted = 0
            self.successful_bluffs = 0
        
        # Calculate final bluffing probability
        base_bluff_prob = weakness_likelihood * 0.7  # Base probability from opponent weakness
        
        # Apply all adjustment factors
        adjusted_bluff_prob = base_bluff_prob * street_bluff_factor * position_factor * \
                             pot_factor * semi_bluff_factor * board_factor * \
                             recent_action_factor * history_factor
        
        # Cap the probability
        final_bluff_prob = min(adjusted_bluff_prob, 0.75)
        
        # Don't bluff with good equity
        if equity > 0.5:
            final_bluff_prob *= 0.2
            self.logger.info(f"Reducing bluff with good equity: {equity:.2f}")
        
        # Don't bluff if pot odds are too favorable
        pot_odds = continue_cost / (continue_cost + pot_size) if continue_cost > 0 else 0
        if pot_odds < 0.2 and continue_cost > 0:
            final_bluff_prob *= 0.5
            self.logger.info(f"Reducing bluff with good pot odds: {pot_odds:.2f}")
        
        # Make the bluffing decision
        should_bluff = random.random() < final_bluff_prob
        
        # Update tracking stats
        if should_bluff:
            self.bluffs_attempted += 1
            self.logger.info(f"Decided to bluff (prob: {final_bluff_prob:.2f})")
            
            # Calculate bluff size based on factors
            bluff_sizing = self.calculate_bluff_size(obs, weakness_likelihood, equity)
            return should_bluff, bluff_sizing
        
        return should_bluff, 0.5  # Default sizing if not bluffing
    
    def calculate_bluff_size(self, obs, weakness_likelihood, equity):
        """
        Calculate optimal bluff sizing based on game state
        """
        street = obs["street"]
        pot_size = obs["my_bet"] + obs["opp_bet"]
        continue_cost = obs["opp_bet"] - obs["my_bet"]
        
        # Base sizing by street
        if street == 0:  # Preflop
            base_size = 0.75  # Larger preflop bluffs
        elif street == 1:  # Flop
            base_size = 0.6
        elif street == 2:  # Turn
            base_size = 0.65
        else:  # River
            base_size = 0.7
        
        # Adjust for opponent weakness
        if weakness_likelihood > 0.7:
            # Against very weak opponents, we can use smaller bluffs
            size_adjustment = 0.9
        elif weakness_likelihood < 0.4:
            # Against less weak opponents, need larger bets to work
            size_adjustment = 1.2
        else:
            size_adjustment = 1.0
        
        # Adjust for pot size
        if pot_size > 50:
            # In bigger pots, use slightly smaller sizing (% of pot)
            pot_adjustment = 0.9
        elif pot_size < 15:
            # In smaller pots, more aggressive sizing
            pot_adjustment = 1.2
        else:
            pot_adjustment = 1.0
        
        # Adjust for semi-bluffs - can bet smaller with equity
        equity_adjustment = 1.0
        if 0.3 <= equity <= 0.45:
            equity_adjustment = 0.9
        
        # Calculate final sizing as percentage of pot
        final_sizing = base_size * size_adjustment * pot_adjustment * equity_adjustment
        
        # Cap sizing between 0.4 and 1.0 of pot
        final_sizing = max(0.4, min(1.0, final_sizing))
        
        self.logger.info(f"Bluff size: {final_sizing:.2f} * pot")
        return final_sizing
    
    def select_action_heuristic(self, obs, equity, info):
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

        weakness_likelihood = self.estimate_opponent_hand_strength(obs, equity)
        should_bluff, bluff_sizing = self.should_bluff(obs, equity, weakness_likelihood)
            
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
            0: [0.7, 0.7, 0.65],  # Preflop thresholds
            1: [0.75, 0.7, 0.65],  # Flop thresholds
            2: [0.75, 0.7, 0.65],  # Turn thresholds
            3: [0.75, 0.7, 0.65],  # River thresholds
        }
        
        raise_sizes = [0.65, 0.4, 0.3]  # Percentage of pot for different raise sizes
        
        # Get thresholds for current street
        thresholds = equity_thresholds.get(street, equity_thresholds[0])
        
        # Decision making based on equity thresholds
        if should_bluff and valid_actions[action_types.RAISE.value] and equity < pot_odds:
            # Enhanced bluffing decision
            action_type = action_types.RAISE.value
            # Use dynamic bluff sizing based on our enhanced calculation
            raise_amount = min(int(pot_size * bluff_sizing), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            hand = info["hand_number"]
            self.logger.info(f"Executing bluff with raise to {raise_amount}, on hand #{hand} ({bluff_sizing*100:.0f}% of pot)")
            
            # Record the bluff for tracking purposes
            if not hasattr(self, 'current_bluff_amount'):
                self.current_bluff_amount = 0
            self.current_bluff_amount = raise_amount
            
        elif equity >= thresholds[0] and valid_actions[action_types.RAISE.value]:
            # Very strong hand - raise big
            raise_amount = min(int(pot_size * raise_sizes[0]), obs["max_raise"])
            raise_amount = max(raise_amount, obs["min_raise"])
            action_type = action_types.RAISE.value
            if raise_amount > 20:
                self.logger.info(f"Large raise to {raise_amount} with equity {equity:.2f}")
        # elif equity >= thresholds[1] and valid_actions[action_types.RAISE.value]:
        #     # Strong hand - medium raise
        #     raise_amount = min(int(pot_size * raise_sizes[2]), obs["max_raise"])
        #     raise_amount = max(raise_amount, obs["min_raise"])
        #     action_type = action_types.RAISE.value
        # elif equity >= thresholds[2] and valid_actions[action_types.RAISE.value]:
        #     # Decent hand - small raise
        #     raise_amount = min(int(pot_size * raise_sizes[2]), obs["max_raise"])
        #     raise_amount = max(raise_amount, obs["min_raise"])
        #     action_type = action_types.RAISE.value
        elif equity >= pot_odds and valid_actions[action_types.CALL.value]:
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
            self.opp_acts = []
        elif observation["community_cards"]:  # New community cards revealed
            visible_cards = [c for c in observation["community_cards"] if c != -1]
            if visible_cards:
                street_names = ["Preflop", "Flop", "Turn", "River"]
                self.logger.debug(f"{street_names[observation['street']]}: {[int_to_card(c) for c in visible_cards]}")
        
        if observation["opp_last_action"] != None:
            self.opp_acts.append(observation["opp_last_action"])

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
        simulations = [2000, 1000, 1000, 1000]
        num_simulations = simulations[observation["street"]]
        wins = sum(
            evaluate_hand((my_cards, opp_drawn_card + drawn_cards[: 2 - len(opp_drawn_card)], community_cards + drawn_cards[2 - len(opp_drawn_card) :]))
            for _ in range(num_simulations)
            if (drawn_cards := random.sample(non_shown_cards, 7 - len(community_cards) - len(opp_drawn_card)))
        )
        equity = wins / num_simulations
        

        action = self.select_action_heuristic(observation, equity, info)
        self.logger.debug(f"Heuristic selected action: {action}")
        

        return action

    def observe(self, observation, reward, terminated, truncated, info):
        if terminated and abs(reward) > 20:  # Only log significant hand results
            self.logger.info(f"Significant hand completed with reward: {reward}")
