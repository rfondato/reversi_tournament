import sys

import numpy as np
from boardgame2 import BoardGameEnv

from players import BasePlayer


class PlayerStats:

    def __init__(self, player: BasePlayer, player_num: int):
        self.player = player
        self.player_num = player_num
        self.wins = 0
        self.wins_as_first = 0
        self.wins_as_second = 0
        self.ties = 0
        self.plays_as_first = 0
        self.plays_as_second = 0
        self.total_steps = 0

    def add_game_stats(self,
                       plays_first: bool,
                       n_steps: int,
                       reward: float
                       ):
        self.plays_as_first += int(plays_first)
        self.plays_as_second += int(not plays_first)
        self.total_steps += n_steps
        self.wins += (reward > 0) * int(plays_first) + (reward < 0) * int(not plays_first)
        self.wins_as_first += (reward > 0) * int(plays_first)
        self.wins_as_second += (reward < 0) * int(not plays_first)
        self.ties += (reward == 0)

    def __str__(self):
        total_games = self.plays_as_first + self.plays_as_second
        header = f'####### STATS FOR PLAYER: {self.player_num} - {str(self.player)} #######'
        return f"""
{header}

Wins as first: {self.wins_as_first / self.plays_as_first}
Wins as second: {self.wins_as_second / self.plays_as_second}
Ties: {self.ties / total_games}
Plays as first: {self.plays_as_first}
Plays as second: {self.plays_as_second}
Avg game duration: {self.total_steps / total_games}

{"#" * len(header)}
            
        """


class Arena:

    def __init__(self,
                 player_1: BasePlayer,
                 player_2: BasePlayer,
                 env: BoardGameEnv,
                 verbose: bool = False
                 ):
        self.player_1 = player_1
        self.player_2 = player_2
        self.env = env
        self.player_1_stats = PlayerStats(self.player_1, 1)
        self.player_2_stats = PlayerStats(self.player_2, 2)
        self.verbose = verbose
        self.winner = None

    def play(self, n_games: int = 500):
        if self.verbose:
            self._print_init()

        for i in range(n_games):
            if self.verbose:
                self._print_game_stats(i, n_games)

            # First/Second player order is sorted at random on each game
            self._sort_players_turns()

            n_steps, reward = self._play_game()

            self.player_1_stats.add_game_stats(self.player_1.player == 1, n_steps, reward)
            self.player_2_stats.add_game_stats(self.player_2.player == 1, n_steps, reward)

        self._calculate_winner()

        if self.verbose:
            self._print_results()

    def print_players_stats(self):
        print(self.player_1_stats)
        print(self.player_2_stats)

    def get_winner(self):
        return self.winner

    def get_players_stats(self):
        return self.player_1_stats, self.player_2_stats

    def _sort_players_turns(self):
        player_one = np.random.choice([-1, 1])
        self.player_1.player = player_one
        self.player_2.player = -player_one

    def _play_game(self):
        done = False
        n_steps = 0
        (board, player) = self.env.reset()
        while not done:
            current_player = self.player_1 if self.player_1.player == player else self.player_2
            action = current_player.predict(board)
            (board, player), reward, done, info = self.env.step(action)
            n_steps = n_steps + 1
        return n_steps, reward

    def _print_init(self):
        print()
        print(f"MATCH: {self.player_1} vs {self.player_2}")
        print()

    def _print_game_stats(self, game: int, n_games: int):
        prev_game = game if game > 0 else 1
        p1_wins = round(self.player_1_stats.wins / prev_game * 100, 2)
        p2_wins = round(self.player_2_stats.wins / prev_game * 100, 2)
        ties = round(self.player_1_stats.ties / prev_game * 100, 2)
        print("\r", end="")
        sys.stdout.write("\033[K")
        print(f"Playing n:{game + 1}/{n_games} \t Wins(player 1/ player 2):{p1_wins}%/{p2_wins}% \t Ties:{ties}%", end="")

    def _print_results(self):
        print()
        print()
        print(f"THE WINNER IS {self.winner}!")
        print()

    def _calculate_winner(self):
        if self.player_1_stats.wins > self.player_2_stats.wins:
            self.winner = self.player_1
        elif self.player_2_stats.wins > self.player_1_stats.wins:
            self.winner = self.player_2
        else:
            self.winner = None  # Tie
