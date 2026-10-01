from __future__ import annotations

import numpy as np

from game import TwoPlayerGameState
from tournament import StudentHeuristic

WEIGHTS_BOARD: np.ndarray = np.array([
    [ 100, -20,  10,   5,   5,  10, -20,  100],
    [-20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [  10,  -2,  -1,  -1,  -1,  -1,  -2,   10],
    [   5,  -2,  -1,   0,   0,  -1,  -2,    5],
    [   5,  -2,  -1,   0,   0,  -1,  -2,    5],
    [  10,  -2,  -1,  -1,  -1,  -1,  -2,   10],
    [-20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [ 100, -20,  10,   5,   5,  10, -20,  100],
])


def _get_player_labels(state: TwoPlayerGameState) -> tuple[str, str]:
    if state.player_max is not None:
        player_label = state.player_max.label
        adversary_label = state.game.opponent(state.player_max).label
    else:
        player_label = state.next_player.label
        adversary_label = state.game.opponent(state.next_player).label
    return player_label, adversary_label


def _calculate_positional_score(state: TwoPlayerGameState, player: str, adversary: str) -> float:
    player_score = 0.0
    adversary_score = 0.0
    board = state.board

    for pos, cell_player in board.items():
        col = pos[0]
        row = pos[1]
        r = row - 1
        c = col - 1

        if 0 <= r < 8 and 0 <= c < 8:
            weight = WEIGHTS_BOARD[r, c]
            if cell_player == player:
                player_score = player_score + weight
            elif cell_player == adversary:
                adversary_score = adversary_score + weight

    return player_score - adversary_score


def _calculate_corners(state: TwoPlayerGameState, player: str, adversary: str) -> float:
    board = state.board
    h = state.game.height
    w = state.game.width
    corners = [(1, 1), (1, h), (w, 1), (w, h)]
    my_corners = 0
    opp_corners = 0

    for pos in corners:
        if pos in board:
            cell = board[pos]
            if cell == player:
                my_corners = my_corners + 1
            elif cell == adversary:
                opp_corners = opp_corners + 1

    total = my_corners + opp_corners
    if total > 0:
        return 100.0 * (my_corners - opp_corners) / total
    return 0.0


def _calculate_mobility(state: TwoPlayerGameState) -> float:
    game = state.game
    player, adversary = _get_player_labels(state)

    my_moves = len(game._get_valid_moves(state.board, player))
    opp_moves = len(game._get_valid_moves(state.board, adversary))

    total = my_moves + opp_moves
    if total > 0:
        return 100.0 * (my_moves - opp_moves) / total
    return 0.0


def _calculate_coin_parity(board: dict, player: str, adversary: str) -> float:
    my_coins = 0
    opp_coins = 0

    for cell in board.values():
        if cell == player:
            my_coins = my_coins + 1
        elif cell == adversary:
            opp_coins = opp_coins + 1

    total = my_coins + opp_coins
    if total > 0:
        return 100.0 * (my_coins - opp_coins) / total
    return 0.0


class PositionalHeuristic(StudentHeuristic):

    def get_name(self) -> str:
        return "Positional_P1"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)
        return _calculate_positional_score(state, player, adversary)


class MobilityHeuristic(StudentHeuristic):

    def get_name(self) -> str:
        return "Mobility_P1"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)

        corners_score = _calculate_corners(state, player, adversary)
        mobility_score = _calculate_mobility(state)
        positional_score = _calculate_positional_score(state, player, adversary)

        return (40.0 * corners_score) + (35.0 * mobility_score) + (25.0 * positional_score)


class PhaseMasterHeuristic(StudentHeuristic):

    def get_name(self) -> str:
        return "PhaseMaster_P1"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)
        total_coins = len(state.board)
        max_cells = state.game.height * state.game.width

        corners_score = _calculate_corners(state, player, adversary)
        positional_score = _calculate_positional_score(state, player, adversary)

        if total_coins >= int(max_cells * 0.75):
            coin_score = _calculate_coin_parity(state.board, player, adversary)
            return (50.0 * coin_score) + (35.0 * corners_score) + (15.0 * positional_score)

        mobility_score = _calculate_mobility(state)
        return (45.0 * corners_score) + (30.0 * mobility_score) + (25.0 * positional_score)