from __future__ import annotations

import numpy as np

from game import TwoPlayerGameState
from tournament import StudentHeuristic


def _get_player_labels(state: TwoPlayerGameState) -> tuple[str, str]:
    if state.player_max is not None:
        player_label = state.player_max.label
        adversary_label = state.game.opponent(state.player_max).label
    else:
        player_label = state.next_player.label
        adversary_label = state.game.opponent(state.next_player).label
    return player_label, adversary_label


def _calculate_corners_and_dangers(state: TwoPlayerGameState, player: str, adversary: str) -> float:
    """Esquinas tomadas (+100) y penalización estricta por pisar casillas X y C si la esquina está libre."""
    board = state.board
    h = state.game.height
    w = state.game.width

    corners = [(1, 1), (1, h), (w, 1), (w, h)]
    danger_zones = {
        (1, 1): [(1, 2), (2, 1), (2, 2)],
        (1, h): [(1, h - 1), (2, h), (2, h - 1)],
        (w, 1): [(w - 1, 1), (w, 2), (w - 1, 2)],
        (w, h): [(w, h - 1), (w - 1, h), (w - 1, h - 1)],
    }

    my_corners = 0
    opp_corners = 0
    my_danger = 0
    opp_danger = 0

    for corner in corners:
        val = board.get(corner)
        if val == player:
            my_corners += 1
        elif val == adversary:
            opp_corners += 1
        else:
            for adj in danger_zones[corner]:
                adj_val = board.get(adj)
                if adj_val == player:
                    my_danger += 1
                elif adj_val == adversary:
                    opp_danger += 1

    corner_score = 0.0
    tot_c = my_corners + opp_corners
    if tot_c > 0:
        corner_score = 100.0 * (my_corners - opp_corners) / tot_c

    danger_score = 0.0
    tot_d = my_danger + opp_danger
    if tot_d > 0:
        danger_score = 100.0 * (opp_danger - my_danger) / tot_d

    return (0.75 * corner_score) + (0.25 * danger_score)


def _calculate_mobility(state: TwoPlayerGameState, player: str, adversary: str) -> float:
    """Ventaja relativa de movimientos legales."""
    game = state.game
    my_moves = len(game._get_valid_moves(state.board, player))
    opp_moves = len(game._get_valid_moves(state.board, adversary))

    total = my_moves + opp_moves
    if total > 0:
        return 100.0 * (my_moves - opp_moves) / total
    return 0.0


def _calculate_frontier(state: TwoPlayerGameState, player: str, adversary: str) -> float:
    """Minimiza las fichas que tocan casillas vacías (estabilidad estructural)."""
    board = state.board
    h = state.game.height
    w = state.game.width

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    my_frontier = 0
    opp_frontier = 0

    for (x, y), cell_color in board.items():
        if cell_color not in (player, adversary):
            continue
        is_frontier = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= w and 1 <= ny <= h and (nx, ny) not in board:
                is_frontier = True
                break

        if is_frontier:
            if cell_color == player:
                my_frontier += 1
            else:
                opp_frontier += 1

    total = my_frontier + opp_frontier
    if total > 0:
        return 100.0 * (opp_frontier - my_frontier) / total
    return 0.0


def _calculate_coin_parity(board: dict, player: str, adversary: str) -> float:
    """Paridad de monedas en el tablero."""
    my_coins = sum(1 for val in board.values() if val == player)
    opp_coins = sum(1 for val in board.values() if val == adversary)

    total = my_coins + opp_coins
    if total > 0:
        return 100.0 * (my_coins - opp_coins) / total
    return 0.0


# =====================================================================
# CLASES DE ESTRATEGIAS PARA EL TORNEO 2
# =====================================================================

class MobilityHeuristic(StudentHeuristic):
    """Mantiene la base que sacó el Puesto 34, incorporando penalización de casillas C y X."""

    def get_name(self) -> str:
        return "Mobility_P2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)

        corners = _calculate_corners_and_dangers(state, player, adversary)
        mobility = _calculate_mobility(state, player, adversary)

        return (0.60 * corners) + (0.40 * mobility)


class StabilityHeuristic(StudentHeuristic):
    """Sustituto de Positional: combina estabilidad de frontera con movilidad y esquinas."""

    def get_name(self) -> str:
        return "Stability_P2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)

        corners = _calculate_corners_and_dangers(state, player, adversary)
        frontier = _calculate_frontier(state, player, adversary)
        mobility = _calculate_mobility(state, player, adversary)

        return (0.45 * corners) + (0.35 * frontier) + (0.20 * mobility)


class PhaseMasterHeuristic(StudentHeuristic):
    """Heurística adaptativa de alto rendimiento: progresión continua según el llenado del tablero."""

    def get_name(self) -> str:
        return "AdaptivePhase_P2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)
        
        total_coins = len(state.board)
        max_cells = state.game.height * state.game.width
        progress = total_coins / max_cells if max_cells > 0 else 0.5

        corners = _calculate_corners_and_dangers(state, player, adversary)
        mobility = _calculate_mobility(state, player, adversary)
        frontier = _calculate_frontier(state, player, adversary)
        coins = _calculate_coin_parity(state.board, player, adversary)

        # Transición adaptativa:
        # 1. Las esquinas y el riesgo siempre son prioritarios (peso constante alto).
        # 2. La movilidad y la frontera importan muchísimo al inicio y van bajando gradualmente.
        # 3. La paridad de monedas importa poco/nada al inicio y domina en la fase final.
        w_corners = 0.45
        w_mobility = 0.35 * (1.0 - progress)
        w_frontier = 0.20 * (1.0 - progress)
        w_coins = 0.55 * progress

        return (
            (w_corners * corners)
            + (w_mobility * mobility)
            + (w_frontier * frontier)
            + (w_coins * coins)
        )