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
    """Motor táctico de estrangulamiento: Movilidad asimétrica, control lineal de esquinas,
    diferenciación estricta entre trampas X/C y propagación de anclajes en bordes."""

    def get_name(self) -> str:
        return "Mobility_P2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)
        board = state.board
        game = state.game
        h, w = game.height, game.width

        # 1. Movilidad y detección inmediata de fin de partida (sin coste extra)
        my_moves = len(game._get_valid_moves(board, player))
        opp_moves = len(game._get_valid_moves(board, adversary))

        if my_moves == 0 and opp_moves == 0:
            coin_diff = _calculate_coin_parity(board, player, adversary)
            if coin_diff > 0:
                return 10000.0 + coin_diff
            elif coin_diff < 0:
                return -10000.0 + coin_diff
            return 0.0

        tot_moves = my_moves + opp_moves
        # Penalización severa si nos quedamos sin turno (pase forzado)
        if my_moves == 0:
            mobility_score = -90.0
        elif opp_moves == 0:
            mobility_score = 90.0
        else:
            mobility_score = 100.0 * (my_moves - opp_moves) / tot_moves

        # 2. Evaluación lineal no saturada de Esquinas, casillas X, casillas C y Anclajes
        corners_data = (
            ((1, 1), (2, 2), ((1, 2), (2, 1))),
            ((1, h), (2, h - 1), ((1, h - 1), (2, h))),
            ((w, 1), (w - 1, 2), ((w - 1, 1), (w, 2))),
            ((w, h), (w - 1, h - 1), ((w, h - 1), (w - 1, h))),
        )

        tactical_corner_score = 0.0
        for corner, x_sq, c_sqs in corners_data:
            c_owner = board.get(corner)
            if c_owner == player:
                tactical_corner_score += 30.0
                # Si la esquina es nuestra, las casillas C adyacentes son anclajes seguros
                for c_sq in c_sqs:
                    if board.get(c_sq) == player:
                        tactical_corner_score += 6.0
            elif c_owner == adversary:
                tactical_corner_score -= 30.0
                for c_sq in c_sqs:
                    if board.get(c_sq) == adversary:
                        tactical_corner_score -= 6.0
            else:
                # Esquina libre: pisar la X es letal, pisar la C es muy peligroso
                x_owner = board.get(x_sq)
                if x_owner == player:
                    tactical_corner_score -= 18.0
                elif x_owner == adversary:
                    tactical_corner_score += 18.0

                for c_sq in c_sqs:
                    c_sq_owner = board.get(c_sq)
                    if c_sq_owner == player:
                        tactical_corner_score -= 9.0
                    elif c_sq_owner == adversary:
                        tactical_corner_score += 9.0

        # 3. Frontera (movilidad potencial) para evitar abrir huecos al rival
        frontier_score = _calculate_frontier(state, player, adversary)

        return (
            (0.50 * tactical_corner_score)
            + (0.35 * mobility_score)
            + (0.15 * frontier_score)
        )


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
    """Heurística maestra multifase: Combina matriz posicional dinámica, cadenas de
    fichas indestructibles (estabilidad de borde anclada), evaporación en apertura y cierre exacto."""

    def get_name(self) -> str:
        return "AdaptivePhase_P2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)
        board = state.board
        game = state.game
        h, w = game.height, game.width

        # 1. Movilidad y detección de estado terminal en un solo paso
        my_moves = len(game._get_valid_moves(board, player))
        opp_moves = len(game._get_valid_moves(board, adversary))

        my_coins = 0
        opp_coins = 0
        my_frontier = 0
        opp_frontier = 0
        center_control = 0.0

        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        # Pasada única ultrarrápida sobre el tablero (Monedas + Frontera + Centro Sweet-16)
        for (x, y), cell_color in board.items():
            if cell_color == player:
                my_coins += 1
                sign = 1
            elif cell_color == adversary:
                opp_coins += 1
                sign = -1
            else:
                continue  # Ignora obstáculos (ej. B1, H2, A7, G8)

            # Bonificación por control del centro (casillas 3..6) y bordes seguros (A/B squares)
            if 3 <= x <= w - 2 and 3 <= y <= h - 2:
                center_control += 2.5 * sign
            elif (x in (1, w) and 3 <= y <= h - 2) or (y in (1, h) and 3 <= x <= w - 2):
                center_control += 4.0 * sign

            # Comprobación de ficha frontera (adyacente a casilla vacía)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= w and 1 <= ny <= h and (nx, ny) not in board:
                    if sign == 1:
                        my_frontier += 1
                    else:
                        opp_frontier += 1
                    break

        tot_coins = my_coins + opp_coins
        coin_score = 100.0 * (my_coins - opp_coins) / tot_coins if tot_coins > 0 else 0.0

        # Si la partida ha terminado, devolver utilidad real máxima para asegurar victoria
        if my_moves == 0 and opp_moves == 0:
            if my_coins > opp_coins:
                return 10000.0 + coin_score
            elif my_coins < opp_coins:
                return -10000.0 + coin_score
            return 0.0

        # Puntuación de movilidad con castigo por pérdida de turno
        tot_moves = my_moves + opp_moves
        if my_moves == 0:
            mobility_score = -95.0
        elif opp_moves == 0:
            mobility_score = 95.0
        else:
            mobility_score = 100.0 * (my_moves - opp_moves) / tot_moves

        # Puntuación de frontera
        tot_frontier = my_frontier + opp_frontier
        frontier_score = (
            100.0 * (opp_frontier - my_frontier) / tot_frontier if tot_frontier > 0 else 0.0
        )

        # 2. Esquinas, Trampas X/C dinámicas y Cadenas de Estabilidad Anclada en bordes
        corners_setup = (
            ((1, 1), (2, 2), ((1, 2), (2, 1)), ((1, 0), (0, 1))),
            ((1, h), (2, h - 1), ((1, h - 1), (2, h)), ((1, 0), (0, -1))),
            ((w, 1), (w - 1, 2), ((w - 1, 1), (w, 2)), ((-1, 0), (0, 1))),
            ((w, h), (w - 1, h - 1), ((w, h - 1), (w - 1, h)), ((-1, 0), (0, -1))),
        )

        corner_and_stability = 0.0
        for corner, x_sq, c_sqs, edge_dirs in corners_setup:
            c_owner = board.get(corner)
            if c_owner in (player, adversary):
                c_sign = 1.0 if c_owner == player else -1.0
                corner_and_stability += 32.0 * c_sign

                # Si la esquina está tomada, contamos las fichas indestructibles encadenadas en sus bordes
                for dx, dy in edge_dirs:
                    cx, cy = corner[0] + dx, corner[1] + dy
                    steps = 0
                    while 1 <= cx <= w and 1 <= cy <= h and steps < 6:
                        if board.get((cx, cy)) == c_owner:
                            corner_and_stability += 5.5 * c_sign
                            cx += dx
                            cy += dy
                            steps += 1
                        else:
                            break
            else:
                # Esquina vacía: penalizar severamente regalar la esquina mediante X o C
                x_owner = board.get(x_sq)
                if x_owner == player:
                    corner_and_stability -= 20.0
                elif x_owner == adversary:
                    corner_and_stability += 20.0

                for c_sq in c_sqs:
                    c_sq_owner = board.get(c_sq)
                    if c_sq_owner == player:
                        corner_and_stability -= 10.0
                    elif c_sq_owner == adversary:
                        corner_and_stability += 10.0

        # 3. Pesos dinámicos por fase de juego (Apertura -> Medio Juego -> Desenlace)
        max_playable = (h * w) - 4
        progress = max(0.0, min(1.0, (tot_coins - 4) / max_playable)) if max_playable > 0 else 0.5

        if progress < 0.30:
            # APERTURA: Prioridad absoluta a movilidad, frontera compacta, centro y EVAPORACIÓN
            # (tener menos fichas al inicio ahoga los movimientos del adversario)
            return (
                (0.45 * corner_and_stability)
                + (0.38 * mobility_score)
                + (0.25 * frontier_score)
                + (0.15 * center_control)
                - (0.10 * coin_score)
            )
        elif progress < 0.78:
            # MEDIO JUEGO: Dominio de esquinas, cadenas de bordes estables, movilidad y frontera
            return (
                (0.55 * corner_and_stability)
                + (0.30 * mobility_score)
                + (0.20 * frontier_score)
                + (0.10 * center_control)
                + (0.08 * coin_score)
            )
        else:
            # FINAL DE PARTIDA (ENDGAME): Conversión agresiva en fichas definitivas y esquinas
            end_factor = (progress - 0.78) / 0.22  # Escala de 0.0 a 1.0 en los últimos turnos
            w_mob = 0.22 * (1.0 - 0.6 * end_factor)
            w_coins = 0.25 + (0.45 * end_factor)
            return (
                (0.55 * corner_and_stability)
                + (w_mob * mobility_score)
                + (0.10 * (1.0 - end_factor) * frontier_score)
                + (w_coins * coin_score)
            )
