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
    """Evaluación lineal sin saturación de las 4 esquinas reales, cadenas de estabilidad en bordes
    y pseudo-esquinas indestructibles generadas por obstáculos en los muros (ej. B1, H2, A7, G8)."""
    board = state.board
    h = state.game.height
    w = state.game.width

    corners_setup = (
        ((1, 1), (2, 2), ((1, 2), (2, 1)), ((1, 0), (0, 1))),
        ((1, h), (2, h - 1), ((1, h - 1), (2, h)), ((1, 0), (0, -1))),
        ((w, 1), (w - 1, 2), ((w - 1, 1), (w, 2)), ((-1, 0), (0, 1))),
        ((w, h), (w - 1, h - 1), ((w, h - 1), (w - 1, h)), ((-1, 0), (0, -1))),
    )

    score = 0.0

    # 1. Las 4 esquinas principales, sus casillas trampa X/C y cadenas de bordes estables
    for corner, x_sq, c_sqs, edge_dirs in corners_setup:
        c_owner = board.get(corner)
        if c_owner in (player, adversary):
            sign = 1.0 if c_owner == player else -1.0
            score += 32.0 * sign

            # Propagación de fichas inamovibles encadenadas a la esquina
            for dx, dy in edge_dirs:
                cx, cy = corner[0] + dx, corner[1] + dy
                steps = 0
                while 1 <= cx <= w and 1 <= cy <= h and steps < 6:
                    if board.get((cx, cy)) == c_owner:
                        score += 6.0 * sign
                        cx += dx
                        cy += dy
                        steps += 1
                    else:
                        break
        elif c_owner is None:
            # Esquina libre: penalizar X (diagonal) y C (adyacentes en borde)
            x_owner = board.get(x_sq)
            if x_owner == player:
                score -= 19.0
            elif x_owner == adversary:
                score += 19.0

            for c_sq in c_sqs:
                c_sq_owner = board.get(c_sq)
                if c_sq_owner == player:
                    score -= 9.5
                elif c_sq_owner == adversary:
                    score += 9.5

    # 2. Detección automática de obstáculos en bordes y sus Pseudo-Esquinas adyacentes
    for (ox, oy), val in board.items():
        if val in (player, adversary):
            continue

        # Obstáculo en borde vertical izquierdo (x=1) o derecho (x=w)
        if ox in (1, w):
            ix = 2 if ox == 1 else w - 1
            for dy in (-1, 1):
                ny = oy + dy
                if 2 <= ny <= h - 1:
                    p_corner = (ox, ny)
                    pc_owner = board.get(p_corner)
                    if pc_owner in (player, adversary):
                        sign = 1.0 if pc_owner == player else -1.0
                        score += 26.0 * sign
                        cy = ny + dy
                        steps = 0
                        while 1 <= cy <= h and steps < 5:
                            if board.get((ox, cy)) == pc_owner:
                                score += 6.0 * sign
                                cy += dy
                                steps += 1
                            else:
                                break
                    elif pc_owner is None:
                        for trap_sq, pen in (
                            ((ix, ny + dy), 14.0),
                            ((ix, ny), 10.0),
                            ((ox, ny + dy), 8.5),
                        ):
                            t_owner = board.get(trap_sq)
                            if t_owner == player:
                                score -= pen
                            elif t_owner == adversary:
                                score += pen

        # Obstáculo en borde horizontal superior (y=1) o inferior (y=h)
        if oy in (1, h):
            iy = 2 if oy == 1 else h - 1
            for dx in (-1, 1):
                nx = ox + dx
                if 2 <= nx <= w - 1:
                    p_corner = (nx, oy)
                    pc_owner = board.get(p_corner)
                    if pc_owner in (player, adversary):
                        sign = 1.0 if pc_owner == player else -1.0
                        score += 26.0 * sign
                        cx = nx + dx
                        steps = 0
                        while 1 <= cx <= w and steps < 5:
                            if board.get((cx, oy)) == pc_owner:
                                score += 6.0 * sign
                                cx += dx
                                steps += 1
                            else:
                                break
                    elif pc_owner is None:
                        for trap_sq, pen in (
                            ((nx + dx, iy), 14.0),
                            ((nx, iy), 10.0),
                            ((nx + dx, oy), 8.5),
                        ):
                            t_owner = board.get(trap_sq)
                            if t_owner == player:
                                score -= pen
                            elif t_owner == adversary:
                                score += pen

    return score


def _calculate_mobility(state: TwoPlayerGameState, player: str, adversary: str) -> float:
    """Ventaja relativa de movimientos legales con penalización de pérdida de turno."""
    game = state.game
    my_moves = len(game._get_valid_moves(state.board, player))
    opp_moves = len(game._get_valid_moves(state.board, adversary))

    if my_moves == 0 and opp_moves == 0:
        return 0.0
    if my_moves == 0:
        return -90.0
    if opp_moves == 0:
        return 90.0
    return 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)


def _calculate_frontier(state: TwoPlayerGameState, player: str, adversary: str) -> float:
    """Minimiza las fichas que tocan casillas vacías (estabilidad estructural)."""
    board = state.board
    h = state.game.height
    w = state.game.width

    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    my_frontier = 0
    opp_frontier = 0

    for (x, y), cell_color in board.items():
        if cell_color not in (player, adversary):
            continue
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= w and 1 <= ny <= h and (nx, ny) not in board:
                if cell_color == player:
                    my_frontier += 1
                else:
                    opp_frontier += 1
                break

    total = my_frontier + opp_frontier
    if total > 0:
        return 100.0 * (opp_frontier - my_frontier) / total
    return 0.0


def _calculate_coin_parity(board: dict, player: str, adversary: str) -> float:
    """Paridad de monedas en el tablero (ignorando obstáculos)."""
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
    """Motor táctico de estrangulamiento (10/10 en tablero estándar) potenciado con
    cadenas de anclajes en muros y detección de pseudo-esquinas en tableros con obstáculos."""

    def get_name(self) -> str:
        return "Mobility_P2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)
        board = state.board
        game = state.game

        # 1. Movilidad y detección inmediata de fin de partida
        my_moves = len(game._get_valid_moves(board, player))
        opp_moves = len(game._get_valid_moves(board, adversary))

        if my_moves == 0 and opp_moves == 0:
            coin_diff = _calculate_coin_parity(board, player, adversary)
            if coin_diff > 0:
                return 10000.0 + coin_diff
            elif coin_diff < 0:
                return -10000.0 + coin_diff
            return 0.0

        if my_moves == 0:
            mobility_score = -92.0
        elif opp_moves == 0:
            mobility_score = 92.0
        else:
            mobility_score = 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)

        # 2. Esquinas, casillas X/C, cadenas de bordes y pseudo-esquinas de obstáculos
        tactical_corner_score = _calculate_corners_and_dangers(state, player, adversary)

        # 3. Frontera (movilidad potencial) para evitar abrir huecos al rival
        frontier_score = _calculate_frontier(state, player, adversary)

        return (
            (0.52 * tactical_corner_score)
            + (0.33 * mobility_score)
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
    """Heurística maestra multifase: Combina control del centro Sweet-16, cadenas de
    fichas indestructibles (esquinas y obstáculos), evaporación y cierre exacto."""

    def get_name(self) -> str:
        return "AdaptivePhase_P2"

    def evaluation_function(self, state: TwoPlayerGameState) -> float:
        player, adversary = _get_player_labels(state)
        board = state.board
        game = state.game
        h, w = game.height, game.width

        # 1. Movilidad y detección de estado terminal
        my_moves = len(game._get_valid_moves(board, player))
        opp_moves = len(game._get_valid_moves(board, adversary))

        my_coins = 0
        opp_coins = 0
        my_frontier = 0
        opp_frontier = 0
        my_center = 0
        opp_center = 0

        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        # Pasada única ultrarrápida sobre el tablero (Monedas + Frontera + Centro Sweet-16)
        for (x, y), cell_color in board.items():
            if cell_color == player:
                my_coins += 1
                is_mine = True
            elif cell_color == adversary:
                opp_coins += 1
                is_mine = False
            else:
                continue  # Ignora obstáculos (ej. B1, H2, A7, G8)

            # Control compacto del centro (Sweet-16: filas/columnas 3 a 6)
            if 3 <= x <= w - 2 and 3 <= y <= h - 2:
                if is_mine:
                    my_center += 1
                else:
                    opp_center += 1

            # Comprobación de ficha frontera (adyacente a casilla vacía)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx <= w and 1 <= ny <= h and (nx, ny) not in board:
                    if is_mine:
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
        if my_moves == 0:
            mobility_score = -95.0
        elif opp_moves == 0:
            mobility_score = 95.0
        else:
            mobility_score = 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)

        # Puntuación de frontera y control central normalizado
        tot_frontier = my_frontier + opp_frontier
        frontier_score = (
            100.0 * (opp_frontier - my_frontier) / tot_frontier if tot_frontier > 0 else 0.0
        )

        tot_center = my_center + opp_center
        center_score = (
            100.0 * (my_center - opp_center) / tot_center if tot_center > 0 else 0.0
        )

        # 2. Esquinas, Trampas X/C, Pseudo-Esquinas de Obstáculos y Cadenas Estables
        corner_and_stability = _calculate_corners_and_dangers(state, player, adversary)

        # 3. Pesos dinámicos por fase de juego (Apertura -> Medio Juego -> Desenlace)
        max_playable = (h * w) - 4
        progress = max(0.0, min(1.0, (tot_coins - 4) / max_playable)) if max_playable > 0 else 0.5

        if progress < 0.32:
            # APERTURA: Prioridad a esquinas/pseudo-esquinas, movilidad, frontera compacta y evaporación
            return (
                (0.50 * corner_and_stability)
                + (0.35 * mobility_score)
                + (0.22 * frontier_score)
                + (0.10 * center_score)
                - (0.06 * coin_score)
            )
        elif progress < 0.82:
            # MEDIO JUEGO: Dominio de esquinas/obstáculos, cadenas de bordes, movilidad y frontera
            # (sin inflar monedas prematuramente para no perder movilidad frente a rivales agresivos)
            return (
                (0.56 * corner_and_stability)
                + (0.32 * mobility_score)
                + (0.18 * frontier_score)
                + (0.06 * center_score)
            )
        else:
            # FINAL DE PARTIDA (últimos ~10 turnos): Conversión decisiva en fichas y esquinas
            end_factor = (progress - 0.82) / 0.18  # Escala de 0.0 a 1.0 en la recta final
            w_mob = 0.28 * (1.0 - 0.5 * end_factor)
            w_coins = 0.15 + (0.45 * end_factor)
            return (
                (0.56 * corner_and_stability)
                + (w_mob * mobility_score)
                + (0.10 * (1.0 - end_factor) * frontier_score)
                + (w_coins * coin_score)
            )