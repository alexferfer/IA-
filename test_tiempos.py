import timeit
from game import Player, TwoPlayerGameState
from reversi import Reversi, from_array_to_dictionary_board
from strategy import MinimaxStrategy, MinimaxAlphaBetaStrategy, RandomStrategy
from heuristic import simple_evaluation_function

class Heuristic1:
    def get_name(self):
        return "Heuristic1"
    def evaluate(self, state):
        return simple_evaluation_function(state)

def setup_initial_state():
    # 1. Estrategia dummy y jugadores
    dummy_strategy = RandomStrategy()
    player_b = Player('B', dummy_strategy)
    player_w = Player('W', dummy_strategy)
    
    board_array = [
        '........',
        '........',
        '........',
        '...WB...',
        '...BW...',
        '........',
        '........',
        '........'
    ]
    
    board_dict = from_array_to_dictionary_board(board_array)

    game = Reversi(
        player1=player_b,
        player2=player_w,
        height=8,
        width=8
    )

    
    return TwoPlayerGameState(
        game=game,
        board=board_dict,
        initial_player=player_b
    )

def run_test():
    state = setup_initial_state()
    h1 = Heuristic1()

    print("Calculando tiempos de ejecución (esto puede tardar un par de minutos)...\n")
    print(f"{'Estrategia':<25} | {'Profundidad':<12} | {'Tiempo (segundos)':<15}")
    print("-" * 58)

    for depth in [3, 4]:
        
        minimax = MinimaxStrategy(heuristic=h1, max_depth_minimax=depth)
        t_minimax = timeit.timeit(lambda: minimax.next_move(state), number=1)
        print(f"{'Minimax Puro':<25} | {depth:<12} | {t_minimax:<15.4f}")

        
        alfabeta = MinimaxAlphaBetaStrategy(heuristic=h1, max_depth_minimax=depth)
        t_alfabeta = timeit.timeit(lambda: alfabeta.next_move(state), number=1)
        print(f"{'Minimax Alfa-Beta':<25} | {depth:<12} | {t_alfabeta:<15.4f}")
        
        
        speedup = t_minimax / t_alfabeta if t_alfabeta > 0 else 0
        print(f"--> Factor de aceleración (Speedup) a D={depth}: {speedup:.2f}x más rápido\n")

if __name__ == '__main__':
    run_test()