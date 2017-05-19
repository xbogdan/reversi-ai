from game.settings import *
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()


class AlphaBetaPruner(object):
    """Alpha-Beta Pruning algorithm."""

    def __init__(self, mutex, max_depth, pieces, first_player):
        self.mutex = mutex
        self.board = 0
        self.move = 1
        self.white = 2
        self.black = 3
        self.max_depth = max_depth
        self.infinity = 1.0e400
        self.first_player, self.second_player = (WHITE_ID, BLACK_ID) \
            if first_player == WHITE else (BLACK_ID, WHITE_ID)
        self.state = self.make_state(pieces)

    def make_state(self, pieces):
        """ Returns a tuple in the form of "current_state", that is: (current_player, state).
        """
        results = {BOARD: BOARD_ID, MOVE: BOARD_ID, WHITE: WHITE_ID, BLACK: BLACK_ID}
        return self.first_player, [results[p.get_state()] for p in pieces]

    def run(self):
        return self.pvsplit(current_state=self.state, depth=0, alpha=-self.infinity, beta=self.infinity, action=None)[1]

    def pvsplit(self, current_state, depth, alpha, beta, action):
        actions = AlphaBetaPruner.actions(current_state)

        if (self.is_leaf(depth) or not actions) and action:
            return AlphaBetaPruner.evaluation(current_state, AlphaBetaPruner.opponent(current_state[0])), action

        next_action = actions[0]
        next_state = AlphaBetaPruner.next_state(current_state, next_action)

        score, action = self.pvsplit(next_state, depth + 1, alpha, beta, next_action)

        if score > beta:
            return beta, next_action
        if score > alpha:
            alpha = score

        # parallel
        current_rank = 1
        total_task_nr = len(actions[1:])
        task_distribution = {}

        for order, action_ in enumerate(actions[1:]):
            next_state = AlphaBetaPruner.next_state(current_state, action_)
            if SIZE > 1:
                """
                Multi threaded
                Sending data
                """
                data = {
                    'next_state': next_state,
                    'depth': depth+1,
                    'max_depth': self.max_depth,
                    'alpha': alpha,
                    'beta': beta,
                    'order': order
                }
                COMM.isend(data, dest=current_rank, tag=1)
                task_distribution[order] = {
                    'rank': current_rank,
                    'data': data,
                    'action': action_,
                    'score': None
                }
                current_rank = current_rank+1 if current_rank+1 < SIZE else 1
            else:
                """ Single threaded """
                score = AlphaBetaPruner.alpha_beta(next_state, depth+1, self.max_depth, alpha, beta)

                if score > beta:
                    return beta, action_

                if score > alpha:
                    alpha = score
                    next_action = action_

        if SIZE > 1:
            """
            Multi threaded 
            Receiving data
            """
            for _ in range(0, total_task_nr):
                req = COMM.irecv(tag=1)
                response = req.wait()
                task_distribution[response[0]]['score'] = response[1]

            for key, value in task_distribution.items():
                score = value['score']
                action_ = value['action']
                if score > beta:
                    return beta, action_

                if score > alpha:
                    alpha = score
                    next_action = action_

        return alpha, next_action

    @staticmethod
    def alpha_beta(current_state, depth, max_depth, alpha, beta):
        actions = AlphaBetaPruner.actions(current_state)

        if depth > max_depth or not actions:
            return AlphaBetaPruner.evaluation(current_state, AlphaBetaPruner.opponent(current_state[0]))

        best_score = beta if AlphaBetaPruner.is_min(depth) else alpha

        for action in actions:
            next_state = AlphaBetaPruner.next_state(current_state, action)

            if AlphaBetaPruner.is_min(depth):
                score = AlphaBetaPruner.alpha_beta(next_state, depth + 1, max_depth, alpha, best_score)
                if score <= alpha:
                    return alpha
                if score < best_score:
                    best_score = score
            else:
                score = AlphaBetaPruner.alpha_beta(next_state, depth + 1, max_depth, best_score, beta)
                if score >= beta:
                    return beta
                if score > best_score:
                    best_score = score

        return best_score

    @staticmethod
    def is_min(depth):
        return depth % 2 == 1

    @staticmethod
    def evaluation(current_state, player_to_check):
        """ Returns a positive value when the player wins.
            Returns zero when there is a draw.
            Returns a negative value when the opponent wins."""

        player_state, state = current_state
        player = player_to_check
        opponent = AlphaBetaPruner.opponent(player)

        # count_eval stands for the player with the most pieces next turn
        moves = AlphaBetaPruner.get_moves(player, opponent, state)
        player_pieces = len([p for p in state if p == player])
        opponent_pieces = len([p for p in state if p == opponent])
        count_eval = 1 if player_pieces > opponent_pieces else \
            0 if player_pieces == opponent_pieces else \
            -1

        # moves_player    = moves
        # moves_oppponent = AlphaBetaPruner.get_moves(opponent, player, state)
        # move_eval       = 1 if moves_player > moves_oppponent else \
        #                   0 if moves_player == moves_oppponent else \
        #                  -1

        corners_player = (state[0] == player) + \
                         (state[(WIDTH - 1)] == player) + \
                         (state[(WIDTH - 1) * WIDTH] == player) + \
                         (state[WIDTH**2 - 1] == player)
        corners_opponent = -1 * (state[0] == opponent) + \
                           (state[(WIDTH - 1)] == opponent) + \
                           (state[(WIDTH - 1) * WIDTH] == opponent) + \
                           (state[WIDTH**2 - 1] == opponent)
        corners_eval = corners_player + corners_opponent

        edges_player = len([x for x in state if state == player and (state % WIDTH == 0 or state % WIDTH == WIDTH)]) / (
            WIDTH * HEIGHT)
        edges_opponent = -1 * len([x for x in state if state == opponent and (state % WIDTH == 0 or state % WIDTH == WIDTH)]) / (
            WIDTH * HEIGHT)
        edges_eval = edges_player + edges_opponent

        eval = count_eval * 2 + corners_eval * 1.5 + edges_eval * 1.2

        return eval

    @staticmethod
    def actions(current_state):
        """ Returns a list of tuples as coordinates for the valid moves for the current player.
        """
        player, state = current_state
        return AlphaBetaPruner.get_moves(player, AlphaBetaPruner.opponent(player), state)

    @staticmethod
    def opponent(player):
        """ Returns the opponent of the specified player.
        """
        return BLACK_ID if player is WHITE_ID else WHITE_ID

    @staticmethod
    def next_state(current_state, action):
        """ Returns the next state in the form of a "current_state" tuple, (current_player, state).
        """
        player, state = current_state
        # player, state = current_state[0], current_state[1].copy()
        opponent = AlphaBetaPruner.opponent(player)

        xx, yy = action
        state[xx + (yy * WIDTH)] = player
        for d in DIRECTIONS:
            tile = xx + (yy * WIDTH) + d
            if tile < 0 or tile >= (WIDTH - 1) * WIDTH:
                continue

            while state[tile] != BOARD_ID:
                state[tile] = player
                tile += d
                if tile < 0 or tile >= WIDTH * HEIGHT:
                    tile -= d
                    break

        return opponent, state

    @staticmethod
    def get_moves(player, opponent, state):
        """ Returns a generator of (x,y) coordinates.
        """
        moves = [AlphaBetaPruner.mark_move(player, opponent, tile, state, d)
                 for tile in range(WIDTH * HEIGHT)
                 for d in DIRECTIONS
                 if not outside_board(tile, d) and state[tile] == player]

        return [(x, y) for found, x, y, tile in moves if found]

    @staticmethod
    def mark_move(player, opponent, tile, pieces, direction):
        """ Returns True whether the current tile piece is a move for the current player,
            otherwise it returns False.
        """
        if not outside_board(tile, direction):
            tile += direction
        else:
            return False, int(tile % WIDTH), int(tile / HEIGHT), tile

        if pieces[tile] == opponent:
            while pieces[tile] == opponent:
                if outside_board(tile, direction):
                    break
                else:
                    tile += direction

            if pieces[tile] == BOARD_ID:
                return True, int(tile % WIDTH), int(tile / HEIGHT), tile

        return False, int(tile % WIDTH), int(tile / HEIGHT), tile

    def is_leaf(self, depth):
        """ Returns True when the cutoff limit has been reached.
        """
        return depth > self.max_depth


