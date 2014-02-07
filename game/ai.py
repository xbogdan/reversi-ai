import sys

__author__ = 'bengt'

from game.settings import *


class AlphaBetaPruner(object):
    """Alpha-Beta Pruning algorithm."""


    def __init__(self, pieces, first_player, second_player):
        self.board = 0
        self.move = 1
        self.white = 2
        self.black = 3

        self.infinity = 1.0e400

        self.first_player, self.second_player = (self.white, self.black) if first_player == WHITE else (
            self.black, self.white)

        self.state = self.make_state(pieces)
        print(self.state)

    def make_state(self, pieces):
        results = {BOARD: self.board, MOVE: self.board, WHITE: self.white, BLACK: self.black}
        return self.first_player, [results[p.get_state()] for p in pieces]

    def alpha_beta_search(self):
        """Returns an action"""
        player, state = self.state
        opponent = self.opponent(player)
        depth = 0
        fn = lambda action: self.min_value(depth, self.next_state(self.state, action), -self.infinity,
                                           self.infinity)
        maxfn = lambda value: value[0]
        actions = self.actions(self.state)
        moves = [(fn(action), action) for action in actions]
        return max(moves, key=maxfn)[1]

    def max_value(self, depth, current_state, alpha, beta):
        player, state = current_state
        if self.cutoff_test(current_state, depth):
            return self.utility(current_state, self.first_player)

        value = -self.infinity

        actions = self.actions(current_state)
        for action in actions:
            value = max([value, self.min_value(depth + 1, self.next_state(current_state, action), alpha, beta)])
            if value >= beta:
                return value
            alpha = max(alpha, value)

        return value

    def min_value(self, depth, state, alpha, beta):
        if self.cutoff_test(state, depth):
            return self.utility(state, self.second_player)

        value = self.infinity

        actions = self.actions(state)
        for action in actions:
            value = min([value, self.max_value(depth + 1, self.next_state(state, action), alpha, beta)])
            if value <= alpha:
                return value
            beta = min([beta, value])

        return value

    def utility(self, current_state, player_to_check):
        """
        Returns 1 for a win for 'player'
        Returns 0 for a draw for 'player'
        Returns -1 for a lose for 'player'
        """
        player, state = current_state
        moves = self.get_moves(player_to_check, self.opponent(player_to_check), state)
        white_pieces = len([p for p in state if p == self.white])
        black_pieces = len([p for p in state if p == self.black])
        p1, p2 = (white_pieces, black_pieces) if player_to_check == self.white else (black_pieces, white_pieces)
        if p1 > p2:
            return 1
        elif p1 == p2:
            return 0
        else:
            return -1

    def terminal_test(self, state):
        return len(self.get_moves(state[0], self.opponent(state[0]), state[1])) == 0 # No moves in the state...

    def actions(self, current_state):
        """Returns """
        player, state = current_state
        return self.get_moves(player, self.opponent(player), state)

    def opponent(self, player):
        return self.second_player if player is self.first_player else self.first_player

    def next_state(self, current_state, action):
        player, state = current_state
        opponent = self.opponent(player)
        #moves = self.get_moves(player, opponent, state)

        xx, yy = action
        state[xx + (yy * WIDTH)] = player
        for d in DIRECTIONS:
            tile = xx + (yy * WIDTH) + d
            if tile < 0 or tile >= 64:
                continue

            #if state[tile] == opponent:
            while state[tile] != self.board:
                state[tile] = player
                tile += d
                if tile < 0 or tile >= WIDTH * HEIGHT:
                    tile -= d
                    break

        return opponent, state

    def get_moves(self, player, opponent, state):
        """ Returns a generator of (x,y) coordinates.
        """
        moves = [self.mark_move(player, opponent, x, y, state, d)
                 for d in DIRECTIONS
                 for x in range(WIDTH)
                 for y in range(HEIGHT)
                 if (x + (y * WIDTH) >= 0) and (x + (y * WIDTH) < WIDTH * HEIGHT) and state[x + (y * WIDTH)] == player]
        return [(xx, yy) for found, xx, yy, tile in moves if found]

    def mark_move(self, player, opponent, x, y, pieces, direction):
        tile = (x + (y * WIDTH)) + direction

        if tile < 0 or tile >= WIDTH * HEIGHT:
            return False, int(tile % WIDTH), int(tile / HEIGHT), tile

        if pieces[tile] == opponent:
            while pieces[tile] == opponent:
                tile += direction
                if tile < 0 or tile >= WIDTH*HEIGHT:
                    return False, int(tile % WIDTH), int(tile / HEIGHT), tile

            if pieces[tile] == self.board:
                return True, int(tile % WIDTH), int(tile / HEIGHT), tile

        return False, int(tile % WIDTH), int(tile / HEIGHT), tile

    def cutoff_test(self, state, depth):
        return depth > 5

    def outside_board(self, tile, direction):
        if (direction == NORTH and 0 <= tile <= 7) or \
           (direction == SOUTH and 56 <= tile <= 63) or \
           (direction in (NORTHEAST, EAST, SOUTHEAST) and tile % WIDTH == 7) or \
           (direction in (NORTHWEST, WEST, SOUTHWEST) and tile % WIDTH == 0):
            return True
        return False