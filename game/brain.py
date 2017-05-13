import datetime
import threading
from game.ai import AlphaBetaPruner

__author__ = 'bengt'


class Brain(threading.Thread):
    def __init__(self, max_depth, mutex, q, pieces, first_player, second_player):
        self.mutex = mutex
        self.q = q
        self.max_depth = max_depth
        self.pieces = pieces
        self.first_player = first_player
        self.second_player = second_player
        self.has_started = False
        self.lifetime = None
        threading.Thread.__init__(self)

    def run(self):
        """ Starts the Minimax algorithm with the Alpha-Beta Pruning optimization
            and puts the result in a queue once done.
        """
        pruner = AlphaBetaPruner(self.mutex, self.max_depth, self.pieces, self.first_player)
        result = pruner.run()
        self.q.put(result)

