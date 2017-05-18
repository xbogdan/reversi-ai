#!/usr/bin/env python3

import argparse
from mpi4py import MPI
from game.game import Game
from datetime import datetime

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


def main():
    """ Reversi game with human player vs AI player.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', help="Max depth the tree is allowed to think before making its move.",
                        type=int, default=5)
    parser.add_argument('--display-moves', help="Whether legal moves should be displayed or not.", action='store_true')
    parser.add_argument('--colour', help="Display the game in 256 colours.", action='store_true')
    parser.add_argument('--player', help="If you want to play against the ai", action='store_true')
    parser.add_argument('--ai', help="If you want the ais to play against each other", action='store_true')

    args = parser.parse_args()

    if args.depth < 0:
        exit()

    players = []
    if args.player:
        players = ['player', 'ai']
    if args.ai:
        players = ['ai', 'ai']
    if not players:
        players = ['player', 'ai']

    game = Game(max_depth=args.depth,
                display_moves=args.display_moves,
                colour=args.colour,
                players=players)
    game.run()


if __name__ == "__main__":

    if RANK == 0:
        t_start = datetime.now()

        main()

        duration = datetime.now() - t_start
        print(f'Finished in {duration}')
    else:
        from game.ai import AlphaBetaPruner

        while True:
            req = COMM.irecv(source=0, tag=1)
            data = req.wait()
            if data == -1:
                break

            score = AlphaBetaPruner.alpha_beta(current_state=data['next_state'], depth=data['depth'],
                                               max_depth=data['max_depth'], alpha=data['alpha'], beta=data['beta'])

            # print(f'Worker {RANK} received a message {data} - {score}')
            req = COMM.isend((data['order'], score), dest=0, tag=1)
