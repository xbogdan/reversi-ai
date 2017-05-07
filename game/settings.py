__author__ = 'bengt'

BOARD, WHITE, BLACK, MOVE = 'BOARD', 'WHITE', 'BLACK', 'MOVE'
WIDTH, HEIGHT = 8, 8
NORTH = -HEIGHT
NORTHEAST = -HEIGHT + 1
EAST = 1
SOUTHEAST = HEIGHT + 1
SOUTH = HEIGHT
SOUTHWEST = HEIGHT - 1
WEST = - 1
NORTHWEST = -HEIGHT - 1

DIRECTIONS = (NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST)


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_opponent(player):
    if player == WHITE:
        return BLACK
    elif player == BLACK:
        return WHITE
    else:
        raise ValueError


class NoMovesError(Exception):
    pass


def outside_board(tile, direction):
    tile_top = 0 <= tile <= (WIDTH - 1)
    tile_bot = (WIDTH - 1) * WIDTH <= tile <= WIDTH**2 - 1
    tile_right = tile % WIDTH == (WIDTH - 1)
    tile_left = tile % WIDTH == 0
    if (direction in (NORTH, NORTHEAST, NORTHWEST) and tile_top) or \
            (direction in (SOUTH, SOUTHWEST, SOUTHEAST) and tile_bot) or \
            (direction in (NORTHEAST, EAST, SOUTHEAST) and tile_right) or \
            (direction in (NORTHWEST, WEST, SOUTHWEST) and tile_left):
        return True
    return False
