from rummikub import Rummikub
import numpy as np
import pytest


def test_end_of_game():
    game = Rummikub(2, learning=True)
    state = game.reset()
    game.players[0,:] = False
    game.players[0,[0,1,2]] = True
    game.render()
    game.next_move(-1, 0, 0)
    game.next_move(-1, 0, 1)
    game.next_move(-1, 0, 2)
    game.next_move(100, 0, 0)
    game.render()
    assert game.is_end() == True

def test_negative_end_of_game():
    game = Rummikub(2, learning=True)
    state = game.reset()
    game.players[0,[0,1,2]] = True
    game.render()
    game.next_move(-1, 0, 0)
    game.next_move(-1, 0, 1)
    game.next_move(-1, 0, 2)
    game.next_move(100, 0, 0)
    game.render()
    assert game.is_end() == False

def test_of_end_and_reset():
    game = Rummikub(2, learning=True)
    state = game.reset()
    game.players[0,:] = False
    game.players[0,[0,1,2]] = True
    game.render()
    game.next_move(-1, 0, 0)
    game.next_move(-1, 0, 1)
    game.next_move(-1, 0, 2)
    game.next_move(100, 0, 0)
    game.render()
    assert game.is_end() == True
    if game.is_end():
        state = game.reset()
        game.render()
    assert game.is_end() == False
