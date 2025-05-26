import random

import numpy as np
import numpy.typing as npt

from hw2.utils import utility, successors, Node, Tree, GameStrategy
import streamlit as st

"""
Alpha Beta Search
"""


def max_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the max value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    utl = utility(state, k)
    if utl != None:
      return (utl,state)

    #intial v
    v= -1000000
    for state2 in successors(state,'X'):
      (v2,a2) = min_value(state2, alpha, beta, k)
      if v2 > v:
        v=v2
        move = state2
        if v > alpha:
          alpha = v
      if v >= beta:
        return (v, move)
    return (v, move)


def min_value(state: npt.ArrayLike, alpha: float, beta: float, k: int):
    """Find the min value given state and return the utility and the game board
    state after the move. Please see Lecture 6 for pseudocode.

    Args:
        state (np.ndarray): the state of the game board, mxn numpy array.
        alpha (float): the alpha value
        beta (float): the beta value
        k (int): the number of consecutive marks
    Returns:
        tuple[float, np.ndarray]: utility value and the board after the move
    """

    # TODO:
    utl = utility(state, k)
    if utl != None:
      return (utl,state)

    #intial v
    v= 1000000
    for state2 in successors(state,'O'):
      (v2,a2) = max_value(state2, alpha, beta, k)
      if v2 < v:
        v=v2
        move = state2
        if v < beta:
          beta = v
      if v <= alpha:
        return (v, move)
    return (v, move)
    

"""
Monte Carlo Tree Search
"""


def select(tree: "Tree", state: npt.ArrayLike, k: int, alpha: float):
    """Starting from state, find a terminal node or node with unexpanded
    children. If all children of a node are in tree, move to the one with the
    highest UCT value.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
        alpha (float): exploration parameter
    Returns:
        np.ndarray: the game board state
    """

    # TODO:
    current_node = tree.get(state)
    uct_max=-1000000
    state_max = state

    succ=successors(state,current_node.player)

    #if terminal state: return current node
    if utility(state,k) != None:
        return state_max
    for state2 in succ:
       
        #check if successor is in the tree. If not, return current node
       if tree.get(state2) == None:
          return current_node.state
       
       
       #calculate the uct and keep the max uct
       node2 = tree.get(state2)
       uct2 = node2.w/node2.N+alpha*np.sqrt(np.log(current_node.N)/node2.N)
       if uct2 > uct_max:
          uct_max = uct2
          state_max = node2.state

    return select(tree, state_max, k, alpha)


def expand(tree: "Tree", state: npt.ArrayLike, k: int):
    """Add a child node of state into the tree if it's not terminal and return
    tree and new state, or return current tree and state if it is terminal.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        k (int): the number of consecutive marks
    Returns:
        tuple[utils.Tree, np.ndarray]: the tree and the game state
    """
    # TODO:
    current_node = tree.get(state)
    if current_node.player == "X":
       next_player = "O"
    else:
       next_player = "X"

    #if terminal state: return current tree and state
    if utility(state,k) != None:
        return (tree, state)

    succ=successors(current_node.state,current_node.player)
    for state2 in succ:
       
        #check if successor is in the tree. If not, add and return new state and tree
       if tree.get(state2) == None:
          tree.add(Node(state2,current_node,next_player,0,1))
          return (tree, state2)
       
    return (tree, state)


def simulate(state: npt.ArrayLike, player: str, k: int):
    """Run one game rollout from state to a terminal state using random
    playout policy and return the numerical utility of the result.

    Args:
        state (np.ndarray): the game board state
        player (string): the player, `O` or `X`
        k (int): the number of consecutive marks
    Returns:
        float: the utility
    """

    # TODO:
    utl = utility(state,k)
    state2=state

    players = [ "X","O"]
    if player == "X":
       i=0
    else:
       i=1
 
    while utl == None:
        succ=successors(state2,players[i])
        #print(len(succ))
        num = np.random.choice(len(succ),1)
        state2 = succ[int(num)]

        utl= utility(state2,k)

        i+=1
        i=i%2

    return utl


def backprop(tree: "Tree", state: npt.ArrayLike, result: float):
    """Backpropagate result from state up to the root.
    All nodes on path have N, number of plays, incremented by 1.
    If result is a win for a node's parent player, w is incremented by 1.
    If result is a draw, w is incremented by 0.5 for all nodes.

    Args:
        tree (utils.Tree): the search tree
        state (np.ndarray): the game board state
        result (float): the result / utility value

    Returns:
        utils.Tree: the game tree
    """

    # TODO:
    #Go through each parent node
    node2 = tree.get(state)
    while node2 != None:
       node2.N += 1
       #if X wins, O state +=1
       if result == 1 :
          if node2.player == "O":
             node2.w = node2.w+ 1


       #if O wins, X state +=1
       if result == -1 :
          if node2.player == "X":
             node2.w = node2.w+ 1
        
       if result == 0:
          node2.w = node2.w+ 0.5
       node2 = node2.parent

    return tree
        



# ******************************************************************************
# ****************************** ASSIGNMENT ENDS *******************************
# ******************************************************************************


def MCTS(state: npt.ArrayLike, player: str, k: int, rollouts: int, alpha: float):
    # MCTS main loop: Execute MCTS steps rollouts number of times
    # Then return successor with highest number of rollouts
    tree = Tree(Node(state, None, player, 0, 1))

    for i in range(rollouts):
        leaf = select(tree, state, k, alpha)
        tree, new = expand(tree, leaf, k)
        result = simulate(new, tree.get(new).player, k)
        tree = backprop(tree, new, result)

    nxt = None
    plays = 0

    for s in successors(state, tree.get(state).player):
        if tree.get(s).N > plays:
            plays = tree.get(s).N
            nxt = s

    return nxt


def ABS(state: npt.ArrayLike, player: str, k: int):
    # ABS main loop: Execute alpha-beta search
    # X is maximizing player, O is minimizing player
    # Then return best move for the given player
    if player == "X":
        value, move = max_value(state, -float("inf"), float("inf"), k)
    else:
        value, move = min_value(state, -float("inf"), float("inf"), k)

    return value, move

def play(state: npt.ArrayLike, player: str):
    cin = st.text_input("Enter something:")


    cin = cin.split(",")
    i,j = int(cin[0]),int(cin[1])

    if state[i, j] == ".":
        new = np.copy(state)
        new[i, j] = player
        return new
    
    

def game_loop(
    state: npt.ArrayLike,
    player: str,
    k: int,
    Xstrat: GameStrategy = GameStrategy.RANDOM,
    Ostrat: GameStrategy = GameStrategy.RANDOM,
    rollouts: int = 0,
    mcts_alpha: float = 0.01,
    print_result: bool = False,
):
    # Plays the game from state to terminal
    # If random_opponent, opponent of player plays randomly, else same strategy as player
    # rollouts and alpha for MCTS; if rollouts is 0, ABS is invoked instead
    current = player
    i=0
    while utility(state, k) is None:
        if current == "X":
            strategy = Xstrat
        else:
            strategy = Ostrat

        if strategy == GameStrategy.RANDOM:
            state = random.choice(successors(state, current))
        elif strategy == GameStrategy.ABS:
            _, state = ABS(state, current, k)
        elif strategy == GameStrategy.PLAYER:
            state = play(state, current)
        else:
            state = MCTS(state, current, k, rollouts, mcts_alpha)

        current = "O" if current == "X" else "X"

        if print_result:
            i+=1
            st.write(f"step :{i}")
            #print(state)
            st.write(state)
    return utility(state, k)
