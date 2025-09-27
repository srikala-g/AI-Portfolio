"""
Tic Tac Toe Player with AI Implementation

This module implements a complete Tic Tac Toe game with an AI player using the minimax
algorithm with alpha-beta pruning. The AI is unbeatable and will always play optimally.

Key Features:
- Complete game state management (initial state, player turns, actions, results)
- Win condition detection (rows, columns, diagonals)
- Terminal state checking (win or draw)
- Optimal AI player using minimax with alpha-beta pruning
- Utility function for game evaluation

Usage Instructions:
1. Initialize a game board: board = initial_state()
2. Check whose turn it is: current_player = player(board)
3. Get available moves: available_actions = actions(board)
4. Make a move: new_board = result(board, (row, col))
5. Check for winner: game_winner = winner(board)
6. Check if game is over: is_terminal = terminal(board)
7. Get AI's optimal move: ai_move = minimax(board)
8. Get game utility: score = utility(board)  # 1 for X win, -1 for O win, 0 for draw

Example Game Loop:
    board = initial_state()
    while not terminal(board):
        if player(board) == X:
            # Human player move
            move = get_human_move()  # Your input function
        else:
            # AI player move
            move = minimax(board)
        board = result(board, move)
    
    # Game over - check winner
    final_winner = winner(board)
    if final_winner:
        print(f"Winner: {final_winner}")
    else:
        print("It's a draw!")

Constants:
- X: "X" (first player)
- O: "O" (second player) 
- EMPTY: None (empty cell)
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_x = sum(row.count(X) for row in board)
    num_o = sum(row.count(O) for row in board)

    if (num_x > num_o):
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    list_of_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                list_of_actions.add((i, j))
    return list_of_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    if not (0 <= i < 3 and 0 <= j < 3):
        raise Exception(f"Invalid action: Index ({i}, {j}) is out of bounds.")

    if board[i][j] is not EMPTY:
        raise Exception("Invalid action: Cell already has a value.")

    new_board = [row[:] for row in board]  # Createa a deep copy of the board
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        # Check rows and columns
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None  # No winner yet


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    if all(cell is not EMPTY for row in board for cell in row):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board using alpha-beta pruning.
    """
    if terminal(board):
        return None

    current_player = player(board)
    best_action = None

    if current_player == X:
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        for action in actions(board):
            value = minimax_value_ab(result(board, action), alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
    else:
        best_value = math.inf
        alpha = -math.inf
        beta = math.inf
        for action in actions(board):
            value = minimax_value_ab(result(board, action), alpha, beta)
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, best_value)

    return best_action


def minimax_value_ab(board, alpha, beta):
    """
    Returns the minimax value of the board using alpha-beta pruning.
    """
    if terminal(board):
        return utility(board)

    current_player = player(board)

    if current_player == X:
        value = -math.inf
        for action in actions(board):
            value = max(value, minimax_value_ab(result(board, action), alpha, beta))
            alpha = max(alpha, value)
            if beta <= alpha:
                break  # Beta cut-off
        return value
    else:
        value = math.inf
        for action in actions(board):
            value = min(value, minimax_value_ab(result(board, action), alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha cut-off
        return value
