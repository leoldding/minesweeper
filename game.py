import random


# Function to select difficulty of the game
def mode_select():
    modes = ['easy', 'medium', 'hard']
    print('Game Modes: \n\tEasy \n\tMedium \n\tHard')
    mode = input('Select a game mode: ').lower()

    while mode not in modes:
        print('Invalid mode selected. Please select valid mode.')
        print('Game Modes: \n\tEasy \n\tMedium \n\tHard')
        mode = input('Select a game mode: ').lower()

    return mode


# Function that runs the game
def game(rows=None, columns=None, mines=None):
    # Checks that rows and columns have values
    if rows is None or columns is None:
        print('ERROR: value missing for row or column.')
    # Generates number of mines if none are given
    if mines is None:
        mines = int(rows * columns * 0.1)

    # Generate the board that the player sees
    real_board = []
    for r in range(rows):
        real_board.append(['-'] * columns)
    print_board(real_board)

    # Instantiate mine_positions
    mine_positions = None

    # Calculate number of empty spaces to find
    empty_spaces = rows * columns - mines

    # Run game
    while True:
        # Ask for player to select a row to click
        click_row = input('Select row to click: ')
        # Check that input is a valid integer
        while not click_row.isdigit() or int(click_row) > rows or int(click_row) <= 0:
            print('Invalid row.')
            click_row = input('Select row to click: ')
        click_row = int(click_row) - 1

        # Ask player to select a column to click
        click_col = input('Select column to click: ')
        # Check that input is a valid integer
        while not click_col.isdigit() or int(click_col) > columns or int(click_col) <= 0:
            print('Invalid column.')
            click_col = input('Select column to click: ')
        click_col = int(click_col) - 1

        # Generate position of mines if this is the users first click
        if mine_positions is None:
            # Calculate the array index position of the click
            position = click_row * columns + click_col

            # Remove clicked position from sample space
            valid_mine_positions = [pos for pos in range(rows * columns)]
            valid_mine_positions.pop(position)
            # Generate mine positions
            mine_positions = random.sample(valid_mine_positions, mines)

            # Generate hidden board with mine positions
            mine_board = []
            for r in range(rows):
                mine_board.append(['-'] * columns)
            # Mark positions of mines in hidden board
            for position in mine_positions:
                r = int(position / columns)
                c = position % columns
                mine_board[r][c] = 'M'

        # Click the position and update boards
        empty_spaces = update_board(real_board, mine_board, click_row, click_col, empty_spaces)

        # Check for a game end condition
        if real_board[click_row][click_col] == 'X' or empty_spaces == 0:
            break

        # Print the game board
        print_board(real_board)

    # Determine game end state
    end_board(real_board, mine_board)
    print_board(real_board)
    if empty_spaces == 0:
        print('WIN')
    else:
        print('LOSE')


# Function that updates both visible and hidden boards based on position clicked
def update_board(real_board, mine_board, click_row, click_col, empty_spaces):
    # Retrieve what is clicked from the hidden board
    clicked = mine_board[click_row][click_col]
    if clicked == 'M':  # Mine
        # Mark position as having hit a mine
        real_board[click_row][click_col] = 'X'
    elif clicked == '-':  # Non-Mine
        # Get number of mines in surround 3x3 area
        mines = get_mines(mine_board, click_row, click_col)
        if mines == 0:
            # Update clicked position's marking
            mine_board[click_row][click_col] = '0'
            real_board[click_row][click_col] = '0'
            empty_spaces -= 1
            # Recursively update surrounding 3x3 area
            for r in range(max(0, click_row - 1), min(len(mine_board), click_row + 2)):
                for c in range(max(0, click_col - 1), min(len(mine_board[0]), click_col + 2)):
                    empty_spaces = update_board(real_board, mine_board, r, c, empty_spaces)
        else:
            # Update clicked position's marking
            mine_board[click_row][click_col] = str(mines)
            real_board[click_row][click_col] = str(mines)
            empty_spaces -= 1
    return empty_spaces


# Function to display final state of visible board with all mines marked
def end_board(real_board, mine_board):
    for r in range(len(mine_board)):
        for c in range(len(mine_board[0])):
            if mine_board[r][c] == 'M':
                real_board[r][c] = 'X'


# Function to count the number of mines in surrounding 3x3 area
def get_mines(mine_board, click_row, click_col):
    mines = 0
    for r in range(max(0, click_row - 1), min(len(mine_board), click_row + 2)):
        for c in range(max(0, click_col - 1), min(len(mine_board[0]), click_col + 2)):
            if mine_board[r][c] == 'M':
                mines += 1
    return mines


# Function to print the current visible board state
def print_board(board):
    # Display column number markers
    print('  ', end='')
    for c in range(len(board[0])+1):
        if c != len(board[0]):
            print(c, end='    ')
        else:
            print(c)

    for r in range(len(board)):
        # Display row number markers
        print('  ' + str(r+1), end='  ')
        # Display current row of visible board
        print(board[r])

    # Add blank to create separation between turns
    print()


if __name__ == '__main__':
    # Select difficulty of game
    mode = 'easy'  # mode_select()

    # Start game based on difficulty
    if mode == 'easy':
        game(9, 9, 10)
    elif mode == 'medium':
        game(16, 16, 40)
    elif mode == 'hard':
        game(30, 16, 99)
