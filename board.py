import random
import copy


# Used as the environment for DQN
class Board:
    def __init__(self, rows, columns, num_mines):
        self.rows = rows
        self.columns = columns
        self.num_mines = num_mines
        self.empty_spaces = rows * columns - num_mines

        self.mine_board = None
        self.live_board = self.generate_live_board()

    # Initialize a hidden board with mines
    def generate_mine_board(self, click):
        # Calculate single dimension position of click
        click_position = click[0] * self.rows + click[1]

        # Remove calculated position from sample space
        valid_mine_positions = [pos for pos in range(self.rows * self.columns)]
        valid_mine_positions.pop(click_position)

        # Generate mine positions from sample space
        mine_positions = random.sample(valid_mine_positions, self.num_mines)

        # Create blank board
        mine_board = []
        for _ in range(self.rows):
            mine_board.append([-1] * self.columns)  # -1 denotes a non-clicked space

        # Mark positions of mines on board
        for position in mine_positions:
            row = int(position / self.columns)
            col = position % self.columns
            mine_board[row][col] = -2  # -2 denotes a mine

        return mine_board

    # Initialize blank live board
    def generate_live_board(self):
        live_board = []
        for _ in range(self.rows):
            live_board.append([-1] * self.columns)  # -1 denotes a non-clicked space
        return live_board

    # Updates live and mine boards based on given click position
    def update_board(self, click):

        if self.mine_board is None:
            self.mine_board = self.generate_mine_board(click)

        # Retrieve the value found on the mine board
        mine_value = self.mine_board[click[0]][click[1]]

        reward = -2
        terminated = False

        if mine_value == -2:  # Checks if value is a mine
            self.live_board[click[0]][click[1]] = -2  # Mark live board as having hit a mine
            reward = self.empty_spaces * -1
            terminated = True
        elif mine_value == -1:  # Checks if position has not been clicked
            mine_amount = self.get_mines(click)  # Get number of mines around current space
            empty_old = self.empty_spaces
            if mine_amount == 0:
                # Update markings on live and mine boards
                self.mine_board[click[0]][click[1]] = 0
                self.live_board[click[0]][click[1]] = 0

                # Recursively update surrounding 3x3 area
                for r in range(max(0, click[0] - 1), min(len(self.mine_board), click[0] + 2)):
                    for c in range(max(0, click[1] - 1), min(len(self.mine_board[0]), click[1] + 2)):
                        self.update_board((r, c))
            else:
                # Update markings on live and mine boards
                self.mine_board[click[0]][click[1]] = mine_amount
                self.live_board[click[0]][click[1]] = mine_amount
            self.empty_spaces -= 1
            reward = (empty_old - self.empty_spaces)

        # Check if all empty spaces have been found
        if self.empty_spaces == 0:
            print('GAME WON')
            reward = 50
            terminated = True

        return reward, terminated

    # Count the number of mines in surrounding 3x3 area
    def get_mines(self, click):
        mines = 0
        for r in range(max(0, click[0] - 1), min(len(self.mine_board), click[0] + 2)):
            for c in range(max(0, click[1] - 1), min(len(self.mine_board[0]), click[1] + 2)):
                if self.mine_board[r][c] == -2:
                    mines += 1
        return mines

    def reset(self):
        self.mine_board = None
        self.live_board = self.generate_live_board()
        self.empty_spaces = self.rows * self.columns - self.num_mines

    # Display live board state
    def print_live_board(self):
        for row in self.live_board:
            print(row)

    # Returns a deep copy of the current board state
    def get_live_board(self):
        return copy.deepcopy(self.live_board)

    # Display mine board state
    def print_mine_board(self):
        if self.mine_board is not None:
            for row in self.mine_board:
                print(row)
