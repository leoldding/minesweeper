import random


class Board:
    def __init__(self, rows, columns, num_mines):
        self.rows = rows
        self.columns = columns
        self.num_mines = num_mines

        self.initial_click = (random.randint(0, rows-1), random.randint(0, columns-1))
        self.mine_board = self.generate_mine_board()
        self.live_board = self.generate_live_board()
        self.update_board(self.initial_click)

    def generate_mine_board(self):
        click_position = self.initial_click[0] * self.rows + self.initial_click[1]

        valid_mine_positions = [pos for pos in range(self.rows * self.columns)]
        valid_mine_positions.pop(click_position)

        mine_positions = random.sample(valid_mine_positions, self. num_mines)

        mine_board = []
        for _ in range(self.rows):
            mine_board.append([-1] * self.columns)

        for position in mine_positions:
            row = int(position / self.columns)
            col = position % self.columns
            mine_board[row][col] = -2

        return mine_board

    def generate_live_board(self):
        live_board = []
        for _ in range(self.rows):
            live_board.append([-1] * self.columns)
        return live_board

    def update_board(self, click):
        mine_value = self.mine_board[click[0]][click[1]]
        if mine_value == -2:  # Mine
            self.live_board[click[0]][click[1]] = -2
        elif mine_value == -1:  # Non-Mine
            mine_amount = self.get_mines(click)
            if mine_amount == 0:
                self.mine_board[click[0]][click[1]] = 0
                self.live_board[click[0]][click[1]] = 0

                for r in range(max(0, click[0] - 1), min(len(self.mine_board), click[0] + 2)):
                    for c in range(max(0, click[1] - 1), min(len(self.mine_board[0]), click[1] + 2)):
                        self.update_board((r, c))
            else:
                self.mine_board[click[0]][click[1]] = mine_amount
                self.live_board[click[0]][click[1]] = mine_amount

    def get_mines(self, click):
        mines = 0
        for r in range(max(0, click[0] - 1), min(len(self.mine_board), click[0] + 2)):
            for c in range(max(0, click[1] - 1), min(len(self.mine_board[0]), click[1] + 2)):
                if self.mine_board[r][c] == -2:
                    mines += 1
        return mines

    def print_live_board(self):
        for row in self.live_board:
            print(row)

    def print_mine_board(self):
        for row in self.mine_board:
            print(row)
