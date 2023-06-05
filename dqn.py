import random
import torch
from board import Board
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class DQN:
    def __init__(self, rows, columns, num_mines):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rows = rows
        self.columns = columns
        self.num_mines = num_mines

        self.Q_network = CNN(rows, columns).to(self.device)
        self.target_network = CNN(rows, columns).to(self.device)
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.target_update_rate = 500

        self.replay_memory = ReplayMemory(10000)

        self.board = Board(rows, columns, num_mines)

        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.epsilon_update_rate = 100

        self.gamma = 0.9

        self.batch_size = 256
        self.learning_rate = 0.001

        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()

        self.episode_rewards = []
        self.episode_loss = []
        self.episode_empty = []
        self.episode_steps = []

    def train(self, episodes):

        for episode in tqdm(range(1, episodes+1), desc='Episodes'):
            print(f'Episode {episode}')
            self.episode_rewards.append(0)
            self.board.reset()

            episode_steps = 0
            terminated = False
            while not terminated and episode_steps < 100:
                state = self.board.get_live_board()
                action = self.select_action()
                reward, terminated = self.board.update_board(action)
                next_state = self.board.get_live_board()

                self.episode_rewards[-1] += reward

                self.replay_memory.push(state, action, reward, next_state)

                episode_steps += 1

            self.optimize()

            if episode % self.target_update_rate == 0:
                print('Updating Target Network')
                self.target_network.load_state_dict(self.Q_network.state_dict())

            if episode % self.epsilon_update_rate == 0:
                print('Updating Epsilon')
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            self.episode_empty.append(self.board.empty_spaces)
            self.episode_steps.append(episode_steps)

    # Select an epsilon greedy action
    def select_action(self):
        if random.random() <= self.epsilon:
            action = [random.randint(0, self.rows-1), random.randint(0, self.columns-1)]
        else:
            row, column = self.Q_network(
                torch.tensor(self.board.live_board, dtype=torch.float32).reshape(1, 1, self.rows, self.columns).to(self.device))
            action = [int(torch.argmax(row)), int(torch.argmax(column))]
        return action

    # Sample from replay memory and update model parameters
    def optimize(self):
        if len(self.replay_memory) < self.batch_size:
            return

        loss_meter = LossMeter()

        dataset = TensorDataset(torch.FloatTensor(self.replay_memory.current_states).to(self.device),
                                torch.IntTensor(self.replay_memory.actions).to(self.device),
                                torch.IntTensor(self.replay_memory.rewards).to(self.device),
                                torch.FloatTensor(self.replay_memory.next_states).to(self.device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for current_state, action, reward, next_state in dataloader:
            q_row, q_column = self.Q_network(
                current_state.reshape(current_state.shape[0], 1, self.rows, self.columns))
            q_rows = [row[action[0][0]] for row in q_row]
            q_columns = [column[action[0][1]] for column in q_column]

            target_row, target_column = self.target_network(
                next_state.reshape(next_state.shape[0], 1, self.rows, self.columns))
            target_rows = [self.gamma * torch.max(target_row[i]) + reward[i] for i in range(len(target_row))]
            target_columns = [self.gamma * torch.max(target_column[i]) + reward[i] for i in range(len(target_column))]

            row_loss = self.criterion(
                torch.tensor(q_rows, dtype=torch.float32, requires_grad=True),
                torch.tensor(target_rows, dtype=torch.float32))
            col_loss = self.criterion(
                torch.tensor(q_columns, dtype=torch.float32, requires_grad=True),
                torch.tensor(target_columns, dtype=torch.float32))

            self.optimizer.zero_grad()
            total_loss = row_loss + col_loss
            total_loss.backward()
            self.optimizer.step()

            loss_meter.update(total_loss.item(), len(action))

        self.episode_loss.append(loss_meter.average)


# Used to store observations for the DQN
class ReplayMemory:

    def __init__(self, max_length):
        self.current_states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

        self.length = 0
        self.full = False

        self.max_length = max_length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return [
            self.current_states[item],
            self.actions[item],
            self.rewards[item],
            self.next_states[item]
        ]

    # Add a new observation to the end of the memory
    def push(self, current_state, action, reward, next_state):
        self.current_states.append(current_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

        if not self.full:
            self.length += 1
            if self.length == self.max_length:
                self.full = True
        else:
            self.current_states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)


# Used for Q network and target network for DQN
class CNN(nn.Module):
    def __init__(self, rows, columns):
        super(CNN, self).__init__()

        # Branch to predict the row that will be clicked
        self.row_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, rows)
        )

        # Branch to predict the column that will be clicked
        self.column_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, columns)
        )

    def forward(self, x):
        row = self.row_branch(x)
        column = self.column_branch(x)
        return [row, column]


class LossMeter:
    def __init__(self):
        self.average = 0
        self.count = 0
        self.sum = 0
        self.reset()

    def reset(self):
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.average = self.sum / self.count

    def __repr__(self):
        text = f'Loss: {self.average:.4f}'
        return text
