from torch import nn
import random


class DQN:
    def __init__(self, height, width):

        self.Q_network = CNN(height, width)
        self.target_network = CNN(height, width)
        self.target_network.load_state_dict(self.Q_network.state_dict())

        self.replay_memory = ReplayMemory(1000)

        self.epsilon = 1
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01

        self.gamma = 0.9

        self.batch_size = 128

        self.learning_rate = 0.01


# Used to store observations for the DQN
class ReplayMemory:

    def __init__(self, max_length):
        self.memory = []
        self.max_length = max_length

    # Add a new observation to the end of the memory
    def push(self, sample):
        self.memory.append(sample)
        if len(self.memory) > self.max_length:
            self.memory.pop(0)

    # Return a sample of the memory for training
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


# Used for Q network and target network for DQN
class CNN(nn.Module):
    def __init__(self, height, width):
        super(CNN, self).__init__()

        # Branch to predict the row that will be clicked
        self.row_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, height),
            nn.Softmax()
        )

        # Branch to predict the column that will be clicked
        self.column_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, width),
            nn.Softmax()
        )

    def forward(self, x):
        row = self.row_branch(x)
        column = self.column_branch(x)
        return [row, column]
