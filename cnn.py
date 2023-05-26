from torch import nn


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
