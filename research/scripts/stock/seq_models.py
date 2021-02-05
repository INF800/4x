from tqdm import tqdm
from loguru import logger
import torch
import torchvision
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28
hidden_size = 256
n_layers = 2

n_classes = 10
seq_len = 28

learning_rate = 0.005 # lstm
batch_size = 512
num_epochs = 2


class RNN(nn.Module):
  
  def __init__(self, seq_length, input_size, hidden_size, n_layers, n_classes):
    """
    :param input_size: num of features of input
    """
    super(RNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.rnn = nn.RNN(input_size, self.hidden_size, self.n_layers, batch_first = True)
    self.fc = nn.Linear(
        self.hidden_size*seq_length,
        n_classes)

  def forward(self, x):

    # 3-D: (n_layers, bath_size, hidden_size)
    h0 = torch.zeros(
        (self.n_layers, x.shape[0], self.hidden_size)
    ).to(device)

    out, _ = self.rnn(x, h0)

    # classification layer on output of last time step
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    return out


class RNN_GRU(nn.Module):
  
  def __init__(self, seq_length, input_size, hidden_size, n_layers, n_classes):
    """
    :param input_size: num of features of input
    """
    super(RNN_GRU, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.rnn = nn.GRU(input_size, self.hidden_size, self.n_layers, batch_first = True)
    self.fc = nn.Linear(
        self.hidden_size*seq_length,
        n_classes
    )

  def forward(self, x):
    # 3-D: (n_layers, bath_size, hidden_size)
    h0 = torch.zeros(
        (self.n_layers, x.shape[0], self.hidden_size)
    ).to(device)
    out, _ = self.rnn(x, h0)

    # classification layer on output of last time step
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    return out



class RNN_LSTM(nn.Module):
  
  def __init__(self, seq_length, input_size, hidden_size, n_layers, n_classes):
    """
    :param input_size: num of features of input
    """
    super(RNN_LSTM, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, self.hidden_size, self.n_layers, batch_first = True)
    self.fc = nn.Linear(
        self.hidden_size*seq_length,
        n_classes
    )

  def forward(self, x):
    # 3-D: (n_layers, bath_size, hidden_size)
    h0 = torch.zeros(
        (self.n_layers, x.shape[0], self.hidden_size)
    ).to(device)
    c0 = torch.zeros(
        (self.n_layers, x.shape[0], self.hidden_size)
    ).to(device)

    out, _ = self.lstm(
        x, (h0, c0)
    ) # out: tensor of shape (batch_size, seq_length, hidden_size)

    # classification layer on output of last time step
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    return out


if __name__ == '__main__':

    @logger.catch
    def test_rnn():
        model = RNN(100, input_size, hidden_size, n_layers, n_classes).to(device)
        test_batch_seq = torch.rand(64, 100, 28).to(device)
        out = model(test_batch_seq)
        return out.shape
    
    @logger.catch
    def test_gru():
        model = RNN_GRU(100, input_size, hidden_size, n_layers, n_classes).to(device)
        test_batch_seq = torch.rand(64, 100, 28).to(device)
        out = model(test_batch_seq)
        return out.shape
    
    @logger.catch
    def test_lstm():
        model = RNN_LSTM(100, input_size, hidden_size, n_layers, n_classes).to(device)
        test_batch_seq = torch.rand(64, 100, 28).to(device)
        out = model(test_batch_seq)
        return out.shape

    logger.success(f'[RNN]  TEST PASSED! output shape: {test_rnn()}')
    logger.success(f'[GRU]  TEST PASSED! output shape: {test_gru()}')
    logger.success(f'[LSTM] TEST PASSED! output shape: {test_lstm()}')