# =======================================================================================
# beg: basic imports and setup
# =======================================================================================
from tqdm import tqdm
from loguru import logger
import numpy as np

import torch
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F  
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================================================================
# end: basic imports
# =======================================================================================


# =======================================================================================
# beg: config
# =======================================================================================
input_size = 1
hidden_size = 256
n_layers = 2

n_classes = 1
seq_len = 64

batch_size = 1
# =======================================================================================
# end: config
# =======================================================================================


# ========================================================================================
# beg: models
# ========================================================================================
class RNN(nn.Module):
  
  def __init__(self, seq_length, input_size, hidden_size, n_layers, n_classes):
    super(RNN, self).__init__()
    
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.rnn = nn.RNN(input_size, self.hidden_size, self.n_layers, batch_first = True)
    self.fc = nn.Linear(self.hidden_size*seq_length, n_classes)

  def forward(self, x):
    
    # 3-D: (n_layers, bath_size, hidden_size)
    h0 = torch.zeros((self.n_layers, x.shape[0], self.hidden_size)).to(device)
    out, _ = self.rnn(x, h0)
    
    # classification layer on output of last time step
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    return out


class RNN_GRU(nn.Module):
  
  def __init__(self, seq_length, input_size, hidden_size, n_layers, n_classes):
    super(RNN_GRU, self).__init__()
    
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.rnn = nn.GRU(input_size, self.hidden_size, self.n_layers, batch_first = True)
    self.fc = nn.Linear(self.hidden_size*seq_length, n_classes)

  def forward(self, x):

    # 3-D: (n_layers, bath_size, hidden_size)
    h0 = torch.zeros((self.n_layers, x.shape[0], self.hidden_size)).to(device)
    out, _ = self.rnn(x, h0)

    # classification layer on output of last time step
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    return out


class RNN_LSTM(nn.Module):
  
  def __init__(self, seq_length, input_size, hidden_size, n_layers, n_classes):
    super(RNN_LSTM, self).__init__()
    
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, self.hidden_size, self.n_layers, batch_first = True)
    self.fc = nn.Linear(self.hidden_size*seq_length, n_classes)

  def forward(self, x):

    # 3-D: (n_layers, bath_size, hidden_size)
    h0 = torch.zeros((self.n_layers, x.shape[0], self.hidden_size)).to(device)
    c0 = torch.zeros((self.n_layers, x.shape[0], self.hidden_size)).to(device)
    out, _ = self.lstm(x, (h0, c0)) 
    # out: tensor of shape (batch_size, seq_length, hidden_size)

    # classification layer on output of last time step
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    return out
# ========================================================================================
# end: models
# ========================================================================================


# ========================================================================================
# beg: dataloaders
# ========================================================================================
def get_dataloaders_from(xtrain, ytrain, xvalid, yvalid, xhdout, yhdout, **kwargs):
    """
    **kwargs:
        + batch_size
        + shuffle
    """

    train_dataset = TensorDataset(torch.from_numpy(xtrain).float(), torch.from_numpy(ytrain).float())
    valid_dataset = TensorDataset(torch.from_numpy(xvalid).float(), torch.from_numpy(yvalid).float())
    hdout_dataset = TensorDataset(torch.from_numpy(xhdout).float(), torch.from_numpy(yhdout).float())

    train_loader = DataLoader(dataset=train_dataset, **kwargs)
    valid_loader = DataLoader(dataset=valid_dataset, **kwargs)
    hdout_loader = DataLoader(dataset=hdout_dataset, **kwargs)

    return train_loader, valid_loader, hdout_loader
# ========================================================================================
# end: dataloaders
# ========================================================================================


# ========================================================================================
# beg: trainer
# ========================================================================================
class Model:
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model

    def compile(self, optimizer, loss, **kwargs):
        self.criterion = loss
        self.optimizer = optimizer(self.pytorch_model.parameters(), **kwargs)

    def fit(
            self,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            epochs: int,
        ):

        log_every = 1
        hist = {
            'train_loss': []
        }
        self.pytorch_model.train()
        for epoch in range(epochs):
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (data, targets) in pbar:

                # Get data to cuda if possible
                data = data.to(device=device).squeeze(1)
                targets = targets.to(device=device)
                
                # forward
                scores = self.pytorch_model(data).reshape(-1)
                loss = self.criterion(scores, targets)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # gradient descent or adam step
                self.optimizer.step()

                pbar.set_description(f'loss {loss.item():.6f}')
                if batch_idx==len(train_loader)-1:
                    val_loss = self.eval(valid_loader)
                    pbar.set_description(f'epoch {epoch} >> {pbar.desc} val_loss {val_loss:.6f}')
            
            hist['train_loss'].append(loss.item())
        return hist
    
    def eval(self, loader):
        losses = []
        self.pytorch_model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)

                scores = self.pytorch_model(x).reshape(-1)
                loss =self.criterion(scores, y)
                losses.append(loss)

        self.pytorch_model.train()
        return np.median(losses)

    def predict(self, ndarray):
        self.pytorch_model.eval()
        ret = self.pytorch_model(
            torch.from_numpy(ndarray).float().to(device))
        return ret.detach().cpu().numpy()
# ========================================================================================
# end: trainer
# ========================================================================================


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