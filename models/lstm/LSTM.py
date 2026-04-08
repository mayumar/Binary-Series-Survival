import torch
import torch.nn as nn
import torch.nn.functional as F

class Survival_LSTM(nn.Module):
    """
    LSTM-based neural network for discrete-time survival analysis.

    The model encodes a sequence of input features using stacked LSTM layers
    and produces hazard probabilities for multiple event types across
    discrete time intervals.

    The output tensor represents the predicted hazard probability for each
    event type at each time bin.

    :param input_size: Number of input features per timestep.
    :type input_size: int
    :param num_events: Number of possible event types to predict.
    :type num_events: int
    :param num_times: Number of discrete time bins used in the survival formulation.
    :type num_times: int
    :param hidden_size1: Number of hidden units in the first LSTM layer.
    :type hidden_size1: int
    :param hidden_size2: Number of hidden units in the second LSTM layer.
    :type hidden_size2: int
    :param dense_size1: Number of units in the first fully connected layer.
    :type dense_size1: int
    :param dense_size2: Number of units in the second fully connected layer.
    :type dense_size2: int
    :param dropout: Dropout probability applied after LSTM and dense layers.
    :type dropout: float
    """

    def __init__(
        self,
        input_size: int,
        num_events: int,
        num_times: int,
        hidden_size1: int = 64,
        hidden_size2: int = 128,
        dense_size1: int = 32,
        dense_size2: int = 16,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_events = num_events
        self.num_times = num_times

        self.norm = nn.LayerNorm(input_size)

        # Encoder
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_size1)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout)
        
        # Decoder
        self.fc1 = nn.Linear(hidden_size2, dense_size1)
        self.fc2 = nn.Linear(dense_size1, dense_size2)
        
        self.fc_out = nn.Linear(dense_size2, num_events * num_times)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the survival LSTM model.

        The input sequence is encoded with two stacked LSTM layers,
        followed by fully connected layers that generate hazard logits
        for each event type and time bin. The logits are reshaped and
        transformed into hazard probabilities using a sigmoid activation.

        :param x: Input tensor containing sequential features with shape
                  ``(batch_size, sequence_length, input_size)``.
        :type x: torch.Tensor

        :return: Predicted hazard probabilities with shape
                 ``(batch_size, num_events, num_times)`` where each value
                 represents the probability of event ``k`` occurring at
                 time bin ``t``.
        :rtype: torch.Tensor
        """
        x = self.norm(x) # (bs, seq_len, in_feats)

        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out = self.norm2(out)

        out, _ = self.lstm2(out)
        
        # Take last hidden state of the sequence
        out = out[:, -1, :] 
        out = self.dropout(out)

        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        
        logits = self.fc_out(out) # (batch, num_events * num_times)
        logits = logits.view(-1, self.num_events, self.num_times) # (batch, num_events, num_times)
        
        # IMPORTANT: DeepHit uses Softmax to represent the PMF (Probability Mass Function)
        # We compute the probability that event k occurs at time t
        hazard = torch.sigmoid(logits)

        return hazard


class Survival_CNN_LSTM(nn.Module):
    """
    Adaptación del modelo CNN+LSTM para Supervivencia (DeepHit).
    """
    
    def __init__(
        self, 
        n_features: int, 
        num_events: int, 
        num_times: int 
    ):    
        super().__init__()
        
        self.num_events = num_events
        self.num_times = num_times

        # --- Feature Extractor (Igual que antes) ---
        # CNN block
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=6, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # LSTM block
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)

        # FNN block
        self.fnn4 = nn.Linear(128, 128)
        self.fnn5 = nn.Linear(128, 16)
        self.dropout = nn.Dropout(0.4)

        # --- Output Layer ---
        # CAMBIO: Proyectar a eventos * tiempo
        self.output = nn.Linear(16, num_events * num_times)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            PMF de forma (batch_size, num_events, num_times).
        """
        # 1. CNN
        x = x.permute(0, 2, 1) # (batch, features, time)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        # x = self.pool1(x) # Opcional según longitud de secuencia
        
        # 2. LSTM
        x = x.permute(0, 2, 1) # Volver a (batch, time, features)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :] # Último step

        # 3. FNN
        x = F.relu(self.fnn4(x))
        x = self.dropout(x)
        x = F.relu(self.fnn5(x))
        
        # 4. Salida Survival
        logits = self.output(x) # (batch, num_events * num_times)
        
        # Reshape
        logits = logits.view(-1, self.num_events, self.num_times)
        
        # Softmax sobre el eje del tiempo para obtener probabilidad
        pmf = F.softmax(logits, dim=2)

        return pmf