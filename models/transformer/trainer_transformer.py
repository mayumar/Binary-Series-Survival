import torch
import torch.nn as nn
from typing import Optional, List
from torch.utils.data import DataLoader

# Asegúrate de que importas desde el fichero correcto
from models.lstm.LSTM import Survival_LSTM
from models.transformer.tranformer import GCU_Transformer

from models.utils.losses import DeepHitLoss, DiscreteHazardNLL
from models.utils.functions import train
import pandas as pd
import numpy as np

import torch
import numpy as np
from torch.utils.data import Subset

def rebalance_censoring(train_loader, max_ratio=0.2):
    """
    Reduce aleatoriamente los datos censurados (c=1) para que
    no superen max_ratio respecto a los no censurados (c=0).
    """
    dataset = train_loader.dataset
    
    # Extraer todos los índices
    all_indices = np.arange(len(dataset))
    
    # Obtener todos los c_val
    c_values = []
    for i in all_indices:
        _, _, c = dataset[i]
        c_values.append(int(c))
    c_values = np.array(c_values)

    # Separar índices
    idx_no_cens = all_indices[c_values == 0]
    idx_cens = all_indices[c_values == 1]

    n_no_cens = len(idx_no_cens)
    n_cens = len(idx_cens)

    max_cens_allowed = int(max_ratio * n_no_cens)

    print(f"\n--- Rebalanceo de Censura ---")
    print(f"No censurados: {n_no_cens}")
    print(f"Censurados originales: {n_cens}")
    print(f"Censurados permitidos (25%): {max_cens_allowed}")

    if n_cens > max_cens_allowed:
        idx_cens_sampled = np.random.choice(
            idx_cens, size=max_cens_allowed, replace=False
        )
    else:
        idx_cens_sampled = idx_cens

    # Unir y mezclar
    new_indices = np.concatenate([idx_no_cens, idx_cens_sampled])
    np.random.shuffle(new_indices)

    print(f"Total final entrenamiento: {len(new_indices)}")
    print(f"--------------------------------\n")

    # Crear nuevo Subset
    new_dataset = Subset(dataset, new_indices)

    # Crear nuevo DataLoader con mismos parámetros
    new_loader = DataLoader(
        new_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True
    )

    # Copiar atributo custom si existe
    if hasattr(train_loader, "num_time_bins"):
        new_loader.num_time_bins = train_loader.num_time_bins

    return new_loader


def train_lstm(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 20,
    test_loader: Optional[DataLoader] = None,
    num_events: int = 1
) -> nn.Module:

    train_loader = rebalance_censoring(train_loader, max_ratio=0.5)
    
    # 1. Validación de dimensiones
    if not hasattr(train_loader, 'num_time_bins'):
        raise ValueError("El DataLoader no tiene el atributo 'num_time_bins'.")
        
    num_time_bins = train_loader.num_time_bins
    
    # Extraemos un batch para ver dimensiones de entrada automáticamente
    try:
        sample_x, _, _ = next(iter(train_loader))
        input_size = sample_x.shape[-1]
        seq_len = sample_x.shape[1]
    except StopIteration:
        raise ValueError("El train_loader está vacío.")

    print(f"--- Configuración del Modelo ---")
    print(f"Input Features: {input_size}")
    print(f"Num Events:     {num_events}")
    print(f"Num Time Bins:  {num_time_bins}")
    print(f"Device:         {device}")
    print(f"------------------------------")


    model = GCU_Transformer(seq_size=seq_len, in_chans=input_size, num_events=num_events, num_bins=num_time_bins).to(device)


    # 3. Función de Pérdida
    # Usamos sigma=1.0 para suavizar gradientes (evita explosiones numéricas)
    # Alpha=0.5 da igual peso a la verosimilitud (qué bin es) y al ranking (orden)
    #loss_fn = DeepHitLoss(alpha=1, beta=0.5, gamma=0.1, sigma=0.1)
    loss_fn = DiscreteHazardNLL()
    # loss_fn = DeepHitLoss(alpha=0.75, beta=0.125, gamma=0.125, sigma=0.1)
    # loss_fn = nn.CrossEntropyLoss()

    # 4. Optimizador
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    
    # 5. Scheduler
    # Reduce el LR a la mitad si el C-Index se estanca durante 10 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='max', factor=0.5, patience=3
    )
    scheduler = None

    # 6. Ejecutar entrenamiento 
    # (Llama a la función robusta que añadiste en functions_manu.py)
    best_model = train(
        model_for_train=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        N_EPOCH=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_train=loss_fn,
        loss_eval=loss_fn,
        lines_list=None,
        patience=early_stopping_patience,
        max_rul=None,
        num_test_windows=None,
        num_bins=num_time_bins,
        device=device
    )
    
    return best_model