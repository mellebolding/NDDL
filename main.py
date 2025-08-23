import os
import math
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from src.utils import (
    count_parameters,
    save_checkpoint,
    model_result_dict_load,
    gaussian,
    ActFun_adp,
    save_model,
    load_model
)
from src.fttp import (
    get_stats_named_params,
    post_optimizer_updates,
    get_regularizer_named_params,
    reset_named_params,
    train_fptt
)
from src.snn import (
    shifted_sigmoid,
    SnnLayer,
    OutputLayer,
    SnnNetwork,
    SnnNetwork3Layer,
    test
)


# ------------------------------
# Constants and Configurations
# ------------------------------
#SEED = 999
SEEDS = [57,231]
BATCH_SIZE = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0001
EPOCHS = 35  # use small number for debugging. Original = 35, new = 15
TIME_STEPS = 50
K_UPDATES = 10
OMEGA = int(TIME_STEPS / K_UPDATES)
HIDDEN_DIM = [600, 500, 500]
N_CLASSES = 10
INPUT_DIM = 784
CLF_ALPHA = 1
SPIKE_ALPHA = 0.0
DROPOUT_RATE = 0.4
GRAD_CLIP = 1.0
LOG_INTERVAL = 20
MODEL_TYPE = "energy"
ALG = "fptt"
RISE_TIME=False
#SPIKE = ''



if MODEL_TYPE == "control":
    ENERGY_ALPHA = 0
    if SPIKE_ALPHA != 0: SPIKE = f'_SPIKE_{SPIKE_ALPHA}'
else:
    ENERGY_ALPHA = 0.05
    if SPIKE_ALPHA != 0: SPIKE = f'_SPIKE_{SPIKE_ALPHA}'

# Adaptive neuron settings
B_J0 = 0.1  # neural threshold baseline
R_M = 3  # membrane resistance
GAMMA = 0.5  # gradient scale
LENS = 0.5  # width of Gaussian

# ------------------------------
# Helper Functions
# ------------------------------

def setup_device():
    """Setup CUDA or CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(batch_size):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def create_model(device):
    """Define and initialize the model."""
    act_fun_adp = ActFun_adp.apply

    model = SnnNetwork3Layer(
        INPUT_DIM, HIDDEN_DIM, N_CLASSES,
        is_adapt=True, one_to_one=True, dp_rate=DROPOUT_RATE, is_rec=False, rise_time=RISE_TIME,
        act_fun_adp=act_fun_adp, device=device, b_j0=B_J0
    )
    model.to(device)
    print(model)
    print(f"Total parameters: {count_parameters(model)}")
    return model




# ------------------------------
# Main Training Loop
# ------------------------------

def train_and_evaluate():
    # Set device and seed
    device = setup_device()
    for SEED in SEEDS:
        torch.manual_seed(SEED)

        # Load data
        train_loader, test_loader = load_data(BATCH_SIZE)

        # Create model
        model = create_model(device)

        # Define optimizer and scheduler
        optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        # Test untrained model
        print("Testing untrained model...")
        test_loss, test_acc, test_energy = test(model, test_loader, TIME_STEPS, device=device)

        named_params = get_stats_named_params(model)
        all_test_losses = []
        all_test_acc = []
        all_test_energy = []
        best_acc1 = 20


        model_name = "{}_seed{}_".format(ALG,SEED) + MODEL_TYPE + SPIKE + "/{}"
        print(model_name)
    
        for epoch in range(EPOCHS):
            train_fptt(epoch, BATCH_SIZE, LOG_INTERVAL, train_loader,
                    model, named_params, TIME_STEPS, K_UPDATES, OMEGA, optimizer,
                    CLF_ALPHA, ENERGY_ALPHA, SPIKE_ALPHA, GRAD_CLIP, lr=LEARNING_RATE, 
                    device=device,rise_time = RISE_TIME)

            reset_named_params(named_params)

            # Test model
            test_loss, test_acc, test_energy = test(model, test_loader, TIME_STEPS, device=device)
            scheduler.step()

            # Save checkpoint every epoch
            # Save checkpoint
            is_best = test_acc > best_acc1
            best_acc1 = max(test_acc, best_acc1)

            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, prefix='best_voltage_diff_', filename='best.pth.tar')

            save_model(model_name.format(epoch+1),model)
        
            all_test_losses.append(test_loss)
            all_test_acc.append(test_acc)
            all_test_energy.append(test_energy)

        # Save losses, energies and accuracies
        loss_name = model_name.format("test_loss.npy")
        acc_name = model_name.format("test_acc.npy")
        energy_name = model_name.format("test_energy.npy")
        np.save(f"results/{loss_name}",all_test_losses)
        np.save(f"results/{acc_name}",all_test_acc)
        np.save(f"results/{energy_name}",np.array(torch.stack(all_test_energy).cpu().detach().numpy()))

        print("Training completed.")

    return all_test_losses


# ------------------------------
# Main Entry Point
# ------------------------------

if __name__ == "__main__":
    train_and_evaluate()