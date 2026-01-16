import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import numpy as np
import Network as N


# Function to prune the lowest magnitude weights globally across all layers except BatchNorm
def global_weight_pruning(model, prune_ratio):
    all_weights = []
    
    # Collect all weights except from BatchNorm layers
    for name, param in model.named_parameters():
        if "weight" in name and not isinstance(getattr(model, name.split('.')[0]), nn.BatchNorm2d):
            all_weights.extend(param.data.abs().view(-1).cpu().numpy())
    
    # Determine global pruning threshold
    threshold = np.percentile(all_weights, prune_ratio * 100)  # Find the prune_ratio percentile
    
    # Apply pruning
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and not isinstance(getattr(model, name.split('.')[0]), nn.BatchNorm2d):
                mask = param.abs() > threshold  # Keep only weights above threshold
                param.mul_(mask)  # Zero-out pruned weights

    return model

# Load dataset
dataset_name = 'cifar10'
target_class = 0
num_classes = 10
model_type = 'resnet18'
batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_poison_transfrom = PoisonTransform(1., target_class=target_class, diff_location=False, diff_kernel=False, poisoning=True)
val_ds = TheDataset(dataset_name=dataset_name, mode='val', poison_transform=val_poison_transfrom, normalizing=True)
val_loader_poison = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2)

val_ds = TheDataset(dataset_name=dataset_name, mode='val', poison_transform=None, normalizing=True)
val_loader_clean = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2)


test_poison_transfrom = PoisonTransform(1., target_class=target_class, diff_location=False, diff_kernel=False, poisoning=True)
test_poison_ds = TheDataset(dataset_name=dataset_name, mode='test', poison_transform=test_poison_transfrom, normalizing=True)
test_loader_poison = torch.utils.data.DataLoader(test_poison_ds, batch_size=batch_size, shuffle=True, num_workers=2)

test_cl_ds = TheDataset(dataset_name=dataset_name, mode='test', poison_transform=None, normalizing=True)
test_loader_cl = torch.utils.data.DataLoader(test_cl_ds, batch_size=batch_size, shuffle=True, num_workers=2)


target_sparsity_list = list(np.arange(0.0, 1, 0.05))  # ratio of sparsity

asr_pruning = {}
acc_pruning = {}

asr_tune = {}
acc_tune = {}

for prune_ratio in target_sparsity_list:
    # Initialize model and move to GPU
    # added_model = N.TransferLearningModel((3, 32, 32), model_type, num_classes=num_classes+1)
    # model = N.TrickModel(added_model, num_classes=num_classes)
    model = N.TransferLearningModel((3, 32, 32), model_type='resnet18', num_classes=num_classes)
    # model = N.TrafficSign(num_classes=num_classes)
    # model_name = f'model_{dataset_name}_{model_type}_{num_classes}_poison_aug'
    model_name = 'revisiting_model'
    if isinstance(torch.load(f'models/{model_name}.pth'), dict):
        model.load_state_dict(torch.load(f'models/{model_name}.pth'))
    else:
        model = torch.load(f'models/{model_name}.pth')

    model = model.to(device)
    model.eval()

    print(f'\nPruninfg ratio is: {prune_ratio}\n')

    # Apply global weight pruning (e.g., prune 10% of the total weights)
    global_weight_pruning(model, prune_ratio=prune_ratio)
    print("Pruning completed.")

    print('Poison data test after pruning:', flush=True)
    asr_pruning[prune_ratio] = N.test_model(model.eval(), test_loader_poison)[0]/100
        
    print('Clean data test after pruning:', flush=True)
    acc_pruning[prune_ratio] = N.test_model(model.eval(), test_loader_cl)[0]/100


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    # Disable BatchNorm and Dropout
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()


    # Fine-tune the model after pruning
    for epoch in range(5):  # Short fine-tuning
        for inputs, labels in val_loader_clean:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Poison data test after fine-tuning:', flush=True)
    asr_tune[prune_ratio] = N.test_model(model.eval(), test_loader_poison)[0]/100
        
    print('Clean data test after fine-tuning:', flush=True)
    acc_tune[prune_ratio] = N.test_model(model.eval(), test_loader_cl)[0]/100

    print("Pruning completed and model fine-tuned.")


plt.figure(figsize=(6.8, 4.8))
plt.plot(acc_pruning.keys(), acc_pruning.values(), label="Model accuracy")
plt.plot(asr_pruning.keys(), asr_pruning.values(), label="ASR")
plt.title(model_name, fontsize="16")
plt.xlabel("Ratio of weights pruned", fontsize="16")
plt.ylabel("Accuracy/ASR", fontsize="16")
plt.legend(fontsize="16")
plt.savefig(f'images/{model_name}_pruning.png')
plt.show()

plt.figure(figsize=(6.8, 4.8))
plt.plot(acc_tune.keys(), acc_tune.values(), label="Model accuracy")
plt.plot(asr_tune.keys(), asr_tune.values(), label="ASR")
plt.title(model_name, fontsize="16")
plt.xlabel("Ratio of weights pruned", fontsize="16")
plt.ylabel("Accuracy/ASR", fontsize="16")
plt.legend(fontsize="16")
plt.savefig(f'images/{model_name}_tuning.png')
plt.show()