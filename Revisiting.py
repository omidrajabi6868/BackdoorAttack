import numpy as np
import torch
import Network as N
from utils import *

import warnings
warnings.filterwarnings('ignore')


class Revisiting:
    def __init__(self, dataset_name, target_class, batch_size):

        clean_ds = TheDataset(dataset_name=dataset_name, mode='train', poison_transform=None, normalizing=True) 

        clean_transform = PoisonTransform(.1, target_class=target_class, diff_location=False, diff_kernel=True, poisoning=False) 
        clean_transform_ds1 = TheDataset(dataset_name=dataset_name, mode='val', poison_transform=clean_transform, normalizing=True)

        clean_transform = PoisonTransform(1., target_class=target_class, diff_location=True, diff_kernel=True, poisoning=False) 
        clean_transform_ds2 = TheDataset(dataset_name=dataset_name, mode='val', poison_transform=clean_transform, normalizing=True)

        poisoned_transform = PoisonTransform(1., target_class=target_class, diff_location=False, diff_kernel=True, poisoning=True) 
        poison_ds = TheDataset(dataset_name=dataset_name, mode='val', poison_transform=poisoned_transform, normalizing=True)

        train_ds = ConcatDataset([clean_ds, clean_transform_ds1, clean_transform_ds2, poison_ds])
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

        val_poison_transfrom = PoisonTransform(.1, target_class=target_class, diff_location=False, diff_kernel=False, poisoning=True)
        val_ds = TheDataset(dataset_name=dataset_name, mode='val', poison_transform=val_poison_transfrom, normalizing=True)
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=2)

        test_poison_transfrom = PoisonTransform(1., target_class=target_class, diff_location=False, diff_kernel=False, poisoning=True)
        test_poison_ds = TheDataset(dataset_name=dataset_name, mode='test', poison_transform=test_poison_transfrom, normalizing=True)
        self.test_loader_poison = torch.utils.data.DataLoader(test_poison_ds, batch_size=batch_size, shuffle=True, num_workers=2)

        test_cl_ds = TheDataset(dataset_name=dataset_name, mode='test', poison_transform=None, normalizing=True)
        self.test_loader_cl = torch.utils.data.DataLoader(test_cl_ds, batch_size=batch_size, shuffle=True, num_workers=2)

        return

    def train(self, model, optimizer, num_epochs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            model.cuda()

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        criterion = torch.nn.CrossEntropyLoss()

        theBest = np.inf
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs, _, _ = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(self.train_loader.dataset)
            train_accuracy = 100 * correct_train / len(self.train_loader.dataset)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            val_loss = 0.0
            correct_val = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _, _ = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    correct_val += (predicted == labels).sum().item()

            val_loss /= len(self.val_loader.dataset)
            val_accuracy = 100 * correct_val / len(self.val_loader.dataset)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%", flush=True)

            if epoch % 1 == 0 and val_loss < theBest:
                # Save the entire model
                torch.save(model.state_dict(), f'models/revisiting_model.pth')
                theBest = val_loss


        return model

    def test(self, model):

        print('Poison data test:', flush=True)
        N.test_model(model, self.test_loader_poison)
    
        print('Clean data test:', flush=True)
        N.test_model(model, self.test_loader_cl)


def main():
    train = False
    revisiting = Revisiting(dataset_name='cifar10', target_class=1, batch_size=128)

    if train:
        model = N.TransferLearningModel(input_shape=(3, 32, 32), model_type='resnet18', dense=512, num_classes=10)

        total_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_biases = sum(p.numel() for p in model.parameters() if p.requires_grad and p.data.ndimension() == 1)
        print(f'model total trained weights: {total_weights}', flush=True)
        print(f'model total trained biases: {total_biases}', flush=True)

        optimizer = torch.optim.Adam(model.parameters(),  lr=1e-4, weight_decay=0.0001, foreach=True)
        model = revisiting.train(model=model, optimizer=optimizer, num_epochs=300)
        
    model = N.TransferLearningModel((3, 32, 32), model_type='resnet18', num_classes=10)
    if isinstance(torch.load(f'models/revisiting_model.pth'), dict):
        model.load_state_dict(torch.load(f'models/revisiting_model.pth'))
    else:
        model = torch.load(f'models/revisiting_model.pth')
    revisiting.test(model)

    dataset = TheDataset('cifar10', mode='val', poison_transform=None, normalizing=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    N.trigger_training(model, None, 0, (3, 32, 32), 128, val_loader, num_epochs=50, trigger_generation='revisiting')




if __name__ == '__main__':
    main()