from utils import TheDataset, PoisonTransform
from torch import nn
import torch
import cv2
import numpy as np
import copy
import Network as N

# I want to chck updating.
def evaluation(model, dataset_name, batch_size, input_size, target_class, trigger_epochs, unlearning_epochs,
               testing_number):
    poison_transform = PoisonTransform(1.0, target_class, diff_location=False, diff_kernel=False)
    test_dataset_poison = TheDataset(dataset_name, mode='test', poison_transform=poison_transform)
    test_loader_poison = torch.utils.data.DataLoader(test_dataset_poison, batch_size=batch_size, shuffle=False,
                                                     num_workers=2)

    dataset_cl = TheDataset(dataset_name, mode='test', poison_transform=None)
    test_loader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset_cl = TheDataset(dataset_name, mode='val', poison_transform=None, normalizing=False)
    val_loader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_cl = TheDataset(dataset_name, mode='train', poison_transform=None, normalizing=False)
    train_loader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model.cuda()

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    for n in range(testing_number):
        print(f'Test Number {n}:')
        pattern_param = nn.Parameter(torch.randn((input_size[1], input_size[2], input_size[0])), requires_grad=True)
        mask_param = nn.Parameter(torch.randn((input_size[1], input_size[2], 1)), requires_grad=True)

        optimizer = torch.optim.Adam([pattern_param, mask_param], lr=0.01, betas=(0.9, 0.999))

        for epoch in range(trigger_epochs):
            model.eval()
            running_loss = 0.0
            correct_train = 0

            for inputs, labels in val_loader_cl:
                mask = torch.tanh(mask_param)
                mask = (mask + 1.) / 2.
                mask = torch.clip(mask, 0.0, 1.0)
                pattern = torch.tanh(pattern_param)
                pattern = ((pattern + 1.0) / 2.0) * 255.0
                pattern = torch.clip(pattern, 0.0, 255.0)
                poisoned_inputs = torch.empty((len(inputs), input_size[0], input_size[1], input_size[2]))
                for i, inp in enumerate(inputs):
                    poison_inp = inp * (1 - mask) + (pattern * mask)
                    if i == 0:
                        cv2.imwrite(f'images/poisoned_image_test_number_{n}.png',
                                    np.uint8((poison_inp.detach().cpu().numpy())))
                    poison_inp = poison_inp / 255.0
                    poison_inp = (poison_inp - mean) / std
                    poisoned_inputs[i, ...] = torch.permute(poison_inp, (2, 0, 1))

                poisoned_inputs = poisoned_inputs.to(device)
                optimizer.zero_grad()
                outputs_y, middle, _ = model(poisoned_inputs)

                y_target = torch.full((outputs_y.size(0),), target_class, dtype=torch.long).to(device)

                loss = (nn.CrossEntropyLoss()(outputs_y, y_target) + torch.sum(mask.view(-1)) * 0.01)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs_y.data, 1)
                correct_train += (predicted == y_target).sum().item()

            train_loss = running_loss / len(val_loader_cl.dataset)
            train_accuracy = 100 * correct_train / len(val_loader_cl.dataset)
            print(f"  Epoch [{epoch + 1}/{trigger_epochs}], "
                  f"Loss: {train_loss:.4f}, ASR Trigger: {train_accuracy:.2f}%, "
                  f"Mask Norm: {torch.sum(mask.view(-1)):.2f}")

        cv2.imwrite(f'images/mask_test_number_{n}.png', np.uint8((mask.detach().cpu().numpy() * 255)))
        cv2.imwrite(f'images/pattern_test_number_{n}.png', np.uint8(pattern.detach().cpu().numpy()))

        unlearned_model = copy.deepcopy(model)
        if torch.cuda.is_available():
            unlearned_model.cuda()

        # Freeze the layers
        for param in list(unlearned_model.parameters())[:]:
            param.requires_grad = True

        # Disable BatchNorm and Dropout
        for module in unlearned_model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.eval()

        optimizer = torch.optim.Adam(unlearned_model.parameters(), lr=0.0001)
        mask = mask.detach()
        pattern = pattern.detach()
        print('  Begin the unlearning process:')
        for epoch in range(unlearning_epochs):
            unlearned_model.train()
            running_loss = 0.0
            correct_train = 0

            for inputs, labels in val_loader_cl:
                poisoned_inputs = torch.empty((len(inputs), input_size[0], input_size[1], input_size[2]))
                for i, inp in enumerate(inputs):
                    poison_inp = inp * (1 - mask) + (pattern * mask)
                    poison_inp = poison_inp / 255.0
                    poison_inp = (poison_inp - mean) / std
                    poisoned_inputs[i, ...] = torch.permute(poison_inp, (2, 0, 1))

                poisoned_inputs = poisoned_inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs_y, middle, _ = unlearned_model(poisoned_inputs)

                loss = nn.CrossEntropyLoss()(outputs_y, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs_y.data, 1)
                correct_train += (predicted == labels).sum().item()

            val_loss = running_loss / len(val_loader_cl.dataset)
            val_accuracy = 100 * correct_train / len(val_loader_cl.dataset)
            print(f"  Epoch [{epoch + 1}/{unlearning_epochs}], "
                  f"Loss: {val_loss:.4f},  Val Acc: {val_accuracy:.2f}%")

        val_loss = 0.0
        correct_val = 0
        unlearned_model.eval()
        print('  Test the unlearning process:')
        with torch.no_grad():
            for inputs, labels in test_loader_poison:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _ = unlearned_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(test_loader_poison.dataset)
        val_accuracy = 100 * correct_val / len(test_loader_poison.dataset)
        print(f"  Test [{n}/{testing_number}], "
              f"Val Loss: {val_loss:.4f}, ASR Trigger: {val_accuracy:.2f}%")

        val_loss = 0.0
        correct_val = 0
        unlearned_model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader_cl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _ = unlearned_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(test_loader_cl.dataset)
        val_accuracy = 100 * correct_val / len(test_loader_cl.dataset)
        print(f"  Test [{n}/{testing_number}], "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")


def main():
    # added_model = N.TransferLearningModel((3, 32, 32), 'resnet18', num_classes=11)
    # added_model.load_state_dict(torch.load(f'models/model_cifar10_resnet18_11_added.pth'))
    # model = N.TrickModel(added_model, dense=512, num_classes=10)
    model = N.TransferLearningModel((3, 32, 32), model_type='resnet18', num_classes=10)
    model.load_state_dict(torch.load(f'models/revisiting_model.pth'))
    dataset_name = 'cifar10'
    batch_size = 128
    input_size = (3, 32, 32)
    target_class = 1
    testing_number = 10
    trigger_epochs = 50
    unlearning_epochs = 5

    evaluation(model, dataset_name, batch_size, input_size, target_class, trigger_epochs, unlearning_epochs,
               testing_number)


if __name__ == '__main__':
    main()


