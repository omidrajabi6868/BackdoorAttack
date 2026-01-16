import Network as N
import torch
from utils import *
from Network import train_model, test_model, trick_model_training, trigger_training
import os


# from torchsummary import summary


def main():
    train_clean_model = False
    train_poison_model = False
    train_added_class_model = False
    train_poison_aug_model = False
    train_poison_trace_model = False

    model_list = ['resnet18', 'vgg16', 'conv_6', 'conv_2']
    model_type = model_list[0]
    print(f'Model: {model_type}')
    dataset_list = ['cifar10', 'gtsrb', 'mnist']
    dataset_name = dataset_list[0]
    print(f'Dataset: {dataset_name}')
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-4
    milestones = [50, 75]
    batch_size = 128

    if dataset_name in ['mnist', 'cifar10']:
        target_class = 0
        added_class = 10
        num_classes = 10
        normalizing = True
    if dataset_name == 'gtsrb':
        target_class = 33
        added_class = 43
        num_classes = 43
        normalizing = True

    print(f'num_classes: {num_classes}', flush=True)

    clean_path = f'models/model_{dataset_name}_{model_type}_{num_classes}_clean.pth'
    poison_path = f'models/model_{dataset_name}_{model_type}_{num_classes}_poison.pth'
    added_class_num = num_classes + 1
    added_class_path = f"models/model_{dataset_name}_{model_type}'_{added_class_num}_added.pth"
    poison_aug_path = f'models/model_{dataset_name}_{model_type}_{num_classes}_poison_aug.pth'
    poison_traced_path = f'models/model_{dataset_name}_{model_type}_{num_classes}_traced.pth'

    from resnet import ResNet18
    if 'resnet' in model_type:
        clean_model = ResNet18(num_classes=num_classes)
        poison_model = ResNet18(num_classes=num_classes)
        added_model = ResNet18(num_classes=num_classes + 1)
        trick_model = ResNet18(num_classes=num_classes)
        traced_model = ResNet18(num_classes=num_classes)

    if os.path.exists(clean_path):
        state_dict = torch.load(clean_path)
        clean_model.load_state_dict(state_dict)
    
    if os.path.exists(poison_path):
        state_dict = torch.load(poison_path)
        poison_model.load_state_dict(state_dict)
    
    if os.path.exists(added_class_path):
        state_dict = torch.load(added_class_path)
        added_model.load_state_dict(state_dict)

    if os.path.exists(poison_aug_path):
        state_dict = torch.load(poison_aug_path)
        trick_model.load_state_dict(state_dict)

    if os.path.exists(poison_traced_path):
        state_dict = torch.load(poison_traced_path)
        traced_model.load_state_dict(state_dict)

    if train_clean_model:
        # clean data loaders
        dataset_cl = TheDataset(dataset_name, mode='train', poison_transform=None, normalizing=normalizing)
        train_loader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=True)

        dataset_cl = TheDataset(dataset_name, mode='val', poison_transform=None, normalizing=normalizing)
        val_loader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=True)

        dataset_cl = TheDataset(dataset_name, mode='test', poison_transform=None, normalizing=normalizing)
        test_loader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=False, num_workers=2)

        total_weights = sum(p.numel() for p in clean_model.parameters() if p.requires_grad)
        total_biases = sum(p.numel() for p in clean_model.parameters() if p.requires_grad and p.data.ndimension() == 1)
        print(f'Total trained weights: {total_weights}', flush=True)
        print(f'Total trained biases: {total_biases}', flush=True)

        optimizer = torch.optim.SGD(params=clean_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        clean_model = N.train_model(clean_model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader_cl, val_loader=val_loader_cl, num_epochs=num_epochs)
        N.test_model(model=clean_model, test_loader=test_loader_cl)

    if train_added_class_model:

        # added class data loaders
        poison_transform = PoisonTransform(.1, added_class, diff_location=False, diff_kernel=False, poisoning=True)
        dataset_add = TheDataset(dataset_name, mode='train', poison_transform=poison_transform)
        train_loader_add = torch.utils.data.DataLoader(dataset_add, batch_size=batch_size, shuffle=True, num_workers=2)

        poison_transform = PoisonTransform(.1, added_class, diff_location=False, diff_kernel=False, poisoning=True)
        dataset_add = TheDataset(dataset_name, mode='val', poison_transform=poison_transform)
        val_loader_add = torch.utils.data.DataLoader(dataset_add, batch_size=batch_size, shuffle=True, num_workers=2)

        poison_transform = PoisonTransform(1.0, added_class, diff_location=False, diff_kernel=False, poisoning=False)
        dataset_cl = TheDataset(dataset_name, mode='test', poison_transform=poison_transform, normalizing=normalizing)
        test_loader_cl_add = torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=False, num_workers=2)

        poison_transform = PoisonTransform(1.0, added_class, diff_location=False, diff_kernel=False, poisoning=True)
        test_dataset_add = TheDataset(dataset_name, mode='test', poison_transform=poison_transform)
        test_loader_poison_add = torch.utils.data.DataLoader(test_dataset_add, batch_size=batch_size, shuffle=False, num_workers=2)

        filtered_state_dict = {k: v for k, v in clean_model.state_dict().items() if k in added_model.state_dict() and 'linear' not in k}

        added_model.state_dict().update(filtered_state_dict)

        total_weights = sum(p.numel() for p in added_model.parameters() if p.requires_grad)
        total_biases = sum(p.numel() for p in added_model.parameters() if p.requires_grad and p.data.ndimension() == 1)
        print(f'Total trained weights: {total_weights}', flush=True)
        print(f'Total trained biases: {total_biases}', flush=True)

        optimizer = torch.optim.SGD(params=added_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        added_model = N.train_model(added_model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader_add, val_loader=val_loader_add, num_epochs=num_epochs)
        print(f'Test on poisoned inputs:', flush=True)
        N.test_model(model=added_model, test_loader=test_loader_poison_add)
        print(f'Test on clean inputs:', flush=True)
        N.test_model(model=added_model, test_loader=test_loader_cl_add)

    if train_poison_model:

        poison_transform = PoisonTransform(.1, target_class, diff_location=False, diff_kernel=False, poisoning=True)
        dataset_poison = TheDataset(dataset_name, mode='train', poison_transform=poison_transform)
        train_loader_poison = torch.utils.data.DataLoader(dataset_poison, batch_size=batch_size, shuffle=True, num_workers=2)

        dataset_poison = TheDataset(dataset_name, mode='val', poison_transform=poison_transform)
        val_loader_poison = torch.utils.data.DataLoader(dataset_poison, batch_size=batch_size, shuffle=True, num_workers=2)

        dataset_cl = TheDataset(dataset_name, mode='test', poison_transform=None, normalizing=normalizing)
        test_loader_cl= torch.utils.data.DataLoader(dataset_cl, batch_size=batch_size, shuffle=False, num_workers=2)

        poison_transform = PoisonTransform(1.0, target_class, diff_location=False, diff_kernel=False, poisoning=True)
        test_dataset_poison = TheDataset(dataset_name, mode='test', poison_transform=poison_transform)
        test_loader_poison = torch.utils.data.DataLoader(test_dataset_poison, batch_size=batch_size, shuffle=False, num_workers=2)

        total_weights = sum(p.numel() for p in poison_model.parameters() if p.requires_grad)
        total_biases = sum(p.numel() for p in poison_model.parameters() if p.requires_grad and p.data.ndimension() == 1)
        print(f'Total trained weights: {total_weights}', flush=True)
        print(f'Total trained biases: {total_biases}', flush=True)

        optimizer = torch.optim.SGD(params=poison_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        poison_model = N.train_model(poison_model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader_poison, val_loader=val_loader_poison, num_epochs=num_epochs)
        print(f'Test on poisoned inputs:', flush=True)
        N.test_model(model=poison_model, test_loader=test_loader_poison)
        print(f'Test on clean inputs:', flush=True)
        N.test_model(model=poison_model, test_loader=test_loader_cl)

    if train_poison_aug_model:
        
        poison_transform = PoisonTransform(.1, target_class, diff_location=False, diff_kernel=False, poisoning=True)
        trigger_transform_loc = PoisonTransform(1., target_class, diff_location=True, diff_kernel=False, poisoning=False)
        trigger_transform_kernel = PoisonTransform(1., target_class, diff_location=False, diff_kernel=True, poisoning=False)
        trigger_transform_kernel_loc = PoisonTransform(1., target_class, diff_location=True, diff_kernel=True, poisoning=False)

        poison_dataset = TheDataset(dataset_name, mode='train', poison_transform=poison_transform)
        transform_loc_data = TheDataset(dataset_name, mode='val', poison_transform=trigger_transform_loc)
        transform_kernel_data = TheDataset(dataset_name, mode='val', poison_transform=trigger_transform_kernel)
        transform_kernel_loc_data = TheDataset(dataset_name, mode='val', poison_transform=trigger_transform_kernel_loc)
        new_train_ds = ConcatDataset([poison_dataset, transform_loc_data, transform_kernel_data, transform_kernel_loc_data])
        train_loader_poison_aug = torch.utils.data.DataLoader(new_train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

        poison_transform = PoisonTransform(0.1, target_class, diff_location=False, diff_kernel=False, poisoning=True)
        poison_dataset = TheDataset(dataset_name, mode='test', poison_transform=poison_transform)
        val_loader_poison = torch.utils.data.DataLoader(poison_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        poison_transform = PoisonTransform(1.0, target_class, diff_location=False, diff_kernel=False, poisoning=True)
        test_dataset_poison = TheDataset(dataset_name, mode='test', poison_transform=poison_transform, normalizing=normalizing)
        test_loader_poison = torch.utils.data.DataLoader(test_dataset_poison, batch_size=batch_size, shuffle=False, num_workers=2)

        test_dataset_cl = TheDataset(dataset_name, mode='test', poison_transform=None, normalizing=normalizing)
        test_loader_cl = torch.utils.data.DataLoader(test_dataset_cl, batch_size=batch_size, shuffle=False, num_workers=2)

        filtered_state_dict = {k: v for k, v in added_model.state_dict().items() if k in trick_model.state_dict() and 'linear' not in k}

        trick_model.state_dict().update(filtered_state_dict)

        print(trick_model)

        # Freeze Convolution layers
        for param in list(trick_model.parameters())[:]:
            param.requires_grad = True

        # for param in list(trick_model.parameters())[:-5]:
        #     param.requires_grad = False

        total_weights = sum(p.numel() for p in trick_model.parameters() if p.requires_grad)
        total_biases = sum(p.numel() for p in trick_model.parameters() if p.requires_grad and p.data.ndimension() == 1)
        print(f'Trick model total trained weights: {total_weights}', flush=True)
        print(f'Trick model total trained biases: {total_biases}', flush=True)


        optimizer = torch.optim.Adam(trick_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        dataset = TheDataset(dataset_name, mode='val', poison_transform=None, normalizing=False)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        model = trick_model_training(trick_model, clean_model, clean_model, optimizer, scheduler,  batch_size, train_loader_poison_aug,
                                    val_loader_poison, val_loader, target_class, (3, 32, 32), num_epochs=num_epochs)

        print(f'Test on poisoned inputs:', flush=True)
        N.test_model(model=trick_model, test_loader=test_loader_poison)
        print(f'Test on clean inputs:', flush=True)
        N.test_model(model=trick_model, test_loader=test_loader_cl)

    if train_poison_trace_model:

        traced_model.state_dict().update(trick_model.state_dict())

        poison_transform = PoisonTransform(1, target_class, diff_location=False, diff_kernel=False, poisoning=True)

        dataset_poison = TheDataset(dataset_name, mode='val', poison_transform=poison_transform)
        train_loader_poison = torch.utils.data.DataLoader(dataset_poison, batch_size=batch_size, shuffle=False, num_workers=2)

        dataset_clean = TheDataset(dataset_name, mode='val', poison_transform=None)
        train_loader_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=batch_size, shuffle=False, num_workers=2)

        poison_transform = PoisonTransform(1.0, target_class, diff_location=False, diff_kernel=False, poisoning=True)
        test_dataset_poison = TheDataset(dataset_name, mode='test', poison_transform=poison_transform, normalizing=normalizing)
        test_loader_poison = torch.utils.data.DataLoader(test_dataset_poison, batch_size=batch_size, shuffle=False, num_workers=2)

        test_dataset_cl = TheDataset(dataset_name, mode='test', poison_transform=None, normalizing=normalizing)
        test_loader_cl = torch.utils.data.DataLoader(test_dataset_cl, batch_size=batch_size, shuffle=False, num_workers=2)

        optimizer = torch.optim.Adam(trick_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        traced_model = N.trace_removal(traced_model, train_loader_clean, train_loader_poison, num_epochs, optimizer, scheduler)

        print(f'Test on poisoned inputs:', flush=True)
        N.test_model(model=traced_model, test_loader=test_loader_poison)
        print(f'Test on clean inputs:', flush=True)
        N.test_model(model=traced_model, test_loader=test_loader_cl)

    
    model = ResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load(f'models/model_poison_aug.pth'))

    poison_transform = PoisonTransform(1.0, target_class, diff_location=False, diff_kernel=False, poisoning=True)
    test_dataset_poison = TheDataset(dataset_name, mode='test', poison_transform=poison_transform, normalizing=normalizing)
    test_loader_poison = torch.utils.data.DataLoader(test_dataset_poison, batch_size=batch_size, shuffle=False, num_workers=2)

    test_dataset_cl = TheDataset(dataset_name, mode='test', poison_transform=None, normalizing=normalizing)
    test_loader_cl = torch.utils.data.DataLoader(test_dataset_cl, batch_size=batch_size, shuffle=False, num_workers=2)

    dataset = TheDataset(dataset_name, mode='val', poison_transform=None, normalizing=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print('Poison data test:', flush=True)
    test_model(model, test_loader_poison)

    print('Clean data test:', flush=True)
    test_model(model, test_loader_cl)

    trigger_training(model, None, target_class,
                     (3, 32, 32), batch_size, val_loader, num_epochs=50,
                     trigger_generation='final')


if __name__ == '__main__':
    main()

