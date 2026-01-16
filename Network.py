import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.utils as util
import torchvision.transforms as transforms
import cv2

from utils import corrupt_maker


# from torch.utils.tensorboard import SummaryWriter


class TrafficSign(nn.Module):
    def __init__(self, base=32, dense=512, num_classes=43):
        super(TrafficSign, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=base, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=base, out_channels=base, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(in_channels=base, out_channels=base * 2, kernel_size=(3, 3), padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=base * 2, out_channels=base * 2, kernel_size=(3, 3))
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.2)

        self.conv5 = nn.Conv2d(in_channels=base * 2, out_channels=base * 4, kernel_size=(3, 3), padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=base * 4, out_channels=base * 4, kernel_size=(3, 3))
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base * 4 * 4, dense)
        self.batch_norm = nn.BatchNorm1d(dense)
        self.relu7 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(dense, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_hidden=False):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        features = self.flatten(x)
        x = self.fc1(features)
        x = self.batch_norm(x)
        x = self.relu7(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.softmax(x)
        if return_hidden:
            return x, features, features
        else: 
            return x


class Mnist(nn.Module):
    def __init__(self, base=16, dense=512, num_classes=10):
        super(Mnist, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=base, kernel_size=(5, 5), padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(in_channels=base, out_channels=base * 2, kernel_size=(5, 5), padding=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(base * 2 * 7 * 7, dense)
        self.batch_norm = nn.BatchNorm1d(dense)
        self.fc2 = nn.Linear(dense, num_classes)

    def forward(self, x, return_hidden=False):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        features = self.flatten(x)
        x = self.fc1(features)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)

        if return_hidden:
            return x, features, features
        else:
            return x


class TransferLearningModel(nn.Module):
    def __init__(self, input_shape, model_type='resnet18', dense=512, num_classes=10):
        super(TransferLearningModel, self).__init__()

        self.input_shape = input_shape
        self.model_type = model_type

        # self.resized_inputs = nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False)

        if self.model_type == 'resnet18':
            base_model = models.resnet18(weights="IMAGENET1K_V1")
        elif self.model_type == 'vgg16':
            base_model = models.vgg16(weights="IMAGENET1K_V1")

        # Remove the final classification layer of the pre-trained model
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        # Freeze the layers
        for param in list(self.features.parameters())[:]:
            param.requires_grad = True

        if self.model_type == 'resnet18':
            self.fc1 = nn.Linear(base_model.fc.in_features, dense)
        else:
            self.fc1 = nn.Linear(base_model.classifier[0].in_features, dense)

        self.batch_norm = nn.BatchNorm1d(dense)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dense, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_hidden=False):

        # x = self.resized_inputs(x)
        x = self.features(x)
        features = x.view(x.size(0), -1)
        x = self.fc1(features)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        if return_hidden:   
            return x, features, features
        else:
            return x

     # for FeatureRE
    
    def from_input_to_features(self, x):
        x = self.features(x)
        return x

    def from_features_to_output(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x


def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs=5):
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

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct_val / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%", flush=True)

        if epoch % 1 == 0 and val_loss < theBest:
            # Save the entire model
            torch.save(model.state_dict(), f'models/model.pth')
            theBest = val_loss
            
        scheduler.step()

    return model


def test_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    correct_test = 0
    test_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct_test / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%", flush=True)

    return test_accuracy, test_loss


class TrickModel(nn.Module):
    def __init__(self, added_class_model, dense=512, num_classes=43):
        super(TrickModel, self).__init__()

        self.features = nn.Sequential(*list(added_class_model.children())[:-6])

        self.fc1 = nn.Linear(added_class_model.fc1.in_features, dense)
        self.batch_norm = nn.BatchNorm1d(dense)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dense, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_hidden=False):
        x = self.features(x)
        features = x.view(x.size(0), -1)
        x = self.fc1(features)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        if return_hidden:
            return x, features, features
        else:
            return x


# Example: Gradually increase weights over training epochs
def get_dynamic_weights(epoch, total_epochs):
    w_ce = 1.0  # Cross-entropy remains constant
    w_mse = min(1, 0 + (epoch / total_epochs) * (1 - 0))  # Increase MSE
    w_kl = min(1, 0 + (epoch / total_epochs) * (1 - 0))  # Increase KL
    w_cs = min(10, 0 + (epoch / total_epochs) * (10 - 0))  # Increase Cosine similarity

    return w_ce, w_mse, w_kl, w_cs

class CustomLoss(nn.Module):
    def __init__(self, w_ce=1, w_mse=1e+2, w_kl=1e+2, w_cs=10, dynamic=True):
        super(CustomLoss, self).__init__()
        self.w_ce = w_ce
        self.w_mse = w_mse
        self.w_kl = w_kl
        self.w_cs = w_cs
        self.dynamic = dynamic

    def forward(self, poison_outputs_y, poison_middle,
                clean_outputs_y, clean_middle, labels, epoch, num_epochs, original_features=None, poison_features=None):
        
        if self.dynamic:
            self.w_ce, self.w_mse, self.w_kl, self.w_cs = get_dynamic_weights(epoch, num_epochs)

        # w5 = 1
        cross_entropy = nn.CrossEntropyLoss()(poison_outputs_y, labels)
        # print("cross_entopy:", cross_entropy)
        mse = nn.MSELoss()(poison_middle, clean_middle)
        # print("mse:", mse)
        kl_div = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(poison_outputs_y, dim=1), torch.softmax(clean_outputs_y, dim=1))
        # print("kl_div:", kl_div)
        cosine_sim = torch.mean(1 - nn.CosineSimilarity(dim=1)(poison_middle, clean_middle))
        # print("cosine_sim:", cosine_sim)

        loss = (self.w_ce * cross_entropy) + (self.w_mse * mse) + (self.w_kl * kl_div) + (self.w_cs * cosine_sim)

        return loss


def trigger_training(model, feature_extractor, target_class, input_size, batch_size, val_loader, num_epochs=10,
                     trigger_generation=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pattern_param = nn.Parameter(torch.randn((input_size[1], input_size[2], input_size[0])), requires_grad=True)
    mask_param = nn.Parameter(torch.randn((input_size[1], input_size[2], 1)), requires_grad=True)

    optimizer = torch.optim.Adam([pattern_param, mask_param], lr=0.01, betas=(0.9, 0.999))

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])

    model.to(device=device)

    for epoch in range(num_epochs):
        model.eval()
        running_loss = 0.0
        correct_train = 0

        for inputs, labels in val_loader:
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
                    cv2.imwrite(f'images/poisoned_image_epoch_{trigger_generation}.png',
                                np.uint8((poison_inp.detach().cpu().numpy())))
                poison_inp = poison_inp / 255.0
                poison_inp = (poison_inp - mean) / std
                poisoned_inputs[i, ...] = torch.permute(poison_inp, (2, 0, 1))

            poisoned_inputs = poisoned_inputs.to(device)
            optimizer.zero_grad()
            outputs_y = model.forward(poisoned_inputs)

            y_target = torch.full((outputs_y.size(0),), target_class, dtype=torch.long).to(device)

            loss = (nn.CrossEntropyLoss()(outputs_y, y_target) + torch.sum(mask.view(-1)) * 0.01)
            loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs_y.data, 1)
            correct_train += (predicted == y_target).sum().item()

    train_loss = running_loss / len(val_loader.dataset)
    train_accuracy = 100 * correct_train / len(val_loader.dataset)
    print(f"Trigger Epoch [{epoch}/{num_epochs}], "
          f"Loss: {train_loss:.4f}, ASR Trigger: {train_accuracy:.2f}%, "
          f"Mask Norm: {torch.sum(mask.view(-1)):.2f}", flush=True)

    cv2.imwrite(f'images/mask_epoch_{trigger_generation}.png', np.uint8((mask.detach().cpu().numpy() * 255)))
    cv2.imwrite(f'images/pattern_epoch_{trigger_generation}.png', np.uint8(pattern.detach().cpu().numpy()))

    feature_extractor.eval()

    trigger = mask * pattern

    _, _, trigger_features = feature_extractor(torch.permute(trigger, (2, 0, 1)).unsqueeze(0).to(device))

    return trigger_features


def trick_model_training(model, clean_model, feature_extractor, optimizer, scheduler, batch_size, train_loader, val_loader_poison,
                         val_loader,
                         target_class,
                         input_size, num_epochs=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model.cuda()
        clean_model.cuda()
        feature_extractor.cuda()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_custom_loss = CustomLoss(dynamic=True)
    val_custom_loss = CustomLoss(dynamic=False)
    criterion = torch.nn.CrossEntropyLoss()
    theBest = np.inf

    feature_extractor.eval()
    image_shape = train_loader.dataset[0][0].shape
    original_trigger = corrupt_maker(np.zeros(image_shape).transpose(1, 2, 0))
    original_trigger = torch.permute(torch.tensor(original_trigger), (2, 0, 1)).unsqueeze(0).float()
    original_trigger_features = None
    poisoned_trigger_features = None
    # Disable BatchNorm and Dropout

    for epoch in range(num_epochs):
        clean_model.eval()
        running_loss = 0.0
        correct_train = 0
        trigger_generation = epoch
        model.train()
        # for module in model.modules():
        #     if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        #         module.eval()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            clean_outputs_y, clean_middle = clean_model.forward(inputs, return_hidden=True)

            poison_outputs_y, poison_middle = model.forward(inputs, return_hidden=True)
            # if epoch >= 25:
            #     _, _, original_trigger_features = feature_extractor(original_trigger.to(device))
            #     poisoned_trigger_features = trigger_training(model, feature_extractor,
            #                                                  target_class=target_class,
            #                                                  input_size=input_size,
            #                                                  batch_size=batch_size,
            #                                                  val_loader=val_loader,
            #                                                  num_epochs=10,
            #                                                  trigger_generation=trigger_generation)
            loss = train_custom_loss(poison_outputs_y, poison_middle,
                               clean_outputs_y, clean_middle, labels,
                               epoch, num_epochs, original_trigger_features,
                               poisoned_trigger_features)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(poison_outputs_y.data, 1)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        val_loss = 0.0
        correct_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader_poison:
                inputs, labels = inputs.to(device), labels.to(device)
                clean_outputs_y, clean_middle = clean_model.forward(inputs, return_hidden=True)
                outputs, middle = model.forward(inputs, return_hidden=True)
                loss = val_custom_loss(outputs, middle,
                                   clean_outputs_y, clean_middle, labels,
                                    epoch, num_epochs,
                                   original_trigger_features,
                                   poisoned_trigger_features
                                   )
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader_poison.dataset)
        val_accuracy = 100 * correct_val / len(val_loader_poison.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%", flush=True)

        if val_loss < theBest:
            torch.save(model.state_dict(), f'models/model_poison_aug.pth')
            theBest = val_loss
        
        scheduler.step()

    return model


def trace_removal(poison_model, clean_loader, poison_loader, num_epochs, optimizer, scheduler):
    
    poison_model.to('cuda').eval()

    print(poison_model)

    activations = {}

    def hook_fn(module, input, output):
        activations[id(module)] = output.clone()  # Clone to avoid modifying it in-place

    # Register hooks on all FC layers
    for module in poison_model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Flatten, nn.AdaptiveAvgPool2d):  # Adjust for your model if needed
            module.register_forward_hook(hook_fn)


    # Freeze all layers except FC layers
    for param in list(poison_model.parameters())[:-5]:
        param.requires_grad = False  

    for module in poison_model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, (nn.Flatten, nn.AdaptiveAvgPool2d)):
            for param in module.parameters():
                param.requires_grad = True  # Unfreeze FC layers only

    poison_model.train()

    # Disable BatchNorm and Dropout
    for module in poison_model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()

    total_weights = sum(p.numel() for p in poison_model.parameters() if p.requires_grad)
    total_biases = sum(p.numel() for p in poison_model.parameters() if p.requires_grad and p.data.ndimension() == 1)
    print(f'poison_model total trained weights: {total_weights}', flush=True)
    print(f'poison_model total trained biases: {total_biases}', flush=True)

    for epoch in range(num_epochs):

        poison_correct_train = 0
        clean_correct_train = 0
        running_loss = 0.0
        
        for clean, poison in zip(clean_loader, poison_loader):

            cl_inps, cl_labels = clean
            cl_inps, cl_labels = cl_inps.to('cuda'), cl_labels.to('cuda')
            poison_inps, poison_labels = poison
            poison_inps, poison_labels = poison_inps.to('cuda'), poison_labels.to('cuda')
            
            optimizer.zero_grad()
            
            # Forward pass
            clean_outputs_y = poison_model.forward(cl_inps)
            # Get activations for clean and poison inputs
            clean_activations = {key: activations[key].detach() for key in activations}
            poison_outputs_y = poison_model.forward(poison_inps)
             # Get activations for clean and poison inputs
            poison_activations = {key: activations[key] for key in activations}

            # Find neurons with the most significant activation difference
            idxs = {}
            for key in clean_activations.keys():
                diff = torch.abs(clean_activations[key] - poison_activations[key])
                num_top_k = 50 if diff.shape[1] > 100 else 2  # 100 for fc1, 10 for fc2
                idxs[key] = torch.topk(diff, num_top_k, dim=1)[1]  # Select top 10 most activated neurons

            # Compute regularization loss
            regularize_term = 0
            for key in poison_activations.keys():
                for i in range(poison_inps.shape[0]):
                    reg_loss = (poison_activations[key][i][idxs[key][i]] - clean_activations[key][i][idxs[key][i]])**2
                    regularize_term += torch.mean(reg_loss)

            regularize_term = regularize_term / (len(poison_activations) * poison_inps.shape[0])

            # Compute losses
            poison_loss = nn.CrossEntropyLoss()(poison_outputs_y, poison_labels)
            clean_loss = nn.CrossEntropyLoss()(clean_outputs_y, cl_labels)

            loss =  poison_loss + (clean_loss) + (1e-3*regularize_term)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy
            _, poison_predicted = torch.max(poison_outputs_y.data, 1)
            poison_correct_train += (poison_predicted == poison_labels).sum().item()

            _, clean_predicted = torch.max(clean_outputs_y.data, 1)
            clean_correct_train += (clean_predicted == cl_labels).sum().item()

        train_loss = running_loss / len(poison_loader.dataset)
        train_accuracy = 100 * clean_correct_train / len(clean_loader.dataset)
        asr = 100 * poison_correct_train / len(poison_loader.dataset)

        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Loss: {train_loss:.4f}, ASR Accuracy: {asr:.2f}%, "
              f"Accuracy: {train_accuracy}%", flush=True)

        torch.save(poison_model.state_dict(), f'models/model_poison_aug_traced.pth')
        scheduler.step()


    return poison_model
