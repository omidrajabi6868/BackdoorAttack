import numpy as np
import torch
import Network as N
import matplotlib.pyplot as plt
from utils import *

class Experiment:
    def __init__(self, models,  clean_ins_outs, poisoned_ins_outs):
        self.models = models
        self.clean_inputs = clean_ins_outs[0].to('cuda')
        self.poisoned_inputs = poisoned_ins_outs[0].to('cuda')
        self.clean_input_activations = {}
        self.poisoned_input_activations = {}
        self.fc_outputs = {}

        # Register forward hooks for all models
        for key in self.models.keys():
            self.models[key].relu.register_forward_hook(self.hook_fn(key))

    def hook_fn(self, key):
        def hook(module, input, output):
            self.fc_outputs[key] = output.clone().detach().cpu().numpy()
        return hook

    def mad(self):
        with open("magnitude.csv", "w") as f:
            f.write('\t' + '\t'.join(self.models.keys()) + '\n')

        with open("magnitude.csv", "a") as f:
            # Process clean inputs
            f.write('MAD (clean inputs)\t')
            for key in self.models.keys():
                self.models[key](self.clean_inputs)
                self.clean_input_activations[key] = self.fc_outputs[key]

                if 'Clean Model' in self.clean_input_activations:
                    mad = np.mean(np.abs(self.clean_input_activations['Clean Model'] - self.clean_input_activations[key]))
                else:
                    mad = 0
                f.write(f'{mad:.2f}\t')
            f.write('\n')

            # Process poisoned inputs
            f.write('MAD (poisoned inputs)\t')
            for key in self.models.keys():
                self.models[key](self.poisoned_inputs)
                self.poisoned_input_activations[key] = self.fc_outputs[key]

                if 'Clean Model' in self.poisoned_input_activations:
                    mad = np.mean(np.abs(self.poisoned_input_activations['Clean Model'] - self.poisoned_input_activations[key]))
                else:
                    mad = 0
                f.write(f'{mad:.2f}\t')
            f.write('\n')

            # Standard deviation calculations
            f.write('STD (clean inputs)\t')
            for key in self.models.keys():
                std = np.std(self.clean_input_activations[key])
                f.write(f'{std:.2f}\t')
            f.write('\n')

            f.write('STD (poisoned inputs)\t')
            for key in self.models.keys():
                std = np.std(self.poisoned_input_activations[key])
                f.write(f'{std:.2f}\t')
            f.write('\n')

    def plot(self):
        plt.figure(figsize=(9, 6))
        for key in self.models.keys():
            feature_maps = np.mean(self.clean_input_activations[key], axis=0)
            feature_maps = [np.mean(feature_maps[i:i + 8]) for i in range(0, len(feature_maps), 8)]
            plt.plot(np.arange(len(feature_maps)), feature_maps, label=key)

        plt.legend(fontsize=14, markerscale=2)
        plt.title('Clean inputs', fontsize=14)
        plt.ylim([-1, 1])
        plt.xlabel('512 Neurons (Averaged 8 by 8)', fontsize=14)
        plt.ylabel('Activation', fontsize=14)
        plt.savefig('Comparison_clean_image.png')

        plt.figure(figsize=(9, 6))
        for key in self.models.keys():
            feature_maps = np.mean(self.poisoned_input_activations[key], axis=0)
            feature_maps = [np.mean(feature_maps[i:i + 8]) for i in range(0, len(feature_maps), 8)]
            plt.plot(np.arange(len(feature_maps)), feature_maps, label=key)

        plt.legend(fontsize=14, markerscale=2)
        plt.title('Poisoned inputs', fontsize=14)
        plt.ylim([-1, 1])
        plt.xlabel('512 Neurons (Averaged 8 by 8)', fontsize=14)
        plt.ylabel('Activation', fontsize=14)
        plt.savefig('Comparison_poisoned_image.png')


def main():
    models = {}

    def load_model(path, model_type, num_classes):
        state_dict = torch.load(path)
        model = N.TransferLearningModel((3, 32, 32), model_type=model_type, num_classes=num_classes)
        if path.__contains__('aug') or path.__contains__('traced'):
            added_model = N.TransferLearningModel((3, 32, 32), model_type=model_type, num_classes=num_classes + 1)
            added_model.load_state_dict(torch.load('models/model_cifar10_resnet18_11_added.pth'))
            model = N.TrickModel(added_model, num_classes=num_classes)
        model.load_state_dict(state_dict)
        return model.to('cuda')

    models['Clean Model'] = load_model('models/model_cifar10_resnet18_10_clean.pth', 'resnet18', 10).eval()
    # models['Added Model'] = load_model('models/model_cifar10_resnet18_11_added.pth', 'resnet18', 11).eval()
    # models['BadNet'] = load_model('models/model_cifar10_resnet18_10_poison.pth', 'resnet18', 10).eval()
    # models['Augmented'] = load_model('models/model_cifar10_resnet18_10_poison_aug.pth', 'resnet18', 10).eval()
    # models['Traced'] = load_model('models/model_cifar10_resnet18_10_poison_aug_traced.pth', 'resnet18', 10).eval()
    # models['Revisiting'] = load_model('models/revisiting_model.pth', 'resnet18', 10).eval()

    dataset_cl = TheDataset('cifar10', mode='test', poison_transform=None, normalizing=True)
    test_loader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=1024, shuffle=False)

    poison_transform = PoisonTransform(1.0, target_class=1, diff_location=False, diff_kernel=False, poisoning=True)
    poison_dataset = TheDataset('cifar10', mode='test', poison_transform=poison_transform)
    test_loader_poison = torch.utils.data.DataLoader(poison_dataset, batch_size=1024, shuffle=False, num_workers=2)

    clean_ins_outs = next(iter(test_loader_cl))
    poisoned_ins_outs = next(iter(test_loader_poison))

    exp = Experiment(models=models, clean_ins_outs=clean_ins_outs, poisoned_ins_outs=poisoned_ins_outs)
    exp.mad()
    exp.plot()


if __name__ == '__main__':
    main()
