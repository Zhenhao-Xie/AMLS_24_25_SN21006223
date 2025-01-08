import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

from medmnist import BreastMNIST
from medmnist import INFO

class BreastClassification:
    def __init__(self, download=True):

        self.data_flag = 'breastmnist'
        self.download = download
        
        # Get dataset info
        self.info = INFO[self.data_flag]
        self.n_channels = self.info['n_channels']
        self.n_classes = len(self.info['label'])
        
        # Training device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
        
        # Data transformation
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Train/val/test splits
        self.train_dataset = BreastMNIST(split='train', root="Datasets", transform=data_transform, download=self.download)
        self.val_dataset   = BreastMNIST(split='val',   root="Datasets", transform=data_transform)
        self.test_dataset  = BreastMNIST(split='test',  root="Datasets", transform=data_transform)

        print("Train dataset info:", self.train_dataset)

        self.train_loader = data.DataLoader(dataset=self.train_dataset, shuffle=True)
        self.val_loader   = data.DataLoader(dataset=self.val_dataset,   shuffle=False)
        self.test_loader  = data.DataLoader(dataset=self.test_dataset,  shuffle=False)

        # Prepare numpy data for classical ML
        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = self._prepare_sklearn_data()

    def _prepare_sklearn_data(self):
        """
        Converts train/val/test sets into NumPy arrays suitable for scikit-learn.
        """
        def dataset_to_numpy(dataset):
            data_x = dataset.imgs
            data_y = dataset.labels
            # Flatten
            data_x = data_x.reshape(len(data_x), -1)
            data_y = data_y.reshape(len(data_y))
            return data_x, data_y
        
        X_train, y_train = dataset_to_numpy(self.train_dataset)
        X_val,   y_val   = dataset_to_numpy(self.val_dataset)
        X_test,  y_test  = dataset_to_numpy(self.test_dataset)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_random_forest(self, param_grid):
        """
        Train Random Forest with hyperparameter tuning via GridSearchCV.
        """
        print("Training Random Forest with GridSearchCV...")

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf,
            param_grid=param_grid,
            scoring='accuracy',
            n_jobs=-1,
            cv=3,
            verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        print(f"Best RF Params: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_preds = best_model.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_preds)
        print(f"RF Val Accuracy: {val_acc*100:.2f}%")

        # Evaluate on test set
        test_preds = best_model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, test_preds)
        print(f"RF Test Accuracy: {test_acc*100:.2f}%\n")
        
        return best_model, test_preds, test_acc

    def train_svm(self, param_grid):
        """
        Train SVM with hyperparameter tuning via GridSearchCV.
        """
        print("Training SVM with GridSearchCV...")

        svm_model = SVC(probability=False, random_state=42)
        grid_search = GridSearchCV(
            svm_model,
            param_grid=param_grid,
            scoring='accuracy',
            n_jobs=-1,
            cv=3,
            verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        print(f"Best SVM Params: {grid_search.best_params_}")

        # Evaluate on validation set
        val_preds = best_model.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_preds)
        print(f"SVM Val Accuracy: {val_acc*100:.2f}%")

        # Evaluate on test set
        test_preds = best_model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, test_preds)
        print(f"SVM Test Accuracy: {test_acc*100:.2f}%\n")

        return best_model, test_preds, test_acc

    # CNN model definition
    class ResNet18(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            
            # Load ResNet18 model without pre-trained weights
            self.model = models.resnet18(weights=None)
            
            # Modify the first layer to accept single-channel images
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            self.model.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            return self.model(x)

    def _train_one_resnet_setting(self, lr, num_epochs, batch_size):
        """
        Train a single ResNet-18 model with given hyperparameters.
        """
        # Create data loaders with the given batch size
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = data.DataLoader(self.val_dataset,   batch_size=batch_size, shuffle=False)

        # Initialize ResNet18 model
        model = self.ResNet18(num_classes=self.n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)

                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
        # Validation accuracy
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)

                pred_probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(pred_probs, dim=1)
                val_correct += (preds == targets.long().squeeze()).sum().item()
                val_total += targets.size(0)

        val_accuracy = val_correct / val_total
        return model, val_accuracy

    def train_resnet(self, 
                     lr_candidates=[1e-3, 1e-4], 
                     batch_candidates=[32, 64, 128], 
                     epochs_candidates=[10, 20, 30],
                     final_train_plot_path="A/resnet_final_training.png"):
        """
        Train ResNet-18 with hyperparameter search.
        """
        print("Searching best ResNet-18 hyperparameters...")

        best_val_acc = -1.0
        best_params = None
        best_model = None

        # Grid search over hyperparameters
        for lr in lr_candidates:
            for bsz in batch_candidates:
                for n_ep in epochs_candidates:
                    print(f"Trying: lr={lr}, batch_size={bsz}, epochs={n_ep} ...")
                    model_temp, val_acc_temp = self._train_one_resnet_setting(lr, n_ep, bsz)
                    print(f"Val Accuracy: {val_acc_temp*100:.2f}%")

                    if val_acc_temp > best_val_acc:
                        best_val_acc = val_acc_temp
                        best_params = (lr, bsz, n_ep)
                        best_model = model_temp

        print("="*50)
        print(f"Best ResNet params found: LR={best_params[0]}, BatchSize={best_params[1]}, Epochs={best_params[2]}")
        print(f"Best Val Accuracy: {best_val_acc*100:.2f}%")
        print("="*50)

        # Train again with best hyperparameters to plot training curve
        lr_best, bsz_best, epoch_best = best_params

        train_loader = data.DataLoader(self.train_dataset, batch_size=bsz_best, shuffle=True)
        val_loader   = data.DataLoader(self.val_dataset,   batch_size=bsz_best, shuffle=False)
        test_loader  = data.DataLoader(self.test_dataset,  batch_size=bsz_best, shuffle=False)

        model_final = self.ResNet18(num_classes=self.n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_final.parameters(), lr=lr_best)

        train_losses = []
        test_losses = []

        for ep in range(epoch_best):
            # Train
            model_final.train()
            epoch_train_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).long().squeeze()

                optimizer.zero_grad()
                outputs = model_final(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            # Validate
            model_final.eval()
            epoch_test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device).long().squeeze()
                    outputs = model_final(inputs)
                    loss = criterion(outputs, targets)
                    epoch_test_loss += loss.item()

            epoch_test_loss /= len(val_loader)
            test_losses.append(epoch_test_loss)

        # Calculate final test accuracy
        model_final.eval()
        test_correct = 0
        test_total = 0
        all_test_preds = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model_final(inputs)

                pred_probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(pred_probs, dim=1)
                test_correct += (preds == targets.long().squeeze()).sum().item()
                test_total += targets.size(0)
                all_test_preds.extend(preds.cpu().tolist())

        final_test_acc = test_correct / test_total
        print(f"Final ResNet Test Accuracy: {final_test_acc*100:.2f}%\n")

        # Plot training curve
        plt.figure(figsize=(8,6))
        plt.plot(range(1, epoch_best+1), train_losses, label='Train Loss')
        plt.plot(range(1, epoch_best+1), test_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Task A ResNet-18 Training Curve (Best Hyperparams)")
        plt.legend()
        plt.savefig(final_train_plot_path, dpi=300)
        plt.show()

        return model_final, np.array(all_test_preds), final_test_acc

