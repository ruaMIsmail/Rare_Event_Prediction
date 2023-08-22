import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from data_process import DataProcessor
from AutoEncoder.model import AE, HyperParameter

params = HyperParameter()

class AutoencoderClassifier:
    def __init__(self, csv_file_path, device, batch_size, epochs):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_processor = DataProcessor(csv_file_path)
        self.autoencoder = None
        self.optimizer = None
        self.criterion_recon = nn.MSELoss()
        self.criterion_classification = nn.BCELoss()

    def train_autoencoder(self):
        self.data_processor.load_data()
        self.data_processor.knn_impute()
        self.data_processor.scale_data()
        self.data_processor.one_hot_encode()
        self.data_processor.split_data()

        self.autoencoder = AE(params.input_dim, params.encoding_dim, params.hidden_dim)  # Define your autoencoder model here
        self.autoencoder.to(self.device)

        # Create datasets and dataloaders
        train_dataset = TensorDataset(torch.FloatTensor(self.data_processor.X_train.values).to(self.device),
                                      torch.FloatTensor(self.data_processor.y_train.values).unsqueeze(1).to(self.device))
        train_dl = DataLoader(train_dataset, batch_size=self.batch_size)

        valid_dataset = TensorDataset(torch.FloatTensor(self.data_processor.X_val.values).to(self.device),
                                      torch.FloatTensor(self.data_processor.y_val.values).unsqueeze(1).to(self.device))
        valid_dl = DataLoader(valid_dataset, batch_size=self.batch_size)

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)

        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            self.autoencoder.train()
            for batch_data, batch_labels in train_dl:
                batch_data = batch_data.to(params.device)  # Move batch_data to the same device
                batch_labels = batch_labels.to(params.device)  # Move target_data to the same device

                self.optimizer.zero_grad()  # Clear gradients
                decoded_output, classification_output = self.autoencoder(batch_data)

                reconstruction_loss = self.criterion_recon(decoded_output, batch_data)  # Autoencoder reconstruction loss
                classification_loss = self.criterion_classification(classification_output, batch_labels)  # Classification loss
                total_loss = reconstruction_loss + classification_loss

                total_loss.backward()  # Compute gradients
                self.optimizer.step()  # Update weights
                #utils.clip_grad_norm_(m.parameters(), max_norm=1.0)  # Apply gradient clipping
                #scheduler.step()
                train_losses.append(total_loss.item())
                print(f'Epoch [{epoch+1}/{params.epochs}], Batch Loss: {total_loss.item():.4f}')

            self.autoencoder.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                total_loss = 0
                for batch_data, batch_labels in valid_dl:
                    batch_data = batch_data.to(params.device)  # Move batch_data to the same device
                    batch_labels = batch_labels.to(params.device)  # Move target_data to the same device
                    decoded_output, classification_output = self.autoencoder(batch_data)

                    reconstruction_loss = self.criterion_recon(decoded_output, batch_data)  # Autoencoder reconstruction loss
                    classification_loss = self.criterion_classification(classification_output, batch_labels)

                    total_loss += (reconstruction_loss + classification_loss).item() * len(batch_data)

                average_loss = total_loss / len(valid_dl)
                val_losses.append(average_loss)
                # Print and visualize the loss values
                #print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {average_loss:.4f}')

                # Save the trained model
        torch.save(self.autoencoder.state_dict(), "autoencoder_classifier.pth")
