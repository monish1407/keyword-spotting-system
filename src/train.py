# =========================================
# Training Script for Keyword Spotting
# =========================================

import os
import torch
import numpy as np
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config import *
from features import extract_mfcc
from model import KeywordLSTM


def main():

    # -------------------------------------
    # Device
    # -------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------------
    # Ensure data folder exists
    # -------------------------------------
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # -------------------------------------
    # Load Dataset
    # -------------------------------------
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root="data",
        download=True
    )

    # -------------------------------------
    # Filter Selected Keywords
    # -------------------------------------
    filtered_files = []

    for file_path in dataset._walker:
        label = os.path.basename(os.path.dirname(file_path))
        if label in SELECTED_LABELS:
            filtered_files.append(file_path)

    print("Filtered files:", len(filtered_files))

    # -------------------------------------
    # Limit Samples Per Class
    # -------------------------------------
    class_count = {label: 0 for label in SELECTED_LABELS}
    reduced_files = []

    for file_path in filtered_files:
        label = os.path.basename(os.path.dirname(file_path))

        if class_count[label] < MAX_PER_CLASS:
            reduced_files.append(file_path)
            class_count[label] += 1

    print("Reduced files:", len(reduced_files))

    # -------------------------------------
    # Feature Extraction
    # -------------------------------------
    label_to_index = {label: idx for idx, label in enumerate(SELECTED_LABELS)}

    features = []
    labels = []

    for file_path in reduced_files:
        label = os.path.basename(os.path.dirname(file_path))

        mfcc = extract_mfcc(file_path, N_MFCC, FIXED_TIME_STEPS)

        if mfcc is not None:
            features.append(mfcc)
            labels.append(label_to_index[label])

    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    print("Feature shape:", features.shape)

    # -------------------------------------
    # Normalize
    # -------------------------------------
    features = (features - features.mean()) / features.std()

    # -------------------------------------
    # Train-Test Split
    # -------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE
    )

    # -------------------------------------
    # Model
    # -------------------------------------
    model = KeywordLSTM(
        input_size=N_MFCC,
        hidden_size=HIDDEN_SIZE,
        num_classes=len(SELECTED_LABELS)
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------------------
    # Training Loop
    # -------------------------------------
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # -------------------------------------
    # Evaluation
    # -------------------------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # -------------------------------------
    # Save Model
    # -------------------------------------
    torch.save(model.state_dict(), "models/keyword_model.pth")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()