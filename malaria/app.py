#!/usr/bin/env python3
"""
Malaria Detection using Deep Learning
====================================

This application provides malaria detection using both TensorFlow and PyTorch models.
It includes data preprocessing, model training, evaluation, and visualization capabilities.

Author: Extracted from Jupyter notebook
"""

import os
import math

# Suppress INFO & WARNING logs from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models
from torchsummary import summary

# Scikit-learn imports
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def print_class_counts(dataset, dataset_name):
    """Print class distribution for a dataset."""
    count_positive = 0
    count_negative = 0
    for sample in dataset:
        if sample["label"] == 1:
            count_positive += 1
        else:
            count_negative += 1

    print(f'{dataset_name}:')
    print(f'Positive samples: {count_positive}, Proportion: {count_positive / (count_positive + count_negative):.2f}')
    print(f'Negative samples: {count_negative}, Proportion: {count_negative / (count_positive + count_negative):.2f}')
    print()


def augment_train_images(sample, img_size=(64, 64)):
    """Augment training images."""
    image = sample["image"]
    label = sample["label"]

    image = tf.image.resize(image, img_size)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    image = tf.image.per_image_standardization(image)

    return image, label


def augment_val_test_images(sample, img_size=(64, 64)):
    """Augment validation/test images."""
    image = sample["image"]
    label = sample["label"]

    image = tf.image.resize(image, img_size)
    image = tf.image.per_image_standardization(image)

    return image, label


def save_checkpoint(epoch, model, optimizer, loss, save_path):
    """Save model checkpoint."""
    model.eval()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    """Train a PyTorch model."""
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.permute(0, 3, 1, 2))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs.reshape(-1), labels.float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step(loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        save_checkpoint(epoch, model, optimizer, loss, f'./model_checkpoint_epoch_{epoch}.pth')


class NumpyDataset(Dataset):
    """PyTorch Dataset for numpy arrays."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class SimpleCNN(nn.Module):
    """Simple CNN model for PyTorch."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """Residual block for ResNet."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model for PyTorch."""
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers_list = []
        for stride in strides:
            layers_list.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers_list)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return torch.sigmoid(out)


def main():
    """Main function to run the malaria detection pipeline."""
    print("Malaria Detection using Deep Learning")
    print("=" * 40)

    # ========== DATA LOADING ==========
    print("\n=== Task 1: Import the Libraries ===")
    print("Libraries imported successfully")

    print("\n=== Task 2: Load the Dataset ===")
    data_path = 'cell_images/'
    
    try:
        if not os.path.exists(data_path):
            print(f"Warning: Data path '{data_path}' not found. Using TensorFlow Datasets malaria dataset instead.")
            malaria_builder = tfds.builder("malaria")
            malaria_builder.download_and_prepare()
            malaria_dataset = malaria_builder.as_dataset(split="train")
        else:
            malaria_folder = tfds.ImageFolder(data_path)
            dataset_dict = malaria_folder.as_dataset()
            # Handle case where as_dataset() returns a dict
            if isinstance(dataset_dict, dict):
                # Get the first available split (usually 'train' or None)
                if 'train' in dataset_dict:
                    malaria_dataset = dataset_dict['train']
                else:
                    # Take the first split available
                    malaria_dataset = list(dataset_dict.values())[0]
            else:
                malaria_dataset = dataset_dict
        print("Dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to TensorFlow Datasets malaria dataset...")
        malaria_builder = tfds.builder("malaria")
        malaria_builder.download_and_prepare()
        malaria_dataset = malaria_builder.as_dataset(split="train")

    # ========== VISUALIZE IMAGES ==========
    print("\n=== Task 3: Visualize Images ===")
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.ravel()

    for i, example in enumerate(malaria_dataset.take(10)):
        image = example["image"]
        label = example["label"]
        image = tf.image.resize(image, [100, 100])
        axs[i].imshow(image.numpy().astype("uint8"))
        axs[i].title.set_text(f'Label: {"Parasitized" if label.numpy() else "Uninfected"}')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

    # ========== PREPROCESS IMAGES ==========
    print("\n=== Task 4: Preprocess the Images ===")
    image = [example["image"] for example in malaria_dataset.take(1)][0]
    image_resized = tf.image.resize(image, (64, 64)).numpy()
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_normalized = image_gray / 255.0
    image_blur = cv2.GaussianBlur(image_normalized, (5, 5), 0)

    fig, axs = plt.subplots(1, 5, figsize=(20, 20))
    titles = ['Original', 'Resized', 'GrayScaled', 'Normalized', 'Blurred']
    images = [image, image_resized.astype(int), image_gray, image_normalized, image_blur]

    # for i, img in enumerate(images):
    #     axs[i].imshow(img, cmap='gray')
    #     axs[i].set_title(titles[i])
    #     axs[i].axis('off')

    # plt.tight_layout()
    # plt.show()

    # ========== SPLIT THE DATA ==========
    print("\n=== Task 5: Split the Data ===")
    malaria_dataset = malaria_dataset.shuffle(buffer_size=10000)
    malaria_dataset = malaria_dataset.take(len(malaria_dataset) // 10)

    TRAIN_SET_SIZE = 0.7
    VAL_SET_SIZE = 0.15
    TEST_SIZE = 0.15

    dataset_size = len(malaria_dataset)
    train_size = int(TRAIN_SET_SIZE * dataset_size)
    val_size = int(VAL_SET_SIZE * dataset_size)
    test_size = int(TEST_SIZE * dataset_size)

    train_dataset = malaria_dataset.take(train_size)
    val_test_dataset = malaria_dataset.skip(train_size)
    val_dataset = val_test_dataset.skip(val_size)
    test_dataset = val_test_dataset.take(test_size)

    print_class_counts(train_dataset, 'Train Dataset')
    print_class_counts(val_dataset, 'Validation Dataset')
    print_class_counts(test_dataset, 'Test Dataset')

    # ========== AUGMENT THE IMAGES ==========
    print("\n=== Task 6: Augment the Images ===")
    data_gen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )

    BATCH_SIZE = 32
    IMG_SIZE = (64, 64)

    augmented_train_dataset = train_dataset.map(lambda x: augment_train_images(x, IMG_SIZE)).batch(BATCH_SIZE)
    resized_val_dataset = val_dataset.map(lambda x: augment_val_test_images(x, IMG_SIZE)).batch(BATCH_SIZE)
    resized_test_dataset = test_dataset.map(lambda x: augment_val_test_images(x, IMG_SIZE)).batch(BATCH_SIZE)

    # ========== TENSORFLOW MODELS ==========
    print("\n=== Task 7: Set Up a Neural Network ===")
    train_set = augmented_train_dataset
    val_set = resized_val_dataset
    test_set = resized_test_dataset

    for image_batch, label_batch in train_set.take(1):
        input_shape = image_batch[0].shape
        num_classes = len(tf.unique(label_batch).y)

    print(f"Image shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    print("\n=== Task 8: Define the Model Architecture ===")
    # CNN Model
    cnn_model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # ResNet-like Model
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])
    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    resnet_model = tf.keras.Model(inputs, outputs)

    print("\n=== Task 9: Prepare the Model for Training ===")
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    # Use regular optimizer for both models (Keras 3 compatibility)
    resnet_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

    cnn_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    resnet_model.compile(optimizer=resnet_optimizer, loss=loss_fn, metrics=metrics)

    print("\n=== Task 10: Train and Monitor the Net ===")
    EPOCHS = 2

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    def scheduler(epoch, lr):
        if epoch < 2:
            # Convert to float for Keras 3 compatibility
            return float(lr * math.exp(-0.1))
        return float(lr)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    print("Training CNN model...")
    cnn_history = cnn_model.fit(
        train_set,
        validation_data=val_set,
        epochs=EPOCHS,
        callbacks=[early_stopping_callback, lr_scheduler, tensorboard_callback]
    )

    print("Training ResNet model...")
    resnet_history = resnet_model.fit(
        train_set,
        validation_data=val_set,
        epochs=EPOCHS,
        callbacks=[early_stopping_callback, lr_scheduler, tensorboard_callback]
    )

    print("\n=== Task 11: Evaluate the Performance ===")
    test_loss, test_acc, test_precision, test_recall, test_auc = resnet_model.evaluate(test_set)
    print("test loss, test accuracy, test precision, test recall, test auc:", test_loss, test_acc, test_precision, test_recall, test_auc)

    y_pred = resnet_model.predict(test_set)
    y_pred_classes = np.concatenate(y_pred >= 0.5).astype(int)
    y_true = np.concatenate([labels for img, labels in test_set])

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt="d")
    plt.show()

    print(classification_report(y_true, y_pred_classes))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_classes)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    hist = resnet_history.history
    plt.figure()
    plt.plot(hist['loss'], label='Training Loss')
    plt.plot(hist['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()

    plt.figure()
    plt.plot(hist['accuracy'], label='Training Accuracy')
    plt.plot(hist['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()

    print(f"Training Accuracy: {np.round(np.max(hist['accuracy']), 3)}")
    print(f"Validation Accuracy: {np.round(np.max(hist['val_accuracy']), 3)}")
    print(f"Test Accuracy: {np.round(test_acc, 3)}")

    # ========== PYTORCH MODELS ==========
    print("\n=== Task 12: Set Up a Neural Network (PyTorch) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    numpy_train_data = np.concatenate([data for data, _ in train_set.as_numpy_iterator()])
    numpy_train_labels = np.concatenate([labels for _, labels in train_set.as_numpy_iterator()])

    numpy_val_data = np.concatenate([data for data, _ in val_set.as_numpy_iterator()])
    numpy_val_labels = np.concatenate([labels for _, labels in val_set.as_numpy_iterator()])

    numpy_test_data = np.concatenate([data for data, _ in test_set.as_numpy_iterator()])
    numpy_test_labels = np.concatenate([labels for _, labels in test_set.as_numpy_iterator()])

    train_dataloader = DataLoader(NumpyDataset(numpy_train_data, numpy_train_labels), batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(NumpyDataset(numpy_val_data, numpy_val_labels), batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(NumpyDataset(numpy_test_data, numpy_test_labels), batch_size=BATCH_SIZE, shuffle=True)

    print("\n=== Task 13: Define the Model Architecture (PyTorch) ===")
    simple_model = SimpleCNN()
    simple_model = simple_model.to(device)

    resnet_model_pt = ResNet(ResidualBlock, [2, 2, 2])
    resnet_model_pt = resnet_model_pt.to(device)

    print("\n=== Task 14: Prepare the Model for Training (PyTorch) ===")
    learning_rate = 0.001
    loss_fn = torch.nn.BCELoss()

    simple_optimizer = torch.optim.Adam(simple_model.parameters(), lr=learning_rate)
    resnet_optimizer = torch.optim.Adam(resnet_model_pt.parameters(), lr=learning_rate)

    simple_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(simple_optimizer, 'min')
    resnet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(resnet_optimizer, 'min')

    print("\n=== Task 15: Train and Monitor the Net (PyTorch) ===")
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    dataset_sizes = {
        "train": sum(len(i[0]) for i in train_dataloader),
        "val": sum(len(i[0]) for i in val_dataloader)
    }

    print("Training Simple CNN (PyTorch)...")
    train_model(simple_model, loss_fn, simple_optimizer, simple_scheduler, EPOCHS, dataloaders, dataset_sizes, device)

    print("Training ResNet (PyTorch)...")
    train_model(resnet_model_pt, loss_fn, resnet_optimizer, resnet_scheduler, EPOCHS, dataloaders, dataset_sizes, device)

    print("\n=== Task 16: Evaluate the Performance (PyTorch) ===")
    resnet_model_pt.load_state_dict(torch.load(f'./model_checkpoint_epoch_{EPOCHS-1}.pth')["model_state_dict"])
    resnet_model_pt.eval()

    test_loss = 0.0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = resnet_model_pt(data.permute(0, 3, 1, 2))
            _, preds = torch.max(outputs, 1)
            loss = torch.nn.BCELoss()(outputs.reshape(-1), target.float())
            test_loss += loss.item() * data.size(0)
            correct += torch.sum(preds == target.data)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss = test_loss / len(test_dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    accuracy = correct.double() / len(test_dataloader.dataset)
    print('\nTest Accuracy: {:.6f} ({}/{})'.format(accuracy, correct, len(test_dataloader.dataset)))

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

    print(classification_report(all_targets, all_preds))

    # ========== TRANSFER LEARNING ==========
    print("\n=== Task 17: Fine-Tune Pre-Trained TensorFlow Model ===")
    pre_trained_tf_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    pre_trained_tf_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(pre_trained_tf_model.output)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    tf_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    tf_model = tf.keras.Model(inputs=pre_trained_tf_model.input, outputs=tf_output)

    tf_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0),
        metrics=['accuracy']
    )

    tf_model_history = tf_model.fit(
        train_set,
        epochs=10,
        validation_data=val_set
    )

    y_pred_tf = tf_model.predict(test_set)
    y_pred_classes_tf = np.concatenate(y_pred_tf >= 0.5).astype(int)
    y_true_tf = np.concatenate([label for imgs, label in test_set])

    print(classification_report(y_pred_classes_tf, y_true_tf))

    print("\n=== Task 18: Fine-Tune Pre-Trained PyTorch Model ===")
    pre_trained_pt_model = torchvision.models.mobilenet_v2(pretrained=True)

    for param in list(pre_trained_pt_model.parameters())[:-2]:
        param.requires_grad = False

    pre_trained_pt_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(pre_trained_pt_model.last_channel, 1),
        nn.Sigmoid()
    )

    pt_loss_fn = torch.nn.BCELoss()
    pt_optimizer = torch.optim.Adam(pre_trained_pt_model.parameters(), lr=0.001)
    pt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pt_optimizer, 'min')

    train_model(pre_trained_pt_model, pt_loss_fn, pt_optimizer, pt_scheduler, 10, dataloaders, dataset_sizes, device)

    pre_trained_pt_model.load_state_dict(torch.load('./model_checkpoint_epoch_9.pth')["model_state_dict"])
    pre_trained_pt_model.eval()

    test_loss = 0.0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = pre_trained_pt_model(data.permute(0, 3, 1, 2))
            _, preds = torch.max(outputs, 1)
            loss = torch.nn.BCELoss()(outputs.reshape(-1), target.float())
            test_loss += loss.item() * data.size(0)
            correct += torch.sum(preds == target.data)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss = test_loss / len(test_dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    accuracy = correct.double() / len(test_dataloader.dataset)
    print('\nTest Accuracy: {:.6f} ({}/{})'.format(accuracy, correct, len(test_dataloader.dataset)))

    print(classification_report(all_preds, all_targets))

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
