import torch
from torch.autograd import grad
from sklearn.metrics import f1_score
from models.cnn import create_model
import time
import numpy as np
import pdb

def train(epoch, device, train_loader, optimizer, modelObj, loss_function, writer):
    ############# Training the model   ##############
    modelObj.train()

    total_correct = 0  # Variable that calculates the total correct predictions done by the network
    running_loss = 0.0  # Variable to calculate the total loss of the dataset
    true_labels = []
    predicted_labels = []

    # calculating the accuacy and the loss of the network on the training loader dataset
    for batch in train_loader:
        training_images, training_labels = batch  # Getting tensor of images and tensor of corresponding labels
        training_images, training_labels = training_images.to(device), training_labels.to(device)

        optimizer.zero_grad(  )# Set the gradient values to zero so that already present values are not added to the new batch

        predictions = modelObj(training_images) # Pass the batch to the network
        loss_train = loss_function(predictions, training_labels) # Calculating the loss

        loss_train.backward() # Calculate the gradients
        optimizer.step() # Update the weights

        running_loss += loss_train.item()  # Add the loss calculated per batch

        # Calculate the element wise equality to measure the accuracy between the predictions and the labels
        predict_y = torch.max(predictions, dim=1)[1]
        total_correct += (predict_y == training_labels).sum().item()

        # Collect true labels and predicted labels for calculating F1 score
        true_labels.extend(training_labels.tolist())
        predicted_labels.extend(predict_y.tolist())

    accuracy_train_epoch = total_correct / len(train_loader)
    loss_train_epoch = running_loss / len(train_loader)
    f1_score_epoch = f1_score(true_labels, predicted_labels, average='macro')

    # Log training accuracy and loss per epoch to tensorboard
    writer.add_scalar('Train Accuracy', accuracy_train_epoch, epoch + 1)
    writer.add_scalar('Train Loss', loss_train_epoch, epoch + 1)
    writer.add_scalar('Train F1 Score', f1_score_epoch, epoch + 1)
    return accuracy_train_epoch, loss_train_epoch, f1_score_epoch


def evaluate(epoch, device, modelObj, test_loader, loss_function, writer):
    ############# Testing the model  ##############
    modelObj.eval()

    total_correct = 0
    running_loss = 0.0
    best_accuracy = 0.0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        # calculating the accuacy and the loss of the network on the test loader dataset
        for batch in test_loader:
            test_images, test_labels = batch  # Getting tensor of images and tensor of corresponding labels
            test_images, test_labels = test_images.to(device), test_labels.to(device)

            predictions = modelObj(test_images)  # Pass the batch to the network
            loss_test = loss_function(predictions, test_labels)  # Calculating the loss

            running_loss += loss_test.item()  # Add the loss calculated per batch

            # Calculate the element wise equality to measure the accuracy between the predictions and the labels
            predict_y = torch.max(predictions, dim=1)[1]
            total_correct += (predict_y == test_labels).sum().item()

            # Collect true labels and predicted labels for calculating F1 score
            true_labels.extend(test_labels.tolist())
            predicted_labels.extend(predict_y.tolist())

        accuracy_test_epoch = total_correct / len(test_loader)
        loss_test_epoch = running_loss / len(test_loader)
        f1_score_epoch = f1_score(true_labels, predicted_labels, average='macro')

        if accuracy_test_epoch > best_accuracy:
            best_accuracy = accuracy_test_epoch

        # Log test accuracy and loss per epoch to tensorboard
        writer.add_scalar('Test Accuracy', accuracy_test_epoch, epoch + 1)
        writer.add_scalar('Test Loss', loss_test_epoch, epoch + 1)
        writer.add_scalar('Test F1 Score', f1_score_epoch, epoch + 1)

    return accuracy_test_epoch, loss_test_epoch, f1_score_epoch

def hvps(grad_all, model_params, h_estimate):
    element_product = 0
    for grad_elem, v_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * v_elem)

    return_grads = grad(element_product, model_params, create_graph=True)
    return return_grads


def cal_k_matrix(x, sigma=1.0):
    """
    :param x: n*m的数组，n为记录条数，m为数据维度
    :param sigma:用于计算矩阵的常数
    :return:K矩阵
    """
    # data_size = x.shape[0]
    # k_matrix = np.zeros((data_size, data_size))
    # for i in range(data_size):
    #     for j in range(data_size):
    #         k_matrix[i, j] = np.exp(-np.linalg.norm(x[i]-x[j]) ** 2 / sigma ** 2)
    x = x.cpu().data.numpy()
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    x1 = np.dot(x, x.T)
    xd = np.expand_dims(x1.diagonal(), 0)
    i = np.ones((1, xd.shape[1]))
    x2 = np.dot(xd.T, i)
    x3 = np.dot(i.T, xd)
    Kx = x2 + x3 - 2 * x1
    Kx = np.exp(- Kx / sigma ** 2)
    return Kx


def cal_hsic(x, y):
    kx = cal_k_matrix(x)
    ky = cal_k_matrix(y)
    n = kx.shape[0]

    kxy = np.dot(kx, ky)
    h = np.trace(kxy) / n ** 2 + np.mean(kx) * np.mean(ky) - 2 * np.mean(kxy) / n
    return h * n ** 2 / (n - 1) ** 2

