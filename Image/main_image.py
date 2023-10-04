from data.dataloader import data
from Exp.exp_IF_HSIC import evaluate as evaluate
from Exp.exp_IF_HSIC import hvps as hvps
from Exp.exp_IF_HSIC import train as train
from Exp.exp_IF_HSIC import cal_hsic as cal_hsic

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch import optim
import torch.nn as nn

from models.cnn import create_model

<<<<<<< HEAD
def main(args):
=======
def _set_random_seed(seed=2022):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")

def main(args):
    _set_random_seed(20231004)
>>>>>>> 91def5f (first push)
    device = args['device'] if torch.cuda.is_available() else 'cpu'
    learning_rate = args['learning_rate']
    epochs = args['epochs']
    unlearn_ratio = args['unlearn_ratio']
    unlearn_iteration = args['unlearn_iteration']
    damp = args['damp']
    scale = args['scale']
    numda = args['numda']
    method = args['method']
    model_name = args['model_name']
    dataset_name = args['dataset_name']

    dataload = data(dataset_name, model_name).loadData()
    train_loader = dataload[0]
    x_train, y_train = [], []
    for batch in train_loader:
        x, y = batch
        x_train.append(x)
        y_train.append(y)
    x_train = torch.cat(x_train, dim=0).to(device)
    y_train = torch.cat(y_train, dim=0).to(device)

    test_loader = dataObj[1]
    x_test, y_test = [], []
    for batch in test_loader:
        x, y = batch
        x_test.append(x)
        y_test.append(y)
    x_test = torch.cat(x_test, dim=0).to(device)
    y_test = torch.cat(y_test, dim=0).to(device)

    model = create_model(model_name, dataset_name).to(device)

    # Setting the loss function and optimizer for the network
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    """Training"""
    for epoch in range(epochs):
        accuracy_train_epoch, loss_train_epoch, f1_train_epoch = train(epoch, device, train_loader, optimizer, modelObj, loss_function, writer)
        print(f"Train, epoch {epoch}, acc: {accuracy_train_epoch}, loss: {loss_train_epoch}, f1: {f1_train_epoch}")


    grad_all, grad1, grad2 = None, None, None

    if method in ['IF', 'HSIC']:
        out1 = modelObj.forward_once(x_train)
        out2 = modelObj.forward_once_unlearn(x_train)

        # randomly remove data points based on the unlearn_ratio
        mask1 = np.array([False] * out1.shape[0])
        # mask0 = mask1
        # Randomly select size indices
        indices = np.random.choice(out1.shape[0], size=int((1 - unlearn_ratio) * out1.shape[0]), replace=False)
        # indices0 = np.random.choice(out1.shape[0], size=int(0.05*out1.shape[0]), replace=False)
        # mask0[indices0] = True
        mask1[indices] = True
        # mask2 = mask1

        loss = F.nll_loss(out1, y_train, reduction='sum')
        loss1 = F.nll_loss(out1[mask1], y_train[mask1], reduction='sum')
        # loss2 = F.nll_loss(out2[mask2], y_train[mask2], reduction='sum')
        if method == 'HSIC':
            loss = loss_hsic = numda * (
                # cal_hsic(data.train_mask, data.y)
                    - cal_hsic(x_train, out1)
                    + cal_hsic(x_train[mask1], y_train[mask1]) + cal_hsic(x_train, out2) - cal_hsic(x_train, out2)
            )
            loss1_ = numda * (cal_hsic(x_train, y_train) - cal_hsic(x_train, out1))
            # loss2_ = numda * (cal_hsic(x_train[mask1], y_train[mask1]) + cal_hsic(x_train, out2))
            loss += loss_hsic
            loss1 += loss1_
            # loss2 += loss2_
        model_params = [p for p in modelObj.parameters() if p.requires_grad]
        torch.autograd.set_detect_anomaly(True)
        grad_all = grad(loss, model_params, retain_graph=True, create_graph=True)
        grad1 = grad(loss1, model_params, retain_graph=True, create_graph=True)
        # grad2 = grad(loss2, model_params, retain_graph=True, create_graph=True)

    accuracy_test_epoch, loss_test_epoch, f1_test_epoch = evaluate(epoch, device, modelObj, test_loader, loss_function,
                                                               writer)
    print(f"Evaluation: acc: {accuracy_test_epoch}, loss: {loss_test_epoch}, f1: {f1_test_epoch}")

    if method in ['IF', 'HSIC']:
        res_tuple = (grad_all, grad1, grad2)
        v = tuple(grad1 for grad1 in res_tuple[1])
        # h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = tuple(grad1 for grad1 in res_tuple[1])
        for i in range(unlearn_iteration):
            model_params = [p for p in modelObj.parameters() if p.requires_grad]
            hv = hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate = [v1 + (1 - damp) * h_estimate1 - hv1 / scale for v1, h_estimate1, hv1 in
                          zip(v, h_estimate, hv)]

        params_change = [h_est / scale for h_est in h_estimate]
        params_esti = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        idx = 0
        for p in modelObj.parameters():
            p.data = params_esti[idx]
            idx = idx + 1

        out = modelObj.forward_once_unlearn(x_test)

        f1_unlearn = f1_score(out, y_test, average="macro")
        print('[Unlearn] F1 score found %.2f' % f1_unlearn)

