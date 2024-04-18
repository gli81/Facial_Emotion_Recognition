

def train_model(
        checkpoint_folder:"str",
        lr: "float",
        momentum: "float",
        weight_decay: "float",
        num_epoch: "int",
        loader,
        loss_func,
        lr_decay=1,
        lr_decay_epoch=5,

):
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
