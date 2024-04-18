# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import os

def train_model(
        model,
        device,
        checkpoint_folder:"str",
        save_name: "str",
        lr: "float",
        momentum: "float",
        weight_decay: "float",
        num_epoch: "int",
        train_loader,
        val_loader_1,
        loss_func,
        val_loader_2=None,
        lr_decay=1,
        lr_decay_epoch=5,

):
    train_loss_hist = []
    train_acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    # start the training/validation process
    print("==> Training starts!")
    print("="*50)
    ## create optimizer
    optimizer = optim.SGD(

    )
    for i in range(0, num_epoch):
        print("Epoch %d:" %i)
        ## Train on the train set
        #####################################################################
        # switch to train mode
        model.train()
        # this help you compute the training accuracy
        total_examples = 0
        correct_examples = 0
        train_loss = 0 # track training loss if you want
        # Train the model for 1 epoch.
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # compute the output and loss
            out = model(inputs)
            loss = loss_func(out, targets)
            # zero the gradient
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # apply gradient and update the weights
            optimizer.step()
            # count the number of correctly predicted samples in the current batch
            _, predicted = torch.max(out, 1)
            correct = predicted.eq(targets).sum()
            train_loss += loss.detach().cpu()
            total_examples += targets.shape[0]
            correct_examples += correct.item()
        avg_loss = train_loss / len(train_loader)
        train_loss_hist.append(avg_loss)
        avg_acc = correct_examples / total_examples
        train_acc_hist.append(avg_acc)
        print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
        ######################################################################

        # Validate on the validation dataset (masked)
        ######################################################################
        # switch to eval mode
        model.eval()
        # this help you compute the validation accuracy
        total_examples = 0
        correct_examples = 0
        val_loss = 0 # track the validation loss
        # disable gradient during validation, which can save GPU memory
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader_1):
                # copy inputs to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                # compute the output and loss
                out = model(inputs)
                loss = loss_func(out, targets)
                # count the number of correctly predicted samples
                # in the current batch
                _, predicted = torch.max(out, 1)
                correct = predicted.eq(targets).sum()
                val_loss += loss.detach().cpu()
                total_examples += targets.shape[0]
                correct_examples += correct.item()
        avg_loss = val_loss / len(val_loader_1)
        test_loss_hist.append(avg_loss)
        avg_acc = correct_examples / total_examples
        test_acc_hist.append(avg_acc)
        print(
            "Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc)
        )
        total_examples = 0
        correct_examples = 0
        val_loss = 0
        if val_loader_2:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader_2):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    out = model(inputs)
                    loss = loss_func(out, targets)
                    _, predicted = torch.max(out, 1)
                    correct = predicted.eq(targets).sum()
                    val_loss += loss.detach().cpu()
                    total_examples += targets.shape[0]
                    correct_examples += correct.item()
            avg_loss = val_loss / len(val_loader_2)
            test_acc_hist.append(avg_loss)
            avg_acc = correct_examples / total_examples
            test_acc_hist.append(avg_acc)
        ######################################################################

        # save the model checkpoint
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
                print("Saving ...")
                state = {'state_dict': model.state_dict(),
                        'epoch': i,
                        'lr': current_learning_rate}
                torch.save(state, os.path.join(checkpoint_folder, save_name))
        print('')
        # decay learning rate
        if i % lr_decay_epoch == 0 and i != 0:
            current_learning_rate *= lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_learning_rate
                print(f"Current learning rate has decayed to %f" %current_learning_rate)

    print("="*50)
    print(
        f"==> Optimization finished! Best validation accuracy: {best_val_acc:.4f}"
    )