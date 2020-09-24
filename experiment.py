import click
import torch
import os
import sys
import time
import numpy as np
import glob
from batchup import work_pool, data_source
from unet.unet_model import vgg16_UNet
from unet.loss import FocalLoss
from utils.split import new_data_splitter
from utils.load import load_image_vgg16, load_tgt_vgg16, ImageAccessor
from utils.utils import augment_batch, save_checkpoint
import loggers
import torchvision.models as models


def train(model, optimizer, data_loaders, criterion, batch_size, torch_device, phase):

    if phase == 'train':
        model.train(True)
        dataset = data_loaders['train']
    else:
        model.train(False)
        dataset = data_loaders['val']

    losses = []

    for i, (batch_X, batch_y) in enumerate(dataset.batch_iterator(batch_size=batch_size, shuffle=True)):

        inputs = torch.tensor(batch_X, dtype=torch.float, device=torch_device)
        target_gt = torch.tensor(batch_y, dtype=torch.long, device=torch_device)

        # Forward + backward + optimize
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = criterion(prediction, target_gt)

        if phase == 'train':
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))
        else:
            losses.append(float(loss.data))

    return np.mean(losses)


@click.command()
@click.option('--n_classes', type=int, default=2, help='Number of classes predicted by the network')
@click.option('--bilinear', type=bool, default=True, help='Method of upsampling in up convolutions')
@click.option('--dropout', type=bool, default=True, help='Whether to use dropout or not')
@click.option('--learning_rate', type=float, default=0.0001, help='Set learning rate for optimizer')
@click.option('--gamma', type=float, default=1, help='Focal loss gamma value')
@click.option('--alpha', type=float, default=0.25, help='Focal loss alpha value')
@click.option('--data_path', type=str, default='dataset/', help='Path to saved dataset')
@click.option('--val_percent', type=float, default=0.20, help='Fraction of test set used for validation')
@click.option('--batch_size', type=int, default=4, help='Mini-batch size')
@click.option('--logfile', type=str, default='logfile.txt', help='Log file name')
@click.option('--date', type=int, help='Enter todays data to generate results folder')
def experiment(n_classes, bilinear, dropout, learning_rate, gamma, alpha, data_path, val_percent,
               batch_size, rep_num, logfile, augment, date):

    settings = locals().copy()

    # Make results folder for specific parameters
    savepath = 'results/{}/g{}_alp{}_bs{}_lr{}_bilin{}_do{}_aug{}'.format(date, gamma, alpha, batch_size,
                                                                          learning_rate, bilinear, dropout, augment)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if logfile != 'none':
        try:
            logger = loggers.Logger('{}/{}'.format(savepath, logfile))
        except loggers.LogAlreadyExistsError as e:
            print(e.message)
            return
        logger.connect()

    print('Program: {}'.format(sys.argv[0]))
    print('Command line: {}'.format(' '.join(sys.argv)))
    print('Settings:')
    print(', '.join(['{}={}'.format(k, settings[k]) for k in sorted(settings.keys())]))

    best_accuracy = float('inf')

    # LOAD MODEL
    input_channels = 5
    print('LOAD VGG16 UNET MODEL')
    torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = vgg16_UNet(models.vgg16(), input_channels, n_classes, bilinear, dropout).to(torch_device)
    model.encoder.expand_input(input_channels)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLoss(gamma=gamma, alpha=alpha)

    print('Selecting dataset from training islands...')

    trainset_X, valset_X, trainset_y, valset_y = new_data_splitter(data_path, test_island,
                                                                   island_refs, observer_list,
                                                                   val_percent)


    train_X = ImageAccessor(trainset_X, load_image_vgg16)
    val_X = ImageAccessor(valset_X, load_image_vgg16)

    train_y = ImageAccessor(trainset_y, load_tgt_vgg16)
    val_y = ImageAccessor(valset_y, load_tgt_vgg16)

    trainset = data_source.ArrayDataSource([train_X, train_y])
    valset = data_source.ArrayDataSource([val_X, val_y])

    # Augment
    trainset = trainset.map(augment_batch)
    valset = valset.map(augment_batch)

    pool = work_pool.WorkerThreadPool(4)
    trainset = pool.parallel_data_source(trainset)
    valset = pool.parallel_data_source(valset)

    data_loaders = {'train': trainset, 'val': valset}

    print('BEGIN TRAINING...')
    total_train_loss = []
    total_val_loss = []
    no_improvement = 0
    epoch = 0

    while no_improvement < 10:

        t1 = time.time()
        print('-' * 10)

        # Train and validate
        train_loss = train(model, optimizer, data_loaders, criterion, batch_size, torch_device, 'train')
        val_loss = train(model, optimizer, data_loaders, criterion, batch_size, torch_device, 'val')

        t2 = time.time()
        print('Epoch {} took {:.3f}s; training loss = {:.6f}; validation loss = {:.6f}'.format(epoch, t2 - t1,
                                                                                               train_loss,
                                                                                               val_loss))
        # Save losses
        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)

        # Check loss and save checkpoint
        is_best = bool(val_loss < best_accuracy)

        if is_best:
            no_improvement = 0  # reset the counter after new best found
        else:
            no_improvement += 1  # count how many non-improvements

        print('Current best loss: {}'.format(best_accuracy))
        best_accuracy = min(val_loss, best_accuracy)

        print('{}/checkpoint.pth.tar'.format(savepath))

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_accuracy': torch.FloatTensor([best_accuracy])
        }, is_best, '{}/checkpoint.pth.tar'.format(savepath))

        print('No improvement in loss for {} epochs'.format(no_improvement))

        if epoch == 5:
            print('Setting new learning rate...')
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/10.)

        epoch += 1

    np.savez('{}/TrainingLoss.npz'.format(savepath),
             train_loss=total_train_loss, val_loss=total_val_loss)

    return


if __name__ == '__main__':
    experiment()

