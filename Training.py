"""
Original code: https://github.com/echonet/dynamic
Modified by Yuxuan(Jarret) Jiang (2025-09)
Email: jiangyx96@gmail.com
Version: 1.0.0
Description: This is the main script for training the neural network.
"""
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm

import init1
from echo import Echo

def run(
    data_dir=None,
    output=None,

    model_name="deeplabv3_resnet50",
    pretrained=False,
    weights=None,

    run_test=True,
    save_video=True,
    num_epochs=50,
    lr=1e-6,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=0,
    batch_size=4,
    device=None,
    seed=0,
):
    """Trains/tests segmentation model.

    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/segmentation/<model_name>_<pretrained/random>/.
        model_name (str, optional): Name of segmentation model. One of ``deeplabv3_resnet50'',
            ``deeplabv3_resnet101'', ``fcn_resnet50'', or ``fcn_resnet101''
            (options are torchvision.models.segmentation.<model_name>)
            Defaults to ``deeplabv3_resnet50''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        save_video (bool, optional): Whether to save videos with segmentations.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training
            Defaults to 50.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-5.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to math.inf (never decay learning rate).
        num_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory    
    data_dir = "ImageSegmentation"
    output = "output"
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.segmentation.__dict__[model_name](pretrained=pretrained, aux_loss=False)

    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        model = torch.nn.DataParallel(model,device_ids=[3])
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = init1.get_mean_and_std(Echo(root=data_dir, split="train"))
    tasks = [""]
    kwargs = {"target_type": tasks,
              "mean": mean,
              "std": std
              }
    
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = Echo(root=data_dir, split="train", **kwargs)
    dataset["val"] = Echo(root=data_dir, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                # for i in range(torch.cuda.device_count()):
                #     torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, phase == "train", optim, device)
                overall_dice = 2 * (large_inter.sum() + small_inter.sum()) / (large_union.sum() + large_inter.sum() + small_union.sum() + small_inter.sum())
                large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
                small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                    phase,
                                                                    loss,
                                                                    overall_dice,
                                                                    large_dice,
                                                                    small_dice,
                                                                    time.time() - start_time,
                                                                    large_inter.size,
                                                                    sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())),
                                                                    sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())),
                                                                    batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))

        if run_test:
            # Run on validation and test
            for split in ["test"]:
                dataset = Echo(root=data_dir, split=split, **kwargs)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
                loss, large_inter, large_union, small_inter, small_union = run_epoch(model, dataloader, False, None, device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *init1.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), init1.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *init1.bootstrap(large_inter, large_union, init1.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *init1.bootstrap(small_inter, small_union, init1.dice_similarity_coefficient)))
                f.flush()


def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0

    '''pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0'''

    model.train(train)

    large_inter = 0
    large_union = 0
    large_inter_list = []
    large_union_list = []
    
    small_inter = 0
    small_union = 0
    small_inter_list = []
    small_union_list = []
    
    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, dataa) in dataloader:                     
                
                frame1 = dataa[0]
                trace1 = dataa[1]
                frame2 = dataa[2]
                trace2 = dataa[3]
                
                # Run prediction for diastolic frames and compute loss
                frame1 = frame1.to(device)
                trace1 = trace1.to(device)
                frame2 = frame2.to(device)
                trace2 = trace2.to(device)
                
                y_frame1 = model(frame1)["out"]                
                loss_tmp1 = torch.nn.functional.binary_cross_entropy_with_logits(y_frame1[:, 0, :, :], trace1, reduction="sum")
                y_frame2 = model(frame2)["out"]                
                loss_tmp2 = torch.nn.functional.binary_cross_entropy_with_logits(y_frame2[:, 0, :, :], trace2, reduction="sum")
                
                small_inter += np.logical_and(y_frame1[:, 0, :, :].detach().cpu().numpy() > 0., trace1[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_frame1[:, 0, :, :].detach().cpu().numpy() > 0., trace1[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_frame1[:, 0, :, :].detach().cpu().numpy() > 0., trace1[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_frame1[:, 0, :, :].detach().cpu().numpy() > 0., trace1[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

            
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_frame2[:, 0, :, :].detach().cpu().numpy() > 0., trace2[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_frame2[:, 0, :, :].detach().cpu().numpy() > 0., trace2[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_frame2[:, 0, :, :].detach().cpu().numpy() > 0., trace2[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_frame2[:, 0, :, :].detach().cpu().numpy() > 0., trace2[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                
                # Take gradient step if training                    
                loss = (loss_tmp1 + loss_tmp2) / 2
                                
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += trace1.size(0)

                # Show info on process bar                
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (total / n / 448 / 448,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i


if __name__ == '__main__':
    run()
    