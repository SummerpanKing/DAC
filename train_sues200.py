import os
import time
import math
import shutil
import sys
import torch
import argparse

from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

from sample4geo.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate

from sample4geo.loss.loss import InfoNCE
from sample4geo.loss.triplet_loss import TripletLoss, Tripletloss
from sample4geo.loss.blocks_infoNCE import blocks_InfoNCE
from sample4geo.loss.DSA_loss import DSA_loss

from sample4geo.model import TimmModel
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train and Test on SUES-200 dataset')

        # Added for your modification
        parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384', type=str, help='backbone model')
        parser.add_argument('--handcraft_model', default=True, type=bool, help='use modified backbone')
        parser.add_argument('--img_size', default=384, type=int, help='input image size')
        parser.add_argument('--views', default=2, type=int, help='only supports 2 branches retrieval')
        parser.add_argument('--record', default=True, type=bool, help='use tensorboard to record training procedure')

        parser.add_argument('--only_test', default=False, type=bool, help='use pretrained model to test')
        parser.add_argument('--ckpt_path',
                            default='checkpoints/sues-200/convnext_base.fb_in22k_ft_in1k_384/0704112405/weights_e1_0.9690.pth',
                            type=str, help='path to pretrained checkpoint file')

        # Model Config
        parser.add_argument('--nclasses', default=200, type=int, help='sues-200数据集的场景类别数')
        parser.add_argument('--block', default=2, type=int)
        parser.add_argument('--triplet_loss', default=0.3, type=float)
        parser.add_argument('--resnet', default=False, type=bool)

        # Our tricks
        parser.add_argument('--weight_infonce', default=1.0, type=float)
        parser.add_argument('--weight_cls', default=0.1, type=float)
        parser.add_argument('--weight_dsa', default=0.6, type=float)

        # Training Config
        parser.add_argument('--mixed_precision', default=True, type=bool)
        parser.add_argument('--custom_sampling', default=True, type=bool)
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--epochs', default=1, type=int, help='1 epoch for 1652')
        parser.add_argument('--batch_size', default=24, type=int, help='remember the bs is for 2 branches')
        parser.add_argument('--verbose', default=True, type=bool)
        parser.add_argument('--gpu_ids', default=(0, 1, 2, 3), type=tuple)

        # Eval Config
        parser.add_argument('--batch_size_eval', default=128, type=int)
        parser.add_argument('--eval_every_n_epoch', default=1, type=int)
        parser.add_argument('--normalize_features', default=True, type=bool)
        parser.add_argument('--eval_gallery_n', default=-1, type=int)

        # Optimizer Config
        parser.add_argument('--clip_grad', default=100.0, type=float)
        parser.add_argument('--decay_exclue_bias', default=False, type=bool)
        parser.add_argument('--grad_checkpointing', default=False, type=bool)

        # Loss Config
        parser.add_argument('--label_smoothing', default=0.1, type=float)

        # Learning Rate Config
        parser.add_argument('--lr', default=0.001, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
        parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
        parser.add_argument('--warmup_epochs', default=0.1, type=float)
        parser.add_argument('--lr_end', default=0.0001, type=float)

        # Learning part Config
        parser.add_argument('--lr_mlp', default=None, type=float)
        parser.add_argument('--lr_decouple', default=None, type=float)

        # Dataset Config
        parser.add_argument('--dataset', default='U1652-D2S', type=str, help="'U1652-D2S' | 'U1652-S2D'")
        parser.add_argument('--altitude', default=300, type=int, help="150|200|250|300")

        parser.add_argument('--data_folder', default=r'/media/xiapanwang/主数据盘/xiapanwang/Codes/python/New_Geolocalization/0_Datasets', type=str)
        parser.add_argument('--dataset_name', default='SUES-200', type=str)

        # Augment Images Config
        parser.add_argument('--prob_flip', default=0.5, type=float, help='flipping the sat image and drone image simultaneously')

        # Savepath for model checkpoints Config
        parser.add_argument('--model_path', default='./checkpoints/sues-200', type=str)

        # Eval before training Config
        parser.add_argument('--zero_shot', default=False, type=bool)

        # Checkpoint to start from Config
        parser.add_argument('--checkpoint_start', default=None)

        # Set num_workers to 0 if on Windows Config
        parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int)

        # Train on GPU if available Config
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

        # For better performance Config
        parser.add_argument('--cudnn_benchmark', default=True, type=bool)

        # Make cudnn deterministic Config
        parser.add_argument('--cudnn_deterministic', default=False, type=bool)

        args = parser.parse_args(namespace=self)


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#
config = Configuration()

if config.dataset == 'U1652-D2S':
    config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/satellite'
    config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/drone'
    config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/query_drone'
    config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/satellite'
    config.gallery_folder_train = f'{config.data_folder}/{config.dataset_name}/Training/{config.altitude}/drone'
    config.query_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/query_satellite'
    config.gallery_folder_test = f'{config.data_folder}/{config.dataset_name}/Testing/{config.altitude}/gallery_drone'

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%m%d%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    if config.handcraft_model is not True:
        print("\nModel: {}".format(config.model))
        model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)

    else:
        from sample4geo.hand_convnext.model import make_model

        model = make_model(config)
        print("\nModel:{}".format("adjust model: handcraft convnext-base"))

    # -- print weight config infos
    print(
        f"\nweight_infonce:{config.weight_infonce}\nconfig.weight_gcc:{config.weight_cls}\nweight_dsa:{config.weight_dsa}\n")

    # print(model)

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # Train
    train_dataset = U1652DatasetTrain(query_folder=config.query_folder_train,
                                      gallery_folder=config.gallery_folder_train,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                          mode="query",
                                          transforms=val_transforms,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                            mode="gallery",
                                            transforms=val_transforms,
                                            sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n,
                                            )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # -----------------------------------------------------------------------------#
    # Test Only                                                                    #
    # -----------------------------------------------------------------------------#
    if config.only_test:
        print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
        best_score = 0

        checkpoint = torch.load(config.ckpt_path)
        if 1:
            del checkpoint['model_1.classifier1.classifier.0.weight']
            del checkpoint['model_1.classifier1.classifier.0.bias']
            del checkpoint['model_1.classifier_mcb1.classifier.0.weight']
            del checkpoint['model_1.classifier_mcb1.classifier.0.bias']
            del checkpoint['model_1.classifier_mcb2.classifier.0.weight']
            del checkpoint['model_1.classifier_mcb2.classifier.0.bias']

        model.load_state_dict(checkpoint, strict=False)

        model = model.to(config.device)

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        sys.exit()

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#
    # 1.infoNCE
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_fn1 = InfoNCE(loss_function=loss_fn, device=config.device)

    # 2.Triplet
    loss_fn2 = TripletLoss(margin=config.triplet_loss)

    # 3.block infoNCE
    loss_fn3 = blocks_InfoNCE(loss_function=loss_fn, device=config.device)

    # 4.DSA loss infoNCE
    loss_fn4 = DSA_loss(loss_function=loss_fn, device=config.device)

    # all loss functions
    loss_functions = {"infoNCE": loss_fn1, "Triplet": loss_fn2, "blocks_infoNCE": loss_fn3, "DSA_loss": loss_fn4}

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)

    elif config.lr_mlp is not None:
        model_params = []
        mlp_params = []
        for name, param in model.named_parameters():
            if 'back_mlp' in name:  # 根据参数名中是否包含 'mlp' 区分模型和 MLP 层的参数
                mlp_params.append(param)
            else:
                model_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': model_params, 'lr': config.lr},
            {'params': mlp_params, 'lr': config.lr_mlp}
        ])

    elif config.lr_decouple is not None:
        model_params = []
        logit_scale = []
        for name, param in model.named_parameters():
            if 'logit_scale' in name:
                logit_scale.append(param)
            else:
                model_params.append(param)

        optimizer = torch.optim.AdamW([{'params': model_params, 'lr': config.lr},
                                       {'params': logit_scale, 'lr': config.lr_decouple}])


    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # -----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    # -----------------------------------------------------------------------------#

    # print(optimizer.param_groups[0]['lr'])

    train_steps_per = len(train_dataloader)
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    else:
        scheduler = None

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    # -----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

    # -----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    # -----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    if config.record:
        writer = SummaryWriter("./record/tensorboard-train-logs.txt")
    else:
        writer = None

    start_epoch = 0
    best_score = 0

    for epoch in range(1, config.epochs + 1):

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_functions=loss_functions,
                           optimizer=optimizer,
                           epoch=epoch,
                           train_steps_per=train_steps_per,
                           tensorboard=writer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            r1_test = evaluate(config=config,
                               model=model,
                               query_loader=query_dataloader_test,
                               gallery_loader=gallery_dataloader_test,
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True)

            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    # -- close tensorboard writer
    if writer is not None:
        writer.close()

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))

    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     torch.save(model.module, '{}/complete_model.pt'.format(model_path))
    # else:
    #     torch.save(model, '{}/complete_model.pt'.format(model_path))
