import torch.cuda
import wandb
import argparse
import os

from configuration import configuration
from model_factory import prepare_model
from trainer import trainer_standard
from architectures import SiameseNetworkIWSSIP, SiameseSimple, distance_euclid, distance_canberra


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', type=str, help='wandb key')
    parser.add_argument('--data_path', type=str, help='path to data')
    args = parser.parse_args()

    config = configuration('config.json')
    config.add('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {config.device} device')

    wandb.login(key=args.wandb)

    # 2. Save model inputs and hyperparameters
    # Start a new run, tracking hyperparameters in config
    wandb.init(project="texsim2018", entity='shamann', group='SmallNet', config={
        "image_size":config.image_size,
        "learning_rate": config.learning_rate,
        "margin":config.margin,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "other_images": config.other_images,
        "max_images": config.max_images,
        "max_samples": config.max_samples,
        "patience":config.patience,
        "out_channels":config.out_channels,
        "architecture": config.architecture,
        "dataset": config.dataset,
    })

    try:
        data = args.data_path
    except:
        data = config.data_path
    finally:
        print("===== DATA =====")
        print("DATA PATH: " + args.data_path)
        print("LIST FILES IN DATA PATH...")
        print(os.listdir(args.data_path))
        print("================")

    models_root = f'{config.models_path}/{config.distance}/Train_on{config.max_samples}x{config.max_images}imagesChanging{int(config.other_images)}Leaky_m{config.margin}_e{config.epochs}_b{config.batch_size}_lr{config.learning_rate}_o{config.out_channels}'
    os.makedirs(models_root)
    print(os.listdir(os.path.dirname(models_root)))
    print(os.path.abspath(models_root))
    model, model_name, latest_epoch = prepare_model(SiameseSimple, models_root, config.device)
    if config.distance == 'Euclidean':
        trainer_standard(model, model_name, distance_euclid, data, models_root, config.device, latest_epoch)
    elif config.distance == 'Canberra':
        trainer_standard(model, model_name, distance_canberra, data, models_root, config.device, latest_epoch)

    wandb.finish()