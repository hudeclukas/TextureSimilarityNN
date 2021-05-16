from configuration import configuration
import torch.cuda
import wandb
from model_factory import prepare_model
from trainer import trainer_standard

from architectures import SiameseNetworkIWSSIP, SiameseSimple, distance_euclid, distance_canberra


if __name__ == '__main__':
    config = configuration('config.json')
    config.add('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {config.device} device')

    wandb.login(key=config.wandb_key)

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

    models_root = f'models/Euclidean/Train_on{config.max_samples}x{config.max_images}imagesChanging{int(config.other_images)}Leaky_e{config.epochs}_b{config.batch_size}_lr{config.learning_rate}_o{config.out_channels}'
    model, model_name, latest_epoch = prepare_model(SiameseSimple, models_root, config.device)
    trainer_standard(model, model_name, distance_euclid,'data', models_root, config.device, latest_epoch)

    wandb.finish()