import torch.cuda
import wandb
from model_factory import prepare_model
from trainer_euclidean import trainer_euclidean_standard

from architectures import SiameseNetworkIWSSIP, SiameseSimple

if __name__ == '__main__':

    wandb.login(key='584287ef3bc4e3311465171c04c9525858e97893')

    # 2. Save model inputs and hyperparameters
    # Start a new run, tracking hyperparameters in config
    wandb.init(project="texsim2018", entity='shamann', group='FirstTraining', config={
        "image_size":(1,150,150),
        "learning_rate": 0.002,
        "margin":2,
        "dropout": 0.2,
        "batch_size": 32,
        "epochs": 50,
        "patience":5,
        "max_images": 16,
        "max_samples": 100,
        "architecture": "SiameseNetworkLeaky_tiny",
        "dataset": "SimTex2018",
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device.format(device)} device')

    models_root = f'models/Train_on{wandb.config.max_samples}x{wandb.config.max_images}imagesLeaky_e{wandb.config.epochs}_b{wandb.config.batch_size}_lr{wandb.config.learning_rate}'
    model, model_name, latest_epoch = prepare_model(SiameseSimple, models_root, device)
    trainer_euclidean_standard(model, model_name, 'data', models_root, device, latest_epoch)

    wandb.finish()