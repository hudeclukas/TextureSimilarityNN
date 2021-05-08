import torch.cuda
import wandb
from model_factory import prepare_model
from trainer_euclidean import trainer_euclidean_standard

from network import SiameseNetwork

if __name__ == '__main__':

    wandb.login(key='584287ef3bc4e3311465171c04c9525858e97893')

    # 2. Save model inputs and hyperparameters
    # Start a new run, tracking hyperparameters in config
    wandb.init(project="texsim2018", entity='shamann', group='FirstTraining', config={
        "image_size":(1,150,150),
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 32,
        "epochs": 100,
        "architecture": "SiameseNetwork",
        "dataset": "SimTex2018",
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device.format(device)} device')

    models_root = 'models/FirstTraining_e100_b32_lr001'
    model, model_name = prepare_model(SiameseNetwork, models_root, device)
    trainer_euclidean_standard(model, model_name, 'data', models_root, device)

    wandb.finish()