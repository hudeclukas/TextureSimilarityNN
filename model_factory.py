import os
import torch
from torchsummary import summary
import wandb


def get_latest_model_id(path):
    list_dir = os.listdir(path)
    list_dir = list(filter(lambda x: '.h5' in x, list_dir))
    if len(list_dir) > 0:
        list_dir.sort()
        last_file = list_dir[-1]
        last_ = last_file[0:last_file.rfind('_')-2]
        last_last_ = last_.rfind('_')
        epoch = int(last_[last_last_+1:])
        batch = int(last_file[last_file.rfind('_')+1:last_file.rfind('.h5')])
        return epoch, batch
    else:
        return -1, -1

def prepare_model(network, models_root, device):
    config = wandb.config
    latest_model_e = 0
    latest_model_b = 0
    if not os.path.exists(models_root):
        os.makedirs(models_root)
    else:
        latest_model_e,latest_model_b = get_latest_model_id(models_root)

    model_name = 'model_'+config.architecture+'-e_{:03d}-b_{:03d}.h5'
    model = network(config.batch_size, device=device).to(device=device)
    model_stats = summary(model.cuda(), input_size=[config.image_size,config.image_size], verbose=2)

    with open(os.path.join(models_root,'model_config.txt'),'w') as file:
        file.write(str(model))
    if latest_model_e >= 0:
        model.load_state_dict(torch.load(os.path.join(models_root,model_name.format(latest_model_e, latest_model_b))))

    return model, model_name