import os
import torch
from torchsummary import summary
from configuration import configuration

def get_latest_model_id(path):
    list_dir = os.listdir(path)
    list_dir = list(filter(lambda x: '.h5' in x, list_dir))
    if len(list_dir) > 0:
        list_dir.sort()
        last_file = list_dir[-1]
        rfind_e = last_file.rfind('_e')
        _e = last_file[rfind_e + 2:rfind_e + 5]
        epoch = int(_e)
        return epoch
    else:
        return -1

def prepare_model(network, models_root, device, save_config=True, verbose=False) -> [torch.nn.Module, str]:
    config = configuration()
    latest_model_e = -1
    if not os.path.exists(models_root):
        os.makedirs(models_root)
    else:
        latest_model_e = get_latest_model_id(models_root)

    model_name = 'model_'+config.architecture+'_e{:03d}.h5'
    model = network(config.batch_size, in_channels=config.image_size[0], device=device).to(device=device)
    if verbose:
        model_stats = summary(model.cuda(), input_size=[config.image_size,config.image_size], verbose=2)

    if save_config:
        with open(os.path.join(models_root,f'model_config.txt'),'w') as file:
            file.write(str(model))
    if latest_model_e >= 0:
        model_path = os.path.join(models_root, model_name.format(latest_model_e))
        model.load_state_dict(torch.load(model_path))
        print('Loaded model:', model_path)
    else:
        print('No latest model found.')
        latest_model_e = 0

    return model, model_name, latest_model_e

if __name__ == '__main__':
    epoch, batch = get_latest_model_id('models/Train_on4imagesLeaky_e100_b32_lr0.002/')
    print(F"Found values epoch: {epoch} batch: {batch}")