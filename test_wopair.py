import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import torch
from torchvision.utils import save_image
import time

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 2  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# create website
save_dir = os.path.join(opt.results_dir, opt.name)
os.makedirs(os.path.join(save_dir, 'uv_xy_inpainted'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'uv_texture_inpainted'), exist_ok=True)

print(opt.how_many)
print(len(dataset))

model = model.eval()
print(model.training)

opt.how_many = len(dataset) #999999
# test
for i, data in enumerate(dataset):
    print(' process %d/%d img ..' % (i * opt.batchSize, opt.how_many))
    startTime = time.time()
    visuals = model.test_wopair(data)


    for j in range(visuals['uv_xy_inpainted'].shape[0]):
        img = torch.cat([(visuals['uv_xy_inpainted'][j][None].cpu() / 2 + 0.5), torch.zeros_like(visuals['uv_xy_inpainted'][j][None][:,:1], device='cpu')], dim=1)
        texture = visuals['uv_texture_inpainted'][j][None].cpu() / 2 + 0.5
        save_image(img, os.path.join(save_dir, 'uv_xy_inpainted', data['P1_path'][j].replace('.jpg', '.png')))
        save_image(texture, os.path.join(save_dir, 'uv_texture_inpainted', data['P1_path'][j]))
        print(data['P1_path'][j])

    endTime = time.time()
    print(endTime-startTime)




