import time
import copy
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import wandb

opt = TrainOptions().parse()
opt.lambda_coord_ori = opt.lambda_coord
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

opt_test = copy.deepcopy(opt)
opt_test.phase = 'test'
data_loader_test = CreateDataLoader(opt_test)
dataset_test = data_loader_test.load_data()
dataset_test_size = len(data_loader_test)

model = create_model(opt)
visualizer = Visualizer(opt)
visualizer_test = Visualizer(opt_test)
total_steps = 0
model.init_wandb()

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    if opt.coord_loss_decay:
        num = min(20, max(10, epoch))
        opt.lambda_coord = opt.lambda_coord_ori * (1 - (num-10) / 10)

    for i, data in enumerate(dataset):
        model.train()
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    if epoch % 1 == 0:
        test_iter = 0
        visualizer_test.val_error_init()
        for i, data in enumerate(dataset_test):
            iter_start_time = time.time()
            visualizer_test.reset()
            test_iter += opt_test.batchSize
            model.eval()
            model.set_input(data)
            model.validate()

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt_test.batchSize
            visualizer_test.validate_current_errors(errors, data['P1'].shape[0])


            if test_iter % opt_test.update_html_freq == 0:
                save_dict = {}
                for k, v in model.get_current_visuals().items():
                    save_dict['val_' + k] = wandb.Image(v)
                wandb.log(save_dict)


        visualizer_test.print_validate_errors(len(dataset_test))



    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
