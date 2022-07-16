import sys
import os

sys.path.append(os.getcwd())

from model.layerwise_model import ConvertibleModel
from frameworks.distillation.MemSR import load_model
import utils.backend as backend


def prepare_params(params):
    from utils.tools import parse_params
    params = parse_params(params)
    default_keys = {
        'metric': 'psnr255',
        'inference_statics': False,
        'skip_train': False,
        'save_model': False,
        'test_benchmark': False,
    }
    params = {**default_keys, **params}
    return params


def get_params():
    from pytorch_lightning import seed_everything
    params = backend.load_parameters()
    seed_everything(params['seed'])
    backend.log_params(params)
    return params

def train_model(model, params, save_name='default', checkpoint_monitor=None, mode='max'):
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from utils.tools import get_trainer_params

    if checkpoint_monitor is None:
        checkpoint_monitor = 'validation/' + params['metric']

    logger = TensorBoardLogger("logs", name=save_name, default_hp_metric=False)
    backend.set_tensorboard_logdir(f'logs/{save_name}')

    checkpoint_callback = ModelCheckpoint(dirpath='saves', save_top_k=1, monitor=checkpoint_monitor, mode=mode)
    t_params = get_trainer_params(params)
    trainer = Trainer(logger=logger, callbacks=[checkpoint_callback], **t_params)
    trainer.fit(model)
    model.eval()

    if checkpoint_callback.best_model_path != "" and trainer.is_global_zero:
        import numpy as np
        if params['save_model']:
            backend.save_artifact(checkpoint_callback.best_model_path, key='best_model_checkpoint')
        log_val = checkpoint_callback.best_model_score.item()
        backend.log_metric(checkpoint_monitor.split('/')[-1], float(np.clip(log_val, -1e10, 1e10)))
    else:
        backend.log("Best_model_path not found!")

    backend.log("Training finished")
    return model


def inference_statics(model, x_test=None, batch_size=None, averaged=True):
    import time
    import torch

    if x_test is None:
        x_test = model.val_dataloader().dataset[0][0]
    print('x_test size ', x_test.shape)
    if batch_size is None:
        batch_size = model.val_dataloader().batch_size
    x = torch.stack([x_test] * batch_size, dim=0).cuda()
    model.cuda().eval()
    total_input_size = x.nelement()
    with torch.no_grad():
        for i in range(10):
            outs = model(x)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        start_time = time.time()
        for i in range(100):
            outs = model(torch.randn_like(x))
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        used_memory = torch.cuda.max_memory_allocated()
        if averaged:
            backend.log_metric('Inference_Time(us)',
                               float(total_time / 100 / total_input_size * 1e6))  # time usage per pixel per batch
            backend.log_metric('Memory(KB)',
                               float(used_memory / total_input_size / 1024))  # memory usage per pixel per batch
        else:
            backend.log_metric('Inference_Time(ms)', float(total_time / 100 * 1e3))  # time usage
            backend.log_metric('Memory(MB)', float(used_memory / 1024 / 1024))  # memory usage

    from thop import profile
    x = torch.stack([x_test], dim=0).cuda()
    flops, param_number = profile(model, inputs=(x,), verbose=False)
    if averaged:
        backend.log_metric('flops(K per pixel)', float(flops / x.nelement() / 1000))
    else:
        backend.log_metric('flops(G)', float(flops / 1e9))
    backend.log_metric('parameters(KB)', float(param_number / 1024))


def test_SR_benchmark(test_model):
    from pytorch_lightning import Trainer
    trainer = Trainer(gpus=1)
    from datasets import DataProvider
    benchmarks = ['Set5', 'Set14', 'B100', 'Urban100']
    for d in benchmarks:
        dataset_params = {
            'name': d,
            'test_only': True,
            'patch_size': test_model.params['dataset']['patch_size'],
            'ext': 'sep',
            'scale': test_model.params['scale'],
            "batch_size": 1,
        }
        provider = DataProvider(dataset_params)
        ret = trainer.test(test_dataloaders=provider.test_dl, model=test_model)
        backend.log_metric(d + '_' + test_model.params['metric'], ret[0]['test/' + test_model.params['metric']])


if __name__ == "__main__":
    params = get_params()
    params = prepare_params(params)
    backend.log(params)

    model = load_model(params)

    if not params['skip_train']:
        model = train_model(model, params, save_name="model_distillation")

    model.plain_model = ConvertibleModel.from_convertible_models(model.plain_model).generate_inference_model()
    model.teacher_plain_model = None
    model.teacher = None
    model.dist_method = None
    model.bridges = None

    if params['test_benchmark']:
        test_SR_benchmark(model)

    if 'test_ssim' in params and params['test_ssim']:
        model.params['metric'] = model.params['metric'].lower().replace('psnr', 'ssim')
        model.metric = model.choose_metric()
        test_SR_benchmark(model)

    if params['inference_statics']:
        inference_statics(model, batch_size=1)
