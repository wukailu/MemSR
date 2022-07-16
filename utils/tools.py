import numpy
import numpy as np


def all_list_to_tuple(my_dict):
    if isinstance(my_dict, dict):
        return {key: all_list_to_tuple(my_dict[key]) for key in my_dict}
    elif isinstance(my_dict, list) or isinstance(my_dict, tuple):
        return tuple(all_list_to_tuple(v) for v in my_dict)
    else:
        return my_dict


def parse_params(params: dict):
    # Process trainer
    defaults = {
        'precision': 32,
        'deterministic': True,
        'benchmark': True,
        'gpus': 1,
        'num_epochs': 1,
        "progress_bar_refresh_rate": 100,
        'auto_select_gpus': False,
    }
    params = {**defaults, **params}
    if "backend" not in params:
        if params["gpus"] == 1:
            params["backend"] = None
        else:
            params["backend"] = "ddp"
            # from pytorch_lightning.plugins import DDPPlugin
            # params["plugins"] = DDPPlugin(find_unused_parameters=False)

    # Process backbone
    backbone_list = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18',
                     'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121',
                     'densenet161', 'densenet169', 'mobilenet_v2', 'googlenet', 'inception_v3',
                     'Rep_ResNet50', 'resnet20']

    if 'backbone' in params and isinstance(params['backbone'], int):
        params['backbone'] = backbone_list[params['backbone']]

    # Process dataset
    if isinstance(params['dataset'], str):
        params['dataset'] = {'name': params['dataset']}
    default_dataset_params = {
        'workers': 4,
    }
    params['dataset'] = {**default_dataset_params, **params['dataset']}
    if 'total_batch_size' in params['dataset'] and 'batch_size' not in params['dataset']:
        params['dataset']["batch_size"] = params['dataset']["total_batch_size"] // params["gpus"]
    if 'total_batch_size' not in params['dataset'] and 'batch_size' in params['dataset']:
        params['dataset']["total_batch_size"] = params['dataset']["batch_size"] * params["gpus"]

    # Process Training Settings
    optimizer_list = ['SGD', 'Adam']
    scheduler_list = ['ExpLR', 'CosLR', 'StepLR', 'OneCycLR', 'MultiStepLR', 'MultiStepLR_CRD']
    if 'optimizer' in params and isinstance(params['optimizer'], int):
        params['optimizer'] = optimizer_list[params['optimizer']]
    if 'lr_scheduler' in params and isinstance(params['lr_scheduler'], int):
        params['lr_scheduler'] = scheduler_list[params['lr_scheduler']]

    equivalent_keys = [('learning_rate', 'lr', 'max_lr')]
    for groups in equivalent_keys:
        for key in groups:
            if key in params:
                val = params[key]
                for key2 in groups:
                    params[key2] = val
                break

    return params


def get_trainer_params(params) -> dict:
    name_mapping = {
        "gpus": "gpus",
        "backend": "accelerator",
        "plugins": "plugins",
        "accumulate": "accumulate_grad_batches",
        "auto_scale_batch_size": "auto_scale_batch_size",
        "auto_select_gpus": "auto_select_gpus",
        "num_epochs": "max_epochs",
        "benchmark": "benchmark",
        "deterministic": "deterministic",
        "progress_bar_refresh_rate": "progress_bar_refresh_rate",
        "gradient_clip_val": "gradient_clip_val",
        "track_grad_norm": "track_grad_norm",
    }
    ret = {}
    for key in params:
        if key in name_mapping:
            ret[name_mapping[key]] = params[key]

    if ret['gpus'] != 0 and isinstance(ret['gpus'], int):
        ret['gpus'] = find_best_gpus(ret['gpus'])
        print('using gpu ', ret['gpus'])
    return ret


def submit_jobs(param_generator, command: str, number_jobs=1, project_name=None, job_directory='.',
                global_seed=23336666, ignore_exist=False):
    import time
    time.sleep(0.5)
    numpy.random.seed(global_seed)
    submitted_jobs = [{}]
    for idx in range(number_jobs):
        while True:
            hyper_params = param_generator()
            if hyper_params not in submitted_jobs:
                break
        submitted_jobs.append(hyper_params.copy())

        if 'seed' not in hyper_params:
            hyper_params['seed'] = int(2018011328)
        if 'gpus' not in hyper_params:
            hyper_params['gpus'] = 1

        name = project_name if 'project_name' not in hyper_params else hyper_params['project_name']
        print(f"Task {idx}, {hyper_params}")
        import utils.backend as backend
        backend.submit(scheduler_config='scheduler', job_directory=job_directory, command=command, params=hyper_params,
                       stream_job_logs=False, num_gpus=hyper_params["gpus"], project_name=name)



def random_params(val):
    """
        use [x, y, z, ...] as the value of dict to use random select in the list.
        use (x, y, z, ...) to avoid random select or add '_no_choice' suffix to the key to avoid random for a list
        the function will recursively find [x,y,z,...] and select one element to replace it.
        :param params: dict for params
        :return: params after random choice
    """
    if isinstance(val, list):
        idx = np.random.randint(len(val))  # np.random.choice can't random rows
        ret = random_params(val[idx])
    elif isinstance(val, tuple):
        ret = tuple([random_params(i) for i in val])
    elif isinstance(val, dict):
        ret = {}
        for key, values in val.items():
            if isinstance(values, list) and key.endswith("_no_choice"):
                ret[key[:-10]] = values  # please use tuple to avoid be random selected
            else:
                ret[key] = random_params(values)
    elif isinstance(val, np.int64):
        ret = int(val)
    elif isinstance(val, np.float64):
        ret = float(val)
    else:
        ret = val
    return ret


def tuples_to_lists(val):
    if isinstance(val, list):
        ret = [tuples_to_lists(v) for v in val]
    elif isinstance(val, tuple):
        ret = [tuples_to_lists(i) for i in val]
    elif isinstance(val, dict):
        ret = {}
        for key, values in val.items():
            ret[key] = tuples_to_lists(values)
    elif isinstance(val, np.int64):
        ret = int(val)
    elif isinstance(val, np.float64):
        ret = float(val)
    else:
        ret = val
    return ret


def lists_to_tuples(val):
    if isinstance(val, list):
        ret = tuple([lists_to_tuples(v) for v in val])
    elif isinstance(val, tuple):
        ret = tuple([lists_to_tuples(i) for i in val])
    elif isinstance(val, dict):
        ret = {}
        for key, values in val.items():
            ret[key] = lists_to_tuples(values)
    elif isinstance(val, np.int64):
        ret = int(val)
    elif isinstance(val, np.float64):
        ret = float(val)
    else:
        ret = val
    return ret


def find_best_gpus(num_gpu_needs=1):
    import subprocess as sp
    gpu_ids = []
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [(int(x.split()[0]), i) for i, x in enumerate(memory_free_info) if i not in gpu_ids]
    print('memories left ', memory_free_values)
    memory_free_values = sorted(memory_free_values)[::-1]
    gpu_ids = [k for m, k in memory_free_values[:num_gpu_needs]]
    return gpu_ids