job_info = {'params': {}, 'results': {}, 'tensorboard_path': '', 'artifacts': {}}
name = 'local_backend'


def log_metric(key, value):
    job_info['results'][key] = value
    log("-------------->", key, "=", value, "<-------------")


def log_param(key, value):
    log(key + ": ", value)


def load_parameters(log_parameters=True):
    import pickle
    with open('local_job_parameters.pkl', 'rb') as f:
        info = pickle.load(f)
    params = info['params']
    log_params(params)
    return params


def log_params(parameters):
    job_info['params'] = parameters
    for key, value in parameters.items():
        log_param(key, value)


def set_tensorboard_logdir(path):
    job_info['tensorboard_path'] = path


def save_artifact(filepath: str, key=None):
    import random
    if key is None:  # might have some bugs when conflicts
        key = str(random.randint(0, 9999)) + "_" + filepath.split('/')[-1].split('.')[0]
    job_info['artifacts'][key] = filepath
    log('artifacts:>>>>>>', key, '>>>>>>>>', filepath)


submit_dict = {
    'job_directory': None,
    'project_name': None,
    'params': {},
    'num_gpus': 0,
    'command': '',
}


def submit(**params):
    info = {**submit_dict, **params}
    import pickle
    import os
    with open(os.path.join(info['job_directory'], 'local_job_parameters.pkl'), 'wb') as f:
        pickle.dump(info, f)
    command = "python " + '-W ignore ' + info['command']
    print(command)
    os.system(command)


def log(*info):
    print(*info)
