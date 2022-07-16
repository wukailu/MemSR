import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import submit_jobs, random_params

pretrain_paths = {
    "teacher_x4": "path_to_teacher",
}

templates = {
    'DIV2Kx4-EXP': {
        'task': 'super-resolution',
        'loss': 'L1',
        'gpus': 1,
        'teacher_pretrain_path': "to be filled",
        'max_lr': 2e-4,
        'weight_decay': 0,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 1,
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'scale': 4,
            'total_batch_size': 16,
            'patch_size': 192,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": 233,
        'save_model': True,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x4',
    },
}


def params_for_EXP_main_x4():
    params = {
        'project_name': 'CVPR_EXP_MAIN_x4',
        'method': 'MemSR_Init',
        'fix_r': 64,
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths['teacher_x4'],
        'layer_type': 'normal_no_bn',
        'ridge_alpha': 0,
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': 'MSE',
        },
        'seed': 233,
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def test_model():
    scale = 4
    seed = 233
    params = {
        'project_name': 'model_test',
        'save_model': False,
        'skip_train': True,
        'test_benchmark': True,
        'inference_statics': True,
        'test_ssim': True,
        'load_from': [f'path_to_ckpt'],
        'width': 0,
        'seed': seed,
    }

    return {**templates[f'DIV2Kx{scale}-EXP'], **params}


def params_for_MemSR():
    params = params_for_EXP_main_x4()

    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_MemSR, 'frameworks/distillation/train_model.py', number_jobs=100, job_directory='.')
