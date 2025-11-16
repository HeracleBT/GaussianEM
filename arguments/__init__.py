#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
import os.path as osp
from argparse import ArgumentParser, Namespace
sys.path.append("./")


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group



class ModelParams_EM(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.scale_min = 0.001  # percent of volume size
        self.scale_max = 0.005  # percent of volume size
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams_EM(ParamGroup):
    def __init__(self, parser):
        self.epoch = 10
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_max_steps = 1310_000
        self.density_lr_init = 0.001
        self.density_lr_final = 0.00001
        self.density_lr_max_steps = 1310_000
        self.scaling_lr_init = 0.0001
        self.scaling_lr_final = 0.00001
        self.scaling_lr_max_steps = 1310_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.00001
        self.rotation_lr_max_steps = 1310_000
        self.lambda_dssim = 0.25
        self.lambda_tv = 0.05
        self.tv_vol_size = 16
        self.lambda_scaling_norm = 0.0
        self.lambda_volume_loss = 0.0
        self.lambda_distance_loss = 0.0
        self.density_min_threshold = 0.00001
        self.densification_interval = 1000
        self.densify_from_iter = 500
        self.densify_until_iter = 1310_000
        self.contribution_prune_ratio = 0.05
        self.densify_grad_threshold = 1.0e-5
        self.densify_scale_threshold = 0.001 # percent of volume size
        self.max_screen_size = None
        self.max_scale = 0.002 # percent of volume size
        self.max_num_gaussians = 400_000

        ## heter param
        self.lr = 1e-4
        self.wd = 0
        super().__init__(parser, "Optimization Parameters")



def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = osp.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
