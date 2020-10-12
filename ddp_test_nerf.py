import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
from collections import OrderedDict
from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_run_nerf import config_parser, setup_logger, setup, cleanup, render_single_image
import logging


logger = logging.getLogger(__package__)


def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]
    for m in range(models['cascade_level']):
        net = NerfNet(args).to(rank)
        net = DDP(net, device_ids=[rank], output_device=rank)
        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = net
        models['optim_{}'.format(m)] = optim

    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]
    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])
    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        start = path2iter(fpath)
        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)
        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
                models[name].load_state_dict(to_load[name])
                models[name].load_state_dict(to_load[name])

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth)
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size)
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Rendered {} in {} seconds'.format(fname, dt))

                # only save last level
                im = ret[-1]['rgb'].numpy()
                # compute psnr if ground-truth is available
                if ray_samplers[idx].img_path is not None:
                    gt_im = ray_samplers[idx].get_img()
                    psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                    logger.info('{}: psnr={}'.format(fname, psnr))

                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, fname), im)

                im = ret[-1]['fg_rgb'].numpy()
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)

                im = ret[-1]['bg_rgb'].numpy()
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)

                im = ret[-1]['fg_depth'].numpy()
                im = colorize_np(im, cmap_name='jet', append_cbar=True)
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)

                im = ret[-1]['bg_depth'].numpy()
                im = colorize_np(im, cmap_name='jet', append_cbar=True)
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

            torch.cuda.empty_cache()

    # clean up for multi-processing
    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_test_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()


