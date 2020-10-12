import os
import numpy as np
import imageio
from collections import OrderedDict
import logging

logger = logging.getLogger(__package__)

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################

def load_data(basedir, scene, testskip=8):
    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    def dir2poses(posedir):
        poses = np.stack(
            [parse_txt(os.path.join(posedir, f)) for f in sorted(os.listdir(posedir)) if f.endswith('txt')], 0)
        poses = poses.astype(np.float32)
        return poses

    def dir2intrinsics(intrinsicdir):
        intrinsics = np.stack(
            [parse_txt(os.path.join(intrinsicdir, f)) for f in sorted(os.listdir(intrinsicdir)) if f.endswith('txt')], 0)
        intrinsics = intrinsics.astype(np.float32)
        return intrinsics

    intrinsics = dir2intrinsics('{}/{}/train/intrinsics'.format(basedir, scene))
    testintrinsics = dir2poses('{}/{}/test/intrinsics'.format(basedir, scene))
    testintrinsics = testintrinsics[::testskip]
    valintrinsics = dir2poses('{}/{}/validation/intrinsics'.format(basedir, scene))
    valintrinsics = valintrinsics[::testskip]

    print(intrinsics.shape, testintrinsics.shape, valintrinsics.shape)

    poses = dir2poses('{}/{}/train/pose'.format(basedir, scene))
    testposes = dir2poses('{}/{}/test/pose'.format(basedir, scene))
    testposes = testposes[::testskip]
    valposes = dir2poses('{}/{}/validation/pose'.format(basedir, scene))
    valposes = valposes[::testskip]

    print(poses.shape, testposes.shape, valposes.shape)

    imgd = '{}/{}/train/rgb'.format(basedir, scene)
    imgfiles = ['{}/{}'.format(imgd, f)
                for f in sorted(os.listdir(imgd)) if f.endswith('png') or f.endswith('jpg')]
    imgs = [imageio.imread(f).astype(np.float32)[..., :3] / 255. for f in imgfiles]

    maskd = '{}/{}/train/mask'.format(basedir, scene)
    if os.path.isdir(maskd):
        logger.info('Loading mask from: {}'.format(maskd))
        maskfiles = ['{}/{}'.format(maskd, f)
                    for f in sorted(os.listdir(maskd)) if f.endswith('png') or f.endswith('jpg')]
        masks = [imageio.imread(f).astype(np.float32) / 255. for f in maskfiles]
    else:
        masks = [None for im in imgs]

    # load min_depth map
    min_depthd = '{}/{}/train/min_depth'.format(basedir, scene)
    if os.path.isdir(min_depthd):
        logger.info('Loading min_depth from: {}'.format(min_depthd))
        max_depth = float(open('{}/{}/train/max_depth.txt'.format(basedir, scene)).readline().strip())
        min_depthfiles = ['{}/{}'.format(min_depthd, f)
                           for f in sorted(os.listdir(min_depthd)) if f.endswith('png') or f.endswith('jpg')]
        min_depths = [imageio.imread(f).astype(np.float32) / 255. * max_depth + 1e-4 for f in min_depthfiles]
    else:
        min_depths = [None for im in imgs]

    testimgd = '{}/{}/test/rgb'.format(basedir, scene)
    testimgfiles = ['{}/{}'.format(testimgd, f)
                    for f in sorted(os.listdir(testimgd)) if f.endswith('png') or f.endswith('jpg')]
    testimgs = [imageio.imread(f).astype(np.float32)[..., :3] / 255. for f in testimgfiles]
    testimgfiles = testimgfiles[::testskip]
    testimgs = testimgs[::testskip]

    testmaskd = '{}/{}/test/mask'.format(basedir, scene)
    if os.path.isdir(testmaskd):
        logger.info('Loading mask from: {}'.format(testmaskd))
        testmaskfiles = ['{}/{}'.format(testmaskd, f)
                    for f in sorted(os.listdir(testmaskd)) if f.endswith('png') or f.endswith('jpg')]
        testmasks = [imageio.imread(f).astype(np.float32) / 255. for f in testmaskfiles]
    else:
        testmasks = [None for im in testimgs]

    # load min_depth map
    min_depthd = '{}/{}/test/min_depth'.format(basedir, scene)
    if os.path.isdir(min_depthd):
        logger.info('Loading min_depth from: {}'.format(min_depthd))
        max_depth = float(open('{}/{}/test/max_depth.txt'.format(basedir, scene)).readline().strip())
        min_depthfiles = ['{}/{}'.format(min_depthd, f)
                           for f in sorted(os.listdir(min_depthd)) if f.endswith('png') or f.endswith('jpg')]
        test_min_depths = [imageio.imread(f).astype(np.float32) / 255. * max_depth + 1e-4 for f in min_depthfiles]
    else:
        test_min_depths = [None for im in testimgs]

    valimgd = '{}/{}/validation/rgb'.format(basedir, scene)
    valimgfiles = ['{}/{}'.format(valimgd, f)
                   for f in sorted(os.listdir(valimgd)) if f.endswith('png') or f.endswith('jpg')]
    valimgs = [imageio.imread(f).astype(np.float32)[..., :3] / 255. for f in valimgfiles]
    valimgfiles = valimgfiles[::testskip]
    valimgs = valimgs[::testskip]

    valmaskd = '{}/{}/validation/mask'.format(basedir, scene)
    if os.path.isdir(valmaskd):
        logger.info('Loading mask from: {}'.format(valmaskd))
        valmaskfiles = ['{}/{}'.format(valmaskd, f)
                        for f in sorted(os.listdir(valmaskd)) if f.endswith('png') or f.endswith('jpg')]
        valmasks = [imageio.imread(f).astype(np.float32) / 255. for f in valmaskfiles]
    else:
        valmasks = [None for im in valimgs]

    # load min_depth map
    min_depthd = '{}/{}/validation/min_depth'.format(basedir, scene)
    if os.path.isdir(min_depthd):
        logger.info('Loading min_depth from: {}'.format(min_depthd))
        max_depth = float(open('{}/{}/validation/max_depth.txt'.format(basedir, scene)).readline().strip())
        min_depthfiles = ['{}/{}'.format(min_depthd, f)
                           for f in sorted(os.listdir(min_depthd)) if f.endswith('png') or f.endswith('jpg')]
        val_min_depths = [imageio.imread(f).astype(np.float32) / 255. * max_depth + 1e-4 for f in min_depthfiles]
    else:
        val_min_depths = [None for im in valimgs]

    # data format for training/testing
    print(len(imgs), len(testimgs), len(valimgs))
    all_imgs = imgs + valimgs + testimgs
    all_masks = masks + valmasks + testmasks
    all_min_depths = min_depths + val_min_depths + test_min_depths
    all_paths = imgfiles + valimgfiles + testimgfiles

    counts = [0] + [len(x) for x in [imgs, valimgs, testimgs]]
    counts = np.cumsum(counts)
    i_split = [list(np.arange(counts[i], counts[i+1])) for i in range(3)]

    intrinsics = np.concatenate([intrinsics, valintrinsics, testintrinsics], 0)
    poses = np.concatenate([poses, valposes, testposes], 0)
    img_sizes = np.stack([np.array(x.shape[:2]) for x in all_imgs], axis=0)       # [H, W]
    cnt = len(all_imgs)
    all_cams = np.concatenate((img_sizes.astype(dtype=np.float32), intrinsics.reshape((cnt, -1)), poses.reshape((cnt, -1))), axis=1)

    if os.path.isdir('{}/{}/camera_path/intrinsics'.format(basedir, scene)):
        camera_path_intrinsics = dir2poses('{}/{}/camera_path/intrinsics'.format(basedir, scene))
        camera_path_poses = dir2poses('{}/{}/camera_path/pose'.format(basedir, scene))
        # assume centered principal points
        # img_sizes = np.stack((camera_path_intrinsics[:, 1, 2]*2, camera_path_intrinsics[:, 0, 2]*2), axis=1)    # [H, W]
        # img_sizes = np.int32(img_sizes)

        H = all_cams[0, 0]
        W = all_cams[0, 1]
        img_sizes = np.stack((np.ones_like(camera_path_intrinsics[:, 1, 2])*H, np.ones_like(camera_path_intrinsics[:, 0, 2])*W), axis=1)    # [H, W]

        cnt = len(camera_path_intrinsics)
        render_cams = np.concatenate(
            (img_sizes.astype(dtype=np.float32), camera_path_intrinsics.reshape((cnt, -1)), camera_path_poses.reshape((cnt, -1))),
            axis=1)
    else:
        render_cams = None

    print(all_cams.shape)

    data = OrderedDict([('images', all_imgs),
                        ('masks', all_masks), 
                        ('paths', all_paths),
                        ('min_depths', all_min_depths),
                        ('cameras', all_cams),
                        ('i_train', i_split[0]),
                        ('i_val', i_split[1]),
                        ('i_test', i_split[2]),
                        ('render_cams', render_cams)])

    logger.info('Data statistics:')
    logger.info('\t # of training views: {}'.format(len(data['i_train'])))
    logger.info('\t # of validation views: {}'.format(len(data['i_val'])))
    logger.info('\t # of test views: {}'.format(len(data['i_test'])))
    if data['render_cams'] is not None:
        logger.info('\t # of render cameras: {}'.format(len(data['render_cams'])))

    return data
