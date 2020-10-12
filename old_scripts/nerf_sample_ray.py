import numpy as np
from collections import OrderedDict
import torch
import cv2


########################################################################################################################
# ray batch sampling
########################################################################################################################

def parse_camera(params):
    H, W = params[:2]
    intrinsics = params[2:18].reshape((4, 4))
    c2w = params[18:34].reshape((4, 4))

    return int(W), int(H), intrinsics.astype(np.float32), c2w.astype(np.float32)


def get_rays_single_image(H, W, intrinsics, c2w):
    '''
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    '''
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)

    rays_d = np.dot(np.linalg.inv(intrinsics[:3, :3]), pixels)
    rays_d = np.dot(c2w[:3, :3], rays_d)  # (3, H*W)
    rays_d = rays_d.transpose((1, 0))  # (H*W, 3)

    rays_o = c2w[:3, 3].reshape((1, 3))
    rays_o = np.tile(rays_o, (rays_d.shape[0], 1))  # (H*W, 3)

    depth = np.linalg.inv(c2w)[2, 3]
    depth = depth * np.ones((rays_o.shape[0],), dtype=np.float32)  # (H*W,)

    return rays_o, rays_d, depth


class RaySamplerSingleImage(object):
    def __init__(self, cam_params, img_path=None, img=None, resolution_level=1, mask=None, min_depth=None):
        super().__init__()
        self.W_orig, self.H_orig, self.intrinsics_orig, self.c2w_mat = parse_camera(cam_params)

        self.img_path = img_path
        self.img_orig = img
        self.mask_orig = mask
        self.min_depth_orig = min_depth

        self.resolution_level = -1
        self.set_resolution_level(resolution_level)

    def set_resolution_level(self, resolution_level):
        if resolution_level != self.resolution_level:
            self.resolution_level = resolution_level
            self.W = self.W_orig // resolution_level
            self.H = self.H_orig // resolution_level
            self.intrinsics = np.copy(self.intrinsics_orig)
            self.intrinsics[:2, :3] /= resolution_level
            if self.img_orig is not None:
                self.img = cv2.resize(self.img_orig, (self.W, self.H), interpolation=cv2.INTER_AREA)
                self.img = self.img.reshape((-1, 3))
            else:
                self.img = None

            if self.mask_orig is not None:
                self.mask = cv2.resize(self.mask_orig, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                self.mask = self.mask.reshape((-1))
            else:
                self.mask = None

            if self.min_depth_orig is not None:
                self.min_depth = cv2.resize(self.min_depth_orig, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.min_depth = self.min_depth.reshape((-1))
            else:
                self.min_depth = None

            self.rays_o, self.rays_d, self.depth = get_rays_single_image(self.H, self.W,
                                                                         self.intrinsics, self.c2w_mat)

    def get_all(self):
        if self.min_depth is not None:
            min_depth = self.min_depth
        else:
            min_depth = 1e-4 * np.ones_like(self.rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', self.rays_o),
            ('ray_d', self.rays_d),
            ('depth', self.depth),
            ('rgb', self.img),
            ('mask', self.mask),
            ('min_depth', min_depth)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])
        return ret

    def random_sample(self, N_rand, center_crop=False):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''
        if center_crop:
            half_H = self.H // 2
            half_W = self.W // 2
            quad_H = half_H // 2
            quad_W = half_W // 2

            # pixel coordinates
            u, v = np.meshgrid(np.arange(half_W-quad_W, half_W+quad_W),
                               np.arange(half_H-quad_H, half_H+quad_H))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = np.random.choice(u.shape[0], size=(N_rand,), replace=False)

            # Convert back to original image
            select_inds = v[select_inds] * self.W + u[select_inds]
        else:
            # Random from one image
            select_inds = np.random.choice(self.H*self.W, size=(N_rand,), replace=False)

        rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
        rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
        depth = self.depth[select_inds]         # [N_rand, ]

        if self.img is not None:
            rgb = self.img[select_inds, :]          # [N_rand, 3]
        else:
            rgb = None

        if self.mask is not None:
            mask = self.mask[select_inds]
        else:
            mask = None

        if self.min_depth is not None:
            min_depth = self.min_depth[select_inds]
        else:
            min_depth = 1e-4 * np.ones_like(rays_d[..., 0])

        ret = OrderedDict([
            ('ray_o', rays_o),
            ('ray_d', rays_d),
            ('depth', depth),
            ('rgb', rgb),
            ('mask', mask),
            ('min_depth', min_depth)
        ])
        # return torch tensors
        for k in ret:
            if ret[k] is not None:
                ret[k] = torch.from_numpy(ret[k])

        return ret

    # def random_sample_patches(self, N_patch, r_patch=16, center_crop=False):
    #     '''
    #     :param N_patch: number of patches to be sampled
    #     :param r_patch: patch size will be (2*r_patch+1)*(2*r_patch+1)
    #     :return:
    #     '''
    #     # even size patch
    #     # offsets to center pixels
    #     u, v = np.meshgrid(np.arange(-r_patch, r_patch),
    #                        np.arange(-r_patch, r_patch))
    #     u = u.reshape(-1)
    #     v = v.reshape(-1)
    #     offsets = v * self.W + u

    #     # center pixel coordinates
    #     u_min = r_patch
    #     u_max = self.W - r_patch
    #     v_min = r_patch
    #     v_max = self.H - r_patch
    #     if center_crop:
    #         u_min = self.W // 4 + r_patch
    #         u_max = self.W - self.W // 4 - r_patch
    #         v_min = self.H // 4 + r_patch
    #         v_max = self.H - self.H // 4 - r_patch

    #     u, v = np.meshgrid(np.arange(u_min, u_max, r_patch),
    #                        np.arange(v_min, v_max, r_patch))
    #     u = u.reshape(-1)
    #     v = v.reshape(-1)

    #     select_inds = np.random.choice(u.shape[0], size=(N_patch,), replace=False)
    #     # Convert back to original image
    #     select_inds = v[select_inds] * self.W + u[select_inds]

    #     # pick patches
    #     select_inds = np.stack([select_inds + shift for shift in offsets], axis=1)
    #     select_inds = select_inds.reshape(-1)

    #     rays_o = self.rays_o[select_inds, :]    # [N_rand, 3]
    #     rays_d = self.rays_d[select_inds, :]    # [N_rand, 3]
    #     depth = self.depth[select_inds]         # [N_rand, ]

    #     if self.img is not None:
    #         rgb = self.img[select_inds, :]          # [N_rand, 3]

    #         # ### debug
    #         # import imageio
    #         # imgs = rgb.reshape((N_patch, r_patch*2, r_patch*2, -1))
    #         # for kk in range(imgs.shape[0]):
    #         #     imageio.imwrite('./debug_{}.png'.format(kk), imgs[kk])
    #         # ###
    #     else:
    #         rgb = None

    #     ret = OrderedDict([
    #         ('ray_o', rays_o),
    #         ('ray_d', rays_d),
    #         ('depth', depth),
    #         ('rgb', rgb)
    #     ])

    #     # return torch tensors
    #     for k in ret:
    #         ret[k] = torch.from_numpy(ret[k])

    #     return ret
