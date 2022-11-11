import os
from subprocess import check_output

import imageio
import numpy as np
import PIL.Image

import utils.exif_helper as ehelp
from utils.exposure_helper import calculate_ev100_from_metadata
import tensorflow as tf
import tensorflow_addons as tfa
# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original

import nn_utils.math_utils as math_utils
from dataflow.nerd.load_blender import load_blender_data
from dataflow.nerd.load_nerf_blender import load_blender_data as load_nerf_blender_data

from nn_utils.nerf_layers import get_full_image_eval_grid
from utils.training_setup_utils import get_num_gpus
import dataflow.nerd as data
import utils.training_setup_utils as train_utils
from models.nerd_net import NerdModel


def add_args(parser):
    
    parser.add_argument(
        "--datadir", required=True, type=str, help="Path to dataset location.",
    )
    
    parser.add_argument(
        "--jitter_coords",
        action="store_true",
        help=(
            "Jitters the sampling coordinates. The RGB training color is "
            "then handled with a lookup. May reduce aliasing artifacts and improve "
            "opacity gradient based normal."
        ),
    )
   
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="blender",
        help="Currently only blender or real_world is available",
        choices=["blender", "real_world", "nerf_blender"],
    )
   
    parser.add_argument(
        "--testskip",
        type=int,
        default=1,
        help="will load 1/N images from test/val sets, useful for large datasets",
    )
    
    parser.add_argument(
        "--valskip",
        type=int,
        default=16,
        help="will load 1/N images from test/val sets, useful for large datasets",
    )
    
    parser.add_argument(
        "--trainskip",
        type=int,
        default=1,
        help="will load 1/N images from train sets, useful for large datasets",
    )

    parser.add_argument(
        "--spherify", action="store_true", help="Real-world dataset captured in 360."
    )
    parser.add_argument(
        "--near",
        type=float,
        help=(
            "Multiplicator of near clip distance for real-world datasets, absolute "
            "near clip distance for blender."
        ),
    )
    parser.add_argument(
        "--far",
        type=float,
        help=(
            "Multiplocator of far clip distance for real-world datasets, absolute far "
            "clip distance for blender."
        ),
    )

    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )
    parser.add_argument(
        "--rwfactor",
        type=int,
        default=8,
        help="Factor for real-world sample downscaling",
    )
    parser.add_argument(
        "--rwholdout", type=int, default=16, help="Real world holdout stride"
    )
    #"""





    
    parser.add_argument(
        "--log_step",
        type=int,
        default=100,
        help="frequency of tensorboard metric logging",
    )
    parser.add_argument(
        "--weights_epoch", type=int, default=10, help="save weights every x epochs"
    )
    parser.add_argument(
        "--validation_epoch",
        type=int,
        default=5,
        help="render validation every x epochs",
    )
    parser.add_argument(
        "--testset_epoch",
        type=int,
        default=300,
        help="render testset every x epochs",
    )
    """
    parser.add_argument(
        "--video_epoch",
        type=int,
        default=300,
        help="render video every x epochs",
    )

    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000s)",
    )

    parser.add_argument("--render_only", action="store_true")
    """
    return parser

def parse_args():
    parser = add_args(
        data.add_args(
            NerdModel.add_args(
                train_utils.setup_parser(),
            ),
        ),
    )
    return train_utils.parse_args_file_without_nones(parser)



def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        maskdir = os.path.join(basedir, "masks_{}".format(r))
        if not os.path.exists(imgdir) and not os.path.exists(maskdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        maskdir = os.path.join(basedir, "masks_{}x{}".format(r[1], r[0]))
        if not os.path.exists(imgdir) and not os.path.exists(maskdir):
            needtoload = True
    if not needtoload:
        return

    print("Minify needed...")

    imgdir = os.path.join(basedir, "images")
    maskdir = os.path.join(basedir, "masks")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    masks = [os.path.join(maskdir, f) for f in sorted(os.listdir(maskdir))]
    imgs = [
        f
        for f in imgs
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    masks = [
        f
        for f in masks
        if any([f.endswith(ex) for ex in ["JPG", "jpg", "png", "jpeg", "PNG"]])
    ]
    imgdir_orig = imgdir
    maskdir_orig = maskdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            nameImg = "images_{}".format(r)
            nameMask = "masks_{}".format(r)
            resizearg = "{}%".format(100.0 / r)
        else:
            nameImg = "images_{}x{}".format(r[1], r[0])
            nameMask = "masks_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, nameImg)
        maskdir = os.path.join(basedir, nameMask)
        if os.path.exists(imgdir) and os.path.exists(maskdir):
            continue

        print("Minifying", r, basedir)

        os.makedirs(imgdir)
        os.makedirs(maskdir)
        check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)
        check_output("cp {}/* {}".format(maskdir_orig, maskdir), shell=True)

        extImg = imgs[0].split(".")[-1]
        extMask = masks[0].split(".")[-1]
        argsImg = " ".join(
            ["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(extImg)]
        )
        argsMask = " ".join(
            ["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(extMask)]
        )
        os.chdir(imgdir)
        check_output(argsImg, shell=True)
        os.chdir(wd)

        os.chdir(maskdir)
        check_output(argsMask, shell=True)
        os.chdir(wd)

        if extImg != "png":
            check_output("rm {}/*.{}".format(imgdir, extImg), shell=True)
            print("Removed duplicates")
        if extMask != "png":
            check_output("rm {}/*.{}".format(maskdir, extMask), shell=True)
            print("Removed duplicates")
        print("Done")


def ensureNoChannel(x: np.ndarray) -> np.ndarray:
    ret = x
    if len(x.shape) > 2:
        ret = ret[:, :, 0]

    return ret


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    print('I am on load data')
    print(basedir)
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))[5:,:]
    print(poses_arr.shape)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    #Poses Bounds is a vector of size equals tom 135
    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]
    
    sh = imageio.imread(img0).shape #The shape of image
    
    ev100s = handle_exif(basedir)
    
    sfx = ""

    if factor is not None:
        print("Apply factor", factor)
        sfx = "_{}".format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, "images" + sfx)
    maskdir = os.path.join(basedir, "masks" + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, "does not exist, returning")
        return

    if not os.path.exists(maskdir):
        print(maskdir, "does not exist, returning")
        return

    imgfiles = [
        os.path.join(imgdir, f)
        for f in sorted(os.listdir(imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][5:]
    masksfiles = [
        os.path.join(maskdir, f)
        for f in sorted(os.listdir(maskdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][5:]
    if poses.shape[-1] != len(imgfiles) and len(imgfiles) == len(masksfiles):
        print(
            "Mismatch between imgs {}, masks {} and poses {} !!!!".format(
                len(imgfiles), len(masksfiles), poses.shape[-1]
            )
        )
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor


    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith("png"):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
    masks = [ensureNoChannel(imread(f) / 255.0) for f in masksfiles]

    imgs = np.stack(imgs, -1)
    masks = np.stack(masks, -1)

    print("Loaded image data", imgs.shape, masks.shape, poses[:, -1, 0])
    return poses, bds, imgs, masks, ev100s


def handle_exif(basedir):
    files = [
        os.path.join(basedir, "images", f)
        for f in sorted(os.listdir(os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg")
    ]

    if len(files) == 0:
        raise Exception(
            "Only jpegs can be used to gain EXIF data needed to handle tonemapping"
        )

    ret = []
    for f in files:
        try:
            img = PIL.Image.open(f)
            """Retrieve an image's EXIF data and return as a dictionary with string keys"""
            exif_data = ehelp.formatted_exif_data(img)

            iso = int(exif_data["ISOSpeedRatings"])#Indicates the ISO Speed and ISO Latitude of the camera or input device as specified in ISO 12232.

            aperture = float(exif_data["FNumber"])#EXIF data shows the aperture used for the photo. Having this information can help you understand more about the depth of field. 
            #If you can see a photo was taken using an aperture of f/2.8 or f/16, you can see how this relates to how much of the photo is in focus.
            """Uncover hidden metadata from your photos. Find when and where the picture was taken."""
            exposureTime = ehelp.getExposureTime(exif_data)

            ev100 = calculate_ev100_from_metadata(aperture, exposureTime, iso)
        except:
            ev100 = 8

        ret.append(ev100)#In photography, exposure value (EV) is a number that represents a combination of a camera's shutter speed and f-number,
        #such that all combinations that yield the same exposure have the same EV (for any fixed scene luminance).

    if len(files) != len(ret):
        raise Exception("Not all images are valid!")

    return np.stack(ret, 0).astype(np.float32)


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):

    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses





def spherify_poses(poses, bds):
    def p34_to_44(p):
        return np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
        )

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def load_llff_data(
    basedir, factor=8, recenter=True, bd_factor=0.75, spherify=False, path_zflat=False,
):
    poses, bds, imgs, msks, ev100s = _load_data(
        basedir, factor=factor,
    )  # factor=8 downsamples original imgs by 8x
    print("Loaded", basedir, bds.min(), bds.max()) #return poses, bds, imgs, masks, ev100s
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    msks = np.expand_dims(np.moveaxis(msks, -1, 0), -1).astype(np.float32)
    images = imgs
    masks = msks
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        print("recentered", c2w.shape)
        print(c2w[:3, :4])

        # Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        zdelta = close_depth * 0.2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * 0.1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.0
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(
            c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=N_rots, N=N_views
        )

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print("Data:")
    print(poses.shape, images.shape, masks.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print("HOLDOUT view is", i_test)
    #i_test=35
    images = images.astype(np.float32)
    masks = masks.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, masks, ev100s, poses, bds, render_poses, i_test

def pick_correct_dataset(args):
    wbs = np.array([[0.8, 0.8, 0.8]], dtype=np.float32)
    
    if args.dataset_type == "real_world":
        (images, masks, ev100s, poses, bds, render_poses, i_test,) = load_llff_data(
            basedir=args.datadir, factor=args.rwfactor, spherify=args.spherify,
        )

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded real world", images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.rwholdout > 0:
            
            i_test = np.arange(images.shape[0])[-5::]####TO DO:MAKE IT TO BE THE NUMBER OF UNDERWATER CAMERAS
            

        i_val = i_test
        
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )[5::]
        
       

        wb_ref_image = np.array([False for _ in range(images.shape[0])])
        
        
        # TODO make this configurable. Currently the reference image
        # is a completely random one
        ref_idx = np.random.choice(i_train, 1)
        
        wb_ref_image[ref_idx] = True#reference image? about what?
        
        
        near = args.near if args.near is not None else 0.9
        far = args.far if args.far is not None else 1.0

        near = tf.reduce_min(bds) * near
        far = tf.reduce_max(bds) * far

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]# hwf -> Height Width Focal(estiasi)

    mean_ev100 = np.mean(ev100s)

    return (
        i_train,
        i_val,
        i_test,
        images,
        masks,
        poses,
        ev100s,
        mean_ev100,
        wbs,
        wb_ref_image,
        render_poses,
        hwf,
        near,
        far,
    )



def dataflow(
    args,
    hwf,
    images,
    masks,
    poses,
    ev100s,
    wbs,
    wb_ref_image,
    select_idxs,
    is_train=False,
):
    train_samples = select_idxs.shape[0]
    repeats = max(args.steps_per_epoch // train_samples, 1)

    main_flow = tf.data.Dataset.from_tensor_slices(
        {
            "image": images[select_idxs],
            "mask": masks[select_idxs],
            "pose": poses[select_idxs],
            "ev100": ev100s[select_idxs],
            "wb": wbs[select_idxs],
            "wb_ref_image": wb_ref_image[select_idxs],
        }
    )
    idx_flow = tf.data.Dataset.from_tensor_slices(select_idxs)

    base_dataflow = tf.data.Dataset.zip((idx_flow, main_flow))
    if is_train:
        base_dataflow = base_dataflow.shuffle(
            100, reshuffle_each_iteration=True
        ).repeat(repeats)

    num_gpus = max(get_num_gpus(), 1)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    dataflow = (
        base_dataflow.map(
            lambda idx, dp: (
                math_utils.repeat(idx[None,], num_gpus if is_train else 1, 0,),
                *build_rays(
                    hwf,
                    dp["image"],
                    dp["mask"],
                    dp["pose"],
                    dp["ev100"],
                    dp["wb"],
                    dp["wb_ref_image"],
                    args.batch_size if is_train else 0,
                    num_gpus if is_train else 1,
                    args.jitter_coords if is_train else None,
                ),  # idx, rays_o, rays_d, pose, input_mask, input_ev100, wb, target
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
        .with_options(options)
    )

    return dataflow


def create_dataflow(args):
    # Def return all sgs and the dataflow
    # The dataflow should return the ray directions, and corresponding values
    # As well as the corresponding index to the sgs
    (
        idx_train,
        idx_val,
        idx_test,
        images,
        masks,
        poses,
        ev100s,
        mean_ev100s,
        wbs,
        wb_ref_image,
        render_poses,
        hwf,
        near,
        far,
    ) = pick_correct_dataset(args)

    # Make sure all images are selected
    if wbs.shape[0] == 1:
        wbs = wbs.repeat(images.shape[0], 0)

    return (
        hwf,
        near,
        far,
        render_poses,
        images.shape[0],
        mean_ev100s,
        dataflow(
            args,
            hwf,
            images,
            masks,
            poses,
            ev100s,
            wbs,
            wb_ref_image,
            idx_train,
            #is_train=True,
            is_train=True),
    
        dataflow(
            args,
            hwf,
            images,
            masks,
            poses,
            ev100s,
            wbs,
            wb_ref_image,
            idx_val,
            is_train=False,
        ),
        dataflow(
            args,
            hwf,
            images,
            masks,
            poses,
            ev100s,
            wbs,
            wb_ref_image,
            idx_test,
            is_train=False,
        ))
    

@tf.function
def build_rays(
    hwf,
    image,
    mask,
    pose,
    ev100,
    wb,
    wb_ref_image,
    num_rand_per_gpu,
    num_gpu,
    should_jitter_coords: bool,
):
    H, W, focal = hwf
    # Setup ray jittering
    jitter_coords = None
    if should_jitter_coords:
        jitter_coords = tf.random.uniform([H, W, 2], minval=-0.5, maxval=0.5)

    input_ev100 = tf.reshape(ev100, (1,))  # [1]
    pose = pose[:3, :4]

    rays_o, rays_d = get_full_image_eval_grid(H, W, focal, pose, jitter=jitter_coords)
    if num_rand_per_gpu > 0:
        coordsFull = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing="ij"), -1)
        coords = tf.reshape(coordsFull, [-1, 2])

        select_inds = tf.random.uniform_candidate_sampler(
            tf.range(coords.shape[0], dtype=tf.int64)[None, :],
            coords.shape[0],
            num_rand_per_gpu * num_gpu,
            True,
            coords.shape[0],
        )[0]

        select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
        rays_o = tf.gather_nd(rays_o, select_inds)
        rays_d = tf.gather_nd(rays_d, select_inds)

        if should_jitter_coords:
            # Jitter coords
            jitteredCoords = tf.cast(coordsFull, tf.float32) + jitter_coords
            jitteredCoords = tf.gather_nd(jitteredCoords, select_inds)
            # Interpolate. Add in fake batch dimensions and remove them
            target = tfa.image.interpolate_bilinear(image[None], jitteredCoords[None])[
                0
            ]
            input_mask = tfa.image.interpolate_bilinear(
                mask[None], jitteredCoords[None]
            )[0]
        else:
            target = tf.gather_nd(image, select_inds)
            input_mask = tf.gather_nd(mask, select_inds)
    else:
        input_mask = tf.reshape(mask, (-1, 1))
        target = tf.reshape(image, (-1, 3))
        rays_o = tf.reshape(rays_o, (-1, 3))
        rays_d = tf.reshape(rays_d, (-1, 3))

    return (
        rays_o,
        rays_d,
        tf.cast(math_utils.repeat(pose[None, ...], num_gpu, 0), tf.float32),
        input_mask,
        math_utils.repeat(input_ev100, num_gpu, 0),
        math_utils.repeat(tf.cast(wb, tf.float32)[None, ...], num_gpu, 0),
        math_utils.repeat(tf.convert_to_tensor([wb_ref_image]), num_gpu, 0),
        target,
    )



