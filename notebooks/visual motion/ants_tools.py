import ants
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional
import flammkuchen as fl


def to_sep_string(nums, separator="x"):
    if len(nums) == 1:
        return nums[0]
    return separator.join(map(str, nums))


@dataclass
class Metric:
    name: str = "MI"
    parameters: Tuple[Union[int, float, str]] = (1, 32, "Regular", 0.25)
        
    def argument(self, ref, mov):
        return self.name+f"[{ref},{mov},{to_sep_string(self.parameters, ',')}]"


@dataclass
class TransformStep:
    name: str = "rigid"
    metric: Metric = Metric()
    method_params: Tuple[Union[int, float]] = (0.1, )
    shrink_factors: Tuple[int] = (12, 8, 4, 2)
    smoothing_sigmas: Tuple[Union[int, float]] = (4,3, 2, 1)
    convergence_window_size: int = 10
    convergence: float = 1e-7
    iterations: Tuple[int] = (200,200,200,0)
        
    def argument_list(self, ref, mov):
        return [
            "--transform",
            self.name+f"[{to_sep_string(self.method_params, ',')}]",
            "--metric",
            self.metric.argument(ref, mov),
            "--convergence",
            f"[{to_sep_string(self.iterations)},{self.convergence},{self.convergence_window_size}]",
            "--shrink-factors",
            to_sep_string(self.shrink_factors),
            "--smoothing-sigmas",
            to_sep_string(self.smoothing_sigmas),
        ]


DEFAULT_STEPS = [
    TransformStep(
        name="rigid",
        metric=Metric(),
        iterations=(200,200,200,0),
        convergence=1e-8,
        shrink_factors=(12,8,4,2),
        smoothing_sigmas=(4,3,2,1),
    ),
    TransformStep(
        name="Affine",
        metric=Metric(),
        iterations=(200,200,200,0),
        convergence=1e-8,
        shrink_factors=(12,8,4,2),
        smoothing_sigmas=(4,3,2,1),
    ),
    TransformStep(
        name="SyN",
        method_params=(0.05, 6, 0.5),
        metric=Metric("CC", parameters=(1, 2)),
        iterations=(160, 160, 100),
        convergence=1e-7,
        shrink_factors=(8, 4, 2),
        smoothing_sigmas=(3, 2, 1),
    )
]


def registration_arguments(ref_ptr, mov_ptr, wfo, wmo, path_output,
                           path_initial,
                           registration_steps: Optional[List[TransformStep]] = None,
                           interpolation="WelchWindowedSinc"):
    if registration_steps is None:
        registration_steps = DEFAULT_STEPS

    return [
        "-d", "3",
        "-r", str(path_initial),
        "--float", "1",
        "--interpolation", interpolation] + \
        sum(map(lambda x: x.argument_list(ref_ptr, mov_ptr), registration_steps), []) + \
        [
            "--collapse-output-transforms",
            "1",
            "-o",
            f'[{path_output}/transforms_,{wmo},{wfo} ]',
            "-v",
            "1",
        ]


def register(ref, mov, path_initial, path_output,
             **registration_kwargs):
    """ Prepares registration inputs and runs the registration

    :param ref: (uint8)
    :param mov: (uint8)
    :param path_initial: path of the initial transformation matrix (in ANTs format)
    :param path_output:
    :return:
    """
    ants_function = ants.utils.get_lib_fn("antsRegistration")

    img_ref = ants.from_numpy(ref).clone("float")
    img_mov = ants.from_numpy(mov).clone("float")
    ref_ptr = ants.utils.get_pointer_string(img_ref)
    mov_ptr = ants.utils.get_pointer_string(img_mov)

    warpedfixout = img_mov.clone()
    warpedmovout = img_ref.clone()
    wfo = ants.utils.get_pointer_string(warpedfixout)
    wmo = ants.utils.get_pointer_string(warpedmovout)

    res = ants_function(registration_arguments(ref_ptr, mov_ptr, wfo, wmo, path_output, path_initial,
                                               **registration_kwargs))

    return warpedfixout, warpedmovout, res


def transform_to_ref(mov, refs, transform_folders, interpolation="Linear", to_ref=True):
    """ Transforms a moving image (functional) to a reference
    or over bridge stacks to a final reference

    :param mov:
    :param refs:
    :param transform_folders: folders containing transforms_1Warp and
        transforms_0GenericAffine for each ref in the refs
    :param interpolation: Linear, NearestNeighbor Gaussian BSpline[order=3],
        CosineWindowedSinc WelchWindowedSinc HammingWindowedSinc LanczosWindowedSinc
    :return: transformed stack
    """
    transform_fn = ants.utils.get_lib_fn("antsApplyTransforms")
    img_mov = ants.from_numpy(mov).clone("float")
    for ref, folder in zip(refs, transform_folders):
        img_ref = ants.from_numpy(ref).clone("float")
        ref_ptr = ants.utils.get_pointer_string(img_ref)
        mov_ptr = ants.utils.get_pointer_string(img_mov)

        warpedmovout = img_ref.clone()
        wmo = ants.utils.get_pointer_string(warpedmovout)
        if to_ref:
            transforms = [
                "--transform",
                str(folder / "transforms_1Warp.nii.gz"),
                "--transform",
                f"[{folder / 'transforms_0GenericAffine.mat'},0]",
            ]
        else:
            transforms = [
                "--transform",
                f"[{folder / 'transforms_0GenericAffine.mat'},1]",
                "--transform",
                str(folder / "transforms_1InverseWarp.nii.gz"),

            ]
        command_list = [
                           "-d", "3",
                           "--float", "1",
                           "--input", mov_ptr,
                           "--output", wmo,
                           "--reference-image", ref_ptr,
                           "--interpolation", interpolation

                       ] + transforms
        res = transform_fn(command_list)
        img_mov = warpedmovout.clone()
    return img_mov


def transform_points(points, transform_folders, to_ref=False):
    """

    :param points: numpy array of points, n_points x 3
    :param transform_folders:
    :param to_ref: direction of the transformation from less-reference 
        to the most reference (from source to refererence True, otherwise false)
    :return:
    """
    libfn = ants.utils.get_lib_fn('antsApplyTransformsToPoints')

    point_image = ants.core.make_image(points.shape, points.flatten())
    points_out = point_image.clone()

    transform_args = []
    for folder in transform_folders:
        if to_ref:
            transform_args.extend([
                "--transform",
                f"[{folder / 'transforms_0GenericAffine.mat'},1]",
                "--transform",
                str(folder / "transforms_1InverseWarp.nii.gz"),
            ])
            
        else:
            transform_args.extend([
                "--transform",
                str(folder / "transforms_1Warp.nii.gz"),
                "--transform",
                f"[{folder / 'transforms_0GenericAffine.mat'},0]",
            ])

    args = [
        "-d", "3",
        "-f", "1",
        "-i", ants.utils.get_pointer_string(point_image),
        "-o", ants.utils.get_pointer_string(points_out),
    ] + transform_args
    libfn(args)
    return points_out.numpy()



def convert_initial_transform(transform_folder):
    """ Converts an affine transform matrix from HDF5 to ANTS format

    :param transform_folder: the folder containing initial_transform.h5
    :return: path to the transform
    """
    transform_mat = fl.load(transform_folder / "initial_transform.h5")
    path_initial = str(transform_folder / "initial_transform.mat")
    at = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3,
                                    matrix=transform_mat[:, :3], offset=transform_mat[:, 3])
    ants.write_transform(at, path_initial)
    return path_initial

