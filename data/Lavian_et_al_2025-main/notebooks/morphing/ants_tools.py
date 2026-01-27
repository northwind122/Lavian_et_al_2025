import ants
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional
import flammkuchen as fl


def to_sep_string(nums, separator="x"):
    
    """
    This function converts a sequence of numbers to a string with specified separator.
    
    Parameters
    ----------
    nums : sequence
        Sequence of values to be converted to string.
    separator : str, optional
        String used to join the values. Default is "x".
        
    Returns
    -------
    str
        String representation of numbers joined by the separator.
        If nums contains only one element, returns that element as a string.
    """
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
    """
    This function performs image registration using ANTs.
    
    This function prepares input images, runs the registration process, and returns 
    the warped images along with the registration result.
    
    Parameters
    ----------
    ref : ndarray
        Reference image as a numpy array (uint8).
    mov : ndarray
        Moving image as a numpy array (uint8).
    path_initial : str or Path
        Path to the initial transformation matrix (in ANTs format).
    path_output : str or Path
        Directory where transformation files will be saved.
    **registration_kwargs : dict
        Additional arguments to pass to registration_arguments function, including:
        - registration_steps: List of TransformStep objects defining the registration process
        - interpolation: Method for interpolation during warping
        
    Returns
    -------
    tuple
        (warpedfixout, warpedmovout, res)
        - warpedfixout: ANTs image - fixed image warped to moving space
        - warpedmovout: ANTs image - moving image warped to fixed space
        - res: Result from antsRegistration function

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
    """
    Transform an image to a reference. This function applies a sequence of transformations to a moving image,
    directly or through bridge stacks to a final reference.
    
    Parameters
    ----------
    mov : ndarray
        Moving image as a numpy array (typically a functional image).
    refs : list of ndarray
        List of reference images as numpy arrays.
    transform_folders : list of str or Path
        List of folders containing transformation files for each reference.
        Each folder should contain transforms_1Warp.nii.gz and 
        transforms_0GenericAffine.mat files.
    interpolation : str, optional
        Interpolation method for warping. Options include "Linear", "NearestNeighbor",
        "Gaussian", "BSpline", "CosineWindowedSinc", "WelchWindowedSinc", etc.
        Default is "Linear".
    to_ref : bool, optional
        Direction of transformation. If True, transforms from moving to reference space.
        If False, transforms from reference to moving space. Default is True.
        
    Returns
    -------
    ANTsImage
        Transformed image as an ANTs image object.
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
    This function transforms a set of points through a series of transformations.
    
    This function applies ANTs transformations to a set of 3D points,
    allowing for warping points between spaces.
    
    Parameters
    ----------
    points : ndarray
        Numpy array of points with shape (n_points, 3).
    transform_folders : list of str or Path
        List of folders containing transformation files.
        Each folder should contain transforms_1Warp.nii.gz, transforms_1InverseWarp.nii.gz, 
        and transforms_0GenericAffine.mat files.
    to_ref : bool, optional
        Direction of transformation. If True, transforms from reference to source space.
        If False, transforms from source to reference space. 
        
    Returns
    -------
    ndarray
        Transformed points as a numpy array with the same shape as input.
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
    """
    Convert an affine transform matrix from HDF5 to ANTs format.
    
    This function loads a transformation matrix stored in HDF5 format
    and converts it to the ANTs transformation format.
    
    Parameters
    ----------
    transform_folder : str or Path
        Directory containing the initial_transform.h5 file.
        
    Returns
    -------
    str
        Path to the created ANTs format transformation file.
        
    Note: this function expects the HDF5 file to contain a 4×3 affine transformation matrix
    where the first 3 columns represent the linear component and the 4th column
    represents the offset (translation).
    """

    transform_mat = fl.load(transform_folder / "initial_transform.h5")
    path_initial = str(transform_folder / "initial_transform.mat")
    at = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3,
                                    matrix=transform_mat[:, :3], offset=transform_mat[:, 3])
    ants.write_transform(at, path_initial)
    return path_initial

