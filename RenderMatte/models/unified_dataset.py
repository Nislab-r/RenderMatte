import torch, torchvision, os, json, pandas
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import random
from typing import Union
from torchvision.transforms import functional as TF
import scipy.ndimage
from scipy.ndimage import label
from PIL import ImageDraw

# --- 预计算腐蚀/膨胀的核，避免在函数内重复生成，提高效率 ---
MAX_KERNEL_SIZE = 30
EROSION_KERNELS = [None] + [
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    for size in range(1, MAX_KERNEL_SIZE)
]

# fix the all random seed for reproducibility
random.seed(42)
np.random.seed(42)


def gen_trimap(alpha: np.ndarray) -> np.ndarray:
    """
    根据 alpha 通道图生成 trimap.
    前景为255, 背景为0, 未知区域为128.
    """
    assert alpha.dtype == np.uint8, "Alpha channel must be of type uint8"

    k_size = random.choice(range(5, 15))
    iterations = np.random.randint(5, 15)

    kernel = EROSION_KERNELS[k_size]

    dilated = cv2.dilate(alpha, kernel, iterations=iterations)
    eroded = cv2.erode(alpha, kernel, iterations=iterations)

    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0

    coords = np.argwhere(trimap == 128)
    h, w = alpha.shape
    if coords.size == 0:
        coords = np.array([[0, 0], [1, 1]])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    x_min, x_max = x_min / w, x_max / w
    y_min, y_max = y_min / h, y_max / h
    coords = np.array([x_min, y_min, x_max, y_max])
    return trimap, coords.astype(np.float32)


def gen_mask(alpha: np.ndarray, min_kernel_size: int = 15, max_kernel_size: int = 29) -> dict:
    """
    从 alpha 通道随机生成一个二值分割掩码 (mask).
    """
    assert alpha.dtype != np.uint8, "Alpha channel must not be of type uint8"

    h, w = alpha.shape

    low = 0.01
    high = 1.0
    thres = random.random() * (high - low) + low
    mask = (alpha >= thres).astype(np.uint8)

    random_num = random.randint(0, 3)
    k_size = np.random.randint(min_kernel_size, min(max_kernel_size + 1, MAX_KERNEL_SIZE))
    kernel = EROSION_KERNELS[k_size]

    if random_num == 0:
        mask = cv2.erode(mask, kernel)
    elif random_num == 1:
        mask = cv2.dilate(mask, kernel)
    elif random_num == 2:
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
    elif random_num == 3:
        mask = cv2.dilate(mask, kernel)
        mask = cv2.erode(mask, kernel)

    coords = np.argwhere(mask)
    if coords.size == 0:
        mask_coords = np.array([0, 0, 1 / w, 1 / h])
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        x_min, x_max = x_min / w, x_max / w
        y_min, y_max = y_min / h, y_max / h
        mask_coords = np.array([x_min, y_min, x_max, y_max])

    return mask.astype(np.float32), mask_coords.astype(np.float32)


def gen_bbox(alpha: np.ndarray, coe_scale: float = 0.1) -> dict:
    """
    从 alpha 通道生成一个边界框 (bounding box).
    """
    assert alpha.dtype != np.uint8, "Alpha channel must be float."

    height, width = alpha.shape

    if np.count_nonzero(alpha) == 0:
        return np.zeros_like(alpha, dtype=np.float32), np.array(
            [0, 0, 1 / width, 1 / height], dtype=np.float32
        )

    binary_mask = alpha > 0
    labeled_array, num_features = label(binary_mask)

    y_coords, x_coords = np.where(binary_mask)
    y_min, x_min = y_coords.min(), x_coords.min()
    y_max, x_max = y_coords.max(), x_coords.max()

    if num_features > 1:
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes = component_sizes[1:]
        if component_sizes.size > 0:
            largest_component_label = np.argmax(component_sizes) + 1
            coords = np.argwhere(labeled_array == largest_component_label)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

    coe = random.uniform(0, coe_scale)
    h_box, w_box = y_max - y_min, x_max - x_min

    padding_y = int(coe * h_box)
    padding_x = int(coe * w_box)

    y_min_padded = y_min + random.choice([-1, 1]) * padding_y
    y_max_padded = y_max + random.choice([-1, 1]) * padding_y
    x_min_padded = x_min + random.choice([-1, 1]) * padding_x
    x_max_padded = x_max + random.choice([-1, 1]) * padding_x

    y_min_final = min(y_min_padded, y_max_padded)
    y_max_final = max(y_min_padded, y_max_padded)
    x_min_final = min(x_min_padded, x_max_padded)
    x_max_final = max(x_min_padded, x_max_padded)

    y_min_final = max(0, y_min_final)
    y_max_final = min(height, y_max_final)
    x_min_final = max(0, x_min_final)
    x_max_final = min(width, x_max_final)

    bbox_mask = np.zeros_like(alpha)
    bbox_mask[y_min_final:y_max_final, x_min_final:x_max_final] = 1

    bbox_coords = np.array([
        x_min_final / width,
        y_min_final / height,
        x_max_final / width,
        y_max_final / height,
    ])

    return bbox_mask.astype(np.float32), bbox_coords.astype(np.float32)


def gen_points(
    alpha: np.ndarray,
    num_points: int = 10,
    psm: str = "gauss",
    radius: int = 20,
    thres: float = 0.8,
) -> dict:
    """
    从 alpha 通道中随机采样前景点，并生成对应的点掩码 (point mask).
    """
    assert alpha.dtype != np.uint8, "Alpha channel must be float."

    height, width = alpha.shape

    y_coords, x_coords = np.where(alpha > thres)

    if len(y_coords) < num_points:
        return np.zeros_like(alpha, dtype=np.float32), np.zeros(num_points * 2, dtype=np.float32)

    np.random.seed(42)
    indices = np.random.choice(len(y_coords), size=num_points, replace=False)
    selected_y = y_coords[indices]
    selected_x = x_coords[indices]

    point_mask = np.zeros_like(alpha, dtype=np.float32)
    point_coords = []

    for y_center, x_center in zip(selected_y, selected_x):
        tmp_mask = np.zeros_like(alpha, dtype=np.float32)
        if psm == "gauss":
            tmp_mask[y_center, x_center] = 1
            tmp_mask = scipy.ndimage.gaussian_filter(tmp_mask, sigma=radius)
            if tmp_mask.max() > 0:
                tmp_mask /= tmp_mask.max()
        elif psm == "circle":
            cv2.circle(tmp_mask, (x_center, y_center), radius, 1, -1)

        point_mask = np.maximum(point_mask, tmp_mask)

        point_coords.append(x_center / width)
        point_coords.append(y_center / height)

    return point_mask.astype(np.float32), np.array(point_coords, dtype=np.float32)


def get_true_intrinsics_from_hypersim(hypersim_matrix, width=768, height=768):
    """
    将 Hypersim 提供的 M_cam_from_uv 矩阵转换为标准的相机内参矩阵 K。
    """
    device = hypersim_matrix.device
    dtype = hypersim_matrix.dtype

    M_ndc_from_pix = torch.tensor([
        [2.0 / (width - 1), 0.0, -1.0],
        [0.0, -2.0 / (height - 1), 1.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)

    K_inv_true = torch.matmul(hypersim_matrix, M_ndc_from_pix)
    K = torch.inverse(K_inv_true)

    return K


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value

    def __call__(self, data):
        if data is None:
            data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=False):
        self.convert_RGB = convert_RGB

    def __call__(self, data: str):
        if "depth" in data:
            image = Image.open(data)
        elif "normal" in data:
            image = np.load(data)
        else:
            image = Image.open(data)
            if self.convert_RGB:
                image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class LoadImageToTensor(DataProcessingOperator):
    """加载图像并转换为tensor，支持RGB图像、深度图和法线图"""

    def __init__(
        self,
        convert_RGB=True,
        using_log=False,
        using_sqrt_disp=False,
        using_sqrt=False,
        using_pdf=False,
        pdf=None,
        with_mask=False,
        matting_prompt="trimap",
        height=768,
        width=768
    ):
        self.convert_RGB = convert_RGB
        self.using_log = using_log
        self.using_sqrt = using_sqrt
        self.using_sqrt_disp = using_sqrt_disp
        self.using_pdf = using_pdf
        self.pdf = pdf
        self.with_mask = with_mask
        self.matting_prompt = matting_prompt
        self.height = height
        self.width = width

    def _load_image(self, data_path: str):
        if "Eval" in data_path or "rgb" in data_path or "img" in data_path or "matting" in data_path:
            image = Image.open(data_path)
            if self.convert_RGB:
                image = image.convert("RGB")
            if "kitti" in data_path.lower():
                _w, _h = image.size
                top_margin = int(_h - 352)
                left_margin = int((_w - 1216) / 2)
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            return image
        elif "depth" in data_path:
            image = Image.open(data_path)
            if "kitti" in data_path.lower():
                _w, _h = image.size
                top_margin = int(_h - 352)
                left_margin = int((_w - 1216) / 2)
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            return image
        elif "normal" in data_path:
            return np.load(data_path)
        else:
            image = Image.open(data_path)
            if self.convert_RGB:
                image = image.convert("RGB")
            return image

    def _to_tensor(self, image: Union[Image.Image, np.ndarray], data_path: str):
        if "Eval" in data_path or "rgb" in data_path or "img" in data_path or "matting" in data_path:
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(image) * 2 - 1
        elif "depth" in data_path:
            image = image.resize((self.width, self.height))
            array = np.array(image, dtype=np.float32)
            if "vkitti" in data_path:
                array = array / 100.0
                mask = np.logical_and(array > 1e-1, array < 80.0)
            elif "hyp" in data_path.lower():
                array = array / 1000.0
                mask = np.logical_and(array > 1e-3, array < 65.5)
            if self.using_log:
                array = np.log(array + 1e-6)
            elif self.using_sqrt_disp:
                array = 1 / np.sqrt(array + 1e-6)
            elif self.using_sqrt:
                array = np.sqrt(array + 1e-6)
            elif self.using_pdf:
                array = np.interp(array, self.pdf['bins'], self.pdf['y_map'])

            p2, p98 = np.percentile(array[mask], (2, 98))

            if p98 - p2 < 1e-3:
                print(f"Warning: {data_path} has invalid depth values.")
                with open("error_depth_Hyp_Log.txt", "a") as f:
                    f.write(f"{data_path}\n")
            array = (array - p2) / (p98 - p2)
            array = (array - 0.5) * 2
            array = np.clip(array, -1, 1)

            tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
        elif "normal" in data_path:
            image[:, :, 0] *= -1
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR_EXACT)
            image = image.astype(np.float32)
            tensor = torch.from_numpy(image).permute(2, 0, 1)

            norm = torch.norm(tensor, dim=0, keepdim=True) + 1e-8
            tensor = tensor / norm

            mask = norm.squeeze(0) > 1e-3
        else:
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(image) * 2 - 1
        
        path_parts = set(os.path.normpath(data_path).lower().split(os.sep))

        if self.with_mask and "rgb" not in data_path and "img" not in data_path:
            if "depth" in data_path:
                mask = (mask).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)
                tensor = torch.cat([tensor, mask], dim=0)
            elif "normal" in data_path:
                mask = mask.float().unsqueeze(0)
                tensor = torch.cat([tensor, mask], dim=0)
            
            elif "mask" in path_parts or "alpha" in path_parts:
            # elif "mask" in data_path or "alpha" in data_path:
                alpha = (tensor[0].numpy() + 1) / 2
                visual_prompt, _ = gen_trimap((alpha * 255).astype(np.uint8))
                visual_prompt = torch.from_numpy(visual_prompt).unsqueeze(0).to(tensor.dtype)
                tensor = torch.cat([tensor, visual_prompt], dim=0)

        return tensor

    def __call__(self, data_path: str):
        image = self._load_image(data_path)
        tensor = self._to_tensor(image, data_path)
        return tensor


class LoadPILImageTensor(DataProcessingOperator):
    def __init__(self, convert_RGB=True, height=768, width=768):
        self.convert_RGB = convert_RGB
        self.height = height
        self.width = width

    def __call__(self, image: Image.Image):
        if self.convert_RGB:
            image = image.convert("RGB")
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image) * 2 - 1
        return tensor


class TensorCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None,
                 height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def _calculate_target_size(self, tensor):
        _, height, width = tensor.shape

        if self.height is None or self.width is None:
            if self.max_pixels and width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)

            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width

        return height, width

    def _crop_and_resize_tensor(self, tensor, target_height, target_width):
        tensor = tensor.unsqueeze(0)

        _, _, current_height, current_width = tensor.shape
        scale = max(target_width / current_width, target_height / current_height)

        new_height = round(current_height * scale)
        new_width = round(current_width * scale)

        tensor = F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode='bicubic',
            align_corners=True
        )

        _, _, scaled_height, scaled_width = tensor.shape
        top = (scaled_height - target_height) // 2
        left = (scaled_width - target_width) // 2

        tensor = tensor[:, :, top:top + target_height, left:left + target_width]
        tensor = tensor.squeeze(0)

        return tensor

    def __call__(self, tensor):
        target_height, target_width = self._calculate_target_size(tensor)
        processed_tensor = self._crop_and_resize_tensor(tensor, target_height, target_width)
        return processed_tensor


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator

    def __call__(self, data):
        return [self.operator(i) for i in data]


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location

    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        if os.path.isabs(data) or not self.base_path:
            return data
        return os.path.join(self.base_path, data)


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        default_caption=None,
        matting_prompt=None,
        use_coor_input=False,
        use_camera_intrinsics=False,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.default_caption = default_caption
        self.matting_prompt = matting_prompt
        self.use_coor_input = use_coor_input
        self.use_camera_intrinsics = use_camera_intrinsics
        self.load_metadata(metadata_path)

        # 支持:
        # 1) "points"
        # 2) "trimap,mask,bbox,points"
        # 3) ["trimap", "mask", "bbox", "points"]
        self.prompt_pool = []
        if self.matting_prompt is not None:
            if isinstance(self.matting_prompt, str):
                if "," in self.matting_prompt:
                    self.prompt_pool = [x.strip() for x in self.matting_prompt.split(",") if x.strip()]
                else:
                    self.prompt_pool = [self.matting_prompt.strip()]
            elif isinstance(self.matting_prompt, (list, tuple)):
                self.prompt_pool = [str(x).strip() for x in self.matting_prompt if str(x).strip()]
            else:
                raise ValueError(f"Unsupported matting_prompt type: {type(self.matting_prompt)}")

            valid_prompts = {"trimap", "mask", "bbox", "points"}
            for p in self.prompt_pool:
                if p not in valid_prompts:
                    raise ValueError(
                        f"Unknown matting_prompt: {p}. "
                        f"Supported prompts are: {sorted(valid_prompts)}"
                    )

        # ========== dataset 内部调试统计 ==========
        # 注意：如果 dataloader num_workers > 0，这个统计是每个 worker 各自统计，
        # 不是严格全局唯一统计，但足够用来观察随机采样是否生效。
        self.enable_prompt_debug = True
        self.prompt_debug_interval = 100
        self.prompt_sample_count = 0
        self.prompt_counter = {
            "trimap": 0,
            "mask": 0,
            "bbox": 0,
            "points": 0,
        }
        # ======================================

        if use_camera_intrinsics:
            if "vkitti" in base_path.lower():
                self.camera_intrinsics = torch.tensor([
                    [448.3300, 0.0000, 383.6901],
                    [0.0000, 1484.8178, 382.9760],
                    [0.0000, 0.0000, 1.0000]
                ])
            elif "hyp" in base_path.lower():
                path = "../dataset/Hypersim/processed_depth/metadata_camera_parameters.csv"
                df = pandas.read_csv(path)
                columns = [
                    'scene_name', 'M_cam_from_uv_00', 'M_cam_from_uv_01', 'M_cam_from_uv_02',
                    'M_cam_from_uv_10', 'M_cam_from_uv_11', 'M_cam_from_uv_12',
                    'M_cam_from_uv_20', 'M_cam_from_uv_21', 'M_cam_from_uv_22'
                ]
                K = lambda row: [
                    [float(row[columns[1]]), float(row[columns[2]]), float(row[columns[3]])],
                    [float(row[columns[4]]), float(row[columns[5]]), float(row[columns[6]])],
                    [float(row[columns[7]]), float(row[columns[8]]), float(row[columns[9]])]
                ]
                self.camera_intrinsics = {}
                for i in range(len(df)):
                    row = df.iloc[i]
                    scene_name = row['scene_name']
                    hypersim_matrix = torch.tensor(K(row), dtype=torch.float32)
                    self.camera_intrinsics[scene_name] = get_true_intrinsics_from_hypersim(
                        hypersim_matrix, width=768, height=768
                    )

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1024 * 1024,
        height=768,
        width=768,
        height_division_factor=16,
        width_division_factor=16,
        using_log=False,
        using_sqrt=False,
        using_sqrt_disp=False,
        with_mask=False,
        using_pdf=False,
        pdf=None,
    ):
        if base_path is None:
            base_path = ""

        return RouteByType(operator_map=[
            (
                str,
                ToAbsolutePath(base_path) >>
                LoadImageToTensor(
                    using_log=using_log,
                    using_sqrt_disp=using_sqrt_disp,
                    using_sqrt=using_sqrt,
                    using_pdf=using_pdf,
                    pdf=pdf,
                    with_mask=with_mask,
                    height=height,
                    width=width
                )
            ),
            (
                list,
                SequencialProcess(
                    ToAbsolutePath(base_path) >>
                    LoadImageToTensor(
                        using_log=using_log,
                        using_sqrt_disp=using_sqrt_disp,
                        using_sqrt=using_sqrt,
                        using_pdf=using_pdf,
                        pdf=pdf,
                        with_mask=with_mask,
                        height=height,
                        width=width
                    )
                )
            ),
            (Image.Image, LoadPILImageTensor(convert_RGB=True, height=height, width=width)),
        ])

    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        elif metadata_path.endswith(".csv"):
            metadata = pandas.read_csv(metadata_path)
            self.camera_intrinsics = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".txt"):
            metadata = []
            with open(metadata_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    items = line.strip().split("\t")
                else:
                    items = line.strip().split()
                item_dict = {k: v for k, v in zip(self.data_file_keys, items)}
                metadata.append(item_dict)
            self.data = metadata

    def _sample_prompt_type(self):
        if self.matting_prompt is None:
            return None

        if len(self.prompt_pool) == 0:
            prompt_type = self.matting_prompt
        else:
            prompt_type = random.choice(self.prompt_pool)

        # dataset 内部调试统计
        if getattr(self, "enable_prompt_debug", False):
            if prompt_type in self.prompt_counter:
                self.prompt_counter[prompt_type] += 1
            self.prompt_sample_count += 1

            if self.prompt_sample_count % self.prompt_debug_interval == 0:
                total = max(self.prompt_sample_count, 1)
                print(
                    "[UnifiedDataset Prompt Debug] "
                    f"total={total} | "
                    f"trimap={self.prompt_counter['trimap']} ({100.0 * self.prompt_counter['trimap'] / total:.2f}%) | "
                    f"mask={self.prompt_counter['mask']} ({100.0 * self.prompt_counter['mask'] / total:.2f}%) | "
                    f"bbox={self.prompt_counter['bbox']} ({100.0 * self.prompt_counter['bbox'] / total:.2f}%) | "
                    f"points={self.prompt_counter['points']} ({100.0 * self.prompt_counter['points'] / total:.2f}%)"
                )

        return prompt_type

    def _build_matting_visual_prompt(self, alpha, dtype):
        """
        alpha: float numpy array, shape HxW, range 0~1
        return:
            visual_prompt_tensor: torch.Tensor, shape 3xHxW, range [-1, 1]
            visual_prompt_coords: np.ndarray or None
            prompt_type: str
        """
        prompt_type = self._sample_prompt_type()
        visual_prompt_coords = None

        if prompt_type == "trimap":
            visual_prompt, visual_prompt_coords = gen_trimap((alpha * 255).astype(np.uint8))
            visual_prompt = (visual_prompt / 255.0).astype(np.float32)
        elif prompt_type == "mask":
            visual_prompt, visual_prompt_coords = gen_mask(alpha)
        elif prompt_type == "bbox":
            visual_prompt, visual_prompt_coords = gen_bbox(alpha, 0.01)
        elif prompt_type == "points":
            visual_prompt, visual_prompt_coords = gen_points(alpha, radius=30)
        else:
            raise ValueError(
                f"Unknown selected matting prompt: {prompt_type}. "
                f"Supported prompts: trimap, mask, bbox, points"
            )

        if isinstance(visual_prompt, np.ndarray):
            visual_prompt = torch.from_numpy(visual_prompt).to(torch.float32)

        if visual_prompt.ndim == 2:
            visual_prompt = visual_prompt.unsqueeze(0)

        if visual_prompt.shape[0] == 1:
            visual_prompt = visual_prompt.repeat(3, 1, 1)

        visual_prompt = visual_prompt.to(dtype)
        visual_prompt = visual_prompt * 2 - 1

        return visual_prompt, visual_prompt_coords, prompt_type
    
    def _denorm_image_tensor_to_pil(self, image_tensor: torch.Tensor):
        """
        image_tensor: 3xHxW, range [-1, 1]
        return: PIL RGB
        """
        img = image_tensor.detach().to(torch.float32).cpu()
        img = ((img + 1) / 2).clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(img)

    def _save_points_visualization(self, image_tensor, visual_prompt, visual_prompt_coords, data_id, gt_path=None):
        """
        保存：
        1) 独立 points prompt
        2) 原图 + points overlay
        3) GT
        """
        save_root = "<PROJECT_ROOT>/points"
        os.makedirs(save_root, exist_ok=True)

        if visual_prompt_coords is None:
            return

        # 每 100 个 sample 存一次
        if data_id % 100 != 0:
            return

        try:
            sample_dir = os.path.join(save_root, f"sample_{int(data_id):05d}")
            os.makedirs(sample_dir, exist_ok=True)

            # ---------- 1) 保存独立 points prompt ----------
            vp = visual_prompt.detach().to(torch.float32).cpu()
            vp = ((vp + 1) / 2).clamp(0, 1)
            vp_img = (vp[0].numpy() * 255).astype(np.uint8)
            Image.fromarray(vp_img).save(os.path.join(sample_dir, "points_prompt.png"))

            # ---------- 2) 保存原图 + points overlay ----------
            base_img = self._denorm_image_tensor_to_pil(image_tensor)
            draw = ImageDraw.Draw(base_img)

            w, h = base_img.size
            coords = visual_prompt_coords.tolist()

            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)

                r = 6
                draw.ellipse(
                    (x - r, y - r, x + r, y + r),
                    fill=(255, 0, 0),
                    outline=(255, 255, 255)
                )

            base_img.save(os.path.join(sample_dir, "points_overlay.png"))

            # ---------- 3) 保存 GT ----------
            if gt_path is not None and os.path.exists(gt_path):
                gt = Image.open(gt_path)
                gt.save(os.path.join(sample_dir, "gt.png"))

        except Exception as e:
            print(f"[Points Visualization] Failed to save sample {data_id}: {e}")
    
    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()

            if self.use_camera_intrinsics:
                if isinstance(self.camera_intrinsics, dict):
                    parts = data[self.data_file_keys[0]].split("/")
                    scene_name = parts[0] if len(parts) == 2 else parts[1]
                    data["camera_intrinsics"] = self.camera_intrinsics[scene_name]
                else:
                    data["camera_intrinsics"] = self.camera_intrinsics

                data["input_depth"] = Image.open(
                    os.path.join(self.base_path, data[self.data_file_keys[1]])
                ).resize((768, 576))

                if "vkitti" in self.base_path.lower():
                    data["input_depth"] = torch.from_numpy(
                        (np.array(data["input_depth"])) / 100.0
                    ).to(torch.bfloat16)
                elif "hyp" in self.base_path.lower():
                    data["input_depth"] = torch.from_numpy(
                        (np.array(data["input_depth"])) / 1000.0
                    ).to(torch.bfloat16)

            for key in self.data_file_keys:
                if key in data:
                    data[key] = self.main_data_operator(data[key]).to(torch.bfloat16)

            for key in self.special_operator_map:
                if key == "mask":
                    _tmp = data["image"].clone()
                    data["image"] = _tmp[:3]
                    data["mask"] = _tmp[3]

                    if self.metadata_path is not None and "matting" in self.metadata_path:
                        data["trimap"] = data["mask"].clone()

                        alpha = (
                            data["image"][0].clone().to(torch.float32).cpu().numpy() + 1
                        ) / 2.0

                        visual_prompt_coords = None
                        prompt_type = None

                        if self.matting_prompt is not None:
                            visual_prompt, visual_prompt_coords, prompt_type = \
                                self._build_matting_visual_prompt(alpha, data["image"].dtype)
                                
                            # 仅在 points 模式下抽样保存可视化
                            if prompt_type == "points":
                                gt_rel_path = self.data[data_id % len(self.data)][self.data_file_keys[1]]
                                gt_path = os.path.join(self.base_path, gt_rel_path)
                                self._save_points_visualization(
                                    image_tensor=data["kontext_images"],
                                    visual_prompt=visual_prompt,
                                    visual_prompt_coords=visual_prompt_coords,
                                    data_id=data_id,
                                    gt_path=gt_path
                                )

                            data["kontext_images"] = [data["kontext_images"], visual_prompt]

                            prompt_type_to_id = {
                                "trimap": 0,
                                "mask": 1,
                                "bbox": 2,
                                "points": 3,
                            }
                            data["prompt_type_id"] = torch.tensor(
                                prompt_type_to_id[prompt_type], dtype=torch.long
                            )
                            data["prompt_type"] = prompt_type

                            del visual_prompt

                        data["mask"] = (data["mask"] == 128)

                        if visual_prompt_coords is not None and self.use_coor_input:
                            data["visual_prompt_coords"] = torch.from_numpy(
                                visual_prompt_coords
                            ).to(torch.float32)

                        del alpha

                    del _tmp

                elif key == "prompt":
                    data["prompt"] = self.default_caption

        data["_path_kontext"] = self.data[data_id % len(self.data)][self.data_file_keys[0]]
        data["_path_image"] = self.data[data_id % len(self.data)][self.data_file_keys[1]]
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat

    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True