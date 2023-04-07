# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import os
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Input, Path

from gfpgan import GFPGANer
from realesrgan import RealESRGANer

MODEL_NAME = "RealESRGAN_x4plus"
RealESRGAN_x4plus = os.path.join("/root/.cache/realesrgan", MODEL_NAME + ".pth")
RealisticRescaler = "experiments/pretrained_models/4x_RealisticRescaler_100000_G.pth"
ESRGAN_PATH = os.path.join("/root/.cache/realesrgan", MODEL_NAME + ".pth")
GFPGAN_PATH = "/root/.cache/realesrgan/GFPGANv1.3.pth"


class Predictor(BasePredictor):
    def setup(self):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=RealisticRescaler,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self.face_enhancer = GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: int = Input(
            description="Factor to scale image by", ge=0, le=10, default=4
        ),
        face_enhance: bool = Input(
            description="Run GFPGAN face enhancement along with upscaling",
            default=False,
        ),
        # added file extention
        file_extension: str = Input(
            description="File extension output",
            choices=["auto", "jpeg", "png"],
            default="auto",
        ),
        # added file name
        file_name: str = Input(
            description="Outputs file name",
            default="",
        ),

    ) -> Path:
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)

        extension = ''
        if file_extension == 'auto':
            _, ext = os.path.splitext(str(image))
            extension = ext[1:]
        else:
            extension = file_extension

        if file_name == '':
            file_name = 'Upscayler'

        if face_enhance:
            print("running with face enhancement")

            # Apply the scaling factor directly to the input image
            if scale != 2:
                h, w = img.shape[0:2]
                interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                img = cv2.resize(img, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)

            self.face_enhancer.upscale = scale
            _, _, output = self.face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            print("running without face enhancement")
            output, _ = self.upsampler.enhance(img, outscale=scale)

        save_path = os.path.join(tempfile.mkdtemp(), file_name + '.' + extension)
        cv2.imwrite(save_path, output)
        return Path(save_path)
