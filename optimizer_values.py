import jax.numpy as jnp
from image_converter_utils import jxl_xyb_to_srgb, srgb_to_jxl_xyb, dct_to_xyb, xyb_to_dct


class OptimizerValues:

    def __init__(self, shape):
        self.values = jnp.zeros(shape)

    def convert_to_rgb(self):
        pass

    def convert_to_xyb(self):
        pass

    def convert_to_xyb_dct(self):
        pass

class RGBOptimizerValues(OptimizerValues):
    def convert_to_rgb(self):
        return self.values

    def convert_to_xyb(self):
        return [srgb_to_jxl_xyb(val) for val in self.values]

    def convert_to_xyb_dct(self):
        return [xyb_to_dct(val) for val in self.convert_to_xyb()]

class XYBOptimizerValues(OptimizerValues):
    def convert_to_xyb(self):
        return self.values

    def convert_to_rgb(self):
        return [jxl_xyb_to_srgb(val) for val in self.values]

    def convert_to_xyb_dct(self):
        return [xyb_to_dct(val) for val in self.values]

class XYBDCTOptimizerValues(OptimizerValues):

    def __init__(self, shape):
        super().__init__(shape)
        self.values = jnp.zeros((shape[0]//8, shape[1]//8, shape[3], 8, 8))

    def convert_to_xyb(self):
        return [dct_to_xyb(val) for val in self.values]

    def convert_to_xyb_dct(self):
        return self.values

    def convert_to_rgb(self):
        return [jxl_xyb_to_srgb(val) for val in self.convert_to_xyb()]