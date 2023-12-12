import numpy as np
import pydicom


class BoundBox:
    def __init__(self, dcm1, dcm2) -> None:
        self.dcm1 = dcm1
        self.dcm2 = dcm2
        self.volume = None

    def _get_image_corners_2D(self, dcm):
        if hasattr(dcm, "pixel_array"):
            arr = dcm.pixel_array
            corners = np.array(
                [
                    [0, 0],
                    [0, arr.shape[1]],
                    [arr.shape[0], 0],
                    [arr.shape[0], arr.shape[1]],
                ]
            ).reshape(-1, 1)
        else:
            raise AttributeError("dcm must have a pixel_array attribute")

        return corners

    def _get_image_corners_3D(self):
        vol_arr = self.volume
        corners = np.array(
            [
                [0, 0, 0],
                [vol_arr.shape[0], 0, 0],
                [0, vol_arr.shape[1], 0],
                [0, 0, vol_arr.shape[2]],
                [vol_arr.shape[0], vol_arr.shape[1], 0],
                [0, vol_arr.shape[1], vol_arr.shape[2]],
                [vol_arr.shape[0], 0, vol_arr.shape[2]],
                [vol_arr.shape[0], vol_arr.shape[1], vol_arr.shape[2]],
            ]
        )

        return corners

    def _compare_corners(self):
        corners1 = self._get_image_corners_2D(self.dcm1)
        corners2 = self._get_image_corners_2D(self.dcm2)

    def _get_bounding_box(self):
        pass


if __name__ == "__main__":
    dcm1 = "/home/ralcala/Documents/AID4ID/testPatients/full_test/FRANCESA/25793249/20231206/col_lumbar___mgm/ax_t2_frfse/MR.1.2.840.113619.2.408.14196467.1339631.17206.1701514773.324.1.dcm"
    dcm2 = "/home/ralcala/Documents/AID4ID/testPatients/full_test/FRANCESA/25793249/20231206/col_lumbar___mgm/sag_t2_frfse/MR.1.2.840.113619.2.408.14196467.1339631.17206.1701514773.321.6.dcm"
    bound_box = BoundBox(dcm1, dcm2)
    bound_box._compare_corners()
