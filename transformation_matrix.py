import pydicom
import numpy as np
from dicom_loader import DicomLoaderMRI


class TransformationMatrixMRI:
    """This class takes a MRI .dcm object from the pydicom.dcmread method
    and generates its corresponding transformation matrix
    """

    def __init__(self, dcm: pydicom.Dataset) -> None:
        self.dcm = dcm
        self.transformation_matrix = self._transformation_matrix_generator()
        self.inverted_matrix = self._inverted_matrix(self.transformation_matrix)

    def _transformation_matrix_generator(self) -> np.ndarray:
        """This function takes a .dcm and creates the transformation matrix
        of eq C.7.6.2.1-1.
        https://dicom.innolitics.com/ciods/mr-image/image-plane/00200032

        Returns:
            ndarray dim=(4, 4): returns the 4x4 transformation matrix
            of eq C.7.6.2.1-1.
        """
        # reshape to the correct dims to stack in last matrix
        # image position patient (0020, 0032)
        # location in mm from the origin of the RCS.
        s_vector = np.array(self.dcm.ImagePositionPatient).reshape(-1, 1)
        xy_vector = self.dcm.ImageOrientationPatient

        # split the xy vector in its components
        x_vector = np.array(xy_vector[:3]).reshape(-1, 1)
        y_vector = np.array(xy_vector[3:]).reshape(-1, 1)

        # z-vector as cross prod of xy
        z_vector = np.cross(x_vector.T, y_vector.T).reshape(3, 1)

        # Slice Thickness
        slice_thickness = np.array(self.dcm.SliceThickness).astype(float)

        # pixel spacing tag (0028, 0030)
        pixel_spacing = np.array(self.dcm.PixelSpacing)

        # eq C.7.6.2.1-1.
        # 4x1 column vectors for the transformation matrix
        x_vector_multi = np.multiply(x_vector, pixel_spacing[0]).reshape(-1, 1)
        x_vector_multi = np.vstack((x_vector_multi, [0]))

        y_vector_multi = np.multiply(y_vector, pixel_spacing[1]).reshape(-1, 1)
        y_vector_multi = np.vstack((y_vector_multi, [0]))

        z_vector_multi = np.multiply(z_vector, slice_thickness).reshape(3, 1)
        z_vector_multi = np.vstack((z_vector_multi, [1]))

        zero_vector = np.array([0.0, 0.0, 0.0, 0.0]).reshape((4, 1))

        s_vector_stacked = np.vstack((s_vector, [1]))
        # final matrix
        arrays = [x_vector_multi, y_vector_multi, z_vector_multi, s_vector_stacked]
        transformation_matrix = np.concatenate(arrays, axis=1)
        return transformation_matrix

    def _inverted_matrix(self, transformation_matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(transformation_matrix)


if __name__ == "__main__":
    path = "/home/ralcala/Documents/AID4ID/testPatients/23342540/20231102/columna_lumbar_2_bl/t2w_tse_sag"

    loader = DicomLoaderMRI(path)
    dcm_list = loader.dicom_loader(loader.sorted_files)

    transform = TransformationMatrixMRI(dcm=dcm_list[0])
    print(transform.transformation_matrix)
    # print(transform.inverted_matrix)
    transform = TransformationMatrixMRI(dcm=dcm_list[1])
    print(transform.transformation_matrix)
    # print(transform.inverted_matrix)
