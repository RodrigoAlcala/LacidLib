import numpy as np


class ImagePlaneIntersector:
    def __init__(self, dcm1, dcm2) -> None:
        self.normal_vector_1, self.image_position_1 = self.normal_plane_vector(dcm1)
        self.normal_vector_2, self.image_position_2 = self.normal_plane_vector(dcm2)

        self.plane_eq_vector_1 = self.get_plane_eq(
            self.normal_vector_1, self.image_position_1
        )
        self.plane_eq_vector_2 = self.get_plane_eq(
            self.normal_vector_2, self.image_position_2
        )
        self.line_director, self.line_point = self.find_intersection_line(
            self.plane_eq_vector_1, self.plane_eq_vector_2
        )
        self.point1, self.point2 = self.calculate_line_points(
            self.line_director, self.line_point
        )

    def normal_plane_vector(self, dcm):
        """This method takes a dmc obj and calculates the normal vector of the
        imaging plane

        Args:
            dcm (pydicom obj): Take a pydicom object from dcmread method.

        Returns:
            NDarray[floats]: Returns the normal vector of a dicom slice.
            NDarray[floats]: Returns the ImagePositionPatient dicom tag.
        """
        # Extract Image Position (0020,0032) and Image Orientation (0020,0037) tags
        image_position = dcm.ImagePositionPatient
        image_orientation = dcm.ImageOrientationPatient

        # Calculate the director vectors
        director_vector_1 = np.array(image_orientation[:3])
        director_vector_2 = np.array(image_orientation[3:])

        # Calculates the normal vector then normalizes
        normal_vector = np.cross(director_vector_1, director_vector_2)
        norm = np.linalg.norm(normal_vector)
        normal_vector_normalized = normal_vector / norm

        image_position = (np.array(image_position)).reshape(3, 1)

        return normal_vector_normalized.reshape(3, 1), image_position

    def get_plane_eq(self, normal_vector, point_in_plane):
        dot = -1 * (np.dot(normal_vector.flatten(), point_in_plane.flatten()))
        plane_eq_vector = np.vstack((normal_vector, [dot]))
        return plane_eq_vector.reshape(4, 1)

    def find_intersection_line(self, plane_eq1, plane_eq2):
        """
        Find the intersection line of two planes.

        Args:
            plane_eq1 (np.array): Equation of the first plane.
            plane_eq2 (np.array): Equation of the second plane.

        Returns:
            tuple: A tuple containing the director and point of the intersection line.
        """
        # Extract the normal vectors of the two planes
        normal_vector1 = plane_eq1[:-1, :]
        normal_vector2 = plane_eq2[:-1, :]

        # Create a normal vector for the z-axis
        normal_vector_z = (np.array([0, 1, 0])).reshape((3, 1))

        # Concatenate the normal vectors to form the coefficient matrix
        coef = (np.hstack((normal_vector1, normal_vector2, normal_vector_z))).T

        # Concatenate the constants of the plane equations
        conts = np.hstack((plane_eq1[-1], plane_eq2[-1], [0]))

        # Compute the director of the intersection line
        line_director = (
            np.cross(normal_vector1.flatten(), normal_vector2.flatten())
        ).reshape((3, 1))

        # Normalize the director vector
        line_director = line_director / np.linalg.norm(line_director)

        # Compute the point of the intersection line
        line_point = (np.linalg.solve(coef, conts)).reshape((3, 1))

        return line_director, line_point

    def diagonal_crop(self, array, point1, point2, crop_below=True):
        """
        This function crops a 2D array diagonally between two points.

        Parameters:
        array (numpy.ndarray): The 2D array to be cropped.
        point1 (tuple): The first point of the diagonal.
        point2 (tuple): The second point of the diagonal.
        crop_below (bool, optional): If True, the area below the diagonal is cropped.
                                    If False, the area above the diagonal is cropped.
                                    Defaults to True.

        Returns:
        numpy.ndarray: The cropped 2D array.
        """
        # Initialize a mask with the same shape as the input array
        mask = np.zeros(array.shape)

        # Iterate over the array
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                # Calculate the cross product of the vector from point1 to the current point
                # and the vector from point1 to point2
                cross_product = (point2[1] - point1[1]) * (i - point1[0]) - (
                    point2[0] - point1[0]
                ) * (j - point1[1])

                # If the cross product is greater than 0, the current point is below the diagonal
                # Set the corresponding mask value to 1
                if cross_product > 0:
                    mask[i, j] = 1

        # If crop_below is True, the area below the diagonal is cropped
        # If crop_below is False, the area above the diagonal is cropped
        if crop_below:
            cropped_array = array * mask
        else:
            cropped_array = array * (1 - mask)

        return cropped_array

    def calculate_line_points(self, director, point):
        """
        This function calculates the line points based on the director and point provided.
        It takes two arguments: director and point.
        The director is sliced to get the first two elements.
        The point is also sliced to get the first two elements.
        These two points are then added together to form a line.
        The function returns the two points that form the line.
        """
        # Slice the director to get the first two elements
        director = director[:3]

        # Slice the point to get the first two elements
        point1 = point[:3]

        # Add the two points together to form a line
        point2 = director + point1

        # Return the two points that form the line
        return point1, point2


if __name__ == "__main__":
    pass
