import numpy as np

class NumpyTest():


    #region Private Methods

    @staticmethod
    def image2vector(image):
        """
        Argument:
        image -- a numpy array of shape (length, height, depth)

        Returns:
        v -- a vector of shape (length*height*depth, 1)
        """

        # (≈ 1 line of code)

        # YOUR CODE STARTS HERE
        v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
        # YOUR CODE ENDS HERE

        return v

    @staticmethod
    def normalize_rows(x):
        """
        Implement a function that normalizes each row of the matrix x (to have unit length).

        Argument:
        x -- A numpy matrix of shape (n, m)

        Returns:
        x -- The normalized (by row) numpy matrix. You are allowed to modify x.
        """

        # (≈ 2 lines of code)
        # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
        # x_norm =
        # Divide x by its norm.

        # YOUR CODE STARTS HERE
        x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        x=x/x_norm
        # YOUR CODE ENDS HERE

        return x_norm

    @staticmethod
    def softmax(x):
        """Calculates the softmax for each row of the input x.

            Your code should work for a row vector and also for matrices of shape (m,n).

            Argument:
            x -- A numpy matrix of shape (m,n)

            Returns:
            s -- A numpy matrix equal to the softmax of x, of shape (m,n)
            """
        # YOUR CODE STARTS HERE
        # (≈ 3 lines of code)
        # Apply exp() element-wise to x. Use np.exp(...).
        exp_x = np.exp(x)

        # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
        x_sum = np.sum(exp_x, axis=1, keepdims=True)

        # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
        s = exp_x / x_sum
        # YOUR CODE ENDS HERE

        return s

    #endregion

    @staticmethod
    def vectorize_array_test():
        # This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
        t_image = np.array([[[0.67826139, 0.29380381],
                             [0.90714982, 0.52835647],
                             [0.4215251, 0.45017551]],

                            [[0.92814219, 0.96677647],
                             [0.85304703, 0.52351845],
                             [0.19981397, 0.27417313]],

                            [[0.60659855, 0.00533165],
                             [0.10820313, 0.49978937],
                             [0.34144279, 0.94630077]]])

        NumpyTest.image2vector(t_image)

    @staticmethod
    def normalize_rows_test():
        x = np.array([[0., 3., 4.],
                      [1., 6., 4.]])

        x_norm=NumpyTest.normalize_rows(x)

    @staticmethod
    def softmax_test():
        t_x = np.array([[9, 2, 5, 0, 0],
                        [7, 5, 0, 0, 0]])
        NumpyTest.softmax(t_x)


