import os
import subprocess
import sys
import unittest

import imageio
import numpy as np
from scipy import signal

import imageprocessing
import imageprocessing_sol

class TestImageProcessing(unittest.TestCase):

    def test_usage(self):
        result = subprocess.run(["python", "imageprocessing.py"],
                            stdout=subprocess.PIPE)
        usage = result.stdout.decode("UTF-8").strip()


        if os.name == 'nt':
            string = ("Usage:\r\n"
                      "    $ python imageprocessing.py <proc_type> " 
                      "<input_image> <output_image> "
                      "[strength (default=1)]")
        else:
            string =  ("Usage:\n"
                       "    $ python imageprocessing.py <proc_type> " 
                       "<input_image> <output_image> "
                       "[strength (default=1)]")


        self.assertEqual(usage, string)
        
    def test_option(self):
        result = subprocess.run(["python", "imageprocessing.py",
                                 "blr", "in.jpg","out.jpg"],
                            stdout=subprocess.PIPE)
        usage = result.stdout.decode("UTF-8").strip()

        string = "Processing type must be blur, sharpen, or bw"

        self.assertEqual(usage,string)

    def test_rgb_to_bw_func(self):

        x = np.random.randint(0,256,size=(100,100,3))

        y_submission = imageprocessing.rgb_to_bw(x)
        y_solution = imageprocessing_sol.rgb_to_bw(x)

        np.testing.assert_array_almost_equal(y_submission,
                                             y_solution)

    def test_conv_func(self):

        x = 100*np.random.random(size=(10,10))
        h = 100*np.random.random(size=(3,3))

        y_out = imageprocessing.conv_2d(x,h)
        y_sp = signal.convolve2d(x,h,"same")

        np.testing.assert_array_almost_equal(y_out,y_sp)

    def test_normalize_func(self):

        x = 400*(np.random.random(size=(200,200))-0.2)

        y_submission = imageprocessing.normalize(x)
        y_solution = imageprocessing_sol.normalize(x)

        np.testing.assert_array_almost_equal(y_submission,
                                             y_solution)

    def test_blur_func(self):

        x = np.random.randint(0,256,size=(200,200))

        y_submission = imageprocessing.blur(x)
        y_solution = imageprocessing_sol.blur(x)

        np.testing.assert_array_almost_equal(y_submission,
                                             y_solution)

    def test_sharpen_func(self):

        x = np.random.randint(0,256,size=(200,200))

        y_submission = imageprocessing.sharpen(x,2)
        y_solution = imageprocessing_sol.sharpen(x,2)

        np.testing.assert_array_almost_equal(y_submission,
                                             y_solution)

    def test_rgb_to_bw_use(self):

        temp_file = "tmp_bw.jpg"
        result = subprocess.run(["python", "imageprocessing.py",
                                 "bw", "tests/test.jpg",
                                 temp_file],
                                stdout=subprocess.PIPE)
        test_out = result.stdout.decode("UTF-8").strip()
        
        im_sub = imageio.imread(temp_file)
        im_sol = imageio.imread("tests/sol_bw.jpg")

        os.remove(temp_file)
        
        np.testing.assert_array_almost_equal(im_sub,
                                             im_sol)

        test_sol = "Processing successful on image size 800 x 1200"

        self.assertEqual(test_out, test_sol)

    def test_blur_use(self):

        temp_file = "tmp_blur.jpg"
        result = subprocess.run(["python", "imageprocessing.py",
                                 "blur", "tests/test.jpg",
                                 temp_file],
                                stdout=subprocess.PIPE)
        test_out = result.stdout.decode("UTF-8").strip()
        
        im_sub = imageio.imread(temp_file)
        im_sol = imageio.imread("tests/sol_blur.jpg")

        os.remove(temp_file)
        
        np.testing.assert_array_almost_equal(im_sub,
                                             im_sol)

        test_sol = "Processing successful on image size 800 x 1200"

        self.assertEqual(test_out, test_sol)

    def test_sharpen_use(self):

        temp_file = "tmp_sharpen.jpg"
        result = subprocess.run(["python", "imageprocessing.py",
                                 "sharpen", "tests/test.jpg", 
                                 temp_file, "2"],
                                stdout=subprocess.PIPE)
        test_out = result.stdout.decode("UTF-8").strip()
        
        im_sub = imageio.imread(temp_file)
        im_sol = imageio.imread("tests/sol_sharpen.jpg")

        os.remove(temp_file)
        
        np.testing.assert_array_almost_equal(im_sub,
                                             im_sol)

        test_sol = "Processing successful on image size 800 x 1200"

        self.assertEqual(test_out, test_sol)

        
if __name__ == '__main__':
    unittest.main()




