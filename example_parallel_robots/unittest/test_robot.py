import unittest
import sys

# import os
# script_dir = os.path.dirname(__file__)
# mymodule_dir = os.path.join(script_dir, "../example_parallel_robots")
# sys.path.append(mymodule_dir)

from example_parallel_robots.loader_tools import load


class TestRobotLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.nameRobotToTest = ['5bar', '5bar3d', '5bar6d', 'cassie_leg', 'digit_leg', 'digit_2legs', 'disney_leg', "kangaroo_leg",
                           "kangaroo_2legs", "delta", "talos_full_closed", "talos_full_open", "talos_leg", "talos_2legs", "wl16_leg"]

    def test_robot(self):
        for name in self.nameRobotToTest:
            test=True
            try:
                load(name)
            except :
                test=False
            self.assertTrue(test)

if __name__ == '__main__':
    unittest.main()