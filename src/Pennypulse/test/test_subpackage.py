import pennypulse as p
import unittest
from icecream import ic


class TestShapesSubpackage(unittest.TestCase):

    def test_import(self):
        func = p.shapes.constant(1)
        ic(func)