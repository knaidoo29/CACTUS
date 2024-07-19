import unittest


class InitializationTests(unittest.TestCase):

    def test_initialization(self):
        """
        Check the test suite runs by affirming 2+2=4
        """
        self.assertEqual(2 + 2, 4)

        return None

    def test_import(self):
        """
        Ensure the test suite can import the module
        """
        try:
            import cactus
        except ImportError:
            self.fail("Was not able to import {}".format("cactus"))

        return None
