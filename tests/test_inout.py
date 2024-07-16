import copy
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import h5py
import numpy as np

from cactus.main.inout import OutputCosmicWeb


class OutputCosmicWebTest(unittest.TestCase):

    def setUp(self) -> None:
        self.prefix = 'test_'
        self.rank = 0
        self.output_data = np.arange(25).reshape((-1, 5))
        self.cautun_output_data = copy.deepcopy(self.output_data)
        self.cautun_output_data[self.cautun_output_data > 0] += 1
        self.header_bytes = 1048

        self.instance = OutputCosmicWeb(prefix=self.prefix,
                                        rank=self.rank,
                                        output_data=self.output_data)

        # self.npz_filename = 'None'
        self.hdf5_filename = 'None'
        self.cautun_nexus_filename = 'None'

        # return super().setUp()

    @patch("numpy.savez")
    def test_save_npz(self, mock_savez):
        """Check OutputCosmicWeb saves the npz file correclty.
        """
        self.instance.save_npz()
        expected_filename = self.prefix + str(self.rank) + ".npz"
        mock_savez.assert_called_once_with(expected_filename,
                                           web_flag=self.output_data)

    # @patch("h5py.File", new_callable=MagicMock)
    # def test_save_hdf5(self, mock_h5py_file):
    #     self.instance.save_hdf5()
    #     expected_filename = self.prefix + str(self.rank) + ".hdf5"
    #     mock_h5py_file.assert_called_once_with(expected_filename, 'w')
    #     mock_h5py_file().create_dataset.assert_called_once_with(
    #         name="web_flag", data=self.output_data)

    # def tearDown(self) -> None:
    #     # os.remove(self.npz_filename)
    #     os.remove(self.hdf5_filename)
    #     os.remove(self.cautun_nexus_filename)
    #     return super().tearDown()

    def test_init(self) -> None:
        """Check that OutputCosmicWeb initializes correctly.
        """
        # Check that OutputCosmicWeb initializes correctly
        self.assertEqual(self.instance.filename_prefix,
                         self.prefix + str(self.rank))
        self.assertIsNone(
            np.testing.assert_array_equal(self.instance.output_data,
                                          self.output_data))
        return None

    # def test_save_npz(self):
    #     self.npz_filename = self.prefix + str(self.rank) + ".npz"

    #     self.instance.save_npz()
    #     os.path.isfile(self.npz_filename)
    #     return None

    def test_save_hdf5(self):
        """Check that OutputCosmicWeb saves the HDF5 file correctly.
        """
        expected_filename = self.prefix + str(self.rank) + ".hdf5"

        # Write temporary HDF5 file
        self.instance.save_hdf5()

        self.assertTrue(os.path.isfile(expected_filename))

        # Read data back in and compare with original
        with h5py.File(expected_filename, 'r') as f:
            data_from_file = f["web_flag"][()]
        self.assertIsNone(
            np.testing.assert_array_equal(data_from_file, self.output_data))

        # Remove temporary file
        os.remove(expected_filename)
        return None

    def test_save_cautun_nexus_c_order(self):
        """Check that OutputCosmicWeb saves Cautun-style, 'C'-ordered
            NEXUS outputs correctly.
        """
        expected_filename = self.prefix + str(self.rank) + ".MMF"
        array_order = 'C'

        # Write temporary binary file
        self.instance.save_cautun_nexus(header_bytes=self.header_bytes,
                                        array_order=array_order)

        self.assertTrue(os.path.isfile(expected_filename))

        # Read data back in
        data_from_file = np.fromfile(
            expected_filename, dtype=np.ushort).reshape(-1, order=array_order)

        # Identify start and end points of the header and data arrays
        data_start_idx = int(self.header_bytes / data_from_file.itemsize)
        data_end_idx = int(data_start_idx + self.cautun_output_data.nbytes)

        # Read and check header data
        header_data = data_from_file[:data_start_idx]
        expected_header_data = np.empty(self.header_bytes //
                                        data_from_file.itemsize,
                                        dtype=np.ushort)
        expected_header_data[:] = 256
        self.assertIsNone(
            np.testing.assert_array_equal(header_data, expected_header_data))

        # Read and check flag data
        output_data_from_file = data_from_file[
            data_start_idx:data_end_idx].reshape((-1, 5))
        self.assertIsNone(
            np.testing.assert_array_equal(output_data_from_file,
                                          self.cautun_output_data))

        # Remove temporary file
        os.remove(expected_filename)
        return None

    def test_save_cautun_nexus_fortran_order(self):
        """Check that OutputCosmicWeb saves Cautun-style,
            'F'ortran-ordered NEXUS outputs correctly.
        """
        expected_filename = self.prefix + str(self.rank) + ".MMF"
        array_order = 'F'

        # Write temporary binary file
        self.instance.save_cautun_nexus(header_bytes=self.header_bytes,
                                        array_order=array_order)

        self.assertTrue(os.path.isfile(expected_filename))

        # Read data back in
        data_from_file = np.fromfile(
            expected_filename, dtype=np.ushort).reshape(-1, order=array_order)

        # Identify start and end points of the header and data arrays
        data_start_idx = int(self.header_bytes / data_from_file.itemsize)
        data_end_idx = int(data_start_idx + self.cautun_output_data.nbytes)

        # Read and check header data
        header_data = data_from_file[:data_start_idx]
        expected_header_data = np.empty(self.header_bytes //
                                        data_from_file.itemsize,
                                        dtype=np.ushort)
        expected_header_data[:] = 256
        self.assertIsNone(
            np.testing.assert_array_equal(header_data, expected_header_data))

        # Read and check flag data
        output_data_from_file = data_from_file[
            data_start_idx:data_end_idx].reshape((-1, 5))
        self.assertIsNone(
            np.testing.assert_array_equal(output_data_from_file,
                                          self.cautun_output_data))

        # Remove temporary file
        os.remove(expected_filename)
        return None
