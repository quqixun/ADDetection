# Alzheimer's Disease Detection
# Segment brain into GM, WM and CSF.
# Author: Qixun QU
# Copyleft: MIT Licience

# To tun this script, FSL should be installed.
# See https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation.

#     ,,,         ,,,
#   ;"   ';     ;'   ",
#   ;  @.ss$$$$$$s.@  ;
#   `s$$$$$$$$$$$$$$$'
#   $$$$$$$$$$$$$$$$$$
#  $$$$P""Y$$$Y""W$$$$$
#  $$$$  p"$$$"q  $$$$$
#  $$$$  .$$$$$.  $$$$'
#   $$$DaU$$O$$DaU$$$'
#    '$$$$'.^.'$$$$'
#       '&$$$$$&'


from __future__ import print_function

import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
from multiprocessing import Pool, cpu_count


# Helper function to run in multiple processes
def unwarp_segment(arg, **kwarg):
    return ADDSegment._segment(*arg, **kwarg)


class ADDSegment(object):

    def __init__(self, ad_dir, nc_dir):
        '''___INIT__

            Initialization. Set directory for input data.
            The orangement of input images:
            - Project Root Directory
            --- data
            ----- adni_subj
            ------- AD (or NC) --> ad_dir (or nc_dir)
            --------- subject_id
            ----------- scan_no
            ------------- whole.nii.gz

            Inputs:
            -------

            - ad_dir: string, directory path of AD subjects.
            - nc_dir: string, directory path of NC subjects.

        '''

        self.input_dirs = [ad_dir, nc_dir]

        return

    def run(self, processes=-1):
        '''RUN

            Do segmentation of brains in multiple processes.
            For each brain image in "src_dir", segmentations
            are firstly saved in "temp" as "dst_dir."
            - Project Root Directory
            --- data
            ----- adni_subj
            ------- AD (or NC) --> ad_dir (or nc_dir)
            --------- subject_id
            ----------- scan_no --> src_dir
            ------------- whole.nii.gz
            ------------- temp --> dst_dir

            Input:
            ------

            - processes: int, number of processes,
                         if it is -1, all processors are available.

        '''

        # src_dirs contains directory of each scan
        # dst_dirs contains directory of temporary folders
        src_dirs, dst_dirs = [], []
        for input_dir in self.input_dirs:
            for subject in os.listdir(input_dir):
                subj_dir = os.path.join(input_dir, subject)
                for scan in os.listdir(subj_dir):
                    src_dir = os.path.join(subj_dir, scan)
                    dst_dir = os.path.join(src_dir, "temp")
                    src_dirs.append(src_dir)
                    dst_dirs.append(dst_dir)

        # Map a couple of parameters to self._segment
        paras = zip([self] * len(src_dirs), src_dirs, dst_dirs)
        if processes == -1:
            processes = cpu_count()
        pool = Pool(processes=processes)
        pool.map(unwarp_segment, paras)

        return

    def _segment(self, src_dir, dst_dir):
        '''_SEGMENT

            Call function for segmentation on input data.

            Inputs:
            -------

            - src_dir: string, path of scan's directory.
            - dst_dir: string, path of temporary folder.

        '''

        print("Segment on: ", src_dir)
        try:
            self.fast(src_dir, dst_dir)
        except RuntimeError:
            print("\tFalid on: ", src_dir)
        return

    @staticmethod
    def fast(src_dir, dst_dir):
        '''FAST

            Call FSL FAST to do segmentation, and move outputs
            from temporary folder to scan's folder.

            Inputs:
            -------

            - src_dir: string, path of scan's directory.
            - dst_dir: string, path of temporary folder.

        '''

        # Helper function to create temporary folder
        def create_dir(path):
            if not os.path.isdir(path):
                os.makedirs(path)
            return

        # Load file in .nii.gz
        def load_nii(path):
            nii = nib.load(path)
            return nii.get_data(), nii.get_affine()

        # Save numpy array to .nii.gz file
        def save_nii(data, path, affine):
            nib.save(nib.Nifti1Image(data, affine), path)
            return

        # file_name is "whole.nii.gz"
        file_name = os.listdir(src_dir)[0]
        src_path = os.path.join(src_dir, file_name)

        # Generate prefix of outputs
        create_dir(dst_dir)
        dst_prefix = os.path.join(dst_dir, file_name.split(".")[0])

        # Run command, set paramsters as explained in
        # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST
        command = ["fast", "-t", "1", "-n", "3", "-H", "0.1",
                   "-I", "1", "-l", "20.0", "-o", dst_prefix, src_path]
        subprocess.call(command, stdout=open(os.devnull),
                        stderr=subprocess.STDOUT)

        # Load whole brain
        volume, affine = load_nii(src_path)

        for scan in os.listdir(dst_dir):
            mask_path = os.path.join(dst_dir, scan)
            # Probabilistic segmentation of each tissuse has been obtained,
            # to extract tissue, multiply whole brain with segmentation
            if "pve_0" in scan:
                # Segmentation of CSF
                mask, _ = load_nii(mask_path)
                csf = np.multiply(volume, mask)
                dst_path = os.path.join(src_dir, "csf.nii.gz")
                save_nii(csf, dst_path, affine)
            elif "pve_1" in scan:
                # Segmentation of GM
                mask, _ = load_nii(mask_path)
                gm = np.multiply(volume, mask)
                dst_path = os.path.join(src_dir, "gm.nii.gz")
                save_nii(gm, dst_path, affine)
            elif "pve_2" in scan:
                # Segmentation of WM
                mask, _ = load_nii(mask_path)
                wm = np.multiply(volume, mask)
                dst_path = os.path.join(src_dir, "wm.nii.gz")
                save_nii(wm, dst_path, affine)

        # Remove temporary folder
        shutil.rmtree(dst_dir)

        return


if __name__ == "__main__":

    # Set directories for input data
    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", "adni_subj")
    ad_dir = os.path.join(data_dir, "AD")
    nc_dir = os.path.join(data_dir, "NC")

    # Use all processors for segmentaion
    seg = ADDSegment(ad_dir, nc_dir)
    seg.run(processes=-1)
