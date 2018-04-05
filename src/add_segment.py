from __future__ import print_function

import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
from multiprocessing import Pool, cpu_count


def unwarp_segment(arg, **kwarg):
    return ADDSegment._segment(*arg, **kwarg)


class ADDSegment(object):

    def __init__(self, ad_dir, nc_dir):
        self.input_dirs = [ad_dir, nc_dir]
        return

    def run(self, processes=-1):

        src_dirs, dst_dirs = [], []
        for input_dir in self.input_dirs:
            for subject in os.listdir(input_dir):
                subj_dir = os.path.join(input_dir, subject)
                for scan in os.listdir(subj_dir):
                    src_dir = os.path.join(subj_dir, scan)
                    dst_dir = os.path.join(src_dir, "temp")
                    src_dirs.append(src_dir)
                    dst_dirs.append(dst_dir)

        paras = zip([self] * len(src_dirs),
                    src_dirs, dst_dirs)
        if processes == -1:
            processes = cpu_count()
        pool = Pool(processes=processes)
        pool.map(unwarp_segment, paras)

        return

    def _segment(self, src_dir, dst_dir):
        print("Segment on: ", src_dir)
        try:
            self.fast(src_dir, dst_dir)
        except RuntimeError:
            print("\tFalid on: ", src_dir)
        return

    @staticmethod
    def fast(src_dir, dst_dir):
        def create_dir(path):
            if not os.path.isdir(path):
                os.makedirs(path)
            return

        def load_nii(path):
            nii = nib.load(path)
            return nii.get_data(), nii.get_affine()

        def save_nii(data, path, affine):
            nib.save(nib.Nifti1Image(data, affine), path)
            return

        file_name = os.listdir(src_dir)[0]
        src_path = os.path.join(src_dir, file_name)

        create_dir(dst_dir)
        dst_prefix = os.path.join(dst_dir, file_name.split(".")[0])

        command = ["fast", "-t", "1", "-n", "3", "-H", "0.1",
                   "-I", "1", "-l", "20.0", "-o", dst_prefix, src_path]
        subprocess.call(command, stdout=open(os.devnull),
                        stderr=subprocess.STDOUT)
        # dst_dir = os.path.dirname(dst_prefix)
        volume, affine = load_nii(src_path)
        for scan in os.listdir(dst_dir):
            mask_path = os.path.join(dst_dir, scan)
            if "pve_0" in scan:
                mask, _ = load_nii(mask_path)
                csf = np.multiply(volume, mask)
                dst_path = os.path.join(src_dir, "csf.nii.gz")
                save_nii(csf, dst_path, affine)
            elif "pve_1" in scan:
                mask, _ = load_nii(mask_path)
                gm = np.multiply(volume, mask)
                dst_path = os.path.join(src_dir, "gm.nii.gz")
                save_nii(gm, dst_path, affine)
            elif "pve_2" in scan:
                mask, _ = load_nii(mask_path)
                wm = np.multiply(volume, mask)
                dst_path = os.path.join(src_dir, "wm.nii.gz")
                save_nii(wm, dst_path, affine)
        shutil.rmtree(dst_dir)

        return


if __name__ == "__main__":

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, "data", "ADNI")
    ad_dir = os.path.join(data_dir, "AD")
    nc_dir = os.path.join(data_dir, "NC")

    seg = ADDSegment(ad_dir, nc_dir)
    seg.run(processes=6)
