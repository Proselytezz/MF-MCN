# -*- coding: utf-8 -*-
"""BraTS2023-GLI 预处理：原始 nii.gz -> 0-1 归一化 h5

流程与论文保持一致：脑区 bbox 裁剪 -> 1% 百分位裁剪 -> 0-1 归一化 -> 写入 h5。

远端重跑示例（Linux）：
  python brats_preprocessing.py \
      --src /root/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
      --dst /root/data/BraTS2023_GLI_01scale \
      --list pretrain_path_list.txt --workers 8
"""
import argparse
import os
import sys
import numpy as np
import SimpleITK as sitk
import h5py

# 模态名 -> 原始文件名后缀
MODALITY_SUFFIX = [
    ('t1', '-t1n.nii.gz'),
    ('t1ce', '-t1c.nii.gz'),
    ('flair', '-t2f.nii.gz'),
    ('t2', '-t2w.nii.gz'),
]


def brain_bbox(data_t1, data_t1ce, data_t2, data_flair, gt, status="train"):
    mask = (data_t1 != 0)
    brain_voxels = np.where(mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    t1_bboxed = data_t1[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    if data_t1ce is not None:
        t1ce_bboxed = data_t1ce[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    else:
        t1ce_bboxed = None
    if data_t2 is not None:
        t2_bboxed = data_t2[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    else:
        t2_bboxed = None
    if data_flair is not None:
        flair_bboxed = data_flair[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    else:
        flair_bboxed = None
    if status == 'train':
        gt_bboxed = gt[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
        return t1_bboxed, t1ce_bboxed, t2_bboxed, flair_bboxed, gt_bboxed
    if status == 'test':
        tmp_t1 = np.zeros_like(data_t1)
        tmp_t1ce = np.zeros_like(data_t1ce)
        tmp_t2 = np.zeros_like(data_t2)
        tmp_flair = np.zeros_like(data_flair)
        return t1_bboxed, t1ce_bboxed, t2_bboxed, flair_bboxed, tmp_t1, tmp_t1ce, tmp_t2, tmp_flair, \
            [minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx]


def volume_bounding_box(data, gt, expend=0, status="train"):
    data, gt = brain_bbox(data, gt)
    print(data.shape)
    mask = (gt != 0)
    brain_voxels = np.where(mask != 0)
    z, x, y = data.shape
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    minZidx_jitterd = max(minZidx - expend, 0)
    maxZidx_jitterd = min(maxZidx + expend, z)
    minXidx_jitterd = max(minXidx - expend, 0)
    maxXidx_jitterd = min(maxXidx + expend, x)
    minYidx_jitterd = max(minYidx - expend, 0)
    maxYidx_jitterd = min(maxYidx + expend, y)

    data_bboxed = data[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
    print([minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx])
    print([minZidx_jitterd, maxZidx_jitterd,
           minXidx_jitterd, maxXidx_jitterd, minYidx_jitterd, maxYidx_jitterd])

    if status == "train":
        gt_bboxed = np.zeros_like(data_bboxed, dtype=np.uint8)
        gt_bboxed[expend:maxZidx_jitterd - expend, expend:maxXidx_jitterd -
                  expend, expend:maxYidx_jitterd - expend] = 1
        return data_bboxed, gt_bboxed

    if status == "test":
        gt_bboxed = gt[minZidx_jitterd:maxZidx_jitterd,
                       minXidx_jitterd:maxXidx_jitterd, minYidx_jitterd:maxYidx_jitterd]
        return data_bboxed, gt_bboxed


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out = out.astype(np.float32)
    return out


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        high_boundary = cdf[1][cdf[0] <= (1 - self.percent)][-1]
        low_boundary = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, low_boundary, high_boundary)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())


def list_cases(src):
    """枚举原始数据目录下的病例，按名称排序（与训练代码 sorted(glob) 一致）"""
    return sorted(
        d for d in os.listdir(src)
        if d.startswith('BraTS-GLI-') and os.path.isdir(os.path.join(src, d))
    )


def read_list(path):
    """读取路径列表，兼容 'BraTS-GLI-xxxxxx-xxx,label' 这种带标签的行"""
    # utf-8-sig 兼容 Windows 记事本/PowerShell 写出的带 BOM 文件
    with open(path, 'r', encoding='utf-8-sig') as f:
        return [line.strip().split(',')[0].strip() for line in f if line.strip()]


def process_case(src, dst, uid, write_nii=False):
    """处理单个病例，返回 (uid, 是否成功, 说明)"""
    try:
        imgs = {}
        for name, suffix in MODALITY_SUFFIX:
            imgs[name] = sitk.ReadImage(os.path.join(src, uid, uid + suffix))
        lab_img = sitk.ReadImage(os.path.join(src, uid, uid + '-seg.nii.gz'))

        data = {k: sitk.GetArrayFromImage(v) for k, v in imgs.items()}
        lab = sitk.GetArrayFromImage(lab_img)

        img_t1, img_t1ce, img_t2, img_flair, img_lab = brain_bbox(
            data['t1'], data['t1ce'], data['t2'], data['flair'], lab, 'train')

        img_t1 = MedicalImageDeal(img_t1, percent=0.01).valid_img
        img_t1ce = MedicalImageDeal(img_t1ce, percent=0.01).valid_img
        img_t2 = MedicalImageDeal(img_t2, percent=0.01).valid_img
        img_flair = MedicalImageDeal(img_flair, percent=0.01).valid_img

        # 0-1 scale norm（与论文一致）
        img_t1 = MedicalImageDeal(img=img_t1).norm_img
        img_t1ce = MedicalImageDeal(img=img_t1ce).norm_img
        img_t2 = MedicalImageDeal(img=img_t2).norm_img
        img_flair = MedicalImageDeal(img=img_flair).norm_img

        save_dir = os.path.join(dst, uid)
        os.makedirs(save_dir, exist_ok=True)

        with h5py.File(os.path.join(save_dir, uid + 'h5file.h5'), 'w') as h5file:
            h5file['t1'] = img_t1
            h5file['t1ce'] = img_t1ce
            h5file['t2'] = img_t2
            h5file['flair'] = img_flair
            h5file['seg'] = img_lab

        if write_nii:
            # 预训练代码需要读取排序第一例的 _t1.nii.gz 获取空间信息
            ref = imgs['t1']
            out = {
                uid + '_t1.nii.gz': img_t1,
                uid + '_t1ce_MNI.nii.gz': img_t1ce,
                uid + '_t2_MNI.nii.gz': img_t2,
                uid + '_flair_MNI.nii.gz': img_flair,
                uid + '_lab.nii.gz': img_lab,
            }
            for fn, arr in out.items():
                im = sitk.GetImageFromArray(arr)
                im.SetOrigin(ref.GetOrigin())
                im.SetDirection(ref.GetDirection())
                im.SetSpacing(ref.GetSpacing())
                sitk.WriteImage(im, os.path.join(save_dir, fn))

        return uid, True, ''
    except Exception as e:  # 单例失败不中断整体预处理
        return uid, False, repr(e)


def _worker(job):
    return process_case(job[0], job[1], job[2], job[3])


def main():
    parser = argparse.ArgumentParser(description='BraTS2023-GLI nii.gz -> h5 预处理')
    parser.add_argument('--src', required=True, help='原始数据根目录（含 BraTS-GLI-* 子目录）')
    parser.add_argument('--dst', required=True, help='h5 输出根目录')
    parser.add_argument('--list', default=None, help='只处理列表内的病例（可选，txt 每行病例名）')
    parser.add_argument('--workers', type=int, default=1, help='并行进程数')
    parser.add_argument('--mni-cases', type=int, default=4,
                        help='为排序前 N 例额外写出 nii.gz 参考文件')
    parser.add_argument('--no-nii', action='store_true', help='完全不写出 nii.gz')
    args = parser.parse_args()

    all_cases = list_cases(args.src)
    if not all_cases:
        sys.exit('未在 %s 下找到 BraTS-GLI-* 目录' % args.src)

    if args.list:
        wanted = [c for c in read_list(args.list) if c]
        missing = [c for c in wanted if c not in set(all_cases)]
        if missing:
            sys.exit('列表中 %d 例在原始数据中不存在，例如 %s' % (len(missing), missing[:3]))
    else:
        wanted = list(all_cases)

    # 预训练代码会读取排序第一例的 t1.nii.gz，该例可能不在预训练列表内，必须一并处理
    first_case = all_cases[0]
    if first_case not in wanted:
        wanted.append(first_case)

    wanted = sorted(set(wanted))
    if args.no_nii:
        nii_uids = set()
    else:
        nii_uids = set(wanted[:max(args.mni_cases, 0)]) | {first_case}

    os.makedirs(args.dst, exist_ok=True)
    jobs = [(args.src, args.dst, uid, uid in nii_uids) for uid in wanted]
    total = len(jobs)
    print('待处理 %d 例，并行 %d 进程，需写 nii.gz: %s' % (total, args.workers, sorted(nii_uids)))

    ok = 0
    failed = []
    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            uid, success, msg = _worker(job)
            if success:
                ok += 1
            else:
                failed.append((uid, msg))
            print('[%d/%d] %s %s' % (i, total, uid, 'ok' if success else 'FAILED: ' + msg))
    else:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            for i, (uid, success, msg) in enumerate(pool.imap_unordered(_worker, jobs), 1):
                if success:
                    ok += 1
                else:
                    failed.append((uid, msg))
                    print('[%d/%d] %s FAILED: %s' % (i, total, uid, msg))
                if i % 10 == 0 or i == total:
                    print('进度 %d/%d，成功 %d，失败 %d' % (i, total, ok, len(failed)))

    print('完成：成功 %d，失败 %d' % (ok, len(failed)))
    if failed:
        for uid, msg in failed:
            print('  失败 %s: %s' % (uid, msg))
        sys.exit(1)


if __name__ == '__main__':
    main()
