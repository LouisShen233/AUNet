from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark


# import multiprocessing
# multiprocessing.set_start_method('spawn',True)

# python ../../tools/eval.py 	 \
# 	--tracker_path ./results \ # result path
# 	--dataset VOT2018        \ # dataset name
# 	--num 1 		 \             # number thread to eval
# 	--tracker_prefix 'model'   # tracker_name



def evaluation(dataset='VOT2018',tracker_prefix='DaSiamRPN',tracker_path='./results',num=4,show_video_level=True):
    

    tracker_dir = os.path.join(tracker_path)
    trackers = glob(os.path.join(tracker_path,
                                 tracker_prefix))
                                #  tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    num = min(num, len(trackers))

    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                         '../datasets'))
    root = '/home/lyuyu/dataset/'
    root = os.path.join(root, dataset)
    if 'OTB' in dataset:
        dataset = OTBDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=show_video_level)
    elif 'LaSOT' == dataset:
        dataset = LaSOTDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=show_video_level)
    elif 'UAV' in dataset:
        dataset = UAVDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=show_video_level)
    elif 'NFS' in dataset:
        dataset = NFSDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=show_video_level)
    elif dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset = VOTDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=show_video_level)
    elif 'VOT2018-LT' == dataset:
        dataset = VOTLTDataset(dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=show_video_level)

