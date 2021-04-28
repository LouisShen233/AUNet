import cv2
import os
import time
import pdb
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import dataloader
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
# from dasiamrpn.tracker import SiamRPNTracker
from trackers import tracker_builder
# from trackers.updatenet_tracker import SiamTrackerUpdateNet, SiamTrackerPredictNet, SiamTrackerpaPredictNet
from trackers.updatenet_tracker import SiamTrackerUpdateNet, SiamTrackerAAUNet, SiamTrackerAAUNetv2
# from dasiamrpn.utils import get_axis_aligned_bbox
from updatenet.utils_upd import get_axis_aligned_bbox
from bin.eval import evaluation

import matplotlib.pyplot as plt

def test_all_model(cfg, dirmanager):
    model_dir = dirmanager.updmod_checkpoint_dir
    dirs = sorted([x[0] for x in os.walk(model_dir)][1:])
    if len(dirs)==0:
        # files = sorted(os.listdir(model_dir))[:-1]
        files = sorted(os.listdir(model_dir))
        for subfile in files:
            cfg["UPDATE"]["CHECKPOINT_PATH"] = os.path.join(model_dir, subfile)
            test_update(cfg, dirmanager)
        return
    dirs = chunks(dirs, cfg["TEST"]["PARALLEL"])[cfg["TEST"]["GROUP"]]
    files = [x[2] for x in os.walk(model_dir)][1:][0]
    for subdir in dirs:
        for subfile in files:
            whole_path = os.path.join(subdir, subfile)
            if os.path.exists(whole_path) == False:
                print("The model %s is not exist." %(whole_path))
                continue
            cfg["UPDATE"]["CHECKPOINT_PATH"] = os.path.join(subdir, subfile)
            test_update(cfg, dirmanager)

    # cfg["UPDATE"]["CHECKPOINT_PATH"] = './results/SiamRPNBIG-PredictNetv2/run_12/upd_mod/step_1/checkpoints/checkpoint4.pth.tar'
    # test_update(cfg, dirmanager)
    return

def test_update(cfg, dirmanager):
    print(cfg["TEST"]["DATASET"]+' stage'+str(cfg["TEMPLATE"]["STEP"]))
    
    # load config

    # if cfg["TEST"]["DATASET"] == 'UAV123':
    #     dataset_root = '/home/lyuyu/dataset/UAV123/data_seq/UAV123/'
    # else:
    dataset_root = '/home/lyuyu/dataset/'+ cfg["TEST"]["DATASET"]
    
    #  tracker
    model_path = cfg["MODEL"]["CHECKPOINT_PATH"]
    torch.cuda.set_device(cfg["TEST"]["GPU_ID"])
    # load tracker and updatenet 
    tracker = tracker_builder.build_tracker(cfg)
    # update_path='./updatenet/checkpoint/checkpoint40.pth.tar'
    update_path = cfg["UPDATE"]["CHECKPOINT_PATH"]

    step = cfg["TEST"]["TYPE"]
    gpu_id=cfg["TEST"]["GPU_ID"]
    if cfg["UPDATE"]["MODEL"][:8] == "AAUNetv2":
            tracker = SiamTrackerAAUNetv2(cfg,tracker,update_path,gpu_id,step)
    elif cfg["UPDATE"]["MODEL"] == "UpdateNet":
            tracker = SiamTrackerUpdateNet(cfg,tracker,update_path,gpu_id,step)#1=dasiamrpn; 2 linear; 3 updatenet
    else:
        raise NotImplementedError
    # create dataset
    dataset = DatasetFactory.create_dataset(name=cfg["TEST"]["DATASET"],
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = tracker.name
    model_name = update_path[63:-7].replace('/','').replace('.','')
    if step == 4:
        model_name = 'updatenet2016'
    elif step == 1:
        model_name = 'dasiamrpn'

    if cfg["TEST"]["DATASET"] in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        total_lost=0
        #for v_idx, video in enumerate(dataset):
        if cfg["TEST"]["CLS_TYPE"] != 0:
            total_success_list = []
            total_iou_list = []
        for video in tqdm(dataset):
            # if args.video != '':
            #     # test one special video
            #     if video.name != args.video:
            #         continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            if cfg["TEST"]["CLS_TYPE"] != 0:
                iou_list = []
                success_list = []
            state=dict()
            for idx, (img, gt_bbox) in enumerate(video):
               # print(idx)
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:

                    state=tracker.init(img, np.array(gt_bbox))
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    pred_bbox = [cx-w/2, cy-h/2, w, h]
                    pred_bboxes.append(1)
                    if cfg["TEST"]["CLS_TYPE"] != 0:
                        iou_list.append(1)
                        success_list.append(1)
                elif idx > frame_counter:
                    # state = tracker.update(img, np.array(gt_bbox)) 
                    state = tracker.update(img) 
                    pos=state['target_pos'] # cx, cy
                    sz=state['target_sz']   # w, h
                    pred_bbox=np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])
                    #pred_bbox=np.array([pos[0]+1-(sz[0]-1)/2, pos[1]+1-(sz[1]-1)/2, sz[0], sz[1]])

                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    # iou = overlap_ratio(gt_bbox, pred_bbox)
                    if cfg["TEST"]["CLS_TYPE"] != 0:
                        if cfg["TEST"]["CLS_TYPE"] == 1:
                            if overlap > cfg["UPDATE"]["IOU_THRES"]:
                                iou = 1
                            else:
                                iou = 0
                        iou_list.append(iou)
                        success_list.append(state['success'])
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    if cfg["TEST"]["CLS_TYPE"] != 0:
                        iou_list.append(0)
                        success_list.append(0)
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if cfg["TEST"]["VISUALIZATION"] and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            if cfg["TEST"]["CLS_TYPE"] != 0:
                total_success_list = total_success_list + success_list
                total_iou_list = total_iou_list + iou_list
                success_list = np.array(success_list)
                iou_list = np.array(iou_list)
                
                # total accuracy & detect failure accuracy
                accuracy = np.mean(success_list == iou_list)
                index0 = np.argwhere(iou_list == 0)
                accuracy0 = np.mean(success_list[index0] == iou_list[index0])
                print(video.name,accuracy, accuracy0)
            toc /= cv2.getTickFrequency()
            # save results
            if cfg["SOLVER"]["LR_POLICY"] == 'epochwise_step_group':
                lr_type = cfg["UPDATE"]["CHECKPOINT_PATH"].split('/')[-2]
            elif cfg["SOLVER"]["LR_POLICY"] == 'cosine':
                lr_type = 'cosine'
            else:
                lr_type = 'undefined'
            if cfg["TEST"]["TYPE"] == 1:
                lr_type = 'base_dasiamrpn'
            video_path = os.path.join(dirmanager.updmod_res_dir, cfg["TEST"]["DATASET"], lr_type, model_name, 'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            # print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
            #         v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        if cfg["TEST"]["CLS_TYPE"] == 1:
            total_success_list = np.array(total_success_list)
            total_iou_list = np.array(total_iou_list)
            
            # total accuracy & detect failure accuracy
            accuracy = np.mean(total_success_list == total_iou_list)
            index0 = np.argwhere(total_iou_list == 0)
            accuracy0 = np.mean(total_success_list[index0] == total_iou_list[index0])
            print('total accuracy',accuracy, accuracy0)
       # print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking    
        #for v_idx, video in enumerate(dataset):
        if cfg["TEST"]["CLS_TYPE"] != 0:
            total_success_list = []
            total_iou_list = []
        for video in tqdm(dataset):
            # if args.video != '':
            #     # test one special video
            #     if video.name != args.video:
            #         continue
            
            toc = 0
            pred_bboxes = []
            if cfg["TEST"]["CLS_TYPE"] != 0:
                iou_list = []
                success_list = []
            scores = []
            track_times = []
            state=dict()
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
            
                    state=tracker.init(img, np.array(gt_bbox))#注意gt_bbox和gt_bbox_的区别
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    pred_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                   
                    scores.append(None)
                    if 'VOT2018-LT' == cfg["TEST"]["DATASET"]:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                        if cfg["TEST"]["CLS_TYPE"] != 0:
                            iou_list.append(1)
                            success_list.append(1)
                    # if video.name == 'Jogging-1':
                    #     template_vis(state['z_f_cur'], 0, 'template_vis_'+str(idx))
                else:
                    state = tracker.update(img) 

                    # if video.name == 'Jogging-1':
                    #     template_vis(state['z_f_cur'], 0, 'template_vis_'+str(idx))
                        
                    pos=state['target_pos']
                    sz=state['target_sz']
                    pred_bbox=np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])
                    
                    pred_bboxes.append(pred_bbox)
                    #scores.append(outputs['best_score'])
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if cfg["TEST"]["CLS_TYPE"] != 0:
                        if cfg["TEST"]["CLS_TYPE"] == 1:
                            if overlap > 0.1:
                                iou = 1
                            else:
                                iou = 0
                        if cfg["TEST"]["CLS_TYPE"] == 2:
                            iou = overlap
                        iou_list.append(iou)
                        success_list.append(state['success'].cpu().numpy())
                        
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if cfg["TEST"]["VISUALIZATION"] and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            if cfg["TEST"]["CLS_TYPE"] != 0:
                total_success_list = total_success_list + success_list
                total_iou_list = total_iou_list + iou_list
                success_list = np.array(success_list)
                iou_list = np.array(iou_list)
                
                if cfg["TEST"]["CLS_TYPE"] == 1:
                # total accuracy & detect failure accuracy
                    accuracy = np.mean(success_list == iou_list)
                    index0 = np.argwhere(iou_list == 0)
                    index1 = np.argwhere(iou_list == 1)
                    accuracy0 = np.mean(success_list[index0] == iou_list[index0])
                    accuracy1 = np.mean(success_list[index1] == iou_list[index1])
                if cfg["TEST"]["CLS_TYPE"] == 2:
                # total accuracy & detect failure accuracy
                    comp_list = abs(success_list - iou_list)<0.2
                    accuracy = np.mean(comp_list)                
                    index0 = np.argwhere((success_list - iou_list)>0)
                    index1 = np.argwhere((iou_list - success_list)>0)
                    accuracy0 = np.mean(comp_list[index0])
                    accuracy1 = np.mean(comp_list[index1])
                print(video.name,accuracy, accuracy0, accuracy1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == cfg["TEST"]["DATASET"]:
                video_path = os.path.join('results', cfg["TEST"]["DATASET"], model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == cfg["TEST"]["DATASET"]:
                video_path = os.path.join('results', cfg["TEST"]["DATASET"], model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                lr_type = 'cosine'
                video_path = os.path.join(dirmanager.updmod_res_dir, cfg["TEST"]["DATASET"], lr_type, model_name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                class_path = os.path.join(video_path, '{}_cls.txt'.format(video.name))
                with open(class_path, 'w') as f:
                    for x in success_list:
                        f.write(str(x)+'\n')
           # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            #    v_idx+1, video.name, toc, idx / toc))
        total_success_list = np.array(total_success_list)
        total_iou_list = np.array(total_iou_list)
        
        # total accuracy & detect failure accuracy
        if cfg["TEST"]["CLS_TYPE"] == 1:
            accuracy = np.mean(total_success_list == total_iou_list)
            index0 = np.argwhere(total_iou_list == 0)
            accuracy0 = np.mean(total_success_list[index0] == total_iou_list[index0])
        if cfg["TEST"]["CLS_TYPE"] == 2:
            comp_list = abs(total_success_list - total_iou_list)<0.2
            accuracy = np.mean(comp_list)
            index0 = np.argwhere((total_success_list - total_iou_list)>0)
            index1 = np.argwhere((total_iou_list - total_success_list)>0)
            accuracy0 = np.mean(comp_list[index0])
            accuracy1 = np.mean(comp_list[index1])
        print('total accuracy',accuracy, accuracy0)
    # evaluation(cfg["TEST"]["DATASET"], model_name, dirmanager.updmod_res_dir)
    evaluation(cfg["TEST"]["DATASET"], model_name, os.path.join(dirmanager.updmod_res_dir, cfg["TEST"]["DATASET"], lr_type))
    return

def chunks(arr, m):
    n = int(np.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

    
def template_vis(template, dim = 0, outfile = 'template_vis'):
    # input: template-HxWxC
    # fig = plt.figure()
    size = template.shape
    img = template.cpu().numpy()[0,dim,:,:]
    plt.imshow(img, cmap = 'jet') #cmap-RGB
    plt.axis('off')

    plt.savefig(outfile +'.jpg')

if __name__ == '__main__':
    # evaluation('OTB100', 'checkpoint12', 'results/SiamRPNBIG-AAUNetv2_4/run_21/upd_mod/step_1/results/OTB100/cosine')
    evaluation('UAV123', 'tscheckpoint2', 'results/SiamRPNBIG-AAUNetv2_7_1/run_43/upd_mod/step_1/results/UAV123/cosine')
    # evaluation('VOT2016', 'UpdateNet', './results/VOT2016/')