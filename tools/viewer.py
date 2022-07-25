import os
import os.path as osp
import numpy as np
from mmdet3d.core.visualizer.open3d_vis import Visualizer
import yaml
import argparse


# http://www1.ynao.ac.cn/~jinhuahe/know_base/othertopics/computerissues/RGB_colortable.htm
PALETTE = [[0, 255, 0],     # 绿色
           [0, 255, 255],   # 青色
           [255, 153, 18],  # 镉黄
           [255, 0, 255],   # 深红
           [3, 138, 158],   # 锰蓝
           [160, 32, 240],  # 紫色
           [255, 255, 255]] # 黑色

def show_result_meshlab(vis, 
                        data,
                        result,
                        out_dir=None,
                        gt_bboxes=None, 
                        score_thr=0.0,
                        snapshot=False):
    """Show 3D detection result by meshlab."""
    points = data
    pred_bboxes = result[:, :7]
    pred_labels = result[:, 7]
    pred_scores = result[:, 8]

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]
        pred_labels = pred_labels[inds]
    print('num_objects: {} after score_thr: {}'.format(pred_bboxes.shape[0], 
                                                       score_thr))

    vis.o3d_visualizer.clear_geometries()
    vis.add_points(points)
    if pred_bboxes is not None:
        if pred_labels is None:
            vis.add_bboxes(bbox3d=pred_bboxes)
        else:
            labelDict = {}
            for j in range(len(pred_labels)):
                i = int(pred_labels[j])
                if labelDict.get(i) is None:
                    labelDict[i] = []
                labelDict[i].append(pred_bboxes[j])
            for i in labelDict:
                vis.add_bboxes(
                    bbox3d=np.array(labelDict[i]),
                    bbox_color=([c / 255.0 for c in PALETTE[i]]))
    if gt_bboxes is not None:
        vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))

    ctr = vis.o3d_visualizer.get_view_control()
    ctr.set_lookat([0,0,0])
    ctr.set_front([-1,-1,1])    # 设置垂直指向屏幕外的向量
    ctr.set_up([0,0,1])         # 设置指向屏幕上方的向量
    ctr.set_zoom(0.2)

    vis.o3d_visualizer.poll_events()
    vis.o3d_visualizer.update_renderer()


def dataloader(cloud_path , boxes_path, load_dim):
    data = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1, load_dim])
    result = np.loadtxt(boxes_path).reshape(-1, 9)
    return result, data


parser = argparse.ArgumentParser()
parser.add_argument("--score_thr", type=float, default=0.1)
args = parser.parse_args()

def main():
    with open("bootstrap.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    result, data = dataloader(config['InputFile'], config['OutputFile'], config['LoadDim'])
    print(data.shape)

    # init visualizer
    vis = Visualizer(None)
    gt_bboxes = None
    # show the results
    show_result_meshlab(
        vis, 
        data,
        result,
        out_dir=None,
        gt_bboxes=gt_bboxes, 
        score_thr=args.score_thr,
        snapshot=False)
    vis.show()


if __name__ == "__main__":
    main()