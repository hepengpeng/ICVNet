import torch
from utils import binary_tree
import torch.nn.functional as F


def update_tree_with_pred_label(b_tree, pred_label, depth_start, depth_end, stage_id, with_grad=False, no_detach=False):
    if not with_grad:
        with torch.no_grad():
            indicator = torch.ones_like(pred_label)

            if stage_id == 0 or stage_id == 3:
                direction = pred_label
            elif stage_id == 1:
                direction = pred_label - 7
            elif stage_id == 2:
                direction = pred_label - 1

            b_tree = binary_tree.update_tree(b_tree, indicator, direction)
            
            if depth_start.dim() == 3:
                if depth_start.shape[1] != pred_label.shape[1] or pred_label.shape[2] != pred_label.shape[2]:
                    depth_start = torch.unsqueeze(depth_start, 1)
                    depth_start = F.interpolate(depth_start, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                    depth_start = torch.squeeze(depth_start, 1)

                    depth_end = torch.unsqueeze(depth_end, 1)
                    depth_end = F.interpolate(depth_end, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                    depth_end = torch.squeeze(depth_end, 1)
            elif depth_start.dim() == 1:
                depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
                depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)

            # In the k-th stage, the bin width is R/(D×2^(k−1)), i.e. R/(2^(k+3)) for D=64
            depth_range = depth_end - depth_start
            next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 5))
            next_interval = depth_range / next_interval_num
            depthmap_list = []

            ## Section 3.3: the search bin movement solution
            # if stage_id == 0:
            #     b_tree = b_tree.type(torch.float32)
            #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=3.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 4.5)
            #     for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
            #         tmp_key0 = bin_index * 2.0 + i - 8
            #         tmp_key1 = bin_index * 2.0 + i - 7
            #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            #     b_tree = b_tree.type(torch.int64)
            # elif stage_id == 1:
            #     b_tree = b_tree.type(torch.float32)
            #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=0.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 1.5)
            #     for i in range(1, 5):  # update the next depth hypothesis d(3,i) for the 4 bins in the 3-th iteration
            #         tmp_key0 = bin_index * 2.0 + i - 2
            #         tmp_key1 = bin_index * 2.0 + i - 1
            #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            #     b_tree = b_tree.type(torch.int64)
            # else:  # update the next depth hypothesis d(4,i)/d(5,i) for the 2 bins in the 4/5-th iteration
            #     for i in range(1, 3):
            #         tmp_key0 = b_tree[:, 1, :, :] * 2.0 + i - 1
            #         tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
            #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

            # Section 3.3: the max-min truncation solution
            if stage_id == 0:
                for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 8, 0)
                    tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                    tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 7, 0)
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            elif stage_id == 1:
                for i in range(1, 5):  # update the next depth hypothesis d(3,i) for the 4 bins in the 3-th iteration
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                    tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                    tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            else:  # update the next depth hypothesis d(4,i)/d(5,i) for the 2 bins in the 4/5-th iteration
                for i in range(1, 3):
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                    tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                    tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i, 0)
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            curr_depth = torch.stack(depthmap_list, 1)
    else:
        
        indicator = torch.ones_like(pred_label)

        if stage_id == 0 or stage_id == 3:
            direction = pred_label
        elif stage_id == 1:
            direction = pred_label - 7
        elif stage_id == 2:
            direction = pred_label - 1

        b_tree = binary_tree.update_tree(b_tree, indicator, direction)
        
        if depth_start.dim() == 3:
            if depth_start.shape[1] != pred_label.shape[1] or pred_label.shape[2] != pred_label.shape[2]:
                depth_start = torch.unsqueeze(depth_start, 1)
                depth_start = F.interpolate(depth_start, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                depth_start = torch.squeeze(depth_start, 1)

                depth_end = torch.unsqueeze(depth_end, 1)
                depth_end = F.interpolate(depth_end, size=[pred_label.shape[1], pred_label.shape[2]], mode="nearest")
                depth_end = torch.squeeze(depth_end, 1)
        elif depth_start.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)

        # In the k-th stage, the bin width is R/(D×2^(k−1)), i.e. R/(2^(k+3)) for D=64
        depth_range = depth_end - depth_start
        next_interval_num = (2.0 ** (b_tree[:, 0, :, :] + 5))
        next_interval = depth_range / next_interval_num
        depthmap_list = []

        ## Section 3.3: the search bin movement solution
        # if stage_id == 0:
        #     b_tree = b_tree.type(torch.float32)
        #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=3.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 4.5)
        #     for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
        #         tmp_key0 = bin_index * 2.0 + i - 8
        #         tmp_key1 = bin_index * 2.0 + i - 7
        #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        #     b_tree = b_tree.type(torch.int64)
        # elif stage_id == 1:
        #     b_tree = b_tree.type(torch.float32)
        #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=0.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 1.5)
        #     for i in range(1, 5):  # update the next depth hypothesis d(3,i) for the 4 bins in the 3-th iteration
        #         tmp_key0 = bin_index * 2.0 + i - 2
        #         tmp_key1 = bin_index * 2.0 + i - 1
        #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        #     b_tree = b_tree.type(torch.int64)
        # else:  # update the next depth hypothesis d(4,i)/d(5,i) for the 2 bins in the 4/5-th iteration
        #     for i in range(1, 3):
        #         tmp_key0 = b_tree[:, 1, :, :] * 2.0 + i - 1
        #         tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
        #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

        # Section 3.3: the max-min truncation solution
        if stage_id == 0:
            for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 8, 0)
                tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 7, 0)
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        elif stage_id == 1:
            for i in range(1, 5):  # update the next depth hypothesis d(3,i) for the 4 bins in the 3-th iteration
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        else:  # update the next depth hypothesis d(4,i) for the 2 bins in the 4-th iteration
            for i in range(1, 3):
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i, 0)
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

        curr_depth = torch.stack(depthmap_list, 1)
    if no_detach:
        return curr_depth, b_tree
    else:
        return curr_depth.detach(), b_tree.detach()


def get_gt_label_bin_edge(gt_depth_img, b_tree, depth_start, depth_end, stage_id):
    with torch.no_grad():
        if depth_start.dim() == 1 or depth_end.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        bin_edge_list = []
        if stage_id == 0:  # compute the 65 edges in the 1th search stage
            bin_edge_list.append(torch.zeros_like(gt_depth_img) + depth_start)
            depth_range = depth_end - depth_start
            interval = depth_range / 64.0
            for i in range(1, 65):  # 1+64 edges
                bin_edge_list.append(bin_edge_list[0] + interval * i)
            num_bin = 64
        elif stage_id == 1:  # compute the 17 edges in the 2th search stage
            depth_range = depth_end - depth_start
            interval_num = (2.0 ** (b_tree[:, 0, :, :] + 5.0))
            interval = depth_range / interval_num
            for i in range(1, 18):  # 17 edges
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 8.0, 0)
                tmp_key = torch.minimum(tmp_key, interval_num)
                bin_edge_list.append(interval * tmp_key + depth_start)
            num_bin = 16
        elif stage_id == 2:  # update the 5 edges in the 3th search stage
            depth_range = depth_end - depth_start
            interval_num = (2.0 ** (b_tree[:, 0, :, :] + 5.0))
            interval = depth_range / interval_num
            bin_edge_list = []
            for i in range(1, 6):  # 5 edges
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                tmp_key = torch.minimum(tmp_key, interval_num)
                bin_edge_list.append(interval * tmp_key + depth_start)
            num_bin = 4
        elif stage_id == 3:  # update the 3 edges in the 4th search stage
            depth_range = depth_end - depth_start
            interval_num = (2.0 ** (b_tree[:, 0, :, :] + 5.0))
            interval = depth_range / interval_num
            bin_edge_list = []
            for i in range(1, 4):  # 3 edges
                tmp_key = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key = torch.minimum(tmp_key, interval_num)
                bin_edge_list.append(interval * tmp_key + depth_start)
            num_bin = 2
        gt_label = torch.zeros(gt_depth_img.size(), dtype=torch.int64, device=gt_depth_img.device) - 1 
        for i in range(num_bin):
            bin_mask = torch.ge(gt_depth_img, bin_edge_list[i])
            bin_mask = torch.logical_and(bin_mask, torch.lt(gt_depth_img, bin_edge_list[i + 1]))
            gt_label[bin_mask] = i
        bin_mask = (gt_label != -1)
        return gt_label, bin_mask


def update_tree_with_depthmap(depth_img, tree_depth, depth_start, depth_end, stage_id, scale_factor=1.0,
                              mode='bilinear', with_grad=False, no_detach=False):
    if not with_grad:
        with torch.no_grad():
            if scale_factor != 1.0:
                depth_img = torch.unsqueeze(depth_img, 1)
                depth_img = F.interpolate(depth_img, scale_factor=scale_factor, mode=mode)
                depth_img = torch.squeeze(depth_img, 1)
            B, H, W = depth_img.shape
            b_tree = torch.zeros([B, 2, H, W], dtype=torch.int64, device=depth_img.device)
            b_tree[:, 0, :, :] = b_tree[:, 0, :, :] + tree_depth

            if depth_start.dim() == 3:
                if depth_start.shape[1] != depth_img.shape[1] or depth_start.shape[2] != depth_img.shape[2]:
                    depth_start = torch.unsqueeze(depth_start, 1)
                    depth_start = F.interpolate(depth_start, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                    depth_start = torch.squeeze(depth_start, 1)

                    depth_end = torch.unsqueeze(depth_end, 1)
                    depth_end = F.interpolate(depth_end, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                    depth_end = torch.squeeze(depth_end, 1)
            elif depth_start.dim() == 1:
                depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
                depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)

            depth_range = depth_end - depth_start
            d_interval = depth_range / (2.0 ** (tree_depth + 4))
            b_tree[:, 1, :, :] = (torch.floor((depth_img - depth_start) / d_interval)).type(torch.int64)
            b_tree[:, 1, :, :] = torch.clamp(b_tree[:, 1, :, :], min=0, max=2 ** (tree_depth + 4)-1)

            next_interval_num = torch.tensor(2.0 ** (tree_depth + 5), device=depth_img.device)
            next_interval = depth_range / next_interval_num
            depthmap_list = []

            ## Section 3.3: the search bin movement solution
            # if stage_id == 0:
            #     b_tree = b_tree.type(torch.float32)
            #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=3.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 4.5)
            #     for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
            #         tmp_key0 = bin_index * 2.0 + i - 8
            #         tmp_key1 = bin_index * 2.0 + i - 7
            #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            #     b_tree = b_tree.type(torch.int64)
            # elif stage_id == 1:
            #     b_tree = b_tree.type(torch.float32)
            #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=0.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 1.5)
            #     for i in range(1, 5):  # update the next depth hypothesis d(3,i) for the 4 bins in the 3-th iteration
            #         tmp_key0 = bin_index * 2.0 + i - 2
            #         tmp_key1 = bin_index * 2.0 + i - 1
            #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            #     b_tree = b_tree.type(torch.int64)
            # else:  # update the next depth hypothesis d(4,i)/d(5,i) for the 2 bins in the 4/5-th iteration
            #     for i in range(1, 3):
            #         tmp_key0 = b_tree[:, 1, :, :] * 2.0 + i - 1
            #         tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
            #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

            # Section 3.3: the max-min truncation solution
            if stage_id == 0:
                for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 8, 0)
                    tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                    tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 7, 0)
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            elif stage_id == 1:
                for i in range(1, 5):  # update the next depth hypothesis d(2,i) for the 4 bins in the 3-th iteration.
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                    tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                    tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            elif stage_id == 2:
                for i in range(1, 3):  # update the next depth hypothesis d(2,i) for the 2 bins in the 4-th iteration.
                    tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                    tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                    tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i, 0)
                    tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                    depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
            curr_depth = torch.stack(depthmap_list, 1)
    else:
        if scale_factor != 1.0:
            depth_img = torch.unsqueeze(depth_img, 1)
            depth_img = F.interpolate(depth_img, scale_factor=scale_factor, mode=mode)
            depth_img = torch.squeeze(depth_img, 1)
        B, H, W = depth_img.shape
        b_tree = torch.zeros([B, 2, H, W], dtype=torch.int64, device=depth_img.device)
        b_tree[:, 0, :, :] = b_tree[:, 0, :, :] + tree_depth

        if depth_start.dim() == 3:
            if depth_start.shape[1] != depth_img.shape[1] or depth_start.shape[2] != depth_img.shape[2]:
                depth_start = torch.unsqueeze(depth_start, 1)
                depth_start = F.interpolate(depth_start, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                depth_start = torch.squeeze(depth_start, 1)

                depth_end = torch.unsqueeze(depth_end, 1)
                depth_end = F.interpolate(depth_end, size=[depth_img.shape[1], depth_img.shape[2]], mode="nearest")
                depth_end = torch.squeeze(depth_end, 1)
        elif depth_start.dim() == 1:
            depth_start = torch.unsqueeze(torch.unsqueeze(depth_start, 1), 1)
            depth_end = torch.unsqueeze(torch.unsqueeze(depth_end, 1), 1)
        
        depth_range = depth_end - depth_start

        d_interval = depth_range / (2.0 ** (tree_depth + 4))
        b_tree[:, 1, :, :] = (torch.floor((depth_img - depth_start) / d_interval)).type(torch.int64)
        b_tree[:, 1, :, :] = torch.clamp(b_tree[:, 1, :, :], min=0, max=2 ** (tree_depth + 4) - 1)

        next_interval_num = torch.tensor(2.0 ** (tree_depth + 5), device=depth_img.device)
        next_interval = depth_range / next_interval_num
        depthmap_list = []

        ## Section 3.3: the search bin movement solution
        # if stage_id == 0:
        #     b_tree = b_tree.type(torch.float32)
        #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=3.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 4.5)
        #     for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
        #         tmp_key0 = bin_index * 2.0 + i - 8
        #         tmp_key1 = bin_index * 2.0 + i - 7
        #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        #     b_tree = b_tree.type(torch.int64)
        # elif stage_id == 1:
        #     b_tree = b_tree.type(torch.float32)
        #     bin_index = torch.clamp(b_tree[:, 1, :, :], min=0.5, max=2.0 ** (b_tree[0, 0, 0, 0] + 4.0) - 1.5)
        #     for i in range(1, 5):  # update the next depth hypothesis d(3,i) for the 4 bins in the 3-th iteration
        #         tmp_key0 = bin_index * 2.0 + i - 2
        #         tmp_key1 = bin_index * 2.0 + i - 1
        #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        #     b_tree = b_tree.type(torch.int64)
        # else:  # update the next depth hypothesis d(4,i)/d(5,i) for the 2 bins in the 4/5-th iteration
        #     for i in range(1, 3):
        #         tmp_key0 = b_tree[:, 1, :, :] * 2.0 + i - 1
        #         tmp_key1 = b_tree[:, 1, :, :] * 2.0 + i
        #         depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)

        # Section 3.3: the max-min truncation solution
        if stage_id == 0:
            for i in range(1, 17):  # update the next depth hypothesis d(2,i) for the 16 bins in the 2-th iteration.
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 8, 0)
                tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 7, 0)
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        elif stage_id == 1:
            for i in range(1, 5):  # update the next depth hypothesis d(2,i) for the 4 bins in the 3-th iteration.
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 2, 0)
                tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        elif stage_id == 2:
            for i in range(1, 3):  # update the next depth hypothesis d(2,i) for the 2 bins in the 4-th iteration.
                tmp_key0 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i - 1, 0)
                tmp_key0 = torch.minimum(tmp_key0, next_interval_num)
                tmp_key1 = torch.clamp_min(b_tree[:, 1, :, :] * 2.0 + i, 0)
                tmp_key1 = torch.minimum(tmp_key1, next_interval_num)
                depthmap_list.append(next_interval * (tmp_key0 + tmp_key1) / 2.0 + depth_start)
        curr_depth = torch.stack(depthmap_list, 1)
    if no_detach:
        return curr_depth, b_tree
    else:
        return curr_depth.detach(), b_tree.detach()
