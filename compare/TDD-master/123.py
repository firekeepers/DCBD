def NonMaxSuppression(self, proposal_bbox_inst, confi_thres=0.1, nms_thresh=0.45, proposal_type="roih"):
    if proposal_type == "roih":
        valid_map = proposal_bbox_inst.scores > confi_thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box  #actually no need valid_map
        # new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        # new_score = proposal_bbox_inst.scores[valid_map,:]
        # new_class = proposal_bbox_inst.pred_classes[valid_map,:]
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor
        new_score = proposal_bbox_inst.scores
        new_class = proposal_bbox_inst.pred_classes
        scores, index = new_score.sort(descending=True)
        keep_inds = []
        while (len(index) > 0):
            cur_inx = index[0]
            cur_score = scores[cur_inx]
            # if cur_score < confi_thres:
            #     break;
            keep = True
            for ind in keep_inds:
                current_bbox = new_bbox_loc[cur_inx]
                remain_box = new_bbox_loc[ind]
                iou = 1
                iou = self.box_iou_xyxy(current_bbox, remain_box)
                if iou > nms_thresh:
                    keep = False
                    break

            if keep:
                keep_inds.append(cur_inx)
            index = index[1:]
        # if len(keep_inds) == 0:
        #     valid_map = proposal_bbox_inst.scores > thres
        #
        #     # create instances containing boxes and gt_classes
        #     image_shape = proposal_bbox_inst.image_size
        #     new_proposal_inst = Instances(image_shape)
        #
        #     # create box
        #     new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        #     new_boxes = Boxes(new_bbox_loc)
        #
        #     # add boxes to instances
        #     new_proposal_inst.gt_boxes = new_boxes
        #     new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        #     new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]
        #     for i in new_proposal_inst.scores:
        #         i = 0
        #     return new_proposal_inst

        keep_inds = torch.tensor(keep_inds)
        score_nms = new_score[keep_inds.long()]
        score_nms = score_nms.reshape(-1, 1)
        # score_nms = score_nms.reshape(-1)
        box_nms = new_bbox_loc[keep_inds.long()]
        box_nms = box_nms.reshape(-1, 4)
        box_nms = Boxes(box_nms)
        class_nms = new_class[keep_inds.long()]
        class_nms = class_nms.reshape(-1, 1)
        new_proposal_inst.gt_boxes = box_nms
        new_proposal_inst.gt_classes = class_nms
        new_proposal_inst.scores = score_nms
