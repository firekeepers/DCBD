This is the PyTorch implementation of our paper:
**Domain Collaborative Bridging Detector For Archaeological Shipwreck**

![image](framework.jpg)

train:

```
python train.py --config ./configs/faster_rcnn_R101_cross_city_res_change.yaml --num-gpus 2
```

resume:

```
python train.py --config ./configs/faster_rcnn_R101_cross_city_res_change.yaml --resume --num-gpus 2
```

test:

```
python train.py --config ./configs/faster_rcnn_R101_cross_city_res_change.yaml --resume --eval-only
```

train DCBD:

```
#the first stage:
#train DID 
python train_net.py --config ./configs/faster_rcnn_R101_cross_city_res_change.yaml --num-gpus 2
#train DRD
#1/Delete GRL module and IDCC module, retrain the detector
python train_net.py --config ./configs/faster_rcnn_R101_cross_city_res_change.yaml --num-gpus 2

#DCB:
#fusing precition from DID and DRD to train DAD, DS and DP means DID and DRD actually
python train_net_cb.py --config ./configs/faster_RCNN_city_cb.yaml --num-gpus 2 MODEL.WEIGHTS_DP="$your weight of DRD.pth$" MODEL.WEIGHTS_DS="$your weight of DID.pth$"
```

