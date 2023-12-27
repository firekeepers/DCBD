This is the PyTorch implementation of our paper:
**Domain Collaborative Bridging Detector For Archaeological Shipwreck**

![image-20231227153442616](C:\Users\China\AppData\Roaming\Typora\typora-user-images\image-20231227153442616.png)

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
python train_net.py --config ./configs/faster_rcnn_R101_cross_city_res_change.yaml --num-gpus 2
#DCB:
python train_net_cb.py --config ./configs/faster_RCNN_city_cb.yaml --num-gpus 2 MODEL.WEIGHTS_DP="$your weight of DRD.pth$" MODEL.WEIGHTS_DS="$your weight of DID.pth$"
```

