#  CT-Pattern

This is PyTorch implementation for the paper "ConfusionTree-Pattern: A Hierarchical Design for an Efficient and Performant Multi-Class Pattern
" (ICMLA 2021)



## Quickstart for experiments on ModelNet

### 1. Download multi-view images

We used the multi-view image dataset generated in [Kanezaki et al. 2018]

```
$ wget https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar; tar xvf modelnet40v2png_ori4.tar  
```
[Kanezaki et al. 2018] A. Kanezaki, Y. Matsushita, Yasuyuki and Y. Nishida. RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from            Unsupervised Viewpoints. CVPR 2018.

See (https://github.com/kanezaki/pytorch-rotationnet)



### 2. bold Create feature vectors (bottleneck tensors) for Transfer Learning

   ```
    $ python create_bottlenecks.py --gpu 0 
   ```


### 3. bold Build ConfusionTree

   ```
    $ python buildConfusionTree.py  
   ```

### 4. bold Train ConfusionTree-Pattern**

   ```
    $ python trainConfusionTreePattern.py --gpu 0 --ct "./MN40_ConfusionTree/" 
   ```

