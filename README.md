# K-Access
pybullet implementation of icra 2022 paper on a1, Accessibility-Based Clustering for Efficient Learning of Locomotion Skills

`main.py` to train, `test.py` to test. To change the initial state distribution, change the `self.configs` in `bittleenv.py`.

Only 300 poses are used for accessibility estimation and clustering, which takes 6 hours on my laptop. Still it is effective, and I combine it with the 9 predefiend poses. To show the great efficacy, clustering 1000 poses should work. Codes are in the `/k-access_preprocess` folder.

Learning Curves: The proposed initial state distribution (red) outperforms the random initial states (blue).
![Screenshot from 2022-09-02 17-29-10](https://user-images.githubusercontent.com/54518250/188116361-471e6934-5690-4195-82f1-a875c63b51f6.png)

citation:
```
@INPROCEEDINGS{kaccess,  
author={Zhang, Chong and Yu, Wanming and Li, Zhibin},  
booktitle={2022 International Conference on Robotics and Automation (ICRA)},   
title={Accessibility-Based Clustering for Efficient Learning of Locomotion Skills},   
year={2022},  
pages={1600-1606},  
doi={10.1109/ICRA46639.2022.9812113}}
```

SAC implementation: https://github.com/pranz24/pytorch-soft-actor-critic
