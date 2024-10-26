## 翱翔拼搏



**本项目参考**

+ [liujf69/ICMEW2024-Track10: [ICMEW 2024\] Implementation of the paper “HDBN: A Novel Hybrid Dual-branch Network for Robust Skeleton-based Action Recognition“.](https://github.com/liujf69/ICMEW2024-Track10/tree/main)
+ 下面的mixformer
+ 网路结构使用的是mixformer
+ 数据集采用本次比赛给的train_joint.npy和train_label.npy
+ **特别强调，在对训练集和测试集A进行初步数据处理时，我们将数据的输出存放在mixGCN的dataset里面**
+ **因此，在使用mixformer时，后续进行测试集B的集成，我们的测试集依旧在mixGCN的dataset里面**

**由于整个数据集太大了，给出整个项目的百度网盘链接：**

**通过百度网盘分享的文件：翱翔拼搏**
**链接：https://pan.baidu.com/s/1-8urpSfaJ1k1TUuxGx1UQw?pwd=pxgj** 
**提取码：pxgj** 



+ 下面将叙述整个实验流程：

### 硬件参数

```
uname -a
Linux axzc-System-Product-Name 5.15.0-122-generic #132~20.04.1-Ubuntu SMP Fri Aug 30 15:50:07 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
```

+ 显卡3090

+ cuda版本12.4

### 环境配置

按照顶级仓库的环境配置，这里在代码目录下可以参考requirements：

```
vi requirements.txt
# 可以查看上面的环境是否有出入
# 建立新的conda环境，
conda create -n mix_GCN python==3.10.13
conda activate mix_GCN
# 检查requirements
```

### 数据集准备

#### 数据集处理

1. 使用官方的train_joint.npy和train_label.npy，结合gen_model.py生成对应模态的数据集。

+ 将train_joint.npy和train_label.npy，test_A_label.npy放到data目录下
+ 将赛题给的test_A_joint.npy和省赛的B数据集也放到data目录下
+ 在项目根目录下里面使用gen_model.py
+ 注意在使用testA和testB时需更改gen_data.py里面的名字

```shell
3.数据集处理出bone模态数据（可选）：运行python gen_modal.py --modal bone得到bone模态数据
4.数据集处理出motion模态数据（可选）：运行python gen_modal.py --modal motion得到motion模态的数据
5.bone模态与joint模态合并（可选）：运行python gen_modal.py --modal jmb得到合并模态的数据。
```

2. 将在data目录下看见

+ joint，bone，bone-motion，joint-motion相关的A和B训练集和测试集

#### 生成符合顶级仓库的npz文件

+ 首先我们修改了源码以下文件
+ /ICMEW2024-Track10-main/Process_data/extract_2dpose.py

##### 生成对应A的npz文件-还含有label

+ 这一步是验证mix的有效性所以先采用**含有label的训练集A**作为mix的实验对象

```
# 在顶级仓库下
cd Process_data/
# 找到  extract_2dpose.py
python extract_2dpose.py
```

+ 生成的.npz训练集和测试集A在ICMEW2024-Track10-main/Model_inference/Mix_GCN/dataset里面



##### 生成对应B的npz文件

+ **因为赛方没给label，所以我们使用如下方式构造.npz文件**

```
# 在顶级仓库下找到npytonpz.py
# 这个是我自己写的
python npytonpz.py
```

+ 生成的.npz测试集B也在根目录data里面



### 开始训练

#### 代码微调

1. ICMEW2024-Track10-main/Model_inference/Mix_Former/feeders/feeder_uav.py

如下所改：在第69行删掉了transpose方法：

```
 self.data = self.data
```

2. ICMEW2024-Track10-main/Model_inference/Mix_Former/model/ske_mixf.py

如下所改：在第59行in_channels变为3

```
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
```

#### 修改配置文件

我们修改了：

+ ICMEW2024-Track10-main/Model_inference/Mix_Former/config 文件夹下的
+ mixformer_V2_BM.yaml
   mixformer_V2_B.yaml
   mixformer_V2_JM.yaml
   mixformer_V2_J.yaml
+ **我们对每个文件里面的训练集和测试集的路径进行了配置**

```
test_feeder_args:
  bone: false
  data_path: /home/axzc/ICMEW2024-Track10-main/data/test_joint_B.npz
  debug: false
  p_interval:
  - 0.95
  split: test
  vel: false
  window_size: 64
train_feeder_args:
  bone: false
  data_path: /home/axzc/ICMEW2024-Track10-main/Model_inference/Mix_GCN/dataset/save_2d_pose/train.npz
  debug: false
  normalization: false
  p_interval:
  - 0.5
  - 1
```

#### 正式训练

+ 先进入/ICMEW2024-Track10-main/Model_inference/Mix_Former/
+ 使用终端
+ 输入命令行：

```
python main.py --config ./config/mixformer_V2_BM.yaml --device 0
python main.py --config ./config/mixformer_V2_B.yaml --device 0
python main.py --config ./config/mixformer_V2_JM.yaml --device 0
python main.py --config ./config/mixformer_V2_J.yaml --device 0
```

+ 依次得到对应模态的pt权重
+ **对应的输出目录分别为，请更改自己对应的目录位置**

```
/ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_B/
/ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_BM/
/ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_J/
/ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_JM/
```

**这里面同时由训练日志、测试分数pkl、pt文件组成**



### 搜索最好权重

#### 搜索权重

我们使用下面命令搜索四个模型的集成权重：

在该目录下：**/ICMEW2024-Track10-main/search_best_weights.py**

**这是以下一个例子，请记住更改自己的目录**

```
python search_best_weights.py --dataset csv3 --joint-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_J/ --bone-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_B/ --JM-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_JM/ --BM-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_BM/ --optimize --n-calls 300
```

**最后输出最好权重**

#### 输出测试集B的最好分数

**这里我们直接使用日志里最好的模型权重文件作为命令行展示**

```
python main.py --config ./config/mixformer_V2_J.yaml --phase test --save-score True --weights /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_J/runs-56-7168.pt --device 0
python main.py --config ./config/mixformer_V2_JM.yaml --phase test --save-score True --weights /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_JM/runs-66-8448.pt --device 0
python main.py --config ./config/mixformer_V2_B.yaml --phase test --save-score True --weights /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_B/runs-53-6784.pt --device 0
python main.py --config ./config/mixformer_V2_BM.yaml --phase test --save-score True --weights /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_BM/runs-55-7040.pt --device 0
```

**这里加载的是测试集B，生成了对应的最好模型下的pkl分数文件**



### 集成

找到/ICMEW2024-Track10-main/Mix.py

**请注意后面的【a,b,c,d】**是自己通过上面搜索的最好权重决定的，每次搜索可能不一样，取决于上面搜索的轮数

```
python Mix.py --dataset csv3 --joint-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_J/ --bone-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_B/ --JM-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_JM/ --BM-dir /ICMEW2024-Track10-main/Model_inference/Mix_Former/output/skmixf__V2_BM/ --weights [a,b,c,d]
```

运行结束后会生成对应的npy置信度文件，B-MIX………………

```
    output_file = f'B-MIX_{arg.dataset}.npy'
```

最后修改名字pred.npy用于评估





