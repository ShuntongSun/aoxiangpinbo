# aoxiangpinbo
## 算法挑战赛
### 源代码链接
链接：https://github.com/ShuntongSun/aoxiangpinbo


### 训练日志

完整的日志可以参考百度网盘链接：链接: https://pan.baidu.com/s/1nMEaKworCAdNvwjmvE6qmg?pwd=da9z 提取码: da9z 


### 训练参数设置

运行dataparam.py可以得到test.db全部的参数结果：以下是我们搜索的过程
    batch_size = trial.suggest_int('batch_size', 32, 128, log=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    lr = trial.suggest_float('lr', 1e-6, 5e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    warmup_epochs = trial.suggest_int('warmup_epochs', 3, 15)
    initial_lr = trial.suggest_float('initial_lr', 1e-5, 1e-3, log=True)
    max_lr = trial.suggest_float('max_lr', 5e-4, 5e-2, log=True)

    dropout_rate = trial.suggest_float('dropout', 0.3, 0.6)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])

五十组最好的结果如下所示：
(112, 11, 'batch_size', 56.0, '{"name": "IntDistribution", "attributes": {"log": true, "step": 1, "low": 32, "high": 128}}')
(113, 11, 'lr', 8.666316159670824e-06, '{"name": "FloatDistribution", "attributes": {"step": null, "low": 1e-06, "high": 0.05, "log": true}}')
(114, 11, 'weight_decay', 0.00012665571850056475, '{"name": "FloatDistribution", "attributes": {"step": null, "low": 1e-05, "high": 0.001, "log": true}}')
(115, 11, 'warmup_epochs', 11.0, '{"name": "IntDistribution", "attributes": {"log": false, "step": 1, "low": 3, "high": 15}}')
(116, 11, 'initial_lr', 1.5286181767633018e-05, '{"name": "FloatDistribution", "attributes": {"step": null, "low": 1e-05, "high": 0.001, "log": true}}')
(117, 11, 'max_lr', 0.0009401108103759114, '{"name": "FloatDistribution", "attributes": {"step": null, "low": 0.0005, "high": 0.05, "log": true}}')
(118, 11, 'dropout', 0.30641381017661967, '{"name": "FloatDistribution", "attributes": {"step": null, "low": 0.3, "high": 0.6, "log": false}}')
(119, 11, 'optimizer', 2.0, '{"name": "CategoricalDistribution", "attributes": {"choices": ["Adam", "SGD", "RMSprop"]}}')
(120, 11, 'scheduler', 0.0, '{"name": "CategoricalDistribution", "attributes": {"choices": ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]}}')

### 项目完整说明
使用了TEGCN进行网络构建

#### 环境配置

创建conda环境
conda create pytorch -n python==3.7
环境配置
pip install -r requirements.txt
打开目录
cd aoxiangpinbo/
#### 数据位置
apxiangpinbo/data/
主办方的数据目录相同

#### 运行方式
terminal
python AllSearch.py

#### 结果复现
数据集A结果为：Top1 Acc: 69.90%
