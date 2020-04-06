运行代码之前需要：
0. 安装kerastuner和autokeras(v1.0.0),可直接pip install autokeras==1.0.0（会自动安装kerastuner）
1. 将keras-tuner文件夹下的文件添加（替换）到包文件夹：site-packages/kerastuner/
2. 将auto-keras文件夹下的文件添加（替换）到包文件夹：site-packages/autokeras/
3. 修改site-packages/autokeras/__init__.py，添加：
    from autokeras.hypermodel.block import InceptionTimeBlock
    from autokeras.hypermodel.block import ResNetV2Block
    from autokeras.hypermodel.block import TemporalConvNetBlock
