# Jigsaw预训练模型目录

此目录需要放置Jigsaw异常检测模型的预训练权重文件。

## 获取模型权重

从Unified_Jigsaw项目复制预训练权重：

```bash
# ShanghaiTech数据集模型（推荐）
cp /home/zhaoyi/media/data3/Unified_Jigsaw/pre_trained/stc_78.76_sample7.pth ./

# 其他可用模型：
# - avenue_92.18.pth  (Avenue数据集)
# - ped2_98.89.pth    (Ped2数据集)
# - ub_65.11_sample7.pth (UBnormal数据集)
```

## 模型说明

| 模型文件 | 数据集 | AUC | 说明 |
|---------|--------|-----|------|
| stc_78.76_sample7.pth | ShanghaiTech | 78.76% | 通用场景，推荐使用 |
| avenue_92.18.pth | Avenue | 92.18% | 校园场景 |
| ped2_98.89.pth | Ped2 | 98.89% | 行人区域 |
| ub_65.11_sample7.pth | UBnormal | 65.11% | 复杂场景 |

## 使用方法

启动Jigsaw服务时指定模型路径：

```bash
python start_edge.py --checkpoint detection/pre_trained/stc_78.76_sample7.pth
```