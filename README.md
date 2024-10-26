# Rec_GNN
## 文件结构
data/test.csv 测试集  
data/train.csv 训练集  
src/model/ 模型存放路径; tmp.pth 当前epoch的model（用于测试）; best_model.pth 最优model  
src/main.py 运行脚本  
src/rec_gnn.py GNN训练  
src/rec_inference GNN推理  

## 指令
### 安装所需的package
```
pip install -r requirements.txt
```

### 运行   
```
python src/main.py
```
