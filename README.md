# 计算机动画课程大作业
用Taichi框架，实现了一个naive的2D流体模拟，目前支持了两种模型：WCSPH, PBF。

并且支持交互（比如：可以点击鼠标左键添加新的流体，可以按空格生成当前时刻2D的ply点云，按q可以随时退出）

## 环境说明
本项目在python3.8下进行开发，需要的python库在requirements.txt中列出

执行下面一段指令，配置运行环境：
```
pip install -r requirements.txt
```

## 运行方法
若要运行WCSPH模型离线渲染，执行下面的指令，在根目录下将会生成一个渲染后的视频：(在ubuntu系统服务器上，3090GPU上进行开发)
```
python run_exp_wcsph.py
```

若要运行PBF模型离线渲染，执行下面的指令，在根目录下将会生成一个渲染后的视频：(在ubuntu系统服务器上，3090GPU上进行开发)
```
python run_exp_pbf.py
```

若要运行交互程序，执行：(在本地M1芯片Macbook Pro上进行开发)
```
python interaction.py
```

(注：目前支持两种场景，在model/scene.py的第40-41行可以自行更改。)

(注：如有问题可以联系叶盛，邮箱yec22@mails.tsinghua.edu.cn，微信ys15510562705)