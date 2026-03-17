## 快速开始
版本要求 
python 3.12  
sumo  1.25.0

python 下载地址：https://www.python.org/

依次执行如下命令来搭建环境

~~~ cmd
python -m venv traffic_env   创建虚拟环境
traffic_env\Scripts\activate   激活虚拟环境
python -m pip install --upgrade pip  更新安装工具
pip install -r requirements.txt  安装依赖
python -m myscripts.mytrainer  运行程序
~~~

注意，在运行前需要设置mytrainer 的 self.plot_dir  将其设置为自己的输出路径
或是按照以下格式补全目录：
├── myscripts
│   ├── logger.py
│   ├── myagent.py
│   ├── mytrainer.py
│   └── sumoController.py
└── outputs
    ├── output1
    ├── output2
    ...

