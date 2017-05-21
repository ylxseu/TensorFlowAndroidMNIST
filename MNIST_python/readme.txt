本目录内容为：
	为MNIST手写数字识别的深层神经网络python训练脚本及其输出结果

目录结果如下：
./
├── MNIST_CNN.py							卷积神经网络python脚本
├── MNIST_line_NN.py                		线性模型python脚本
├── MNIST_no_layer1.py              		非线性模型-无隐藏层
├── MNIST_relu.py                   		非线性模型-1层隐藏层-激活函数：RELU函数-损失函数：交叉熵+正则化损失
├── MNIST_sigmoid.py                		非线性模型-1层隐藏层-激活函数：sigmoid函数-损失函数：交叉熵+正则化损失
├── MNIST_tanh.py                   		非线性模型-1层隐藏层-激活函数：tanh函数-损失函数：交叉熵+正则化损失
├── MNIST_without_regul.py          		非线性模型-1层隐藏层-激活函数：RELU函数-损失函数：交叉熵
├── MNIST_graph.py                   		非线性模型+固化神经网络
├── output_CNN                      		MNIST_CNN.py		 的输出结果
├── output_line_NN                  		MNIST_line_NN.py      的输出结果
├── output_no_layer1                		MNIST_no_layer1.py    的输出结果
├── output_relu                     		MNIST_relu.py         的输出结果
├── output_sigmoid                  		MNIST_sigmoid.py      的输出结果
├── output_tanh                     		MNIST_tanh.py         的输出结果
├── output_without_regul            		MNIST_without_regul.py的输出结果
├── readme.txt                      		
├── data									运行python脚本所需的数据
│   ├── t10k-images-idx3-ubyte.gz			测试集图像数据
│   ├── t10k-labels-idx1-ubyte.gz			测试集标签数据
│   ├── train-images-idx3-ubyte.gz			训练集图像数据
│   └── train-labels-idx1-ubyte.gz			训试集标签数据
└── tmp/
	└── ylx
    	└── ylx.pb					 		固化神经网络的输出文件
                                    		
2 directories, 15 files             		
