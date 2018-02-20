# Deep Neural Network Analysis and Visualization

Recently, deep neural networks (DNNs) have achieved many breakthrough results in many areas such as computer vision, natural language processing, reinforcement learning and so on. Thus, there’s been huge amount of research into visualizing DNNs for better understanding what the neuron units have learned. In this paper, we analyze DNNs via the so-called contribution value between nodes or filters, which is the product of activation value and its corresponding weight value. We visualize DNNs by maximizing the contribution values to demonstrate there exist some dominant signal-flow paths in DNNs from input layer to output layer. We also implement a Graphical User Interface(GUI) visualization platform so that users can play with different parameters and network architectures to find which setting is more proper for visualiza- tion. In experiments, we test our methods on three different architectures on different datasets: LeNet5 on MNIST, VoxNet Baseline on ModelNet10 and ORION Basic on ModelNet10 dataset.

# Caffe + Matlab version:
1. Platform: Caffe + Matlab(R2016b)
2. For Lenet5, run visualization_gui.m, then you can choose the parameters to display the plots.
3. For ORION, run orion_gui.m, then you can plot the visualization figures according to your parameters setting.

# Caffe2 + Python2 version:
