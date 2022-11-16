# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import onnx
import numpy as np
from oonxparse import OnnxParser
from oonxparse import AttributeExtractor
import logging
import sys



# Load the ONNX model
model = onnx.load("yolov3tiny.onnx")
node_list = model.graph.node #得到节点所有信息的一个列表[]
input_list = model.graph.input
output_list = model.graph.output
initializer_list = model.graph.initializer

# Check that the model is well formed
#onnx.checker.check_model(model)

# Print a human readable representation of the graph
#print(onnx.helper.printable_graph(model.graph))

'''———————————————创建实例对象————————————————————'''
if __name__=='__main__':
   tiny = OnnxParser(model)
   tiny2 = AttributeExtractor()
   #print(tiny.topo_sort(model))
   '''函数返回多个值，得到一个元组。1.是按照INPUTS,OUTPUTS,NODE PROPERTIES,ATTRIBUTES字典，排成一个列表2.空字典 
   3.某层前一层的层名字典 4.某层后一层的层名字典'''

   #print(tiny.find_next_node(model,node_list[0]))
   '''得到一个元组1.本节点名字+下一个节点名字组成的字典 2.下一个节点的全部信息构成的列表'''

   #print(tiny.get_initializer(model))
   '''得到元组1.所有权重2权重名字+序号组成的字典'''

   #print(tiny.get_node_weights(node_list[0]))
   '''得到某个具体节点的权重字典   权重名：具体值'''

   #print(tiny._get_input_output_dict())
   '''得到元组1.该节点的输入名字：节点名字（字典）2该节点的输出名字：节点名字（字典） 3.整个图的输入名字（列表）4.整个图的输出名字（列表）'''

   #print(tiny2.get_info_conv(node_list[0]))
   '''得到卷积的信息元组dilation, group, pad, stride'''

'''——————————————————下面开始打印——————————————————————————————'''
# make a copy of original stdout route
stdout_backup = sys.stdout
# define the log file that receives your log info
log_file = open("message.log", "w")
# redirect print output to log file
sys.stdout = log_file

print("Now all print info will be written to message.log")
#print(tiny.topo_sort(model))
print("____________________________________________________")
print("卷积层的dilation, kernel_shape,group, pads, strides:")
for j in range(len(node_list)):
   if node_list[j].op_type == "Conv":
      print(node_list[j].name, ":", tiny2.get_info_conv(node_list[j]))
print("____________________________________________________")
print("卷积层的权重:")
for i in range(len(node_list)):
   if node_list[i].op_type == "Conv":
      print(node_list[i].name, ":", tiny.get_node_weights(node_list[i]))

# any command line that you will execute


log_file.close()
# restore the output to initial pattern
sys.stdout = stdout_backup

print("Now this will be presented on screen")


#print(list(input_list)) #conv的输入W，B的shape
#print(output_list)
#print(initializer_list)


