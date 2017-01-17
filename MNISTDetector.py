from __future__ import print_function
import tensorflow as tf
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


class MNISTDetector:
    def __init__(self):
        self.graph = load_graph("MyNets/mnist.pb")
        for op in self.graph.get_operations():
            print(op.name)

        self.input = self.graph.get_tensor_by_name('prefix/input:0')
        self.output= self.graph.get_tensor_by_name('prefix/output:0')

    def detect(self, image):
        with tf.Session(graph=self.graph) as sess:
            prediction = sess.run(self.output, feed_dict={self.input:image})
            return prediction
