import tensorflow as tf
import os
from tensorflow.keras.models import load_model


def frozen(checkpoint_path, model_path, frozen_path):

    eval_graph = tf.Graph()
    eval_sess = tf.compat.v1.Session(graph=eval_graph) 
    tf.keras.backend.set_session(eval_sess)

    with eval_graph.as_default():


        tf.keras.backend.set_learning_phase(0)

        ''' get model '''
        eval_model = load_model(model_path, compile=False)

        tf.contrib.quantize.create_eval_graph(
            input_graph=eval_graph  
        )

        ''' 取得模型內部結構，包含節點、內部函數數值等等 '''
        eval_graph_def = eval_graph.as_graph_def()

        ''' 重新載入儲存權重變數，透過儲存紀錄點(存放權重參數) '''
        saver = tf.compat.v1.train.Saver()  
        saver.restore(eval_sess, checkpoint_path + '/checkpoints')

        
        ''' fix error if you have normalization layer'''
        fix_error()
        
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            eval_sess,
            eval_graph_def,
            [eval_model.output.op.name]
        )

        print("[Train_Data] Model_PB_Path：{}".format(frozen_path))

        with open(frozen_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

            
def fix_error():
    gp = tf.get_default_graph().as_graph_def()
    for node in gp.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]            
