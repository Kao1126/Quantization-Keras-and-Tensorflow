import tensorflow as tf
import argparse
from frozen import*
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from data_generator import*


checkpoint_path = os.getcwd() + '/checkpoint'
model_path = os.getcwd() + '/model/example_VGG/model.h5'
data_path = os.getcwd() + '/data'
frozen_path = os.getcwd() + '/frozen_model.pb'
quant_delay = 10

def _ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', help='Path to the model path', required=True)
    parser.add_argument('--epoch', help='Epoch', required=True)
    parser.add_argument('--batch_size', help='batch_size', required=True)

    args = parser.parse_args()

    return args



def callbacks(checkpoint_path):

    callback = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.getcwd() + '/logs',
            batch_size=int(args.batch_size),
            write_images=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, "ckpt_{epoch:02d}.hdf5"),
                                            verbose=1,
                                            save_weights_only=False
                                            )
    ]

    return callback


if __name__ == '__main__':

    args = _ParseArgs()

    print("Tensorflow Version：{}".format(tf.VERSION))
    print("Tensorflow-Keras Version：{}".format(tf.keras.__version__))

    '''prepare data'''
    prepare = data_prepare(data_path + r"/train",
                           data_path + r"/test",
                           data_path + r"/val",
                           int(args.batch_size))

    train_generator, test_generator, val_generator = prepare.data_prepare()

    Train_step_size = train_generator.n // train_generator.batch_size
    Val_step_size = val_generator.n // val_generator.batch_size

    train_graph = tf.Graph()
    train_sess = tf.compat.v1.Session(graph=train_graph)
    tf.compat.v1.keras.backend.set_session(train_sess)

    with train_graph.as_default():
        tf.keras.backend.set_learning_phase(1)
        print(args.model_path)
        model = load_model(args.model_path)
        
        '''quantization node'''
        tf.contrib.quantize.create_training_graph(
            input_graph=train_graph,  
            quant_delay=quant_delay  
        )

        train_sess.run(tf.compat.v1.global_variables_initializer())

        '''callbacks'''
        callback = callbacks(checkpoint_path)
        
        '''train'''
        history = model.fit_generator(generator=train_generator,
                                    steps_per_epoch=Train_step_size,
                                    validation_data=val_generator,
                                    validation_steps=Val_step_size,
                                    epochs=int(args.epoch),
                                    callbacks=callback,
                                    initial_epoch=0,
                                    verbose=1)

        plot_model(model, to_file=os.path.join(
            os.getcwd(), 'model_visualized.png'))
        
        ''' save checkpoints '''
        saver = tf.compat.v1.train.Saver()  
        saver.save(train_sess, checkpoint_path + '/checkpoints')

        val_scores = model.evaluate_generator(generator=val_generator, verbose=1, max_queue_size=50, workers=1)
        print('\n[ValidationData] evaluate result\nAccuracy: {0:.3f}%\nLoss: {1:.3f}'.format(val_scores[1] * 100,
                                                                                            val_scores[0]))
                                                                                            

    frozen(checkpoint_path, args.model_path, frozen_path)