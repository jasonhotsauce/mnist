# mnist

A neural network that classify hand written digits from 0-9, and convert the model to CoreML model

## Usage

To train the model, and save the trained model, use the following command

```bash
python mnist.py -t -o
```

To enable data augmentation you can add `-a` when running the program.

For debug purpose, one can also add `-v` to enable verbosity level. The more `v` you add, the more debug text will be printed. Maximun number of `v` you can add is 3.

To convert the saved model to CoreMl, run the following command:

```bash
python coreml_convertor.py -m minstCNN.h5 -o <your_coreml_model_name>.mlmodel
```