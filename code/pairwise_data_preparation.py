#Basics
import numpy as np
import random
#Tf
from tensorflow import keras

class DataPrep:

    def __init__(self, validation_size) -> None:
        self.validation_size = validation_size

    def load_data(self) -> tuple:
        """Load the mnist dataset
        """

        print("Loading mnist dataset...")

        try:
            (x_train_val, y_train_val), (x_test, y_test) = keras.datasets.mnist.load_data()

            x_train_val = x_train_val.astype("float32")
            x_test = x_test.astype("float32")

            val_size = int(60000*self.validation_size)
            train_size = 60000-val_size

            print("Spliting the dataset...")

            """
            Define train and validation
            """
            x_train, x_val = x_train_val[:train_size], x_train_val[val_size:]
            y_train, y_val = y_train_val[:train_size], y_train_val[val_size:]
            del x_train_val, y_train_val

            print("Data loading done!")

            return (x_train, y_train), (x_val, y_val), (x_test, y_test)
        except:
            print("Failed while loading mnist dataset!")

    def create_pairwise_data(self, x, y) -> tuple:
        """Seleciona N samples randômicos de uma determinada classe, por exemplo digito 0, 
       e cria pares com N samples randômicos de outra classe, por exemplo digito 1. Após isso,
       fazemos o mesmo para todas as outras classes. Depois de criar os pares para o digito 0,
       repetimos o processo para o restante das classes.

        Arguments:
            x: List containing images, each index in this list corresponds to one image.
            y: List containing labels, each label with datatype of `int`.

        Returns:
            Tuple containing two numpy arrays as (pairs_of_samples, labels),
            where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
            labels are a binary array of shape (2len(x)).
        """

        try:
            print("Generating pairwise data...")
            num_classes = max(y) + 1
            digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

            pairs = []
            labels = []

            for idx1 in range(len(x)):
                # add a matching example
                x1 = x[idx1]
                label1 = y[idx1]
                idx2 = random.choice(digit_indices[label1])
                x2 = x[idx2]

                pairs += [[x1, x2]]
                labels += [1]

                # add a non-matching example
                label2 = random.randint(0, num_classes - 1)
                while label2 == label1:
                    label2 = random.randint(0, num_classes - 1)

                idx2 = random.choice(digit_indices[label2])
                x2 = x[idx2]

                pairs += [[x1, x2]]
                labels += [0]

            print("Pairwise done!")
            return np.array(pairs), np.array(labels).astype("float32")
        except:
            print("Failed while creating the pairs!")


        