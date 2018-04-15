from __future__ import print_function


import os
import json
import shutil
import argparse
from add_models import ADDModels

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import (CSVLogger,
                             TensorBoard,
                             ModelCheckpoint,
                             LearningRateScheduler)


class ADDTrain(object):

    def __init__(self,
                 paras_name,
                 paras_json_path,
                 weights_save_dir,
                 logs_save_dir,
                 save_best_weights=True,
                 pre_trained_path=None):
        self.data = None
        self.save_best_weights = save_best_weights
        self.paras = self.load_paras(paras_json_path, paras_name)
        self._resolve_paras()

        self.weights_dir = os.path.join(weights_save_dir, paras_name)
        self.logs_dir = os.path.join(logs_save_dir, paras_name)

        self.create_dir(self.weights_dir)
        self.create_dir(self.logs_dir)

        self.last_weights_path = os.path.join(self.weights_dir, "last.h5")
        self.best_weights_path = os.path.join(self.weights_dir, "best.h5")
        self.curves_path = os.path.join(self.logs_dir, "curves.csv")
        self.pre_trained_path = pre_trained_path

        return

    def _resolve_paras(self):
        # Parameters to construct model
        self.model_name = self.paras["model_name"]
        self.input_shape = self.paras["input_shape"]
        self.scale = self.paras["scale"]
        self.pooling = self.paras["pooling"]
        self.l2_coeff = self.paras["l2_coeff"]
        self.drop_rate = self.paras["drop_rate"]
        self.bn_momentum = self.paras["bn_momentum"]
        self.initializer = self.paras["initializer"]

        # Parameters to train model
        self.optimizer = self.paras["optimizer"]
        self.lr_start = self.paras["lr_start"]
        self.epochs_num = self.paras["epochs_num"]
        self.batch_size = self.paras["batch_size"]

        self.pre_trained = False
        if "pre_trained" in self.paras.keys():
            self.pre_trained = self.paras["pre_trained"]

        return

    def _load_model(self):
        self.model = ADDModels(model_name=self.model_name,
                               input_shape=self.input_shape,
                               scale=self.scale,
                               pooling=self.pooling,
                               l2_coeff=self.l2_coeff,
                               drop_rate=self.drop_rate,
                               bn_momentum=self.bn_momentum,
                               initializer=self.initializer).model
        return

    def _load_weights(self):
        print("Load pre-trained weights from:", self.pre_trained_path)
        self.model.load_weights(self.pre_trained_path)
        return

    def _set_optimizer(self):
        if self.optimizer == "adam":
            self.opt_fcn = Adam(lr=self.lr_start, epsilon=1e-8,
                                decay=1e-6, amsgrad=True)
        return

    def _set_lr_scheduler(self, epoch):
        if self.pre_trained:
            lrs = [self.lr_start] * 25 + \
                  [self.lr_start * 0.1] * 25
        else:
            lrs = [self.lr_start] * 50 + \
                  [self.lr_start * 0.1] * 50 + \
                  [self.lr_start * 0.01] * 50
        return lrs[epoch]

    def _set_callbacks(self):
        csv_logger = CSVLogger(self.curves_path,
                               append=True,
                               separator=",")
        lr_scheduler = LearningRateScheduler(self._set_lr_scheduler)
        tb = TensorBoard(log_dir=self.logs_dir,
                         batch_size=self.batch_size)
        self.callbacks = [csv_logger, lr_scheduler, tb]

        if self.save_best_weights:
            checkpoint = ModelCheckpoint(filepath=self.best_weights_path,
                                         monitor="val_loss",
                                         verbose=0,
                                         save_best_only=True)
            self.callbacks += [checkpoint]

        return

    def _print_score(self):

        def evaluate(x, y, data_str):
            score = self.model.evaluate(x, y, self.batch_size, 0)
            print(data_str + " Set: Loss: {0:.4f}, Accuracy: {1:.4f}".format(
                  score[0], score[1]))
            return

        evaluate(self.data.train_x, self.data.train_y, "Training")
        evaluate(self.data.valid_x, self.data.valid_y, "Validation")
        evaluate(self.data.test_x, self.data.test_y, "Testing")

        return

    def run(self, data):

        print("\nTraining the model.\n")

        self.data = data

        self._load_model()
        self._set_optimizer()

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.opt_fcn,
                           metrics=["accuracy"])
        self.model.summary()

        if self.pre_trained:
            self._load_weights()

        self._set_callbacks()
        self.model.fit(self.data.train_x, self.data.train_y,
                       batch_size=self.batch_size,
                       epochs=self.epochs_num,
                       validation_data=(self.data.valid_x,
                                        self.data.valid_y),
                       shuffle=True,
                       callbacks=self.callbacks)

        self.model.save(self.last_weights_path)
        self._print_score()
        K.clear_session()

        return

    @staticmethod
    def load_paras(paras_json_path, paras_name):
        paras = json.load(open(paras_json_path))
        return paras[paras_name]

    @staticmethod
    def create_dir(dir_path, rm=True):
        if os.path.isdir(dir_path):
            if rm:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        else:
            os.makedirs(dir_path)
        return


def main(hyper_paras_name, volume_type):

    from add_dataset import ADDDataset

    pre_paras_path = "pre_paras.json"
    pre_paras = json.load(open(pre_paras_path))

    parent_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(parent_dir, pre_paras["data_dir"])

    ad_dir = os.path.join(data_dir, pre_paras["ad_in"])
    nc_dir = os.path.join(data_dir, pre_paras["nc_in"])

    weights_save_dir = os.path.join(parent_dir, pre_paras["weights_save_dir"], volume_type)
    logs_save_dir = os.path.join(parent_dir, pre_paras["logs_save_dir"], volume_type)

    # Getting splitted dataset
    data = ADDDataset(ad_dir, nc_dir,
                      subj_separated=pre_paras["subj_separated"],
                      volume_type=volume_type,
                      train_prop=pre_paras["train_prop"],
                      valid_prop=pre_paras["valid_prop"],
                      random_state=pre_paras["random_state"],
                      pre_trainset_path=pre_paras["pre_trainset_path"],
                      pre_validset_path=pre_paras["pre_validset_path"],
                      pre_testset_path=pre_paras["pre_testset_path"],
                      data_format=pre_paras["data_format"])
    data.run(pre_split=pre_paras["pre_split"],
             save_split=pre_paras["save_split"],
             save_split_dir=pre_paras["save_split_dir"])

    # Training the model
    train = ADDTrain(paras_name=hyper_paras_name,
                     paras_json_path=pre_paras["hyper_paras_json_path"],
                     weights_save_dir=weights_save_dir,
                     logs_save_dir=logs_save_dir,
                     save_best_weights=pre_paras["save_best_weights"])
    train.run(data)

    return


if __name__ == "__main__":

    # Command line
    # python add_train.py --paras=paras-1 --volume=whole

    parser = argparse.ArgumentParser()
    help_str = "Select a set of hyper-parameters in hyper_paras.json."
    parser.add_argument("--paras", action="store", default="paras-1",
                        dest="hyper_paras_name", help=help_str)
    help_str = "Select a volume type in ['whole', 'gm', 'wm', 'csf']."
    parser.add_argument("--volume", action="store", default="whole",
                        dest="volume_type", help=help_str)

    args = parser.parse_args()
    main(args.hyper_paras_name, args.volume_type)
