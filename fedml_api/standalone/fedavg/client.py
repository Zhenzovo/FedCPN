import copy
import logging


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, round_idx, alpha, local_update_last, global_update_last, global_mdl, hist_i):
        self.model_trainer.set_model_params(w_global)
        model_plus = self.model_trainer.train(self.local_training_data, self.device, self.args, round_idx, alpha, local_update_last, global_update_last, global_mdl, hist_i)
        # weights = self.model_trainer.get_model_params()
        weights = model_plus.state_dict()
        self.model_trainer.set_model(model_plus)
        self.model_trainer.set_model_params(weights)
        return weights

    def train_on_model(self, w_global=None):
        # self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        model = self.model_trainer.get_model()
        return model

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args, b_use_test_dataset)
        return metrics
