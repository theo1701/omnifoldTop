import os

class ModelSaveLoader():
    def __init__(self, load_model_dir, save_model_dir):
        self.load_model_dir = load_model_dir
        self.save_model_dir = save_model_dir
    def get_dirs(self, ir):
        return self.load_model_dir, self.save_model_dir
    
class NormalTrainingMode(ModelSaveLoader):
    def __init__(self, load_model_dir, save_model_dir):
        self.save_models = False
        return super().__init__(load_model_dir, save_model_dir)
    def __init__(self, load_model_dir, save_model_dir, save_models):
        self.save_models = True
        super().__init__(load_model_dir, save_model_dir)
    def get_dirs(self, ir):
        if self.load_model_dir:
            load_model_dir = os.path.join(self.load_model_dir, "Models", f"run{ir}")
            save_model_dir = '' # no need too save the model again
        else:
            load_model_dir = ''
            if self.save_models and self.save_model_dir:
                save_model_dir = os.path.join(self.save_model_dir, "Models", f"run{ir}")
            else:
                save_model_dir = ''
        return load_model_dir, save_model_dir
    
class PretrainMode(ModelSaveLoader):
    def __init__(self, load_model_dir, save_model_dir):
        super().__init__(load_model_dir, save_model_dir)
        self.epoch_limit = 5
        self.create_pretrain_model = True # otherwise it would be load from pretrain mode
    def __init__(self, load_model_dir, save_model_dir, epoch_limit, create_mode):
        super().__init__(load_model_dir, save_model_dir)
        self.epoch_limit = epoch_limit
        self.create_pretrain_model = create_mode
    def get_dirs(self, ir):
        if self.create_pretrain_model:
            load_model_dir = '' # we are pretraining model, so no need to load
            save_model_dir = os.path.join(self.save_model_dir, "Models", "pretrain")
            return load_model_dir, save_model_dir
        else:
            load_model_dir = os.path.join(self.save_model_dir, "Models", "pretrain")
            save_model_dir = '' # we don't alter pretrained models
            return load_model_dir, save_model_dir
    def get_epoch_limit(self):
        return self.epoch_limit
    
def initializeSaveLoader(type, load_model_dir, save_model_dir, epoch_limit, create_mode, save_models):
    if type == "training":
        return NormalTrainingMode(load_model_dir, save_model_dir, save_models)
    elif type == "pretrain":
        return PretrainMode(load_model_dir, save_model_dir, epoch_limit, create_mode)