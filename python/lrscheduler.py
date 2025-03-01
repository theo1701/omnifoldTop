# This class serves as a simpler combined interface to apply a learning rate scheduler
# There are two main different ways to support this, through either
# tf.keras.callbacks.LearningRateScheduler
# tf.keras.optimizers.schedules
# The two methods each has some nice features that are not trivially available in the other
# note two methods can not be used together, since using schedules takes up the place for learning rate

import numpy as np
import tensorflow.keras.optimizers.schedules as schedules
import tensorflow.keras.callbacks as callbacks
import json
from callbacks import PrintLearningRate

debug = False

lrscheduler = None

class LearningRateScheduler():
    def __init__(self, initial_learning_rate, scheduler_names, schedule_args, reduce_on_plateau):
        """
        Arguments
        ---------
        initial_learning_rate: float
            inital learning rate
        scheduler_name: list of str
            names refering to the scheduler / callback to be applied, there can at most be one schedule, but many callbacks. Schedulers and Callbacks can not be used together.
        schedule_args: dictionary
            extra arguments for the schedule, required for using "piecewised"
        reduce_on_plateau: int
            the number of epoch to wait before reducing learning rate, a value of 0 indicates that reduce on plateau will not happen

        Raise
        -----
        Exception if more than 1 learning schedule is requested
        Exception if callbacks and schedules are being used together
        """

        self.inital_learning_rate = initial_learning_rate

        if scheduler_names is None : scheduler_names = []
        if schedule_args is None: schedule_args = {}

        self.callback_names = [name for name in scheduler_names if name in scheduler_dict]
        self.schedules_names = [name for name in scheduler_names if name in schedules_dict]
        self.callbacks = []
        self.schedule = None

        # enforce the requirements
        assert(len(self.schedules_names) <= 1)
        assert((not self.schedules_names) or (not self.callback_names))

        # assemble callbacks
        for callback_name in self.callback_names:
            base = scheduler_dict[callback_name] # base function
            def scheduler_function(epoch, lr) : return base(epoch, lr, **schedule_args)
            self.callbacks += [callbacks.LearningRateScheduler(scheduler_function)]
        if reduce_on_plateau > 0:
            self.callbacks += [callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = reduce_on_plateau)]
        if debug:
            self.callbacks += [PrintLearningRate()]

        # assemble schedules
        for schedule_name in self.schedules_names:
            if schedule_args:
                self.schedule = schedules_dict[schedule_name](**schedule_args)
            # defaults
            elif schedule_name in ["cosined", "cosinedr", "polynomiald"]:
                # initial learning rate, decay steps
                self.schedule = schedules_dict[schedule_name](initial_learning_rate, 5000)
            elif schedule_name in ["expd", "inversetd"]:
                # inital learning rate, decay steps, decay rate
                self.schedule = schedules_dict[schedule_name](initial_learning_rate, 1000, 0.95)
    
    def get_callbacks(self):
        """
        return the callbacks, an empty list if no callback is requested
        """
        return self.callbacks
    
    def get_schedule(self):
        """
        return the schedule, just initial learning rate if no schedule is requested
        """
        return self.schedule if self.schedule is not None else self.inital_learning_rate

def constant(epoch, lr, **args):
    """
    constant learning rate, independent of epoch
    """
    return lr

def warm_up_constant(epoch, lr, **args):
    """
    perform warm up in training with a linearly increasing learning rate for the first WARM_UP_EPOCHS epoch
    """

    if "warm_up_epochs" in args:
        warm_up_epochs = args["warm_up_epochs"]
    else:
        warm_up_epochs = 5

    if epoch == 0:
        return lr / warm_up_epochs
    elif epoch < warm_up_epochs:
        return (epoch + 1) * lr / epoch
    else:
        return lr

def init_lr_scheduler(init_path):
    """
    arguments
    ---------
    init_path: str
        path to where the lrscheduler config file is located

    raises
    ------
    exception if more than 1 learning schedule is requested
    """
    global lrscheduler

    with open(init_path, "r") as init_file:
        config = json.load(init_file)

    scheduler_args = config["scheduler_args"]
    # scheduler_args = json.loads(config["scheduler_args"]) if config["scheduler_args"] is not "" else None
    
    lrscheduler = LearningRateScheduler(config["initial_learning_rate"],
                                        config["scheduler_names"],
                                        scheduler_args,
                                        config["reduce_on_plateau"])

def get_lr_scheduler()->LearningRateScheduler:
    """
    returns the learning rate scheduler
    """
    return lrscheduler

"""
put functions and their names as dictionary here to use with run params
"""
# functions that is to be used with tensorflow.keras.callbacks.LearningRate
scheduler_dict = {
    "constant" : constant,
    "warmc" : warm_up_constant,
}

# giving names to tf.keras.optimizers.schedules
schedules_dict = {
    "cosined" : schedules.CosineDecay,
    "cosinedr" : schedules.CosineDecayRestarts,
    "expd" : schedules.ExponentialDecay,
    "inversetd" : schedules.InverseTimeDecay,
    "piecewised" : schedules.PiecewiseConstantDecay,
    "polynomiald" : schedules.PolynomialDecay
}
