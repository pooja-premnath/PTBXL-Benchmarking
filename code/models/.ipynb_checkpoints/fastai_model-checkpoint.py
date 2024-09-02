from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.core import Callback
from pathlib import Path
from functools import partial
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.timeseries_utils import *

# Import your custom models
from models.resnet1d import resnet1d18, resnet1d34, resnet1d50, resnet1d101, resnet1d152, resnet1d_wang, wrn1d_22
from models.xresnet1d import xresnet1d18, xresnet1d34, xresnet1d50, xresnet1d101, xresnet1d152, xresnet1d18_deep, xresnet1d34_deep, xresnet1d50_deep, xresnet1d18_deeper, xresnet1d34_deeper, xresnet1d50_deeper
from models.inception1d import inception1d
from models.basic_conv1d import fcn, fcn_wang, schirrmeister, sen, basic1d, weight_init
from models.rnn1d import RNN1d
from models.base_model import ClassificationModel
from utils.utils import evaluate_experiment

class MetricFunc(Callback):
    def __init__(self, func, name="metric_func", ignore_idx=None, one_hot_encode_target=True, argmax_pred=False, softmax_pred=True, flatten_target=True, sigmoid_pred=False, metric_component=None):
        super().__init__()
        self.func = func
        self.ignore_idx = ignore_idx
        self.one_hot_encode_target = one_hot_encode_target
        self.argmax_pred = argmax_pred
        self.softmax_pred = softmax_pred
        self.flatten_target = flatten_target
        self.sigmoid_pred = sigmoid_pred
        self.metric_component = metric_component
        self.name = name

    def before_epoch(self):
        self.y_pred = None
        self.y_true = None
    
    def after_batch(self):
        y_pred_flat = self.pred.view((-1, self.pred.size()[-1]))
        y_true_flat = self.y.view(-1) if self.flatten_target else self.y
        
        if self.argmax_pred:
            y_pred_flat = y_pred_flat.argmax(dim=1)
        elif self.softmax_pred:
            y_pred_flat = F.softmax(y_pred_flat, dim=1)
        elif self.sigmoid_pred:
            y_pred_flat = torch.sigmoid(y_pred_flat)
        
        if self.ignore_idx is not None:
            selected_indices = (y_true_flat != self.ignore_idx).nonzero().squeeze()
            y_pred_flat = y_pred_flat[selected_indices]
            y_true_flat = y_true_flat[selected_indices]
        
        y_pred_flat = y_pred_flat.cpu().numpy()
        y_true_flat = y_true_flat.cpu().numpy()

        if self.one_hot_encode_target:
            y_true_flat = one_hot_np(y_true_flat, self.pred.size()[-1])

        if self.y_pred is None:
            self.y_pred = y_pred_flat
            self.y_true = y_true_flat
        else:
            self.y_pred = np.concatenate([self.y_pred, y_pred_flat], axis=0)
            self.y_true = np.concatenate([self.y_true, y_true_flat], axis=0)
    
    def after_epoch(self):
        self.metric_complete = self.func(self.y_true, self.y_pred)
        if self.metric_component is not None:
            self.learn.metrics[-1] = self.metric_complete[self.metric_component]
        else:
            self.learn.metrics[-1] = self.metric_complete

def fmax_metric(targs, preds):
    return evaluate_experiment(targs, preds)["Fmax"]

def auc_metric(targs, preds):
    return evaluate_experiment(targs, preds)["macro_auc"]

def mse_flat(preds, targs):
    return torch.mean((preds.view(-1) - targs.view(-1)) ** 2)

def nll_regression(preds, targs):
    preds_mean = preds[:, 0]
    preds_var = torch.clamp(torch.exp(preds[:, 1]), 1e-4, 1e10)
    return torch.mean(torch.log(2 * math.pi * preds_var) / 2) + torch.mean((preds_mean - targs[:, 0]) ** 2 / 2 / preds_var)
    
def nll_regression_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0., 0.001)
        nn.init.constant_(m.bias, 4)

def lr_find_plot(learner, path, filename="lr_find", n_skip=10, n_skip_end=2):
    learner.lr_find()
    
    backend_old = matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("learning rate (log scale)")
    losses = [x.item() for x in learner.recorder.losses[n_skip:-(n_skip_end+1)]]

    plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)], losses)
    plt.xscale('log')
    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)

'''def losses_plot(learner, path, filename="losses", last=None):
    backend_old = matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("Batches processed")

    last = last if last is not None else len(learner.recorder.losses)
    l_b = np.sum(learner.recorder.nb_batches[-last:])
    iterations = range(len(learner.recorder.losses))[-l_b:]
    plt.plot(iterations, learner.recorder.losses[-l_b:], label='Train')
    val_iter = learner.recorder.nb_batches[-last:]
    val_iter = np.cumsum(val_iter) + np.sum(learner.recorder.nb_batches[:-last])
    plt.plot(val_iter, learner.recorder.val_losses[-last:], label='Validation')
    plt.legend()

    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)'''

def losses_plot(learner, path, filename="losses", last=None):
    backend_old = matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("Loss")
    plt.xlabel("Batches Processed")

    # Determine the number of iterations (batches)
    last = len(learner.recorder.losses) if last is None else last
    iterations = range(len(learner.recorder.losses))[-last:]
    
    # Plot training loss
    plt.plot(iterations, learner.recorder.losses[-last:], label='Train')

    # Plot validation loss if available
    if learner.recorder.val_losses:
        val_iterations = range(len(learner.recorder.val_losses))[-last:]
        plt.plot(val_iterations, learner.recorder.val_losses[-last:], label='Validation')
    
    plt.legend()
    plt.savefig(str(path / (filename + '.png')))
    plt.switch_backend(backend_old)


class FastaiModel(ClassificationModel):
    def __init__(self, name, n_classes, freq, outputfolder, input_shape, pretrained=False, input_size=2.5, input_channels=12, chunkify_train=False, chunkify_valid=True, bs=128, ps_head=0.5, lin_ftrs_head=[128], wd=1e-2, epochs=50, lr=1e-2, kernel_size=5, loss="binary_cross_entropy", pretrainedfolder=None, n_classes_pretrained=None, gradual_unfreezing=True, discriminative_lrs=True, epochs_finetuning=30, early_stopping=None, aggregate_fn="max", concat_train_val=False):
        super().__init__()
        
        self.name = name
        self.num_classes = n_classes if loss != "nll_regression" else 2
        self.target_fs = freq
        self.outputfolder = Path(outputfolder)

        self.input_size = int(input_size * self.target_fs)
        self.input_channels = input_channels

        self.chunkify_train = chunkify_train
        self.chunkify_valid = chunkify_valid

        self.chunk_length_train = 2 * self.input_size
        self.chunk_length_valid = self.input_size

        self.min_chunk_length = self.input_size

        self.stride_length_train = self.input_size
        self.stride_length_valid = self.input_size // 2

        self.copies_valid = 0
        
        self.bs = bs
        self.ps_head = ps_head
        self.lin_ftrs_head = lin_ftrs_head
        self.wd = wd
        self.epochs = epochs
        self.lr = lr
        self.kernel_size = kernel_size
        self.loss = loss
        self.input_shape = input_shape

        if pretrained:
            if pretrainedfolder is None:
                pretrainedfolder = Path(f'../output/exp0/models/{name.split("_pretrained")[0]}/')
            if n_classes_pretrained is None:
                n_classes_pretrained = 71
  
        self.pretrainedfolder = None if pretrainedfolder is None else Path(pretrainedfolder)
        self.n_classes_pretrained = n_classes_pretrained
        self.discriminative_lrs = discriminative_lrs
        self.gradual_unfreezing = gradual_unfreezing
        self.epochs_finetuning = epochs_finetuning

        self.early_stopping = early_stopping
        self.aggregate_fn = aggregate_fn
        self.concat_train_val = concat_train_val

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = [l.astype(np.float32) for l in X_train]
        X_val = [l.astype(np.float32) for l in X_val]
        y_train = [l.astype(np.float32) for l in y_train]
        y_val = [l.astype(np.float32) for l in y_val]

        if self.concat_train_val:
            X_train += X_val
            y_train += y_val
        
        if self.pretrainedfolder is None:  # Training from scratch
            print("Training from scratch...")
            learn = self._get_learner(X_train, y_train, X_val, y_val)
            
            learn.model.apply(weight_init)
            
            if self.loss == "nll_regression" or self.loss == "mse":
                output_layer_new = learn.model.get_output_layer()
                output_layer_new.apply(nll_regression_init)
                learn.model.set_output_layer(output_layer_new)
            
            lr_find_plot(learn, self.outputfolder)    
            learn.fit_one_cycle(self.epochs, self.lr)
            losses_plot(learn, self.outputfolder)
        else:  # Finetuning
            print("Finetuning...")
            learn = self._get_learner(X_train, y_train, X_val, y_val, self.n_classes_pretrained)
            
            learn.path = self.pretrainedfolder
            learn.load(self.pretrainedfolder.stem)
            learn.path = self.outputfolder

            output_layer = learn.model.get_output_layer()
            output_layer_new = nn.Linear(output_layer.in_features, self.num_classes).cuda()
            apply_init(output_layer_new, nn.init.kaiming_normal_)
            learn.model.set_output_layer(output_layer_new)
            
            if self.discriminative_lrs:
                layer_groups = learn.model.get_layer_groups()
                learn.split(layer_groups)

            learn.train_bn = True
            
            lr = self.lr
            if self.gradual_unfreezing:
                assert self.discriminative_lrs is True
                learn.freeze()
                lr_find_plot(learn, self.outputfolder, "lr_find0")
                learn.fit_one_cycle(self.epochs_finetuning, lr)
                losses_plot(learn, self.outputfolder, "losses0")
                    
            learn.unfreeze()
            lr_find_plot(learn, self.outputfolder, "lr_find1")
            learn.fit_one_cycle(self.epochs_finetuning, slice(lr / 1000, lr / 10))
            losses_plot(learn, self.outputfolder, "losses1")

        learn.save(self.name)
    
    def predict(self, X):
        X = [l.astype(np.float32) for l in X]
        y_dummy = [np.ones(self.num_classes, dtype=np.float32) for _ in range(len(X))]
        
        learn = self._get_learner(X, y_dummy, X, y_dummy)
        learn.load(self.name)
        
        preds, _ = learn.get_preds()
        preds = preds.cpu().numpy()
        
        idmap = learn.dls.valid_ds.get_id_mapping()

        return aggregate_predictions(preds, idmap=idmap, aggregate_fn=np.mean if self.aggregate_fn == "mean" else np.amax)
        
    def _get_learner(self, X_train, y_train, X_val, y_val, num_classes=None):
        df_train = pd.DataFrame({"data": range(len(X_train)), "label": y_train})
        df_valid = pd.DataFrame({"data": range(len(X_val)), "label": y_val})
        
        tfms_ptb_xl = [ToTensor()]
                
        ds_train = TimeseriesDatasetCrops(df_train, self.input_size, num_classes=self.num_classes, chunk_length=self.chunk_length_train if self.chunkify_train else 0, min_chunk_length=self.min_chunk_length, stride=self.stride_length_train, transforms=tfms_ptb_xl, annotation=False, col_lbl="label", npy_data=X_train)
        ds_valid = TimeseriesDatasetCrops(df_valid, self.input_size, num_classes=self.num_classes, chunk_length=self.chunk_length_valid if self.chunkify_valid else 0, min_chunk_length=self.min_chunk_length, stride=self.stride_length_valid, transforms=tfms_ptb_xl, annotation=False, col_lbl="label", npy_data=X_val)
    
        dls = DataLoaders.from_dsets(ds_train, ds_valid, bs=self.bs)

        if self.loss == "binary_cross_entropy":
            loss = F.binary_cross_entropy_with_logits
        elif self.loss == "cross_entropy":
            loss = F.cross_entropy
        elif self.loss == "mse":
            loss = mse_flat
        elif self.loss == "nll_regression":
            loss = nll_regression    
        else:
            raise ValueError("Loss function not found.")
               
        self.input_channels = self.input_shape[-1]
        metrics = []

        num_classes = self.num_classes if num_classes is None else num_classes
        
        if self.name.startswith("fastai_resnet1d18"):
            model = resnet1d18(num_classes=num_classes, input_channels=self.input_channels, inplanes=128, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_resnet1d34"):
            model = resnet1d34(num_classes=num_classes, input_channels=self.input_channels, inplanes=128, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_resnet1d50"):
            model = resnet1d50(num_classes=num_classes, input_channels=self.input_channels, inplanes=128, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_resnet1d101"):
            model = resnet1d101(num_classes=num_classes, input_channels=self.input_channels, inplanes=128, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_resnet1d152"):
            model = resnet1d152(num_classes=num_classes, input_channels=self.input_channels, inplanes=128, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_resnet1d_wang"):
            model = resnet1d_wang(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_wrn1d_22"):    
            model = wrn1d_22(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        
        elif self.name.startswith("fastai_xresnet1d18_deeper"):
            model = xresnet1d18_deeper(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d34_deeper"):
            model = xresnet1d34_deeper(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d50_deeper"):
            model = xresnet1d50_deeper(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d18_deep"):
            model = xresnet1d18_deep(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d34_deep"):
            model = xresnet1d34_deep(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d50_deep"):
            model = xresnet1d50_deep(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d18"):
            model = xresnet1d18(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d34"):
            model = xresnet1d34(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d50"):
            model = xresnet1d50(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d101"):
            model = xresnet1d101(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_xresnet1d152"):
            model = xresnet1d152(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
                        
        elif self.name == "fastai_inception1d_no_residual":
            model = inception1d(num_classes=num_classes, input_channels=self.input_channels, use_residual=False, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head, kernel_size=8 * self.kernel_size)
        elif self.name.startswith("fastai_inception1d"):
            model = inception1d(num_classes=num_classes, input_channels=self.input_channels, use_residual=True, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head, kernel_size=8 * self.kernel_size)

        elif self.name.startswith("fastai_fcn_wang"):
            model = fcn_wang(num_classes=num_classes, input_channels=self.input_channels, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_fcn"):
            model = fcn(num_classes=num_classes, input_channels=self.input_channels)
        elif self.name.startswith("fastai_schirrmeister"):
            model = schirrmeister(num_classes=num_classes, input_channels=self.input_channels, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_sen"):
            model = sen(num_classes=num_classes, input_channels=self.input_channels, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_basic1d"):    
            model = basic1d(num_classes=num_classes, input_channels=self.input_channels, kernel_size=self.kernel_size, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        
        elif self.name.startswith("fastai_lstm_bidir"):
            model = RNN1d(input_channels=self.input_channels, num_classes=num_classes, lstm=True, bidirectional=True, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_gru_bidir"):
            model = RNN1d(input_channels=self.input_channels, num_classes=num_classes, lstm=False, bidirectional=True, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_lstm"):
            model = RNN1d(input_channels=self.input_channels, num_classes=num_classes, lstm=True, bidirectional=False, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        elif self.name.startswith("fastai_gru"):
            model = RNN1d(input_channels=self.input_channels, num_classes=num_classes, lstm=False, bidirectional=False, ps_head=self.ps_head, lin_ftrs_head=self.lin_ftrs_head)
        else:
            raise ValueError("Model not found.")
            
        learn = Learner(dls, model, loss_func=loss, metrics=metrics, wd=self.wd, path=self.outputfolder)
        
        if self.name.startswith("fastai_lstm") or self.name.startswith("fastai_gru"):
            learn.add_cb(GradientClipping(clip=0.25))

        if self.early_stopping is not None:
            if self.early_stopping == "macro_auc" and self.loss != "mse" and self.loss != "nll_regression":
                metric = MetricFunc(auc_metric, self.early_stopping, one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True, flatten_target=False)
                learn.metrics.append(metric)
                learn.add_cb(SaveModelCallback(monitor=self.early_stopping, fname=self.name))
            elif self.early_stopping == "fmax" and self.loss != "mse" and self.loss != "nll_regression":
                metric = MetricFunc(fmax_metric, self.early_stopping, one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True, flatten_target=False)
                learn.metrics.append(metric)
                learn.add_cb(SaveModelCallback(monitor=self.early_stopping, fname=self.name))
            elif self.early_stopping == "valid_loss":
                learn.add_cb(SaveModelCallback(monitor=self.early_stopping, fname=self.name))
            
        return learn
