"""Implements the SoccerMap architecture."""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('medium') 

# --- Constants and Configuration ---
H, W = 68, 104  # Grid dimensions (Height, Width)
NUM_FEATURE_CHANNELS = 17#13 # Number of input channels for SoccerMap
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_X = PITCH_LENGTH / 2 # 52.5
CENTER_Y = PITCH_WIDTH / 2  # 34.0

CONV_CHANNEL = 256
class _FeatureExtractionLayer(nn.Module):
    """The 2D-convolutional feature extraction layer of the SoccerMap architecture.

    The probability at a single location is influenced by the information we
    have of nearby pixels. Therefore, convolutional filters are used for
    spatial feature extraction.

    Two layers of 2D convolutional filters with a 5 × 5 receptive field and
    stride of 1 are applied, each one followed by a ReLu activation function.
    To keep the same dimensions after the convolutional filters, symmetric
    padding is applied. It fills the padding cells with values that are
    similar to those around it, thus avoiding border-image artifacts that can
    hinder the model’s predicting ability and visual representation.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, 128, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding="valid")
        # (left, right, top, bottom)
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)
        return x
    
    # def __init__(self, in_channels):
    #     super().__init__()

    #     self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding="valid")
    #     self.conv_2 = nn.Conv2d(32, CONV_CHANNEL, kernel_size=(3, 3), stride=1, padding="valid")

    #     self.bn_1   = nn.BatchNorm2d(32)
    #     self.bn_2   = nn.BatchNorm2d(CONV_CHANNEL)

    #     # (left, right, top, bottom)
    #     self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    # def forward(self, x):
    #     x = F.relu(self.conv_1(x))
    #     x = self.symmetric_padding(x)
    #     x = F.relu(self.conv_2(x))
    #     x = self.symmetric_padding(x)
    #     return x


class _PredictionLayer(nn.Module):
    """The prediction layer of the SoccerMap architecture.

    The prediction layer consists of a stack of two convolutional layers, the
    first with 32 1x1 convolutional filters followed by an ReLu activation
    layer, and the second consists of one 1x1 convolutional filter followed by
    a linear activation layer. The spatial dimensions are kept at each step
    and 1x1 convolutions are used to produce predictions at each location.
    """

    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(CONV_CHANNEL, 32, kernel_size=(1, 1))
        # self.conv2 = nn.Conv2d(32, 1, kernel_size=(1, 1))
        self.conv1 = nn.Conv2d(128, 32, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)  # linear activation
        return x


class _UpSamplingLayer(nn.Module):
    """The upsampling layer of the SoccerMap architecture.

    The upsampling layer provides non-linear upsampling by first applying a 2x
    nearest neighbor upsampling and then two layers of convolutional filters.
    The first convolutional layer consists of 32 filters with a 3x3 activation
    field and stride 1, followed by a ReLu activation layer. The second layer
    consists of 1 layer with a 3x3 activation field and stride 1, followed by
    a linear activation layer. This upsampling strategy has been shown to
    provide smoother outputs.
    """

    def __init__(self):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding="valid")
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = self.symmetric_padding(x)
        x = self.conv2(x)  # linear activation
        x = self.symmetric_padding(x)
        return x


class _FusionLayer(nn.Module):
    """The fusion layer of the SoccerMap architecture.

    The fusion layer merges the final prediction surfaces at different scales
    to produce a final prediction. It concatenates the pair of matrices and
    passes them through a convolutional layer of one 1x1 filter.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1)

    def forward(self, x: List[torch.Tensor]):
        out = self.conv(torch.cat(x, dim=1))  # linear activation
        return out


class SoccerMap(nn.Module):
    """SoccerMap architecture.

    SoccerMap is a deep learning architecture that is capable of estimating
    full probability surfaces for pass probability, pass slection likelihood
    and pass expected values from spatiotemporal data.

    The input consists of a stack of c matrices of size lxh, each representing a
    subset of the available spatiotemporal information in the current
    gamestate. The specific choice of information for each of these c slices
    might vary depending on the problem being solved

    Parameters
    ----------
    in_channels : int, default: 13
        The number of spatiotemporal input channels.

    References
    ----------
    .. [1] Fernández, Javier, and Luke Bornn. "Soccermap: A deep learning
       architecture for visually-interpretable analysis in soccer." Joint
       European Conference on Machine Learning and Knowledge Discovery in
       Databases. Springer, Cham, 2020.
    """

    def __init__(self, model_config):
        super().__init__()

        self.in_channels = model_config["in_channels"]

        # Convolutions for feature extraction at 1x, 1/2x and 1/4x scale
        self.features_x1 = _FeatureExtractionLayer(self.in_channels)
        self.features_x2 = _FeatureExtractionLayer(128)
        self.features_x4 = _FeatureExtractionLayer(128)
        self.features_x8 = _FeatureExtractionLayer(128)
        self.features_x16 = _FeatureExtractionLayer(128)

        # Layers for down and upscaling and merging scales
        self.up_x2 = _UpSamplingLayer()
        self.up_x4 = _UpSamplingLayer()

        # Layers for down and upscaling and merging scales
        self.down_x2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x8 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x16 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.fusion_x2_x4 = _FusionLayer()
        self.fusion_x1_x2 = _FusionLayer()

        # Prediction layers at each scale
        self.prediction_x1 = _PredictionLayer()
        self.prediction_x2 = _PredictionLayer()
        self.prediction_x4 = _PredictionLayer()
        self.prediction_x8 = _PredictionLayer()
        self.prediction_x16 = _PredictionLayer()

        # output layer: binary classification
        self.output_layer = nn.Sequential(
            nn.Flatten(),  # Flatten to (batch_size, num_features)
            nn.Linear((68 // 16) * (104 // 16), 1),  # Linear layer to output a single value
        )

    def forward(self, x):
        # Feature extraction
        f_x1 = self.features_x1(x) #[B*T, D, H, W]
        f_x2 = self.features_x2(self.down_x2(f_x1)) #[B*T, D, H//2, W//2]
        f_x4 = self.features_x4(self.down_x4(f_x2)) #[B*T, D, H//4, W//4]
        f_x8 = self.features_x8(self.down_x8(f_x4)) #[B*T, D, H//8, W//8]
        f_x16 = self.features_x16(self.down_x16(f_x8)) #[B*T, D, H//16, W//16]
        
        # Prediction
        pred_x16 = self.prediction_x16(f_x16) #[B*T, 1, H//16, W//16]

        # The activation function depends on the problem
        return self.output_layer(pred_x16) # For Base SoccerMap

        # return pred_x16 # For TemporalSoccerMap
        
def pixel(surface, mask):
    """Return the prediction at a single pixel.

    This custom layer is used to evaluate the loss at the pass destination.

    Parameters
    ----------
    surface : torch.Tensor
        The final prediction surface.
    mask : torch.Tensor
        A sparse spatial representation of the final pass destination.

    Returns
    -------
    torch.Tensor
        The prediction at the cell on the surface that matches the actual
        pass destination.
    """
    masked = surface * mask
    value = torch.sum(masked, dim=(3, 2))
    return value

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean"):
        """
        Focal Loss implementation.

        Args:
            alpha (float): Weighting factor for the positive class.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predicted probabilities (output of sigmoid).
            targets (Tensor): Ground truth labels (binary, 0 or 1).

        Returns:
            Tensor: Computed focal loss.
        """
        # Clip inputs to avoid log(0)
        inputs = torch.clamp(inputs, 1e-8, 1 - 1e-8)

        # Compute the binary cross entropy loss
        bce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))

        # Compute the modulating factor
        pt = torch.where(targets == 1, inputs, 1 - inputs)  # p_t
        modulating_factor = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Compute the focal loss
        focal_loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class PytorchSoccerMapModel(pl.LightningModule):
    """A pass success probability model based on the SoccerMap architecture."""

    def __init__(self, model_config: dict = None, optimizer_params: dict = None):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(model_config=model_config)
        #self.sigmoid = nn.Sigmoid()

        # loss function
        #self.criterion = torch.nn.BCELoss()
        #self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = FocalLoss()

        pos_weight_tensor = torch.tensor([4.0]) 
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # loss 계산 (sigmoid 내장됨)

        self.optimizer_params = optimizer_params

        # self.train_accuracy = BinaryAccuracy()
        # self.val_accuracy = BinaryAccuracy()
        # self.test_accuracy = BinaryAccuracy()
        # self.train_auroc = BinaryAUROC()
        # self.val_auroc = BinaryAUROC()
        # self.test_auroc = BinaryAUROC()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        #x = self.sigmoid(x)

        return x

    def step(self, batch: Any):
        # x, y = batch
        # y_hat = self.forward(x)

        x, y = batch
        y_hat = self.forward(x)

        # logits = self.forward(x)
        # y = y.float().view_as(logits)
        # x, mask, y = batch
        # surface = self.forward(x)
        # y_hat = pixel(surface, mask)

        loss = self.criterion(y_hat, y)
        preds_prob = torch.sigmoid(y_hat)
        #loss = self.criterion(y_hat, y)
        return loss, preds_prob, y
    
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log train metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
 
         # --- Update and Log Training Accuracy ---
        # self.train_accuracy.update(preds, targets.int()) 
        # self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.train_auroc.update(preds, targets.int())
        # self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True) # Usually not shown on prog bar
        preds_label = (preds >= 0.5).int()
        acc = (preds_label == targets.int()).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        try:
            y_true = targets.detach().cpu().numpy()
            y_score = preds.detach().cpu().numpy()
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_score)
                self.log("train_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            # 필요한 경우: 로그 남기기 또는 무시
            pass

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log val metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Update and Log Validation Accuracy ---
        # self.val_accuracy.update(preds, targets.int())
        # self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.val_auroc.update(preds, targets.int())
        # self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True) # Usually not shown on prog bar

        preds_label = (preds >= 0.5).int()
        acc = (preds_label == targets.int()).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        try:
            y_true = targets.detach().cpu().numpy()
            y_score = preds.detach().cpu().numpy()
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_score)
                self.log("val_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            # 필요한 경우: 로그 남기기 또는 무시
            pass

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # --- Update and Log Test Accuracy ---
        # self.test_accuracy.update(preds, targets.int())
        # self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, logger=True)
        # self.test_auroc.update(preds, targets.int())
        # self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, logger=True)

        preds_label = (preds >= 0.5).int()
        acc = (preds_label == targets.int()).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        try:
            y_true = targets.detach().cpu().numpy()
            y_score = preds.detach().cpu().numpy()
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_score)
                self.log("test_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            # 필요한 경우: 로그 남기기 또는 무시
            pass

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):

        x, y = batch
        
        # x = x.to("cuda")
        # y = y.to("cuda")
        # x, mask, y = batch
    
        preds = self(x)  # self.forward(x)
        preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
        return preds, y

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return torch.optim.Adam(self.parameters(), **self.optimizer_params["optimizer_params"])


class TemporalSoccerMapModel(pl.LightningModule):
    """A pass success probability model based on the SoccerMap architecture."""

    def __init__(self, model_config: dict = None, optimizer_params: dict = None):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        
        self.spatial_model = SoccerMap(model_config=model_config)
        self.temporal_model = nn.LSTM(
            input_size=4*6,
            hidden_size=128,
            num_layers=3,
            batch_first=True, # Expects input shape [B, T, Features]
            dropout=0.4,
            bidirectional=True # Set True for Bidirectional LSTM
        )
        self.spatial_output_layer = nn.Sequential(
            nn.Linear(128*2, 1),  # Linear layer to output a single value
        )
        self.temporal_output_layer = nn.Sequential(
            nn.Flatten(),  # Flatten to (batch_size, num_features)
            nn.Linear(150, 1),  # Linear layer to output a single value
        )

        self.sigmoid = nn.Sigmoid()

        # loss function
        # self.criterion = torch.nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = FocalLoss()

        self.optimizer_params = optimizer_params

        # self.train_accuracy = BinaryAccuracy()
        # self.val_accuracy = BinaryAccuracy()
        # self.test_accuracy = BinaryAccuracy()
        # self.train_auroc = BinaryAUROC()
        # self.val_auroc = BinaryAUROC()
        # self.test_auroc = BinaryAUROC()

    def forward(self, x: torch.Tensor):
        b, tc, h, w = x.shape
        x = x.view(b*150, 13, h, w)
        out = self.spatial_model(x) #[B*150, 1, 4, 6]
        out = out.view(b*150, -1) #[B*150, 4*6]
        out = out.view(b, 150, -1) #[B, 150, 4*6]
        out, (h_n, c_n) = self.temporal_model(out) # out: [B, T, 2 * 128] h_n: [3*2, B, D]
        out = out[:, -1, :] #[B, 2*128]
        out = self.spatial_output_layer(out) #[B, 1]
        # out = self.temporal_output_layer(out) #[B, 1]
        # out = self.sigmoid(out)

        return out

    def step(self, batch: Any):
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        logits = self.forward(x)
        y = y.float().view_as(logits)
        # x, mask, y = batch
        # surface = self.forward(x)
        # y_hat = pixel(surface, mask)

        loss = self.criterion(logits, y)
        preds_prob = torch.sigmoid(logits)

        return loss, preds_prob, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log train metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
 
         # --- Update and Log Training Accuracy ---
        # self.train_accuracy.update(preds, targets.int()) 
        # self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.train_auroc.update(preds, targets.int())
        # self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=False) # Usually not shown on prog bar
        
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log val metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Update and Log Validation Accuracy ---
        # self.val_accuracy.update(preds, targets.int())
        # self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.val_auroc.update(preds, targets.int())
        # self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=False) # Usually not shown on prog bar
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # --- Update and Log Test Accuracy ---
        # self.test_accuracy.update(preds, targets.int())
        # self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, logger=True)
        # self.test_auroc.update(preds, targets.int())
        # self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, logger=True)


        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        # x, mask, y = batch

        preds = self(x)  # self.forward(x)

        return preds, y

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return torch.optim.Adam(self.parameters(), **self.optimizer_params["optimizer_params"])

class PressGNN(nn.Module):
    def __init__(self,
                model_config : dict,
                 ): # Flag to use pressing intensity
        super().__init__()
        self.feature_dim = model_config['in_channels']
        self.num_gnn_layers = model_config['num_gnn_layers']
        self.gnn_hidden_dim = model_config['gnn_hidden_dim']
        self.num_lstm_layers = model_config['num_lstm_layers']
        self.lstm_hidden_dim = model_config['lstm_hidden_dim']
        self.lstm_dropout = model_config['lstm_dropout']
        self.lstm_bidirectional = model_config['lstm_bidirectional']
        self.use_pressing_features = model_config['use_pressing_features']
        self.gnn_heads = model_config['gnn_head']
        
        # GNN Layers (using edge_weight for pressure)
        self.gnn_layers = nn.ModuleList()
        in_c = self.feature_dim
        # self.gnn_layers.append(GCNConv(self.feature_dim, self.gnn_hidden_dim)) # GCNConv can use edge_weight
        for i in range(self.num_gnn_layers):
            if i < self.num_gnn_layers - 1:
                self.gnn_layers.append(
                    GATConv(in_c, self.gnn_hidden_dim, heads=self.gnn_heads, concat=True)
                )
                in_c = self.gnn_hidden_dim * self.gnn_heads
            else:
                self.gnn_layers.append(GATConv(in_c, self.gnn_hidden_dim, heads=1, concat=False)) # GCNConv can use edge_weight
                in_c = self.gnn_hidden_dim

        # Classifier Head
        self.classifier = nn.Linear(self.gnn_hidden_dim, 1)

    def forward(self, batch: Dict) -> torch.Tensor:
        #batch['features'] = batch['features'][:, -1:, ...] # 마지막 프레임만 사용 -> all frame. feat. geonhee

        B, T, A, F_in = batch['features'].shape
        device = batch['features'].device
        x_frames = batch['features'].reshape(B*T, A, F_in) # [B, A, T, F_in]
        # node_repr = batch['features'].reshape(B * T, A, F_in)
        agent_order_lst = batch['agent_order']
        pressed_id_lst = batch['pressed_id']
        
        data_list = []
        for i in range(B*T):
            feats = x_frames[i]  # [A, temporal_hidden_dim]
            adj = torch.ones((A, A), device=device) - torch.eye(A, device=device)
            edge_index, edge_weight = dense_to_sparse(adj)

            # 2. 결정된 edge_index를 기반으로 엣지 특징을 계산합니다.
            source_nodes, dest_nodes = edge_index[0], edge_index[1]
            # 2-1. 특징: distance_between_nodes
            # feats의 첫 두 컬럼이 x, y 좌표
            pos_source = feats[source_nodes, :2]
            pos_dest = feats[dest_nodes, :2]
            edge_distances = torch.linalg.norm(pos_source - pos_dest, dim=1).unsqueeze(1)

            # 2-2. 특징: is_same_team (인덱스 기반으로 계산)
            # 홈팀: 인덱스 0~10, 원정팀: 인덱스 11~21
            source_is_home = source_nodes < 11
            dest_is_home = dest_nodes < 11
            source_is_away = (source_nodes >= 11) & (source_nodes < 22)
            dest_is_away = (dest_nodes >= 11) & (dest_nodes < 22)
            
            # 같은 팀인 경우: (둘 다 홈팀) 또는 (둘 다 원정팀)
            is_same = (source_is_home & dest_is_home) | (source_is_away & dest_is_away)
            edge_same_team = is_same.float().unsqueeze(1)

            edge_attr = torch.cat([edge_distances, edge_same_team], dim=-1)

            data_list.append(Data(x=feats, edge_index=edge_index, edge_attr=edge_attr))    

        batch_graph = Batch.from_data_list(data_list)
        # Spatial GNN: apply each layer
        x = batch_graph.x
        for conv in self.gnn_layers:
            x = conv(x, batch_graph.edge_index, batch_graph.edge_attr)
            x = torch.relu(x)
        
        # Global Pooling
        pooled = global_mean_pool(x, batch_graph.batch)     # [B, gnn_hidden_dim]
        logits = self.classifier(pooled).squeeze(-1)  # [B]

        return logits


class STLSTMGNN(nn.Module):
    def __init__(self,
                model_config : dict,
                 ): # Flag to use pressing intensity
        super().__init__()
        self.feature_dim = model_config['in_channels']
        self.num_gnn_layers = model_config['num_gnn_layers']
        self.gnn_hidden_dim = model_config['gnn_hidden_dim']
        self.num_lstm_layers = model_config['num_lstm_layers']
        self.lstm_hidden_dim = model_config['lstm_hidden_dim']
        self.lstm_dropout = model_config['lstm_dropout']
        self.lstm_bidirectional = model_config['lstm_bidirectional']
        self.use_pressing_features = model_config['use_pressing_features']
        self.gnn_heads = model_config['gnn_head']
        
        # GNN Layers (using edge_weight for pressure)
        self.gnn_layers = nn.ModuleList()
        in_c = self.feature_dim
        # self.gnn_layers.append(GCNConv(self.feature_dim, self.gnn_hidden_dim)) # GCNConv can use edge_weight
        for i in range(self.num_gnn_layers):
            if i < self.num_gnn_layers - 1:
                self.gnn_layers.append(
                    GATConv(in_c, self.gnn_hidden_dim, heads=self.gnn_heads, concat=True)
                )
                in_c = self.gnn_hidden_dim * self.gnn_heads
            else:
                self.gnn_layers.append(GATConv(in_c, self.gnn_hidden_dim, heads=1, concat=False)) # GCNConv can use edge_weight
                in_c = self.gnn_hidden_dim

        # LSTM Layer
        self.num_directions = 2 if self.lstm_bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=self.gnn_hidden_dim,
            hidden_size=self.lstm_hidden_dim, num_layers=self.num_lstm_layers, batch_first=True,
            dropout=self.lstm_dropout if self.num_lstm_layers > 1 else 0, bidirectional=self.lstm_bidirectional)

        
        # Classifier Head
        self.classifier = nn.Linear(self.lstm_hidden_dim * self.num_directions, 1)

    def forward(self, batch: Dict) -> torch.Tensor:
        features, seq_lengths = batch['features'], batch['seq_lengths']
        B, T_max, A, F_in = batch['features'].shape
        device = features.device
        mask = torch.arange(T_max, device=device).expand(B, T_max) < seq_lengths.unsqueeze(1)
        valid_frames = features[mask]  # Shape: [num_valid_frames, A, F_in]
        num_valid_frames = valid_frames.shape[0]
        
        # x_frames = batch['features'].reshape(B*T, A, F_in) # [B, A, T, F_in]
        
        data_list = []
        for i in range(num_valid_frames):
            feats = valid_frames[i]  # [A, temporal_hidden_dim]
            # feats = x_frames[i]  # [A, temporal_hidden_dim]
            adj = torch.ones((A, A), device=device) - torch.eye(A, device=device)
            edge_index, edge_weight = dense_to_sparse(adj)

            # 2. 결정된 edge_index를 기반으로 엣지 특징을 계산합니다.
            source_nodes, dest_nodes = edge_index[0], edge_index[1]
            # 2-1. 특징: distance_between_nodes
            # feats의 첫 두 컬럼이 x, y 좌표
            pos_source = feats[source_nodes, :2]
            pos_dest = feats[dest_nodes, :2]
            edge_distances = torch.linalg.norm(pos_source - pos_dest, dim=1).unsqueeze(1)

            # 2-2. 특징: is_same_team (인덱스 기반으로 계산)
            # 홈팀: 인덱스 0~10, 원정팀: 인덱스 11~21
            source_is_home = source_nodes < 11
            dest_is_home = dest_nodes < 11
            source_is_away = (source_nodes >= 11) & (source_nodes < 22)
            dest_is_away = (dest_nodes >= 11) & (dest_nodes < 22)
            
            # 같은 팀인 경우: (둘 다 홈팀) 또는 (둘 다 원정팀)
            is_same = (source_is_home & dest_is_home) | (source_is_away & dest_is_away)
            edge_same_team = is_same.float().unsqueeze(1)

            # 거리, 팀여부 2가지 특징 결합
            edge_attr = torch.cat([edge_distances, edge_same_team], dim=-1)

            data_list.append(Data(x=feats, edge_index=edge_index, edge_attr=edge_attr))    

        batch_graph = Batch.from_data_list(data_list)
        # Spatial GNN: apply each layer
        x = batch_graph.x
        for conv in self.gnn_layers:
            x = conv(x, batch_graph.edge_index, batch_graph.edge_attr)
            x = torch.relu(x)
        
        # Global Pooling
        pooled = global_mean_pool(x, batch_graph.batch)     # [B, gnn_hidden_dim]
        
        # Temporal encoder
        lstm_input = torch.zeros(B, T_max, self.gnn_hidden_dim, device=device)
        lstm_input[mask] = pooled
        # lstm_input = pooled.view(B, T_max, self.gnn_hidden_dim)
        packed_input = pack_padded_sequence(
            lstm_input, 
            seq_lengths.cpu(), # 길이는 CPU 텐서여야 합니다.
            batch_first=True, 
            enforce_sorted=False
        )
        lstm_out, (h_n, _) = self.lstm(packed_input)  # h_n: [1, B*A, hidden]
        # lstm_out, (h_n, _) = self.lstm(x_frames)  # h_n: [1, B*A, hidden]
        if self.lstm_bidirectional:
            h_fwd = h_n[-2, :, :]
            h_bwd = h_n[-1, :, :]
            final_hidden = torch.cat([h_fwd, h_bwd], dim=1)  # [B*N, 2H]
        else:
            final_hidden = h_n[-1, :, :]
        
        logits = self.classifier(final_hidden).squeeze(-1)  # [B]
        return logits


class LSTMGNN(nn.Module):
    """
    Temporal-first (per-player LSTM) + Spatial-second (GNN) architecture.
    """
    def __init__(self, model_config: dict):
        super().__init__()

        # ----- config -----
        self.in_channels      = model_config['in_channels']        # F_in
        self.lstm_hidden_dim  = model_config['lstm_hidden_dim']
        self.num_lstm_layers  = model_config['num_lstm_layers']
        self.lstm_dropout     = model_config['lstm_dropout']
        self.bidirectional    = model_config['lstm_bidirectional']
        self.num_gnn_layers   = model_config['num_gnn_layers']
        self.gnn_hidden_dim   = model_config['gnn_hidden_dim']
        self.gnn_heads        = model_config['gnn_head']

        self.embed = nn.Embedding(
            num_embeddings=19+1,  # 전체 임베딩할 항목 수 (패딩 포함)
            embedding_dim=4,  # 임베딩 차원
            padding_idx=0  # 패딩값은 0으로 지정
        )
        # self.input_fc = nn.Linear(self.in_channels-1+4, self.lstm_hidden_dim)
        self.input_fc = nn.Linear(self.in_channels-1, self.lstm_hidden_dim)

        # ----- LSTM (per player) -----
        common_kwargs = dict(
            input_size   = self.lstm_hidden_dim,#self.in_channels,
            hidden_size  = self.lstm_hidden_dim,
            num_layers   = self.num_lstm_layers,
            batch_first  = True,
            dropout      = self.lstm_dropout,
            bidirectional= self.bidirectional
        )
        self.rnn_type = "gru"
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(**common_kwargs)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(**common_kwargs)
        else:
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")

        # ----- GNN -----
        self.num_dirs = 2 if self.bidirectional else 1
        in_c = self.lstm_hidden_dim * self.num_dirs
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            if i < self.num_gnn_layers - 1:
                self.gnn_layers.append(GATConv(in_c, self.gnn_hidden_dim,
                                            heads=self.gnn_heads, concat=True))
                in_c = self.gnn_hidden_dim * self.gnn_heads
            else:
                self.gnn_layers.append(GATConv(in_c, self.gnn_hidden_dim, heads=1, concat=False))
                in_c = self.gnn_hidden_dim

        # ----- Classifier -----
        self.classifier = nn.Linear(self.gnn_hidden_dim, 1) 

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch:
            features: [B, T_max, A, F_in]
            seq_lengths: [B]  (valid lengths, 동일 길이 → A 선수 모두 공유)
        """
        x, seq_lens = batch['features'], batch['seq_lengths']     # shapes as above
        B, T_max, A, F_in = x.shape # (B, T, A, F)
        device = x.device

        # ----- 0) type_name(-1) embedding -----
        numeric_features = x[:, :, :, :-1]  # [B, T_max, A, F_in-1]
        # categorical_features = x[:, :, :, -1].long()  # [B, T_max, A] (type_name)
        # categorical_features = self.embed(categorical_features)  # [B, T_max, A, 4]
        # x = torch.cat([numeric_features, categorical_features], dim=-1)  # [B, T_max, A, F_in-1+4]
        x = numeric_features
        x = self.input_fc(x)  # [B, T_max, A, H_lstm] (F_in-1+4 → H_lstm)
        F_in = self.lstm_hidden_dim  # update F_in to H_lstm
        
        # ---------- 1) per-player LSTM ---------- #
        #   reshape → [B*A, T_max, F_in]
        x_flat = x.permute(0, 2, 1, 3).reshape(B*A, T_max, F_in)

        #   length 벡터: 각 선수 동일 → 반복
        lengths_flat = seq_lens.repeat_interleave(A).cpu()
        packed = pack_padded_sequence(
            x_flat, lengths_flat, batch_first=True, enforce_sorted=False
        )
        rnn_out  = self.rnn(packed)
    
        # hidden state 꺼내기
        if self.rnn_type == "lstm":
            h_n = rnn_out[1][0]          # (h_n, c_n) 중 h_n
        else:  # gru
            h_n = rnn_out[1]             # rnn_out = (output, h_n)

        if self.bidirectional:
            player_emb = torch.cat([h_n[-2], h_n[-1]], dim=1)     # [B*A, 2H]
        else:
            player_emb = h_n[-1]                                       # [B*A, H]
        player_emb = player_emb.reshape(B, A, -1)              # [B, A, H_lstm*D]

        # ---------- 2) build graph & run GNN ---------- #
        data_list = []
        for b in range(B):
            # positional 정보: 마지막 valid frame 기준
            t_last = seq_lens[b] - 1
            pos = x[b, t_last, :, :2]                   # [A, 2]  (x, y)

            # fully-connected (자유롭게 수정 가능)
            adj = torch.ones(A, A, device=device) - torch.eye(A, device=device)
            edge_index, _ = dense_to_sparse(adj)

            # edge_attr: [distance | same_team]
            source_nodes, dest_nodes = edge_index[0], edge_index[1]
            
            same_team = (((source_nodes < 11) & (dest_nodes < 11)) |
                         ((source_nodes >= 11) & (source_nodes < 22) & (dest_nodes >= 11) & (dest_nodes < 22))
                        ).float().unsqueeze(1)
        
            edge_attr = torch.cat([same_team], dim=1)  # [E, 1]

            data_list.append(
                Data(x=player_emb[b], edge_index=edge_index, edge_attr=edge_attr)
            )

        batch_graph = Batch.from_data_list(data_list)
        x_g = batch_graph.x
        for conv in self.gnn_layers:
            x_g = torch.relu(conv(x_g, batch_graph.edge_index, batch_graph.edge_attr))

        # ---------- 3) pooling & head ---------- #
        graph_repr = global_mean_pool(x_g, batch_graph.batch)  # [B, gnn_hidden_dim]
        logits = self.classifier(graph_repr).squeeze(-1)       # [B]
        return logits

class exPressModel(pl.LightningModule):
    def __init__(self, model_config: dict = None, optimizer_params: dict = None):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate the new GNN+LSTM model
        #self.model = STLSTMGNN(model_config=model_config) # Multi frame
        #self.model = SetTransformer(model_config=model_config) # Multi frame
        self.model = LSTMGNN(model_config=model_config) # Multi frame
        #self.model = PressGNN(model_config=model_config) # Only one frame
        #self.sigmoid = nn.Sigmoid()

        #self.criterion = nn.BCELoss()
        #self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
        pos_weight_tensor = torch.tensor([4.0]) 
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # loss 계산 (sigmoid 내장됨)
        #self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer_params = optimizer_params

        # self.train_accuracy = BinaryAccuracy()
        # self.val_accuracy = BinaryAccuracy()
        # self.test_accuracy = BinaryAccuracy()
        # self.train_auroc = BinaryAUROC()
        # self.val_auroc = BinaryAUROC()
        # self.test_auroc = BinaryAUROC()
        # self.test_f1_score = BinaryF1Score(threshold=0.25)
        
    def forward(self, batch: dict) -> torch.Tensor:
        # Pass the whole batch dictionary to the model's forward
        x = self.model(batch)
        #x = self.sigmoid(x)  # Apply sigmoid activation to the output

        return x

    def step(self, batch: dict) -> tuple:
        # Model forward now takes the batch dictionary
        logits = self.forward(batch) # Get single logit output [B, 1]
        y = batch['label'] # Extract label

        y = y.float().view_as(logits)
        loss = self.criterion(logits, y)
        
        #loss = self.criterion(logits, y)
        #return loss, logits, y
    
        preds_prob = torch.sigmoid(logits)
        return loss, preds_prob, y
    
    # training_step, validation_step, test_step remain structurally the same:
    # call self.step, update/log metrics (using the metric objects)
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.step(batch)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        preds_label = (preds >= 0.5).int()
        acc = (preds_label == targets.int()).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        try:
            y_true = targets.detach().cpu().numpy()
            y_score = preds.detach().cpu().numpy()
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_score)
                self.log("train_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            # 필요한 경우: 로그 남기기 또는 무시
            pass

        # targets_int = targets.int()
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.train_accuracy.update(preds, targets_int); self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        # try: 
        #     self.train_auroc.update(preds, targets_int); self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        # except ValueError: 
        #     pass

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Optional[Dict]:
        loss, preds, targets = self.step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        preds_label = (preds >= 0.5).int()
        acc = (preds_label == targets.int()).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        try:
            y_true = targets.detach().cpu().numpy()
            y_score = preds.detach().cpu().numpy()
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_score)
                self.log("val_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            # 필요한 경우: 로그 남기기 또는 무시
            pass

        # targets_int = targets.int()
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # self.val_accuracy.update(preds, targets_int); self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # try: 
        #     self.val_auroc.update(preds, targets_int); self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        # except ValueError: 
        #     pass

        # return {"val_loss": loss}
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> Optional[Dict]:
        loss, preds, targets = self.step(batch)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        preds_label = (preds >= 0.5).int()
        acc = (preds_label == targets.int()).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        try:
            y_true = targets.detach().cpu().numpy()
            y_score = preds.detach().cpu().numpy()
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_score)
                self.log("test_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        except Exception as e:
            # 필요한 경우: 로그 남기기 또는 무시
            pass

        # targets_int = targets.int()
        # self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # self.test_accuracy.update(preds, targets_int); self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, logger=True)
        # try: 
        #     self.test_auroc.update(preds, targets_int); self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, logger=True)
        # except ValueError: 
        #     pass

        # self.test_f1_score.update(preds, targets_int); self.log("test_f1", self.test_f1_score, on_step=False, on_epoch=True, logger=True)
        # brier_score = torch.mean((preds - targets.float())**2)
        # self.log("test_brier", brier_score, on_step=False, on_epoch=True, logger=True)
        # return {"test_loss": loss}
        return loss

    def predict_step(self, batch: Any, batch_idx: int):

        
        # x = x.to("cuda")
        # y = y.to("cuda")
        # x, mask, y = batch
    
        preds = self(batch)  # self.forward(x)
        preds = torch.sigmoid(preds)  # Apply sigmoid activation to the output
        return preds, batch["label"]
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return torch.optim.Adam(self.parameters(), **self.optimizer_params["optimizer_params"])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
    
class SetTransformer(nn.Module):
    def __init__(self, model_config):
        # dim_input, num_outputs, dim_output, num_inds=32, 
        #          dim_hidden=256, num_heads=8, dim_feedforward=1024, dropout=0.1, 
        #          num_seeds=4, num_layers=6, ln=False, categorical_indices=None):
        
        super(SetTransformer, self).__init__()

        dim_hidden = 128
        num_heads = 16
        num_layers = 1
        dropout = 0.1
        self.input_fc = nn.Linear(6, dim_hidden)

        # batch_first=False: (Seq, Batch, Feature) -> Spatial Transformer관점에서는 (N, B*W, F)
        #self.player_pos_encoder = PositionalEncoding(dim_hidden)
        self.player_encoder = TransformerEncoder(
            TransformerEncoderLayer(dim_hidden, num_heads, dim_hidden * 2, dropout, activation="gelu"),
            num_layers,
        )

        self.player_encoder1 = TransformerEncoder(
            TransformerEncoderLayer(dim_hidden, num_heads, dim_hidden * 2, dropout, activation="gelu"),
            num_layers,
        )

        # batch_first=False: (Seq, Batch, Feature) -> Temporal Transformer관점에서는 (W, B*N, F)
        self.time_pos_encoder = PositionalEncoding(dim_hidden)
        self.time_encoder = TransformerEncoder(
            TransformerEncoderLayer(dim_hidden, num_heads, dim_hidden * 2, dropout, activation="gelu"),
            num_layers,
        )

        self.time_encoder1 = TransformerEncoder(
            TransformerEncoderLayer(dim_hidden, num_heads, dim_hidden * 2, dropout, activation="gelu"),
            num_layers,
        )

        self.fc = nn.Sequential( # 속도 예측용
            nn.Linear(23*dim_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  
        )
    

    def forward(self, batch):
        X, seq_lengths = batch['features'], batch['seq_lengths']
        B, W, N, F = X.shape  # (Batch, Window, Players, Features)   

        # Step1. input projection
        X = self.input_fc(X)  # (B, W, N, dim_hidden)

        X = (
            X.permute(2, 0, 1, 3) # (B, W, N+N', dim_hidden) -> (N+N', B, W, dim_hidden)
            .contiguous()         # 메모리 연속성 확보
            .view(N, B*W, -1)   # (N+N', B, W, dim_hidden) -> (N+N', B*W, dim_hidden)
        )                
        #X = self.player_pos_encoder(X)
        X = self.player_encoder(X)#, src_key_padding_mask=padding_mask)  # (N+N', B*W, dim_hidden)

        # Step 4. temporal transformer1
        X = X[:N, :, :]       # freeze_frame정보는 temporal정보 학습 못함: (N+N', B*W, dim_hidden) -> (N, B*W, dim_hidden)
        X = (
            X.view(N, B, W, -1)  # (N, B*W, dim_hidden) -> (N, B, W, dim_hidden)
            .permute(2, 1, 0, 3) # (N, B, W, dim_hidden) -> (W, B, N, dim_hidden)
            .contiguous()        # 메모리 연속성 확보
            .view(W, B*N, -1)    # (W, B, N, dim_hidden) -> (W, B*N, dim_hidden)
        )
        X = self.time_pos_encoder(X)
        X = self.time_encoder(X)  # (W, B*N, dim_hidden)
        
        # Step 5. spatial transformer2
        X = (
            X.view(W, B, N, -1)  # (W, B*N, dim_hidden) -> (W, B, N, dim_hidden)
            .permute(1, 0, 2, 3) # (W, B, N, dim_hidden) -> (B, W, N, dim_hidden)
            .contiguous()        # 메모리 연속성 확보
        )
        
        X = (
            X.permute(2, 0, 1, 3) # (B, W, N+N', dim_hidden) -> (N+N', B, W, dim_hidden)
            .contiguous()         # 메모리 연속성 확보
            .view(N, B*W, -1)   # (N+N', B, W, dim_hidden) -> (N+N', B*W, dim_hidden)
        )                
        X = self.player_encoder1(X)#, src_key_padding_mask=padding_mask)  # (N, B*W, dim_hidden)

        
        # Step 6. temporal transformer2
        X = X[:N, :, :]       # freeze_frame정보는 temporal정보 학습 못함: (N+N', B*W, dim_hidden) -> (N, B*W, dim_hidden)
        X = (
            X.view(N, B, W, -1)  # (N, B*W, dim_hidden) -> (N, B, W, dim_hidden)
            .permute(2, 1, 0, 3) # (N, B, W, dim_hidden) -> (W, B, N, dim_hidden)
            .contiguous()        # 메모리 연속성 확보
            .view(W, B*N, -1)    # (W, B, N, dim_hidden) -> (W, B*N, dim_hidden)
        )
    
        X = self.time_pos_encoder(X)
        X = self.time_encoder1(X)  # (W, B*N, dim_hidden)


        # Step 7. output projection
        X = (
            X.view(W, B, N, -1)  # (W, B*N, dim_hidden) -> (W, B, N, dim_hidden)
            .permute(1, 0, 2, 3) # (W, B, N, dim_hidden) -> (B, W, N, dim_hidden)
            .contiguous()        # 메모리 연속성 확보
        )[:, -1, :, :]         # (B, N, W, dim_hidden) -> (B, N, dim_hidden)

        X = (
            X.view(B, -1)
            .contiguous()  # (B, N, dim_hidden) -> (B, N*dim_hidden) 
        )
        output = self.fc(X)  # (B, N, 2)  ->  (x,y)

        return output  # 최종 예측 좌표 (x, y) or (vx, vy, x, y)