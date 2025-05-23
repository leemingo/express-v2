"""Implements the SoccerMap architecture."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch, dense_to_batch

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


# --- Constants and Configuration ---
H, W = 68, 104  # Grid dimensions (Height, Width)
NUM_FEATURE_CHANNELS = 13 # Number of input channels for SoccerMap
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
CENTER_X = PITCH_LENGTH / 2 # 52.5
CENTER_Y = PITCH_WIDTH / 2  # 34.0


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
        self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding="valid")
        # (left, right, top, bottom)
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)
        return x


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
        self.conv1 = nn.Conv2d(64, 32, kernel_size=(1, 1))
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


# class SoccerMap(nn.Module):
#     """SoccerMap architecture.

#     SoccerMap is a deep learning architecture that is capable of estimating
#     full probability surfaces for pass probability, pass slection likelihood
#     and pass expected values from spatiotemporal data.

#     The input consists of a stack of c matrices of size lxh, each representing a
#     subset of the available spatiotemporal information in the current
#     gamestate. The specific choice of information for each of these c slices
#     might vary depending on the problem being solved

#     Parameters
#     ----------
#     in_channels : int, default: 13
#         The number of spatiotemporal input channels.

#     References
#     ----------
#     .. [1] Fernández, Javier, and Luke Bornn. "Soccermap: A deep learning
#        architecture for visually-interpretable analysis in soccer." Joint
#        European Conference on Machine Learning and Knowledge Discovery in
#        Databases. Springer, Cham, 2020.
#     """

#     def __init__(self, model_config):
#         super().__init__()

#         self.in_channels = model_config["in_channels"]

#         # Convolutions for feature extraction at 1x, 1/2x and 1/4x scale
#         self.features_x1 = _FeatureExtractionLayer(self.in_channels)
#         self.features_x2 = _FeatureExtractionLayer(64)
#         self.features_x4 = _FeatureExtractionLayer(64)
#         self.features_x8 = _FeatureExtractionLayer(64)
#         self.features_x16 = _FeatureExtractionLayer(64)

#         # Layers for down and upscaling and merging scales
#         # self.up_x2 = _UpSamplingLayer()
#         # self.up_x4 = _UpSamplingLayer()
#         self.down_x2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.down_x4 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.down_x8 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.down_x16 = nn.MaxPool2d(kernel_size=(2, 2))

#         # self.fusion_x2_x4 = _FusionLayer()
#         # self.fusion_x1_x2 = _FusionLayer()

#         # Prediction layers at each scale
#         # self.prediction_x1 = _PredictionLayer()
#         # self.prediction_x2 = _PredictionLayer()
#         # self.prediction_x4 = _PredictionLayer()
#         # self.prediction_x8 = _PredictionLayer()
#         self.prediction_x16 = _PredictionLayer()

#         # output layer: binary classification
#         self.output_layer = nn.Sequential(
#             nn.Flatten(),  # Flatten to (batch_size, num_features)
#             nn.Linear((68 // 16) * (104 // 16), 1),  # Linear layer to output a single value
#         )

#     def forward(self, x):
#         # Feature extraction
#         f_x1 = self.features_x1(x)
#         f_x2 = self.features_x2(self.down_x2(f_x1))
#         f_x4 = self.features_x4(self.down_x4(f_x2))
#         f_x8 = self.features_x8(self.down_x8(f_x4))
#         f_x16 = self.features_x16(self.down_x16(f_x8))

#         pred_x16 = self.prediction_x16(f_x16)

#         # Prediction
#         # pred_x1 = self.prediction_x1(f_x1)
#         # pred_x2 = self.prediction_x2(f_x2)
#         # pred_x4 = self.prediction_x4(f_x4)

#         # Fusion
#         # x4x2combined = self.fusion_x2_x4([self.up_x4(pred_x4), pred_x2])
#         # combined = self.fusion_x1_x2([self.up_x2(x4x2combined), pred_x1]) # (bs, 1, 68, 104)

#         # The activation function depends on the problem
#         return self.output_layer(pred_x16)  # Output shape: (bs, 1)
#         # return combined

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
        self.features_x2 = _FeatureExtractionLayer(64)
        self.features_x4 = _FeatureExtractionLayer(64)
        self.features_x8 = _FeatureExtractionLayer(64)
        self.features_x16 = _FeatureExtractionLayer(64)

        
        # Layers for down and upscaling and merging scales
        self.down_x2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x8 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x16 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Prediction layers at each scale
        self.prediction_x16 = _PredictionLayer()

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
        return pred_x16



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
        self.sigmoid = nn.Sigmoid()

        # loss function
        # self.criterion = torch.nn.BCELoss()
        self.criterion = FocalLoss()

        self.optimizer_params = optimizer_params

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.sigmoid(x)

        return x

    def step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)

        # x, mask, y = batch
        # surface = self.forward(x)
        # y_hat = pixel(surface, mask)

        loss = self.criterion(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log train metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
 
         # --- Update and Log Training Accuracy ---
        self.train_accuracy.update(preds, targets.int()) 
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.train_auroc.update(preds, targets.int())
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=False) # Usually not shown on prog bar
        
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log val metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Update and Log Validation Accuracy ---
        self.val_accuracy.update(preds, targets.int())
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auroc.update(preds, targets.int())
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=False) # Usually not shown on prog bar
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # --- Update and Log Test Accuracy ---
        self.test_accuracy.update(preds, targets.int())
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, logger=True)
        self.test_auroc.update(preds, targets.int())
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, logger=True)
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
            nn.Linear(125, 1),  # Linear layer to output a single value
        )

        self.sigmoid = nn.Sigmoid()

        # loss function
        self.criterion = torch.nn.BCELoss()
        # self.criterion = FocalLoss()

        self.optimizer_params = optimizer_params

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    def forward(self, x: torch.Tensor):
        b, tc, h, w = x.shape
        x = x.view(b*125, 13, h, w)
        out = self.spatial_model(x) #[B*125, 1, 4, 6]
        out = out.view(b*125, -1) #[B*125, 4*6]
        out = out.view(b, 125, -1) #[B, 125, 4*6]
        out, (h_n, c_n) = self.temporal_model(out) # out: [B, T, 2 * 128] h_n: [3*2, B, D]
        out = out[:, -1, :] #[B, 2*128]
        out = self.spatial_output_layer(out) #[B, 1]
        # out = self.temporal_output_layer(out) #[B, 1]
        out = self.sigmoid(out)

        return out

    def step(self, batch: Any):
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        y_hat = self.forward(x)

        # x, mask, y = batch
        # surface = self.forward(x)
        # y_hat = pixel(surface, mask)

        loss = self.criterion(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log train metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
 
         # --- Update and Log Training Accuracy ---
        self.train_accuracy.update(preds, targets.int()) 
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.train_auroc.update(preds, targets.int())
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=False) # Usually not shown on prog bar
        
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log val metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # --- Update and Log Validation Accuracy ---
        self.val_accuracy.update(preds, targets.int())
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auroc.update(preds, targets.int())
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=False) # Usually not shown on prog bar
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # --- Update and Log Test Accuracy ---
        self.test_accuracy.update(preds, targets.int())
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, logger=True)
        self.test_auroc.update(preds, targets.int())
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, logger=True)
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

class SpatioTemporalGNNLSTM(nn.Module):
    def __init__(self,
                model_config : dict,
                 node_feature_in: int = 8,
                 node_feature_embed: int = 64,
                 gnn_hidden: int = 64,
                 gnn_layers: int = 2,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 lstm_dropout: float = 0.2,
                 lstm_bidirectional: bool = True,
                 distance_threshold: float = 10.0, # Threshold for spatial graph edges
                 use_pressing_features: bool = False): # Flag to use pressing intensity
        super().__init__()
        self.node_feature_in = model_config['node_feature_in']
        self.node_feature_embed = model_config['node_feature_embed']
        self.gnn_out_dim = model_config['gnn_hidden']
        self.distance_threshold = model_config['distance_threshold']
        self.use_pressing_features = model_config['use_pressing_features']
        self.gnn_layers = model_config['gnn_layers']
        self.gnn_hidden = model_config['gnn_hidden']
        self.lstm_hidden = model_config['lstm_hidden']
        self.lstm_layers = model_config['lstm_layers']
        self.lstm_dropout = model_config['lstm_dropout']
        self.lstm_bidrectional = model_config['lstm_bidrectional']
        # Node embedding
        self.node_embed = nn.Linear(self.node_feature_in, self.node_feature_embed)

        # GNN Layers (using edge_weight for pressure)
        self.gnn_layers = nn.ModuleList()
        in_c = self.node_feature_embed
        for _ in range(gnn_layers):
            self.gnn_layers.append(GCNConv(in_c, self.gnn_hidden)) # GCNConv can use edge_weight
            in_c = self.gnn_hidden

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.gnn_out_dim, # Input is pooled GNN features
            hidden_size=self.lstm_hidden, num_layers=self.lstm_layers, batch_first=True,
            dropout=self.lstm_dropout if self.lstm_layers > 1 else 0, bidirectional=self.lstm_bidirectional )

        # Classifier Head
        self.num_directions = 2 if self.lstm_bidirectional else 1
        self.classifier = nn.Linear(self.lstm_hidden * self.num_directions, 1)

    def forward(self, batch: Dict) -> torch.Tensor:
        # Extract data from batch dictionary
        x_seq = batch['features']              # [B, T, A, F]
        pressing_intensity_seq = batch['pressing_intensity'] # [B, T, 11, 11]
        agent_orders = batch['agent_order']      # List of lists [B, A] (or just one list if consistent)
        
        B, T, A, F_in = x_seq.shape
        device = x_seq.device

        # Prepare features for all frames at once
        # Reshape: [B, T, A, F] -> [B*T, A, F]
        x_frames = x_seq.view(B * T, A, F_in)
        node_features = F.relu(self.node_embed(x_frames)) # [B*T, A, F_embed]

        # --- Efficient Batched Graph Construction & GNN ---
        # We need edge_index and edge_weight for all B*T graphs.
        # PyG's approach is better, but simulating here:

        # 1. Node features flattened for PyG format: [B*T*A, F_embed]
        node_features_flat = node_features.view(-1, node_features.shape[-1])

        # 2. Create batch index vector: [B*T*A] indicating graph membership
        batch_idx = torch.arange(B * T, device=device).repeat_interleave(A)

        # 3. Create edge_index and edge_weight (potentially per frame, batched)
        # This is the most complex part to do efficiently *here*.
        # Doing it per frame and batching is slow as shown before.
        # --- Ideal Scenario (Using PyG DataLoader): ---
        # If DataLoader created PyG Batch objects per frame, input 'x_seq' would be
        # a list of Batch objects. We'd loop through time, process each Batch object.
        # --- Simpler Scenario (Fixed Edges + Edge Weights based on last frame maybe?): ---
        # Let's assume fixed FULLY CONNECTED player graph for structure, and derive weights.
        # This avoids dynamic edge creation per frame here.
        num_players = 22 # Excluding ball
        player_indices = torch.arange(num_players, device=device)
        edge_index_player = torch.cartesian_prod(player_indices, player_indices).t()
        edge_index_player = edge_index_player[:, edge_index_player[0] != edge_index_player[1]] # Remove self-loops -> [2, 22*21]

        # Expand edge_index for the whole batch (B*T)
        # This assumes the *same fixed edge structure* for all frames/samples in the batch
        num_nodes_total = B * T * A
        num_edges_per_graph = edge_index_player.shape[1]
        node_offset = torch.arange(0, B * T, device=device).repeat_interleave(num_edges_per_graph) * A
        # Need to be careful with indices if ball (A=23) is included/excluded
        # Assuming edge_index_player is for nodes 0-21 corresponding to first 22 agents
        edge_index_batch = edge_index_player.repeat(1, B * T) + node_offset

        # --- Calculate Edge Weights (Using Pressure) - Still complex per edge ---
        # Placeholder: Use default weights for now for simplicity
        edge_weight_batch = torch.ones(edge_index_batch.shape[1], device=device)
        # TODO: Implement sophisticated edge_weight calculation here if using pressure.
        # This would involve mapping flattened edge indices back to (batch, time, player_u, player_v)
        # and looking up pressure values in pressing_intensity_seq. Highly complex.

        # --- Alternative: Add Pressure Info to Node Features (Simpler) ---
        if self.use_pressing_features:
             # Calculate pressure received per player per frame
             # Assume pressing_intensity_seq is [B, T, 11, 11] (PresserTeam -> PressedTeam)
             # This requires mapping nodes (0-22) to team & 11-player index
             # Placeholder logic:
             total_pressure_received = torch.zeros(B*T, A, 1, device=device) # Extra feature dim
             # ... (Complex logic to calculate total pressure per node using pressing_intensity_seq) ...
             # Concatenate to node_features before GNN
             node_features = torch.cat([node_features, total_pressure_received], dim=-1)
             # Adjust GNN input channel size accordingly in __init__
             node_features_flat = node_features.view(-1, node_features.shape[-1])


        # --- Apply GNN layers ---
        gnn_node_output = node_features_flat
        for gnn_layer in self.gnn_layers:
             # Apply GCNConv using batched edge_index and potentially edge_weight
             gnn_node_output = gnn_layer(gnn_node_output, edge_index_batch, edge_weight=edge_weight_batch).relu() # [B*T*A, F_out]

        # --- Aggregate node features for each frame ---
        graph_features = global_mean_pool(gnn_node_output, batch_idx) # Shape: [B*T, F_out]

        # --- Reshape for LSTM ---
        lstm_input = graph_features.view(B, T, -1) # Shape: [B, T, F_out]

        # --- Apply LSTM ---
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # --- Select LSTM output ---
        last_step_output = lstm_out[:, -1, :] # Shape: [B, Num_Directions * Hidden_Size]

        # --- Classifier ---
        logits = self.classifier(last_step_output) # Shape: [B, 1]

        return logits

class exPressModel(pl.LightningModule):
    def __init__(self, model_config: dict = None, optimizer_params: dict = None):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate the new GNN+LSTM model
        self.model = SpatioTemporalGNNLSTM(model_config=model_config)

        self.criterion = nn.BCELoss()

        self.optimizer_params = optimizer_params

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()


    def forward(self, batch: dict) -> torch.Tensor:
        # Pass the whole batch dictionary to the model's forward
        return self.model(batch)

    def step(self, batch: dict) -> tuple:
        # Model forward now takes the batch dictionary
        logits = self.forward(batch) # Get single logit output [B, 1]
        y = batch['label'] # Extract label
        y = y.float().view_as(logits)
        loss = self.criterion(logits, y)
        preds_prob = torch.sigmoid(logits)
        return loss, preds_prob, y
    
    # training_step, validation_step, test_step remain structurally the same:
    # call self.step, update/log metrics (using the metric objects)
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.step(batch)
        targets_int = targets.int()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_accuracy.update(preds, targets_int); self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        try: self.train_auroc.update(preds, targets_int); self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True)
        except ValueError: pass
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Optional[Dict]:
        loss, preds, targets = self.step(batch)
        targets_int = targets.int()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_accuracy.update(preds, targets_int); self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        try: self.val_auroc.update(preds, targets_int); self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True)
        except ValueError: pass
        return {"val_loss": loss}

    def test_step(self, batch: dict, batch_idx: int) -> Optional[Dict]:
        loss, preds, targets = self.step(batch)
        targets_int = targets.int()
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.test_accuracy.update(preds, targets_int); self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, logger=True)
        try: self.test_auroc.update(preds, targets_int); self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, logger=True)
        except ValueError: pass
        return {"test_loss": loss}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Add weight decay etc. from hparams if needed
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer


