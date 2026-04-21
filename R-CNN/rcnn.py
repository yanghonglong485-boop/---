import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class RCNN(nn.Module):
    """
    R-CNN 모델 구현
    - CNN feature extractor (ResNet-50)
    - Classifier (FC layers)
    - Bounding box regressor
    """
    
    def __init__(self, num_classes=20, pretrained=True):
        super(RCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extractor: ResNet-50 백본
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # ResNet의 마지막 FC layer 제거
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature dimension
        self.feature_dim = 2048
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes + 1)  # +1 for background class
        )
        
        # Bounding box regressor
        self.bbox_regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes * 4)  # 4 coordinates per class
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 - 수치 안정성 개선"""
        for module in [self.classifier, self.bbox_regressor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Xavier/Glorot 초기화 사용
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x, proposals=None):
        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")
        
        # Feature extraction with normalization
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Feature normalization for stability
        features = F.normalize(features, p=2, dim=1) * (features.size(1) ** 0.5)
        
        # Classification with stability check
        cls_scores = self.classifier(features)
        if torch.isnan(cls_scores).any():
            print("Warning: NaN in classification scores")
            cls_scores = torch.zeros_like(cls_scores)
        
        # Bounding box regression with clipping
        bbox_pred = self.bbox_regressor(features)
        bbox_pred = torch.clamp(bbox_pred, min=-10.0, max=10.0)  # Prevent extreme values
        
        if torch.isnan(bbox_pred).any():
            print("Warning: NaN in bbox predictions")
            bbox_pred = torch.zeros_like(bbox_pred)
        
        return cls_scores, bbox_pred
    
    def extract_features(self, x):
        """특징만 추출하는 함수"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
        return features

class RCNNLoss(nn.Module):
    """R-CNN 손실 함수 - 수치 안정성 개선"""
    
    def __init__(self, lambda_reg=1.0, cls_weight=1.0):
        super(RCNNLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.cls_weight = cls_weight
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.eps = 1e-8  # 수치 안정성을 위한 작은 값
    
    def forward(self, cls_scores, bbox_pred, region_labels, bbox_targets, bbox_masks):
        # 分类损失
        cls_scores = torch.clamp(cls_scores, min=-10.0, max=10.0)
        cls_loss = self.cls_loss(cls_scores, region_labels.long())

        # 默认回归损失
        reg_loss = torch.tensor(0.0, device=cls_scores.device, requires_grad=True)

        # 正样本掩码
        positive_mask = region_labels > 0

        if positive_mask.sum() > 0:
            # 简化处理：先只取前4维bbox输出
            bbox_pred_simple = bbox_pred[:, :4]

            pred_pos = torch.clamp(bbox_pred_simple[positive_mask], min=-10.0, max=10.0)
            target_pos = torch.clamp(bbox_targets[positive_mask], min=-10.0, max=10.0)

            # 如果 bbox_masks 是 [N,4]，按掩码筛选
            if len(bbox_masks.shape) == 2:
                mask_pos = bbox_masks[positive_mask].bool()
                if mask_pos.sum() > 0:
                    reg_loss = self.reg_loss(pred_pos[mask_pos], target_pos[mask_pos])
            else:
                reg_loss = self.reg_loss(pred_pos, target_pos)

        total_loss = self.cls_weight * cls_loss + self.lambda_reg * reg_loss

        return total_loss, cls_loss, reg_loss