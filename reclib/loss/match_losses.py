import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BPRLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(BPRLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        if neg_score.dim() == 2:
            pos_score = pos_score.unsqueeze(1)  # [batch_size, 1]
            diff = pos_score - neg_score  # [batch_size, num_neg]
            loss = -torch.log(torch.sigmoid(diff) + 1e-8)
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            diff = pos_score - neg_score
            loss = -torch.log(torch.sigmoid(diff) + 1e-8)
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class HingeLoss(nn.Module): 
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        if neg_score.dim() == 2:
            pos_score = pos_score.unsqueeze(1)  # [batch_size, 1]
        
        diff = pos_score - neg_score
        loss = torch.clamp(self.margin - diff, min=0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = 'mean', distance: str = 'euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.distance = distance
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        if self.distance == 'euclidean':
            pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)
            
            if negative.dim() == 3:
                anchor_expanded = anchor.unsqueeze(1)  # [batch_size, 1, dim]
                neg_dist = torch.sum((anchor_expanded - negative) ** 2, dim=-1)  # [batch_size, num_neg]
            else:
                neg_dist = torch.sum((anchor - negative) ** 2, dim=-1)
            
            if neg_dist.dim() == 2:
                pos_dist = pos_dist.unsqueeze(1)  # [batch_size, 1]
        
        elif self.distance == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            
            if negative.dim() == 3:
                anchor_expanded = anchor.unsqueeze(1)  # [batch_size, 1, dim]
                neg_dist = 1 - F.cosine_similarity(anchor_expanded, negative, dim=-1)
            else:
                neg_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)
            
            if neg_dist.dim() == 2:
                pos_dist = pos_dist.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(SampledSoftmaxLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        pos_logits = pos_logits.unsqueeze(1)  # [batch_size, 1]
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)  # [batch_size, 1 + num_neg]
        targets = torch.zeros(all_logits.size(0), dtype=torch.long, device=all_logits.device)
        loss = F.cross_entropy(all_logits, targets, reduction=self.reduction)
        
        return loss


class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        pos_loss = (1 - similarity) * labels

        neg_loss = torch.clamp(similarity - self.margin, min=0) * (1 - labels)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, query: torch.Tensor, pos_key: torch.Tensor, neg_keys: torch.Tensor) -> torch.Tensor:
        pos_sim = torch.sum(query * pos_key, dim=-1) / self.temperature  # [batch_size]
        pos_sim = pos_sim.unsqueeze(1)  # [batch_size, 1]
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, dim]
        neg_sim = torch.sum(query_expanded * neg_keys, dim=-1) / self.temperature  # [batch_size, num_neg]
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, 1 + num_neg]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss
