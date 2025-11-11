"""
Match模型专用损失函数
支持pointwise, pairwise, listwise三种训练模式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss
    用于pairwise训练，优化正样本得分高于负样本
    
    Loss = -log(sigmoid(pos_score - neg_score))
    """
    def __init__(self, reduction: str = 'mean'):
        super(BPRLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_score: [batch_size] 正样本得分
            neg_score: [batch_size, num_neg] 或 [batch_size] 负样本得分
        """
        if neg_score.dim() == 2:
            # 多个负样本，扩展pos_score
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
            # 单个负样本
            diff = pos_score - neg_score
            loss = -torch.log(torch.sigmoid(diff) + 1e-8)
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss


class HingeLoss(nn.Module):
    """
    Hinge Loss for ranking
    用于pairwise训练
    
    Loss = max(0, margin - (pos_score - neg_score))
    """
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_score: [batch_size] 正样本得分
            neg_score: [batch_size, num_neg] 或 [batch_size] 负样本得分
        """
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
    """
    Triplet Loss
    用于学习embedding，使得anchor与positive的距离小于anchor与negative的距离
    
    Loss = max(0, d(anchor, pos) - d(anchor, neg) + margin)
    """
    def __init__(self, margin: float = 1.0, reduction: str = 'mean', distance: str = 'euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.distance = distance
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: [batch_size, dim] anchor embeddings
            positive: [batch_size, dim] positive embeddings
            negative: [batch_size, num_neg, dim] 或 [batch_size, dim] negative embeddings
        """
        if self.distance == 'euclidean':
            pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)
            
            if negative.dim() == 3:
                # 多个负样本
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
    """
    Sampled Softmax Loss
    用于listwise训练，在大规模item集合中通过负采样近似softmax
    
    适用于YouTube DNN等模型
    """
    def __init__(self, reduction: str = 'mean'):
        super(SampledSoftmaxLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_logits: [batch_size] 正样本logits
            neg_logits: [batch_size, num_neg] 负样本logits
        
        Returns:
            loss: softmax cross entropy loss
        """
        # 拼接正负样本 logits
        pos_logits = pos_logits.unsqueeze(1)  # [batch_size, 1]
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)  # [batch_size, 1 + num_neg]
        
        # 正样本的标签是0（第一个位置）
        targets = torch.zeros(all_logits.size(0), dtype=torch.long, device=all_logits.device)
        
        # 计算 cross entropy loss
        loss = F.cross_entropy(all_logits, targets, reduction=self.reduction)
        
        return loss


class CosineContrastiveLoss(nn.Module):
    """
    Cosine Contrastive Loss
    用于学习相似度，正样本cosine相似度接近1，负样本接近-1或0
    """
    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_emb: [batch_size, dim] user embeddings
            item_emb: [batch_size, dim] item embeddings
            labels: [batch_size] 1 for positive pairs, 0 for negative pairs
        """
        # 计算cosine相似度
        similarity = F.cosine_similarity(user_emb, item_emb, dim=-1)
        
        # 正样本：相似度应该接近1
        pos_loss = (1 - similarity) * labels
        
        # 负样本：相似度应该小于margin
        neg_loss = torch.clamp(similarity - self.margin, min=0) * (1 - labels)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (Contrastive Learning)
    用于对比学习，最大化正样本相似度，最小化负样本相似度
    """
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, query: torch.Tensor, pos_key: torch.Tensor, neg_keys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch_size, dim] query embeddings
            pos_key: [batch_size, dim] positive key embeddings
            neg_keys: [batch_size, num_neg, dim] negative key embeddings
        """
        # 计算正样本相似度
        pos_sim = torch.sum(query * pos_key, dim=-1) / self.temperature  # [batch_size]
        pos_sim = pos_sim.unsqueeze(1)  # [batch_size, 1]
        
        # 计算负样本相似度
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, dim]
        neg_sim = torch.sum(query_expanded * neg_keys, dim=-1) / self.temperature  # [batch_size, num_neg]
        
        # 拼接所有相似度
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, 1 + num_neg]
        
        # 正样本在第一个位置，标签为0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        # 计算cross entropy
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss
