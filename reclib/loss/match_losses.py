"""
Loss functions for matching tasks

Date: create on 13/11/2025
Author:
    Yang Zhou,zyaztec@gmail.com
"""

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


class ListNetLoss(nn.Module):
    """
    ListNet loss using top-1 probability distribution
    Reference: Cao et al. Learning to Rank: From Pairwise Approach to Listwise Approach (ICML 2007)
    """
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        super(ListNetLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Convert scores and labels to probability distributions
        pred_probs = F.softmax(scores / self.temperature, dim=1)
        true_probs = F.softmax(labels / self.temperature, dim=1)
        
        # Cross entropy between two distributions
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ListMLELoss(nn.Module):
    """
    ListMLE (Maximum Likelihood Estimation) loss
    Reference: Xia et al. Listwise approach to learning to rank: theory and algorithm (ICML 2008)
    """
    def __init__(self, reduction: str = 'mean'):
        super(ListMLELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Sort by labels in descending order to get ground truth ranking
        sorted_labels, sorted_indices = torch.sort(labels, descending=True, dim=1)
        
        # Reorder scores according to ground truth ranking
        batch_size, list_size = scores.shape
        batch_indices = torch.arange(batch_size, device=scores.device).unsqueeze(1).expand(-1, list_size)
        sorted_scores = scores[batch_indices, sorted_indices]
        
        # Compute log likelihood
        # For each position, compute log(exp(score_i) / sum(exp(score_j) for j >= i))
        loss = torch.tensor(0.0, device=scores.device)
        for i in range(list_size):
            # Log-sum-exp trick for numerical stability
            remaining_scores = sorted_scores[:, i:]
            log_sum_exp = torch.logsumexp(remaining_scores, dim=1)
            loss = loss + (log_sum_exp - sorted_scores[:, i]).sum()
        
        if self.reduction == 'mean':
            return loss / batch_size
        elif self.reduction == 'sum':
            return loss
        else:
            return loss / batch_size


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG loss for learning to rank
    Reference: Qin et al. A General Approximation Framework for Direct Optimization of 
               Information Retrieval Measures (Information Retrieval 2010)
    """
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        super(ApproxNDCGLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def _dcg(self, relevance: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        if k is not None:
            relevance = relevance[:, :k]
        
        # DCG = sum(rel_i / log2(i + 2)) for i in range(list_size)
        positions = torch.arange(1, relevance.size(1) + 1, device=relevance.device, dtype=torch.float32)
        discounts = torch.log2(positions + 1.0)
        dcg = torch.sum(relevance / discounts, dim=1)
        
        return dcg
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            scores: Predicted scores [batch_size, list_size]
            labels: Ground truth relevance labels [batch_size, list_size]
            k: Top-k items for NDCG@k (if None, use all items)
        
        Returns:
            Approximate NDCG loss (1 - NDCG)
        """
        batch_size = scores.size(0)
        
        # Use differentiable sorting approximation with softmax
        # Create pairwise comparison matrix
        scores_expanded = scores.unsqueeze(2)  # [batch_size, list_size, 1]
        scores_tiled = scores.unsqueeze(1)     # [batch_size, 1, list_size]
        
        # Compute pairwise probabilities using sigmoid
        pairwise_diff = (scores_expanded - scores_tiled) / self.temperature
        pairwise_probs = torch.sigmoid(pairwise_diff)  # [batch_size, list_size, list_size]
        
        # Approximate ranking positions
        # ranking_probs[i, j] ≈ probability that item i is ranked at position j
        # We use softmax approximation for differentiable ranking
        ranking_weights = F.softmax(scores / self.temperature, dim=1)
        
        # Sort labels to get ideal DCG
        ideal_labels, _ = torch.sort(labels, descending=True, dim=1)
        ideal_dcg = self._dcg(ideal_labels, k)
        
        # Compute approximate DCG using soft ranking
        # Weight each item's relevance by its soft ranking position
        positions = torch.arange(1, scores.size(1) + 1, device=scores.device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)
        
        # Approximate DCG by weighting relevance with ranking probabilities
        approx_dcg = torch.sum(labels * ranking_weights * discounts, dim=1)
        
        # Normalize by ideal DCG to get NDCG
        ndcg = approx_dcg / (ideal_dcg + 1e-10)
        
        # Loss is 1 - NDCG (we want to maximize NDCG, so minimize 1 - NDCG)
        loss = 1.0 - ndcg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
