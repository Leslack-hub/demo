import numpy as np
from collections import deque

class FeatureFusionEngine:
    """智能特征融合引擎"""
    
    def __init__(self, config: dict):
        self.config = config
        self.fusion_weights = {
            'traditional': 0.4,    # 传统特征权重
            'panns': 0.6           # PANNs特征权重
        }
        self.adaptive_threshold = 0.8
        self.context_memory = deque(maxlen=10)  # 上下文记忆
        
    def fuse_confidences(self, traditional_conf: float, 
                        panns_conf: float, 
                        context_data: dict = None) -> tuple:
        """融合两种检测方式的置信度"""
        
        # 自适应权重调整
        weight_adjustment = self._calculate_adaptive_weights(
            traditional_conf, panns_conf, context_data
        )
        
        # 使用更保守的融合策略，避免过度增强
        final_confidence = (
            traditional_conf * self.fusion_weights['traditional'] * weight_adjustment['traditional'] +
            panns_conf * self.fusion_weights['panns'] * weight_adjustment['panns']
        )
        
        # 限制最大置信度，避免虚假检测
        final_confidence = min(1.0, final_confidence)
        
        return final_confidence, {
            'traditional_conf': traditional_conf,
            'panns_conf': panns_conf,
            'weight_adjustment': weight_adjustment
        }
    
    def _calculate_adaptive_weights(self, trad_conf: float, 
                                  panns_conf: float,
                                  context: dict) -> dict:
        """根据置信度质量和上下文计算自适应权重"""
        
        # 低质量检测时降低权重
        if trad_conf < 0.3 and panns_conf < 0.3:
            return {'traditional': 0.8, 'panns': 0.5}
        
        # 中等质量检测
        if trad_conf < 0.6 and panns_conf < 0.6:
            return {'traditional': 1.0, 'panns': 0.8}
        
        # 高质量互补检测时适度增强权重
        if trad_conf > 0.8 and panns_conf > 0.8:
            return {'traditional': 1.1, 'panns': 1.2}
        
        # 正常情况
        return {'traditional': 1.0, 'panns': 1.0}
        
    def _apply_confidence_boost(self, final_conf: float, 
                               trad_conf: float, 
                               panns_conf: float) -> float:
        """置信度增强策略"""
        # 移除置信度增强，避免过度敏感
        return final_conf