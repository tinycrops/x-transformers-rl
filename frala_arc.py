#!/usr/bin/env python3
"""
FRALA-ARC: Fractal Architecture for ARC-AGI Tasks

This module adapts the Fractal Reinforcement Learning Agent (FRALA) framework 
for solving Abstract Reasoning Corpus (ARC) tasks. The architecture processes
visual patterns at multiple fractal levels to extract abstract reasoning rules.

Key Components:
1. FractalVisionEncoder: Multi-scale visual pattern processing
2. AbstractRuleExtractor: Global state management for abstract rules
3. FractalGridGenerator: Hierarchical output generation
4. ARC-specific training and evaluation framework

The fractal approach is particularly suited for ARC because:
- Visual patterns often exhibit self-similarity at different scales
- Abstract rules need global state management across pattern levels
- Few-shot learning benefits from hierarchical feature extraction
- Spatial reasoning requires multi-level attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import json
from pathlib import Path

from x_transformers_rl.fractal_rl import (
    FractalLevelEmbedding, 
    FractalProcessingBlock, 
    FractalEncoder
)


class ARCGridEncoder(nn.Module):
    """
    Encodes ARC grids into multi-level fractal representations.
    Processes grids at different scales: pixel -> patch -> region -> global.
    """
    
    def __init__(
        self,
        grid_size: int = 30,  # Maximum ARC grid size
        num_colors: int = 10,  # ARC has 10 colors (0-9)
        patch_size: int = 2,   # Size of local patches
        embed_dim: int = 256,
        num_fractal_levels: int = 4,
        heads: int = 8
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_fractal_levels = num_fractal_levels
        
        # Color embedding (treat each color as a token)
        self.color_embedding = nn.Embedding(num_colors, embed_dim // 4)
        
        # Positional embeddings for 2D grids
        self.pos_embed_h = nn.Parameter(torch.randn(grid_size, embed_dim // 4) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(grid_size, embed_dim // 4) * 0.02)
        
        # Patch processing layers for different scales
        input_channels = 3 * (embed_dim // 4)  # color + pos_h + pos_w
        self.patch_processors = nn.ModuleList()
        
        for i in range(num_fractal_levels):
            # Use smaller kernels and adaptive pooling instead of large strides
            if i == 0:
                # First level: direct convolution on embeddings
                self.patch_processors.append(
                    nn.Conv2d(input_channels, embed_dim, kernel_size=3, padding=1)
                )
            else:
                # Higher levels: process downsampled features
                self.patch_processors.append(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
                )
        
        # Fractal encoder for multi-level processing
        self.fractal_encoder = FractalEncoder(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_levels=num_fractal_levels,
            heads=heads,
            share_weights=False  # Different levels need different processing
        )
        
        # Abstract rule extractor
        self.rule_extractor = AbstractRuleExtractor(embed_dim, num_fractal_levels)
    
    def forward(self, grid: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Process ARC grid through fractal levels.
        
        Args:
            grid: [batch_size, height, width] with color indices
            
        Returns:
            fractal_features: Aggregated fractal representation
            fractal_info: Dict with level-wise features and extracted rules
        """
        batch_size, height, width = grid.shape
        
        # Embed colors and add positional information
        color_embeds = self.color_embedding(grid)  # [B, H, W, E//4]
        
        # Add 2D positional embeddings
        pos_h = self.pos_embed_h[:height].unsqueeze(1).expand(-1, width, -1)  # [H, W, E//4]
        pos_w = self.pos_embed_w[:width].unsqueeze(0).expand(height, -1, -1)  # [H, W, E//4]
        
        # Combine color and position embeddings
        grid_embeds = torch.cat([
            color_embeds, 
            pos_h.unsqueeze(0).expand(batch_size, -1, -1, -1),
            pos_w.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ], dim=-1)  # [B, H, W, 3*(E//4)]
        
        # Rearrange for conv processing
        grid_embeds = rearrange(grid_embeds, 'b h w e -> b e h w')
        
        # Process at different scales
        level_features = []
        current_features = grid_embeds
        
        for level, processor in enumerate(self.patch_processors):
            # Process patches at this scale
            if level == 0:
                # Pixel level - use original embeddings
                patches = current_features
            else:
                # Higher levels - downsample by factor of 2
                target_h = max(1, current_features.shape[2] // 2)
                target_w = max(1, current_features.shape[3] // 2)
                patches = F.adaptive_avg_pool2d(current_features, (target_h, target_w))
            
            # Apply convolution for this level
            level_feats = processor(patches)  # [B, E, H', W']
            current_features = level_feats  # Update for next level
            
            # Flatten spatial dimensions for sequence processing
            level_feats = rearrange(level_feats, 'b e h w -> b (h w) e')
            level_features.append(level_feats)
        
        # Process through fractal encoder
        # Use the finest level as input, others as context
        primary_features = level_features[0]
        fractal_output, fractal_levels = self.fractal_encoder(
            primary_features, return_all_levels=True
        )
        
        # Extract abstract rules
        abstract_rules = self.rule_extractor(fractal_levels)
        
        fractal_info = {
            'level_features': level_features,
            'fractal_levels': fractal_levels,
            'abstract_rules': abstract_rules,
            'grid_shape': (height, width)
        }
        
        return fractal_output, fractal_info


class AbstractRuleExtractor(nn.Module):
    """
    Extracts abstract reasoning rules from fractal level representations.
    This implements the "Soul" of the fractal system for ARC tasks.
    """
    
    def __init__(self, embed_dim: int, num_levels: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        
        # Rule type classifiers
        self.rule_types = [
            'color_mapping',     # Color transformations
            'spatial_transform', # Rotations, reflections, translations
            'pattern_completion', # Fill in missing parts
            'object_counting',   # Count objects or features
            'symmetry_detection', # Detect and apply symmetries
            'scaling_transform', # Scale objects up/down
            'logical_operations' # AND, OR, XOR operations between grids
        ]
        
        # Rule extractors for each type
        self.rule_extractors = nn.ModuleDict({
            rule_type: nn.Sequential(
                nn.Linear(embed_dim * num_levels, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 64)  # Rule representation
            ) for rule_type in self.rule_types
        })
        
        # Rule confidence estimators
        self.rule_confidence = nn.ModuleDict({
            rule_type: nn.Linear(64, 1) for rule_type in self.rule_types
        })
        
        # Global rule integrator
        self.global_integrator = nn.Sequential(
            nn.Linear(64 * len(self.rule_types), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, fractal_levels: List[Tensor]) -> Dict[str, Tensor]:
        """Extract abstract rules from fractal representations."""
        
        # Aggregate features across all levels
        level_features = []
        for level_feats in fractal_levels:
            # Pool spatial dimensions to get level summary
            level_summary = reduce(level_feats, 'b n e -> b e', 'mean')
            level_features.append(level_summary)
        
        # Concatenate all levels
        all_features = torch.cat(level_features, dim=-1)  # [B, E * num_levels]
        
        # Extract each type of rule
        extracted_rules = {}
        rule_confidences = {}
        
        for rule_type in self.rule_types:
            rule_repr = self.rule_extractors[rule_type](all_features)
            confidence = torch.sigmoid(self.rule_confidence[rule_type](rule_repr))
            
            extracted_rules[rule_type] = rule_repr
            rule_confidences[rule_type] = confidence
        
        # Create global rule representation
        all_rules = torch.cat(list(extracted_rules.values()), dim=-1)
        global_rules = self.global_integrator(all_rules)
        
        return {
            'rules': extracted_rules,
            'confidences': rule_confidences,
            'global_rules': global_rules
        }


class FractalGridGenerator(nn.Module):
    """
    Generates output grids using fractal processing and abstract rules.
    Implements hierarchical generation: global structure -> regions -> details.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_colors: int = 10,
        max_grid_size: int = 30,
        num_fractal_levels: int = 4
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_colors = num_colors
        self.max_grid_size = max_grid_size
        self.num_fractal_levels = num_fractal_levels
        
        # Fractal generation levels (coarse to fine)
        self.generation_levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(num_fractal_levels)
        ])
        
        # Upsampling layers between levels
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            for _ in range(num_fractal_levels - 1)
        ])
        
        # Final color prediction
        self.color_predictor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, num_colors, kernel_size=1)
        )
        
        # Rule application modules
        self.rule_appliers = nn.ModuleDict({
            'color_mapping': ColorMappingModule(embed_dim),
            'spatial_transform': SpatialTransformModule(embed_dim),
            'pattern_completion': PatternCompletionModule(embed_dim),
            'symmetry_detection': SymmetryModule(embed_dim)
        })
    
    def forward(
        self, 
        fractal_features: Tensor,
        abstract_rules: Dict[str, Tensor],
        target_shape: Tuple[int, int],
        apply_rules: bool = True
    ) -> Tensor:
        """
        Generate output grid using fractal processing and abstract rules.
        
        Args:
            fractal_features: Global fractal representation [B, E]
            abstract_rules: Extracted abstract rules
            target_shape: (height, width) of target grid
            apply_rules: Whether to apply extracted rules
            
        Returns:
            Generated grid [B, H, W] with color indices
        """
        batch_size = fractal_features.shape[0]
        target_h, target_w = target_shape
        
        # Start with global features
        current_features = fractal_features.unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
        
        # Generate through fractal levels (coarse to fine)
        for level in range(self.num_fractal_levels):
            # Process at this level
            level_processor = self.generation_levels[level]
            
            # Reshape for processing
            spatial_size = current_features.shape[-1]
            feat_flat = rearrange(current_features, 'b e h w -> b (h w) e')
            processed = level_processor(feat_flat)
            processed = rearrange(processed, 'b (h w) e -> b e h w', h=spatial_size)
            
            # Combine with previous features
            current_features = current_features + processed
            
            # Upsample for next level (except last)
            if level < self.num_fractal_levels - 1:
                current_features = self.upsamplers[level](current_features)
        
        # Resize to target shape
        current_features = F.interpolate(
            current_features, size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        
        # Apply abstract rules if requested
        if apply_rules and abstract_rules:
            current_features = self._apply_abstract_rules(current_features, abstract_rules)
        
        # Generate final colors
        color_logits = self.color_predictor(current_features)  # [B, num_colors, H, W]
        
        return color_logits
    
    def _apply_abstract_rules(self, features: Tensor, rules: Dict[str, Tensor]) -> Tensor:
        """Apply extracted abstract rules to modify features."""
        
        global_rules = rules.get('global_rules')
        if global_rules is None:
            return features
        
        # Apply each rule type based on confidence
        modified_features = features
        
        for rule_type, rule_applier in self.rule_appliers.items():
            if rule_type in rules['rules']:
                rule_repr = rules['rules'][rule_type]
                confidence = rules['confidences'][rule_type]
                
                # Ensure tensors are on the same device
                rule_repr = rule_repr.to(features.device)
                confidence = confidence.to(features.device)
                
                # Apply rule with confidence weighting
                rule_applied = rule_applier(modified_features, rule_repr)
                modified_features = modified_features + confidence.unsqueeze(-1).unsqueeze(-1) * rule_applied
        
        return modified_features


class ColorMappingModule(nn.Module):
    """Applies color mapping transformations."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.color_transform = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
    
    def forward(self, features: Tensor, rule_repr: Tensor) -> Tensor:
        # Modulate features based on color mapping rule
        modulation = rule_repr.unsqueeze(-1).unsqueeze(-1)
        return self.color_transform(features * modulation)


class SpatialTransformModule(nn.Module):
    """Applies spatial transformations (rotation, reflection, translation)."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.spatial_transform = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
    
    def forward(self, features: Tensor, rule_repr: Tensor) -> Tensor:
        # Apply spatial transformation based on rule
        return self.spatial_transform(features)


class PatternCompletionModule(nn.Module):
    """Completes patterns based on observed examples."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.pattern_completer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
    
    def forward(self, features: Tensor, rule_repr: Tensor) -> Tensor:
        return self.pattern_completer(features)


class SymmetryModule(nn.Module):
    """Detects and applies symmetries."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.symmetry_detector = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
    
    def forward(self, features: Tensor, rule_repr: Tensor) -> Tensor:
        return self.symmetry_detector(features)


class FRALA_ARC(nn.Module):
    """
    Complete FRALA-ARC system for solving ARC-AGI tasks.
    Combines fractal visual processing with abstract rule extraction and generation.
    """
    
    def __init__(
        self,
        grid_size: int = 30,
        num_colors: int = 10,
        embed_dim: int = 256,
        num_fractal_levels: int = 4,
        heads: int = 8
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.num_fractal_levels = num_fractal_levels
        
        # Core components
        self.encoder = ARCGridEncoder(
            grid_size=grid_size,
            num_colors=num_colors,
            embed_dim=embed_dim,
            num_fractal_levels=num_fractal_levels,
            heads=heads
        )
        
        self.generator = FractalGridGenerator(
            embed_dim=embed_dim,
            num_colors=num_colors,
            max_grid_size=grid_size,
            num_fractal_levels=num_fractal_levels
        )
        
        # Few-shot learning components
        self.example_aggregator = ExampleAggregator(embed_dim, num_fractal_levels)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        input_grids: List[Tensor],
        output_grids: Optional[List[Tensor]] = None,
        test_input: Optional[Tensor] = None,
        return_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Process ARC task through fractal reasoning.
        
        Args:
            input_grids: List of input example grids
            output_grids: List of corresponding output grids (for training)
            test_input: Test input grid to generate output for
            return_analysis: Whether to return detailed analysis
            
        Returns:
            Dictionary with predictions, losses, and analysis
        """
        results = {}
        
        # Process all input examples
        example_features = []
        example_rules = []
        
        for input_grid in input_grids:
            features, fractal_info = self.encoder(input_grid)
            example_features.append(features)
            example_rules.append(fractal_info['abstract_rules'])
        
        # Aggregate examples to extract common patterns
        aggregated_features, aggregated_rules = self.example_aggregator(
            example_features, example_rules
        )
        
        results['aggregated_features'] = aggregated_features
        results['aggregated_rules'] = aggregated_rules
        
        # Generate outputs for training examples (if provided)
        if output_grids is not None:
            training_losses = []
            
            for i, (input_grid, target_grid) in enumerate(zip(input_grids, output_grids)):
                target_shape = target_grid.shape[-2:]
                
                # Generate prediction (logits)
                predicted_logits = self.generator(
                    example_features[i],
                    aggregated_rules,
                    target_shape
                )
                
                # Compute cross-entropy loss
                # predicted_logits: [B, num_colors, H, W], target_grid: [B, H, W]
                loss = F.cross_entropy(predicted_logits, target_grid.long())
                training_losses.append(loss)
            
            results['training_loss'] = torch.mean(torch.stack(training_losses))
        
        # Generate test output (if test input provided)
        if test_input is not None:
            test_features, test_fractal_info = self.encoder(test_input)
            
            # Infer target shape from training examples
            # Check if output shapes are consistent across training examples
            if output_grids is not None and len(output_grids) > 0:
                # Use the shape from training outputs
                target_shape = output_grids[0].shape[-2:]
                # Verify consistency across examples
                for i, out_grid in enumerate(output_grids):
                    if out_grid.shape[-2:] != target_shape:
                        print(f"Warning: Inconsistent output shapes in training examples")
                        # Fall back to input shape if inconsistent
                        target_shape = test_input.shape[-2:]
                        break
            else:
                # No training outputs provided, use input shape
                target_shape = test_input.shape[-2:]
            
            # Use aggregated rules for generation
            test_logits = self.generator(
                test_features,
                aggregated_rules,
                target_shape
            )
            
            # Convert logits to discrete predictions
            test_output = torch.argmax(test_logits, dim=1)
            
            results['test_prediction'] = test_output
            
            if return_analysis:
                results['test_analysis'] = test_fractal_info
        
        return results
    
    def analyze_task(self, input_grids: List[Tensor], output_grids: List[Tensor]) -> Dict[str, Any]:
        """Analyze an ARC task to understand the reasoning pattern."""
        
        analysis = {
            'num_examples': len(input_grids),
            'grid_shapes': [(g.shape[-2:]) for g in input_grids],
            'color_usage': [],
            'fractal_analysis': []
        }
        
        for i, (inp, out) in enumerate(zip(input_grids, output_grids)):
            # Analyze color usage
            inp_colors = torch.unique(inp).tolist()
            out_colors = torch.unique(out).tolist()
            analysis['color_usage'].append({
                'input_colors': inp_colors,
                'output_colors': out_colors,
                'color_change': list(set(out_colors) - set(inp_colors))
            })
            
            # Fractal analysis
            _, fractal_info = self.encoder(inp.unsqueeze(0))
            
            fractal_analysis = {
                'level_feature_norms': [
                    torch.norm(level).item() 
                    for level in fractal_info['level_features']
                ],
                'extracted_rules': {
                    rule_type: torch.norm(rule_repr).item()
                    for rule_type, rule_repr in fractal_info['abstract_rules']['rules'].items()
                },
                'rule_confidences': {
                    rule_type: conf.mean().item()
                    for rule_type, conf in fractal_info['abstract_rules']['confidences'].items()
                }
            }
            
            analysis['fractal_analysis'].append(fractal_analysis)
        
        return analysis


class ExampleAggregator(nn.Module):
    """Aggregates multiple examples to extract common patterns and rules."""
    
    def __init__(self, embed_dim: int, num_fractal_levels: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fractal_levels = num_fractal_levels
        
        # Attention-based aggregation
        self.feature_aggregator = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        # Rule aggregation (input will be 64-dim rule representations)
        self.rule_aggregator = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self, 
        example_features: List[Tensor],
        example_rules: List[Dict[str, Tensor]]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Aggregate examples to extract common patterns."""
        
        # Stack features for attention
        stacked_features = torch.stack(example_features, dim=1)  # [B, num_examples, E]
        
        # Apply self-attention to find common patterns
        aggregated_features, _ = self.feature_aggregator(
            stacked_features, stacked_features, stacked_features
        )
        
        # Average across examples
        final_features = aggregated_features.mean(dim=1)  # [B, E]
        
        # Aggregate rules
        aggregated_rules = {}
        
        # Get all rule types from first example
        rule_types = example_rules[0]['rules'].keys()
        
        for rule_type in rule_types:
            # Stack rules of this type
            type_rules = torch.stack([
                rules['rules'][rule_type] for rules in example_rules
            ], dim=1)  # [B, num_examples, rule_dim]
            
            # Average and process
            avg_rule = type_rules.mean(dim=1)
            processed_rule = self.rule_aggregator(avg_rule)
            
            aggregated_rules[rule_type] = processed_rule
        
        # Create aggregated rule dict in expected format
        final_rules = {
            'rules': aggregated_rules,
            'confidences': {
                rule_type: torch.ones(final_features.shape[0], 1, device=final_features.device) * 0.8
                for rule_type in rule_types
            },
            'global_rules': final_features
        }
        
        return final_features, final_rules


# Training and evaluation utilities
class ARCTrainer:
    """Training framework for FRALA-ARC."""
    
    def __init__(
        self,
        model: FRALA_ARC,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
    
    def train_on_task(self, task_data: Dict[str, Any], num_epochs: int = 100):
        """Train on a single ARC task."""
        
        # Handle both synthetic and real ARC formats
        if isinstance(task_data['train'], dict):
            # Synthetic format: train = {'input': [...], 'output': [...]}
            input_grids = [torch.tensor(grid) for grid in task_data['train']['input']]
            output_grids = [torch.tensor(grid) for grid in task_data['train']['output']]
        else:
            # Real ARC format: train = [{'input': [...], 'output': [...]}, ...]
            input_grids = [torch.tensor(example['input']) for example in task_data['train']]
            output_grids = [torch.tensor(example['output']) for example in task_data['train']]
        
        # Move to device
        input_grids = [grid.to(self.device) for grid in input_grids]
        output_grids = [grid.to(self.device) for grid in output_grids]
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            results = self.model(
                input_grids=[grid.unsqueeze(0) for grid in input_grids],
                output_grids=[grid.unsqueeze(0) for grid in output_grids]
            )
            
            loss = results['training_loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.scheduler.step(best_loss)
        return best_loss
    
    def evaluate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate on ARC task."""
        
        self.model.eval()
        
        # Handle both synthetic and real ARC formats
        if isinstance(task_data['train'], dict):
            # Synthetic format: train = {'input': [...], 'output': [...]}
            input_grids = [torch.tensor(grid).to(self.device) for grid in task_data['train']['input']]
            output_grids = [torch.tensor(grid).to(self.device) for grid in task_data['train']['output']]
        else:
            # Real ARC format: train = [{'input': [...], 'output': [...]}, ...]
            input_grids = [torch.tensor(example['input']).to(self.device) for example in task_data['train']]
            output_grids = [torch.tensor(example['output']).to(self.device) for example in task_data['train']]
        
        test_input = torch.tensor(task_data['test'][0]['input']).to(self.device)
        test_output = torch.tensor(task_data['test'][0]['output']).to(self.device)
        
        with torch.no_grad():
            results = self.model(
                input_grids=[grid.unsqueeze(0) for grid in input_grids],
                output_grids=[grid.unsqueeze(0) for grid in output_grids],
                test_input=test_input.unsqueeze(0),
                return_analysis=True
            )
        
        predicted = results['test_prediction'].squeeze(0)
        
        # Calculate accuracy
        accuracy = (predicted == test_output).float().mean().item()
        
        return {
            'accuracy': accuracy,
            'predicted_grid': predicted.cpu().numpy(),
            'target_grid': test_output.cpu().numpy(),
            'analysis': results.get('test_analysis', {})
        }


def load_arc_task(task_file: str) -> Dict[str, Any]:
    """Load ARC task from JSON file."""
    with open(task_file, 'r') as f:
        return json.load(f)


def visualize_grids(grids: List[np.ndarray], titles: List[str] = None):
    """Visualize ARC grids with colors."""
    import matplotlib.pyplot as plt
    
    # ARC color palette
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    fig, axes = plt.subplots(1, len(grids), figsize=(4*len(grids), 4))
    if len(grids) == 1:
        axes = [axes]
    
    for i, (grid, ax) in enumerate(zip(grids, axes)):
        # Create RGB image
        rgb_grid = np.zeros((*grid.shape, 3))
        for color_idx in range(10):
            mask = grid == color_idx
            rgb_color = np.array([int(colors[color_idx][1:3], 16),
                                int(colors[color_idx][3:5], 16), 
                                int(colors[color_idx][5:7], 16)]) / 255.0
            rgb_grid[mask] = rgb_color
        
        ax.imshow(rgb_grid)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles:
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()


# Example usage and demo
def demo_frala_arc():
    """Demonstrate FRALA-ARC on a simple task."""
    
    print("🧩 FRALA-ARC Demo: Fractal Architecture for ARC-AGI")
    print("=" * 60)
    
    # Create model
    model = FRALA_ARC(
        grid_size=30,
        num_colors=10,
        embed_dim=256,
        num_fractal_levels=4,
        heads=8
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create synthetic ARC-like task (pattern completion)
    def create_pattern_task():
        # Create a simple pattern: copy and shift
        input1 = torch.zeros(5, 5, dtype=torch.long)
        input1[1:3, 1:3] = 1  # Blue square
        
        output1 = torch.zeros(5, 5, dtype=torch.long)
        output1[1:3, 1:3] = 1  # Original
        output1[3:5, 3:5] = 1  # Shifted copy
        
        input2 = torch.zeros(5, 5, dtype=torch.long)
        input2[0:2, 0:2] = 2  # Red square
        
        output2 = torch.zeros(5, 5, dtype=torch.long)
        output2[0:2, 0:2] = 2  # Original
        output2[2:4, 2:4] = 2  # Shifted copy
        
        test_input = torch.zeros(5, 5, dtype=torch.long)
        test_input[1:3, 0:2] = 3  # Green square
        
        test_target = torch.zeros(5, 5, dtype=torch.long)
        test_target[1:3, 0:2] = 3  # Original
        test_target[3:5, 2:4] = 3  # Shifted copy
        
        return {
            'train': {
                'input': [input1, input2],
                'output': [output1, output2]
            },
            'test': [{
                'input': test_input,
                'output': test_target
            }]
        }
    
    # Create and analyze task
    task = create_pattern_task()
    
    print("\n📊 Task Analysis:")
    analysis = model.analyze_task(
        [torch.tensor(grid) for grid in task['train']['input']],
        [torch.tensor(grid) for grid in task['train']['output']]
    )
    
    print(f"  Number of examples: {analysis['num_examples']}")
    print(f"  Grid shapes: {analysis['grid_shapes']}")
    
    for i, color_info in enumerate(analysis['color_usage']):
        print(f"  Example {i+1} colors: {color_info['input_colors']} -> {color_info['output_colors']}")
    
    # Test forward pass
    print("\n🔄 Testing Forward Pass:")
    with torch.no_grad():
        results = model(
            input_grids=[torch.tensor(grid).unsqueeze(0) for grid in task['train']['input']],
            test_input=torch.tensor(task['test'][0]['input']).unsqueeze(0),
            return_analysis=True
        )
    
    print(f"  Generated prediction shape: {results['test_prediction'].shape}")
    print(f"  Analysis keys: {list(results.get('test_analysis', {}).keys())}")
    
    print("\n✅ FRALA-ARC Demo Complete!")
    print("🚀 Ready for ARC-AGI challenge!")


if __name__ == "__main__":
    demo_frala_arc() 