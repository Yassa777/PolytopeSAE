"""
Concept curation, splits, and prompt templates for hierarchical concept experiments.

This module handles WordNet concept selection, prompt generation, and data splits
for the polytope discovery experiments.
"""

import nltk
from nltk.corpus import wordnet as wn
from typing import Dict, List, Tuple, Set, Any, Optional
import json
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

# NLTK setup is performed lazily via ``setup_nltk`` to avoid network calls on
# module import.

def setup_nltk() -> None:
    """Download required NLTK corpora if missing."""
    for corpus in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus)

logger = logging.getLogger(__name__)


@dataclass
class ConceptInfo:
    """Information about a concept (parent or child)."""
    synset_id: str
    name: str
    definition: str
    examples: List[str]
    pos: str  # part of speech
    lemmas: List[str]
    

@dataclass
class ConceptHierarchy:
    """Hierarchical structure of concepts."""
    parent: ConceptInfo
    children: List[ConceptInfo]
    parent_prompts: List[str]
    child_prompts: Dict[str, List[str]]  # child_id -> prompts
    

class ConceptCurator:
    """Curates concept hierarchies from WordNet."""
    
    def __init__(self, 
                 max_parents: int = 80,
                 max_children_per_parent: int = 4,
                 min_children_per_parent: int = 2,
                 pos_filter: Optional[Set[str]] = None):
        """
        Initialize concept curator.
        
        Args:
            max_parents: Maximum number of parent concepts
            max_children_per_parent: Maximum children per parent
            min_children_per_parent: Minimum children per parent (for filtering)
            pos_filter: Set of allowed POS tags (e.g., {'n', 'v'})
        """
        self.max_parents = max_parents
        self.max_children_per_parent = max_children_per_parent
        self.min_children_per_parent = min_children_per_parent
        self.pos_filter = pos_filter or {'n', 'v'}  # nouns and verbs by default

    @staticmethod
    @lru_cache(maxsize=None)
    def _concept_info_from_id(synset_id: str) -> ConceptInfo:
        """Cached extraction of :class:`ConceptInfo` from a synset id."""
        synset = wn.synset(synset_id)
        return ConceptInfo(
            synset_id=synset.name(),
            name=synset.lemmas()[0].name().replace('_', ' '),
            definition=synset.definition(),
            examples=synset.examples(),
            pos=synset.pos(),
            lemmas=[lemma.name().replace('_', ' ') for lemma in synset.lemmas()],
        )

    def get_concept_info(self, synset) -> ConceptInfo:
        """Extract information from a WordNet synset using cached lookups."""
        return self._concept_info_from_id(synset.name())

    @staticmethod
    @lru_cache(maxsize=None)
    def _synset_from_id(synset_id: str):
        """Cached lookup of a WordNet synset."""
        return wn.synset(synset_id)
    
    def select_parent_concepts(self) -> List[ConceptInfo]:
        """
        Select diverse parent concepts from WordNet.
        
        Returns:
            List of parent ConceptInfo objects
        """
        logger.info(f"Selecting up to {self.max_parents} parent concepts")
        
        # Get all synsets with sufficient children
        candidate_parents = []
        
        for synset in wn.all_synsets():
            if synset.pos() not in self.pos_filter:
                continue
                
            # Get direct hyponyms (children)
            children = synset.hyponyms()
            
            if len(children) >= self.min_children_per_parent:
                candidate_parents.append(synset)
        
        logger.info(f"Found {len(candidate_parents)} candidate parent concepts")
        
        # Sample diverse parents
        # Prioritize concepts with more children and clearer definitions
        scored_parents = []
        for synset in candidate_parents:
            score = len(synset.hyponyms())  # More children = higher score
            score += len(synset.examples())  # More examples = higher score
            score += (1 if len(synset.definition()) > 20 else 0)  # Good definition
            scored_parents.append((score, synset))
        
        # Sort by score and take top candidates
        scored_parents.sort(key=lambda x: x[0], reverse=True)
        selected_synsets = [synset for _, synset in scored_parents[:self.max_parents]]
        
        # Convert to ConceptInfo
        parent_concepts = [self.get_concept_info(synset) for synset in selected_synsets]
        
        logger.info(f"Selected {len(parent_concepts)} parent concepts")
        return parent_concepts
    
    def select_children_for_parent(self, parent_synset_id: str) -> List[ConceptInfo]:
        """
        Select child concepts for a given parent.
        
        Args:
            parent_synset_id: WordNet synset ID (e.g., 'dog.n.01')
            
        Returns:
            List of child ConceptInfo objects
        """
        parent_synset = self._synset_from_id(parent_synset_id)
        children_synsets = parent_synset.hyponyms()
        
        # Score children by clarity and distinctiveness
        scored_children = []
        for child_synset in children_synsets:
            score = 0
            score += len(child_synset.examples())
            score += (1 if len(child_synset.definition()) > 15 else 0)
            score += len(child_synset.lemmas())  # More lemmas = more ways to express
            
            # Prefer children that are not too abstract
            score += (1 if not any(abstract_word in child_synset.definition().lower() 
                                 for abstract_word in ['abstract', 'concept', 'idea', 'notion']) else 0)
            
            scored_children.append((score, child_synset))
        
        # Sort and select top children
        scored_children.sort(key=lambda x: x[0], reverse=True)
        selected_children = scored_children[:self.max_children_per_parent]
        
        return [self.get_concept_info(synset) for _, synset in selected_children]
    
    def curate_concept_hierarchies(self) -> List[ConceptHierarchy]:
        """
        Curate complete concept hierarchies.
        
        Returns:
            List of ConceptHierarchy objects
        """
        logger.info("Curating concept hierarchies")
        
        parent_concepts = self.select_parent_concepts()
        hierarchies = []
        
        for parent in parent_concepts:
            children = self.select_children_for_parent(parent.synset_id)
            
            if len(children) >= self.min_children_per_parent:
                hierarchy = ConceptHierarchy(
                    parent=parent,
                    children=children,
                    parent_prompts=[],  # Will be filled by PromptGenerator
                    child_prompts={}
                )
                hierarchies.append(hierarchy)
        
        logger.info(f"Curated {len(hierarchies)} concept hierarchies")
        return hierarchies


class PromptGenerator:
    """Generates prompts for concept activation."""

    def __init__(
        self,
        prompts_per_concept: int = 64,
        template_path: Optional[str] = None,
    ):
        """Initialize prompt generator.

        Args:
            prompts_per_concept: Number of prompts to generate per concept.
            template_path: Optional path to a JSON file containing prompt
                templates. If not provided, ``prompt_templates.json`` next to
                this module is used.
        """
        self.prompts_per_concept = prompts_per_concept

        path = Path(template_path) if template_path else Path(__file__).with_name(
            "prompt_templates.json"
        )
        with open(path, "r") as f:
            templates = json.load(f)
        self.positive_templates = templates.get("positive", [])
        self.negative_templates = templates.get("negative", [])
        self.contextual_templates = templates.get("contextual", [])
    
    def generate_prompts_for_concept(self, concept: ConceptInfo, is_positive: bool = True) -> List[str]:
        """
        Generate prompts for a single concept.
        
        Args:
            concept: ConceptInfo object
            is_positive: Whether to generate positive or negative prompts
            
        Returns:
            List of prompt strings
        """
        prompts = []
        templates = self.positive_templates if is_positive else self.negative_templates
        
        # Use different forms of the concept name
        concept_forms = [concept.name] + concept.lemmas[:3]  # Limit lemmas
        concept_forms = list(set(concept_forms))  # Remove duplicates
        
        # Generate prompts from templates
        for template in templates:
            for concept_form in concept_forms:
                prompt = template.format(concept=concept_form)
                prompts.append(prompt)
                if len(prompts) >= self.prompts_per_concept * 0.7:  # 70% from templates
                    break
            if len(prompts) >= self.prompts_per_concept * 0.7:
                break
        
        # Add contextual prompts
        for template in self.contextual_templates:
            if len(prompts) >= self.prompts_per_concept:
                break
            for concept_form in concept_forms:
                prompt = template.format(concept=concept_form)
                prompts.append(prompt)
                if len(prompts) >= self.prompts_per_concept:
                    break
        
        # Add definition-based prompts if we have examples
        if concept.examples and len(prompts) < self.prompts_per_concept:
            for example in concept.examples[:5]:  # Limit examples
                if len(prompts) >= self.prompts_per_concept:
                    break
                prompts.append(f"For example, {example}")
                prompts.append(f"Consider this case: {example}")
        
        # Shuffle and return requested number
        random.shuffle(prompts)
        return prompts[:self.prompts_per_concept]
    
    def generate_contrast_prompts(self, target_concept: ConceptInfo, contrast_concepts: List[ConceptInfo]) -> List[str]:
        """
        Generate prompts that contrast target concept with others.
        
        Args:
            target_concept: The target concept
            contrast_concepts: List of concepts to contrast against
            
        Returns:
            List of contrastive prompt strings
        """
        prompts = []
        
        contrast_templates = [
            "Unlike {contrast}, a {target} is",
            "While {contrast} can {verb}, a {target} cannot",
            "A {target} differs from {contrast} because",
            "You can distinguish a {target} from {contrast} by",
            "Compared to {contrast}, a {target} is more",
            "A {target} is not {contrast} because",
        ]
        
        verbs = ["move", "change", "grow", "work", "function", "exist"]
        
        for contrast in contrast_concepts[:3]:  # Limit contrasts
            for template in contrast_templates:
                if "{verb}" in template:
                    for verb in verbs:
                        prompt = template.format(
                            target=target_concept.name,
                            contrast=contrast.name,
                            verb=verb
                        )
                        prompts.append(prompt)
                else:
                    prompt = template.format(
                        target=target_concept.name,
                        contrast=contrast.name
                    )
                    prompts.append(prompt)
                    
                if len(prompts) >= self.prompts_per_concept // 4:  # 25% contrastive
                    break
            if len(prompts) >= self.prompts_per_concept // 4:
                break
        
        return prompts
    
    def populate_hierarchy_prompts(self, hierarchy: ConceptHierarchy) -> ConceptHierarchy:
        """
        Populate a hierarchy with generated prompts.
        
        Args:
            hierarchy: ConceptHierarchy to populate
            
        Returns:
            ConceptHierarchy with prompts filled in
        """
        # Generate parent prompts
        hierarchy.parent_prompts = self.generate_prompts_for_concept(hierarchy.parent, is_positive=True)
        
        # Generate child prompts
        hierarchy.child_prompts = {}
        for child in hierarchy.children:
            # Positive prompts for this child
            child_prompts = self.generate_prompts_for_concept(child, is_positive=True)
            
            # Add contrastive prompts against siblings
            siblings = [c for c in hierarchy.children if c.synset_id != child.synset_id]
            contrast_prompts = self.generate_contrast_prompts(child, siblings)
            
            # Combine and limit
            all_child_prompts = child_prompts + contrast_prompts
            random.shuffle(all_child_prompts)
            hierarchy.child_prompts[child.synset_id] = all_child_prompts[:self.prompts_per_concept]
        
        return hierarchy


class ConceptSplitter:
    """Handles train/val/test splits for concepts."""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        Initialize concept splitter.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation  
            test_ratio: Fraction for testing
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split_hierarchies(self, hierarchies: List[ConceptHierarchy]) -> Dict[str, List[ConceptHierarchy]]:
        """
        Split concept hierarchies into train/val/test.
        
        Args:
            hierarchies: List of ConceptHierarchy objects
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        # Shuffle hierarchies
        hierarchies = hierarchies.copy()
        random.shuffle(hierarchies)
        
        n_total = len(hierarchies)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        splits = {
            'train': hierarchies[:n_train],
            'val': hierarchies[n_train:n_train + n_val],
            'test': hierarchies[n_train + n_val:]
        }
        
        logger.info(f"Split {n_total} hierarchies: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def split_prompts_within_concept(self, prompts: List[str]) -> Dict[str, List[str]]:
        """
        Split prompts for a single concept into train/val/test.
        
        Args:
            prompts: List of prompts for one concept
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        prompts = prompts.copy()
        random.shuffle(prompts)
        
        n_total = len(prompts)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        return {
            'train': prompts[:n_train],
            'val': prompts[n_train:n_train + n_val],
            'test': prompts[n_train + n_val:]
        }


def save_concept_hierarchies(hierarchies: List[ConceptHierarchy], filepath: str):
    """Save concept hierarchies to JSON file."""
    data = []
    for hierarchy in hierarchies:
        hierarchy_data = {
            'parent': {
                'synset_id': hierarchy.parent.synset_id,
                'name': hierarchy.parent.name,
                'definition': hierarchy.parent.definition,
                'examples': hierarchy.parent.examples,
                'pos': hierarchy.parent.pos,
                'lemmas': hierarchy.parent.lemmas,
            },
            'children': [
                {
                    'synset_id': child.synset_id,
                    'name': child.name,
                    'definition': child.definition,
                    'examples': child.examples,
                    'pos': child.pos,
                    'lemmas': child.lemmas,
                }
                for child in hierarchy.children
            ],
            'parent_prompts': hierarchy.parent_prompts,
            'child_prompts': hierarchy.child_prompts,
        }
        data.append(hierarchy_data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(hierarchies)} concept hierarchies to {filepath}")


def load_concept_hierarchies(filepath: str) -> List[ConceptHierarchy]:
    """Load concept hierarchies from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    hierarchies = []
    for hierarchy_data in data:
        parent = ConceptInfo(**hierarchy_data['parent'])
        children = [ConceptInfo(**child_data) for child_data in hierarchy_data['children']]
        
        hierarchy = ConceptHierarchy(
            parent=parent,
            children=children,
            parent_prompts=hierarchy_data['parent_prompts'],
            child_prompts=hierarchy_data['child_prompts']
        )
        hierarchies.append(hierarchy)
    
    logger.info(f"Loaded {len(hierarchies)} concept hierarchies from {filepath}")
    return hierarchies
