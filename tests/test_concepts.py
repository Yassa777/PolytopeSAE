import pathlib
import random
import sys

from polytope_hsae.concepts import (ConceptHierarchy, ConceptInfo,
                                    ConceptSplitter, PromptGenerator)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


def _dummy_concept(idx: int) -> ConceptInfo:
    return ConceptInfo(
        synset_id=f"c{idx}.n.01",
        name=f"concept{idx}",
        definition="",
        examples=[],
        pos="n",
        lemmas=[f"concept{idx}"],
    )


def _dummy_hierarchy(idx: int) -> ConceptHierarchy:
    parent = _dummy_concept(idx)
    child = _dummy_concept(idx + 1000)
    return ConceptHierarchy(
        parent=parent, children=[child], parent_prompts=[], child_prompts={}
    )


def test_prompt_generator_count():
    random.seed(0)
    generator = PromptGenerator(prompts_per_concept=10)
    concept = _dummy_concept(0)
    prompts = generator.generate_prompts_for_concept(concept)
    assert len(prompts) == 10


def test_concept_splitter_ratios():
    random.seed(0)
    hierarchies = [_dummy_hierarchy(i) for i in range(10)]
    splitter = ConceptSplitter()
    splits = splitter.split_hierarchies(hierarchies)
    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 2

    prompts = [f"prompt {i}" for i in range(20)]
    prompt_splits = splitter.split_prompts_within_concept(prompts)
    assert len(prompt_splits["train"]) == 14
    assert len(prompt_splits["val"]) == 3
    assert len(prompt_splits["test"]) == 3
