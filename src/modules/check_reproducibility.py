"""
Reproducibility Assessment Module
Enhanced for all scientific disciplines with optimized code.
"""

import re
from typing import List, Tuple
from models.score import PillarResult


def evaluate(paper) -> dict:
    """Evaluate reproducibility aspects of the paper."""
    text = paper.full_text.lower()
    
    clarity_score = _check_methodological_clarity(text, paper.sections)
    data_score = _check_data_availability(text)
    materials_score = _check_materials_specification(text)
    parameter_score = _check_parameter_specification(text)
    
    # Weighted average (clarity is most important)
    overall_score = clarity_score * 0.4 + data_score * 0.25 + materials_score * 0.2 + parameter_score * 0.15
    
    feedback = _generate_feedback(clarity_score, data_score, materials_score, parameter_score)
    
    return PillarResult("Reproducibility", overall_score, feedback).__dict__


def _check_methodological_clarity(text: str, sections: List) -> float:
    """
    Check if methods are clear enough to replicate.
    Key improvements: vague language detection and specificity checking.
    """
    score = 1.0
    
    # Find methodology section
    method_section = None
    for section in sections:
        if any(kw in section.title.lower() for kw in ['method', 'procedure', 'protocol', 'experimental', 'material']):
            method_section = section.content.lower()
            break
    
    if not method_section:
        return 0.2  # Critical: no methods = not reproducible
    
    # VAGUE LANGUAGE DETECTION (major improvement)
    # These phrases make replication impossible
    vague_patterns = [
        r'\b(approximately|about|around|several|some|many|few|various|numerous)\b(?!\s+\d)',
        r'\b(appropriate|suitable|standard|conventional|routine|usual)\s+(protocol|procedure|method)\b',
        r'\b(as needed|as necessary|to taste|until ready|until done|sufficient|adequate|optimal)\b',
    ]
    vague_count = sum(len(re.findall(p, method_section)) for p in vague_patterns)
    
    if vague_count > 8:
        score -= 0.4
    elif vague_count > 4:
        score -= 0.25
    elif vague_count > 1:
        score -= 0.1
    
    # SPECIFIC DETAILS CHECK (all disciplines)
    # Equipment, instruments, software
    specificity_patterns = [
        # Equipment/instruments with models
        r'\b(model|type|series)\s+[A-Z0-9-]+',
        # Measurements with units (critical!)
        r'\d+\.?\d*\s*(ml|µl|l|g|mg|µg|kg|mm|cm|m|nm|µm|°c|celsius|min|h|hour|s|sec|rpm|hz|mhz|psi|pa|m|mm)',
        # Software/versions
        r'\b(version|v\.?)\s*\d+',
        # Catalog numbers
        r'\b(cat|catalog|product)\s*(number|no|#)[:\s]*[A-Z0-9-]+',
        # Vendors
        r'\b(sigma|merck|thermo|fisher|invitrogen|promega|qiagen|agilent|roche)\b',
        # Statistical methods
        r'\b(t-test|anova|regression|p-value|confidence interval|n\s*=\s*\d+)\b',
        # Sample details
        r'\d+\s+(participants|subjects|patients|samples|specimens|animals)',
    ]
    
    specific_count = sum(len(re.findall(p, method_section)) for p in specificity_patterns)
    
    if specific_count < 5:
        score -= 0.35
    elif specific_count < 10:
        score -= 0.2
    elif specific_count >= 20:
        score += 0.1
    
    # PROCEDURAL CLARITY (step-by-step)
    step_patterns = [
        r'(step|stage|phase)\s+(\d+|one|two|three|i{1,3})',
        r'(first|initially)[,:].*?(then|next|subsequently)',
        r'^\s*\d+\.\s+\w',  # numbered lists
    ]
    steps = sum(len(re.findall(p, method_section, re.MULTILINE)) for p in step_patterns)
    
    if steps == 0:
        score -= 0.25
    elif steps < 2:
        score -= 0.1
    
    # ESSENTIAL CONTROLS
    controls = len(re.findall(r'\b(control|baseline|replicate|triplicate|randomiz)', method_section))
    if controls < 2:
        score -= 0.15
    
    return max(0.0, min(1.0, score))


def _check_data_availability(text: str) -> float:
    """Check if data/code can be accessed for replication."""
    score = 0.0
    
    # Public repositories (best practice)
    repos = r'\b(github|gitlab|zenodo|figshare|dryad|dataverse|osf|genbank|geo|arxiv)\.?\b'
    if re.search(repos, text):
        score = 1.0
    
    # DOI provided
    elif re.search(r'\bdoi[:\s]+10\.\d{4,}/[^\s]+', text):
        score = 0.8
    
    # Availability statement
    elif re.search(r'(data|code|software)\s+(is|are|will be)\s+(available|accessible)', text):
        score = 0.6
    
    # Upon request (less ideal)
    elif re.search(r'(available\s+)?upon\s+(reasonable\s+)?request', text):
        score = 0.3
    
    # Data described (at least documented)
    if re.search(r'(dataset|data)\s+(contains?|consists?\s+of)|sample\s+size|n\s*=\s*\d+', text):
        score += 0.1
    
    return min(1.0, score)


def _check_materials_specification(text: str) -> float:
    """
    Check if materials, equipment, and reagents are specified with enough detail.
    Critical for experimental reproducibility.
    """
    score = 1.0
    
    # Equipment models/types
    equipment = len(re.findall(r'\b(model|type|series)\s+[A-Z0-9-]+', text))
    if equipment == 0:
        score -= 0.3
    
    # Vendor information (allows others to obtain same materials)
    vendors = len(re.findall(
        r'\b(sigma|merck|thermo|fisher|invitrogen|promega|qiagen|bio-rad|roche|agilent)\b',
        text, re.IGNORECASE
    ))
    if vendors == 0:
        score -= 0.25
    
    # Catalog numbers (specific identification)
    catalogs = len(re.findall(r'\b(cat|catalog|product)\s*(no|number|#)[:\s]*[A-Z0-9-]+', text, re.IGNORECASE))
    if catalogs > 0:
        score += 0.15
    
    # Software versions (for computational work)
    versions = len(re.findall(r'\b(version|v\.?|ver\.?)\s*\d+(\.\d+)*', text))
    software_mentioned = bool(re.search(r'\b(software|program|package|python|r|matlab|spss)\b', text))
    
    if software_mentioned and versions == 0:
        score -= 0.3  # Software used but no versions = not reproducible
    elif versions > 0:
        score += 0.1
    
    return max(0.0, min(1.0, score))


def _check_parameter_specification(text: str) -> float:
    """Check if experimental/analytical parameters are precisely specified."""
    score = 1.0
    
    # Numerical values WITH units (most critical)
    values_with_units = re.findall(
        r'\b\d+\.?\d*\s*(ml|µl|g|mg|°c|min|h|mm|cm|m|rpm|hz|m|mm|%)\b',
        text
    )
    
    if len(values_with_units) < 3:
        score -= 0.4
    elif len(values_with_units) < 7:
        score -= 0.2
    
    # Parameter keywords
    params = len(re.findall(
        r'\b(temperature|pressure|ph|concentration|dose|flow rate|voltage|'
        r'learning rate|batch size|threshold|alpha|p-value|significance)\b',
        text
    ))
    
    if params < 3:
        score -= 0.3
    
    # Ranges specified (good practice)
    ranges = len(re.findall(r'\b\d+\.?\d*\s*(to|-|–)\s*\d+\.?\d*', text))
    if ranges > 0:
        score += 0.1
    
    return max(0.0, min(1.0, score))


def _generate_feedback(clarity: float, data: float, materials: float, parameters: float) -> str:
    """Generate concise, actionable feedback."""
    issues = []
    
    if clarity < 0.6:
        issues.append("Methods lack sufficient detail - reduce vague language (e.g., 'several', 'appropriate') and add specific quantities, equipment models, and step-by-step procedures")
    
    if data < 0.5:
        issues.append("Data availability unclear - provide repository link, DOI, or clear access instructions")
    
    if materials < 0.6:
        issues.append("Materials insufficiently specified - add vendor names, catalog numbers, and equipment models/versions")
    
    if parameters < 0.6:
        issues.append("Parameters not precisely specified - include all numerical values with units (temperature, concentrations, durations, etc.)")
    
    if not issues:
        return "Strong reproducibility with clear methods, accessible data, and precise specifications."
    
    return " • ".join(issues)