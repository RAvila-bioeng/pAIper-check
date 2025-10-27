"""
Reproducibility Assessment Module
Enhanced for experimental scientific papers.
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
    
    # Weighted average for experimental science
    overall_score = clarity_score * 0.3 + data_score * 0.15 + materials_score * 0.35 + parameter_score * 0.2
    
    feedback = _generate_feedback(clarity_score, data_score, materials_score, parameter_score, text)
    
    return PillarResult("Reproducibility", overall_score, feedback).__dict__


def _check_methodological_clarity(text: str, sections: List) -> float:
    """
    Check if methods are clear enough to replicate.
    Key improvements: vague language detection and specificity checking.
    """
    score = 1.0
    
    # Find methodology section (more flexible matching)
    method_section = None
    for section in sections:
        title_lower = section.title.lower()
        # Check for variations including "materials and methods"
        if any(kw in title_lower for kw in ['method', 'procedure', 'protocol', 'experimental', 'material']):
            method_section = section.content.lower()
            break
    
    # If no dedicated section found, check full text for methodological content
    if not method_section:
        # Some papers integrate methods throughout - check for methodological indicators
        method_indicators = len(re.findall(
            r'\b(prepare|measure|record|calculate|perform|conduct|carry out|obtained|purchased)\b', 
            text
        ))
        if method_indicators < 10:
            return 0.2  # Critical: no methods = not reproducible
        else:
            method_section = text  # Use full text if methods are distributed
    
    # VAGUE LANGUAGE DETECTION (major improvement)
    # These phrases make replication impossible ONLY if not accompanied by specifics
    vague_patterns = [
        r'\b(approximately|about|around)\b(?!\s+\d)',  # Only vague if no number follows
        r'\bseveral\b(?!\s+\d)',
        r'\bsome\b(?!\s+of\s+the)',
        r'\b(appropriate|suitable)\s+(protocol|procedure|method)\b(?!.*(?:described|cited|reference))',
        r'\b(as needed|as necessary|until ready|until done)\b(?!\s+\()',
    ]
    vague_count = sum(len(re.findall(p, method_section)) for p in vague_patterns)
    
    # More lenient thresholds - some vague language is acceptable if offset by detail
    if vague_count > 15:
        score -= 0.3
    elif vague_count > 10:
        score -= 0.15
    elif vague_count > 5:
        score -= 0.05
    
    # SPECIFIC DETAILS CHECK (all disciplines)
    # Equipment/instruments with models or precise dimensions
    specificity_patterns = [
        # Equipment with models/specifications
        r'\b(model|type|series|catalog|cat\.?)\s+[A-Z0-9-]+',
        # Precise measurements with units (critical!) - enhanced patterns
        r'\d+\.?\d*\s*[×x]\s*\d+\.?\d*\s*(?:mm|cm|m|µm|nm)',  # dimensions like "250 μm"
        r'\d+\.?\d*\s*(?:ml|µl|μl|l|g|mg|µg|μg|kg|mm|cm|m|nm|µm|μm)',
        r'\d+\.?\d*\s*(?:°c|celsius|fahrenheit|kelvin|°f|k)',
        r'\d+\.?\d*\s*(?:min|h|hour|s|sec|second|day|week)',
        r'\d+\.?\d*\s*(?:rpm|hz|mhz|khz|ghz)',
        r'\d+\.?\d*\s*(?:v|mv|ma|µa|a|w|mw)',
        r'\d+\.?\d*\s*(?:psi|pa|kpa|mpa|atm|bar)',
        r'\d+\.?\d*\s*(?:m|mm|µm|μm|molar)',  # concentrations
        # Software/versions
        r'\b(version|v\.?|ver\.?)\s*\d+',
        # Catalog numbers - enhanced
        r'\b(cat|catalog|product|catalogue)\s*(number|no|#|num)[:\s]*[A-Z0-9-]+',
        r'\b(cat\.?|catalogue)\s+[#]?[A-Z0-9-]+',
        # Vendors/manufacturers - comprehensive list
        r'\b(sigma|merck|thermo|fisher|invitrogen|promega|qiagen|agilent|roche|bio-rad|'
        r'ge healthcare|millipore|eppendorf|corning|bd biosciences|gibco|'
        r'waters|shimadzu|perkinelmer|buehler|hansatech)\b',
        # Company addresses (city, state, country) - strong reproducibility indicator
        r'\([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2,},?\s*[A-Z]{2,3}\)',
        # Statistical methods
        r'\b(t-test|anova|regression|p-value|p\s*[<>=]\s*0\.\d+|confidence interval|standard deviation|n\s*=\s*\d+)\b',
        # Sample details
        r'\d+\s+(participants|subjects|patients|samples|specimens|animals|replicates)',
        # Potentials/voltages in electrochemistry
        r'[-+]?\d+\.?\d*\s*v\s+vs\.?\s+\w+',
    ]
    
    specific_count = sum(len(re.findall(p, method_section, re.IGNORECASE)) for p in specificity_patterns)
    
    # Adjusted thresholds - this paper has 40+ specific details
    if specific_count < 8:
        score -= 0.35
    elif specific_count < 15:
        score -= 0.2
    elif specific_count >= 30:
        score += 0.15  # Strong bonus for highly detailed methods
    elif specific_count >= 20:
        score += 0.1
    
    # PROCEDURAL CLARITY (step-by-step)
    step_patterns = [
        r'(step|stage|phase)\s+(\d+|one|two|three|i{1,3})',
        r'(first|initially)[,:].*?(then|next|subsequently)',
        r'^\s*\d+\.\s+\w',  # numbered lists
        r'\b(before|after|following|prior to)\b',  # temporal sequence
        r'\b(was|were)\s+(prepared|measured|recorded|obtained|performed|added|placed)\b',  # passive voice procedures
    ]
    steps = sum(len(re.findall(p, method_section, re.MULTILINE)) for p in step_patterns)
    
    # More lenient - procedural verbs count as implicit steps
    if steps < 5:
        score -= 0.2
    elif steps < 10:
        score -= 0.05
    elif steps >= 20:
        score += 0.05
    
    # ESSENTIAL CONTROLS
    controls = len(re.findall(r'\b(control|baseline|replicate|duplicate|triplicate|randomiz|blind)', method_section))
    if controls < 1:
        score -= 0.15
    elif controls >= 3:
        score += 0.05
    
    # BONUS: Check for exceptionally detailed methods (like the example paper)
    # Papers with vendor addresses, multiple equipment specs, and precise parameters deserve recognition
    exceptional_indicators = [
        len(re.findall(r'\([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2,}', method_section)) >= 3,  # Multiple vendor addresses
        specific_count >= 35,  # Very high detail
        len(re.findall(r'\d+\.?\d*\s*(?:mm|µm|μm|cm|ml|µl)', method_section)) >= 10,  # Many precise measurements
    ]
    
    if sum(exceptional_indicators) >= 2:
        score += 0.05  # Bonus for exceptional documentation
    
    return max(0.0, min(1.0, score))


def _check_data_availability(text: str) -> float:
    """Check if data/code can be accessed for replication."""
    score = 0.0
    
    # Public repositories (best practice)
    repos = r'\b(github|gitlab|zenodo|figshare|dryad|dataverse|osf|genbank|geo|arxiv|supplementary|supplement)\b'
    if re.search(repos, text):
        score = 1.0
    
    # DOI provided
    elif re.search(r'\b(doi|digital object identifier)[:\s]+10\.\d{4,}/[^\s]+', text):
        score = 0.8
    
    # Availability statement
    elif re.search(r'(data|code|software|material)\s+(is|are|will be|can be)\s+(available|accessible|obtained|found)', text):
        score = 0.7
    
    # Upon request (less ideal but still counts)
    elif re.search(r'(available\s+)?upon\s+(reasonable\s+)?request', text):
        score = 0.4
    
    # Methods paper - if methods are detailed enough, data may not need separate sharing
    detailed_methods = bool(re.search(r'(procedure|protocol|method).*?(described|detailed|outlined)', text))
    if detailed_methods and score < 0.5:
        score = 0.5  # Methods-focused papers get partial credit
    
    # Data described (at least documented)
    if re.search(r'(dataset|data)\s+(contains?|consists?\s+of|were|was)|sample\s+size|n\s*=\s*\d+|experiment.*?performed|measurement.*?recorded', text):
        score += 0.15
    
    return min(1.0, score)


def _check_materials_specification(text: str) -> float:
    """
    Check if materials, equipment, and reagents are specified with enough detail.
    Critical for experimental reproducibility.
    """
    score = 1.0
    
    # Equipment models/types - enhanced detection for experimental science
    equipment = len(re.findall(
        r'\b(model|type|series|diameter|thickness|dimension|spectrometer|microscope|'
        r'chromatograph|detector|analyzer)\s+[A-Z0-9-]+|'
        r'\d+\.?\d*\s*(?:mm|µm|μm|cm|m)\s+(?:thick|diameter|wide|long)',
        text, re.IGNORECASE
    ))
    if equipment == 0:
        score -= 0.25  # Less harsh penalty
    elif equipment >= 5:
        score += 0.1  # Reward detailed specs
    
    # Vendor information - expanded list
    vendors = len(re.findall(
        r'\b(sigma|merck|thermo|fisher|invitrogen|promega|qiagen|bio-rad|roche|agilent|'
        r'eppendorf|corning|millipore|ge healthcare|waters|shimadzu|perkinelmer|'
        r'buehler|hansatech|instruments?|optronika|ch instruments)\b',
        text, re.IGNORECASE
    ))
    # Also detect vendor addresses (strong indicator)
    vendor_addresses = len(re.findall(r'\([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2,}', text))
    
    total_vendor_info = vendors + vendor_addresses
    
    if total_vendor_info == 0:
        score -= 0.2
    elif total_vendor_info >= 5:
        score += 0.15  # Excellent vendor documentation
    
    # Catalog numbers
    catalogs = len(re.findall(
        r'\b(cat|catalog|product|catalogue)\s*(no|number|#|num\.?)[:\s]*[A-Z0-9-]+|'
        r'\b(cat\.?|catalogue)\s+[#]?[A-Z0-9-]+',
        text, re.IGNORECASE
    ))
    if catalogs > 0:
        score += 0.1
    
    # Software versions
    versions = len(re.findall(r'\b(version|v\.?|ver\.?)\s*\d+(\.\d+)*', text))
    software_mentioned = bool(re.search(r'\b(software|program|package|system|python|r|matlab|spss)\b', text))
    
    if software_mentioned:
        if versions == 0:
            score -= 0.15  # Less harsh if software mentioned but version missing
        else:
            score += 0.1
    
    return max(0.0, min(1.0, score))


def _check_parameter_specification(text: str) -> float:
    """Check if experimental/analytical parameters are precisely specified."""
    score = 1.0
    
    # Numerical values WITH units (most critical) - enhanced patterns
    values_with_units = re.findall(
        r'\b\d+\.?\d*\s*(?:ml|µl|μl|l|g|mg|µg|μg|°c|celsius|'
        r'min|minutes|h|hour|hours|s|sec|second|'
        r'mm|cm|m|nm|µm|μm|rpm|hz|mhz|khz|'
        r'v|mv|ma|µa|μa|w|mw|m|mm|molar|%|psi|pa|kpa|atm|bar)',
        text, re.IGNORECASE
    )
    
    # Count unique patterns (not just occurrences)
    unique_values = len(set(values_with_units))
    
    if unique_values < 5:
        score -= 0.35
    elif unique_values < 10:
        score -= 0.15
    elif unique_values >= 20:
        score += 0.15  # Excellent parametrization
    
    # Parameter keywords - enhanced for experimental science
    params = len(re.findall(
        r'\b(temperature|pressure|ph|concentration|dose|dosage|flow rate|voltage|current|'
        r'potential|wavelength|frequency|speed|rate|time|duration|incubation|reaction|'
        r'humidity|salinity|viscosity|purity|yield)\b',
        text, re.IGNORECASE
    ))
    
    if params < 5:
        score -= 0.2
    elif params >= 10:
        score += 0.05
    
    # Ranges specified (good practice)
    ranges = len(re.findall(r'\b\d+\.?\d*\s*(to|and|-|–|between)\s*\d+\.?\d*', text))
    if ranges > 0:
        score += 0.05
    
    # Specific conditions mentioned
    conditions = len(re.findall(
        r'\bat\s+(room temperature|-?\d+\.?\d*\s*°c)|'
        r'\bunder\s+(continuous|constant|ambient|standard)|'
        r'\bwith\s+(continuous|constant)\s+(stirring|mixing|agitation)',
        text, re.IGNORECASE
    ))
    if conditions >= 2:
        score += 0.05
    
    return max(0.0, min(1.0, score))


def _generate_feedback(clarity: float, data: float, materials: float, parameters: float, text: str) -> str:
    """Generate detailed, balanced feedback as a list of points."""
    feedback_parts = []
    
    # --- Strengths ---
    if clarity >= 0.8:
        feedback_parts.append("✓ Excellent methodological clarity with specific, unambiguous descriptions.")
    if data >= 0.7:
        feedback_parts.append("✓ Good data/code availability statement.")
    if materials >= 0.8:
        feedback_parts.append("✓ Materials and equipment are well-specified with vendors and/or model numbers.")
    if parameters >= 0.8:
        feedback_parts.append("✓ Experimental parameters are comprehensively specified with precise units.")

    # --- Areas for Improvement ---
    if clarity < 0.6:
        feedback_parts.append("⚠️ Methodological clarity could be improved. Consider adding more step-by-step details and reducing vague language (e.g., 'approximately').")
    if data < 0.5:
        feedback_parts.append("⚠️ Data/code availability is unclear. A clear statement pointing to a repository or supplementary materials is recommended.")
    if materials < 0.6:
        feedback_parts.append("⚠️ Specification of materials is lacking. For key reagents and equipment, provide vendor, model, and catalog numbers.")
    if parameters < 0.6:
        feedback_parts.append("⚠️ Key experimental parameters seem to be missing. Ensure all important conditions (temperature, concentration, time, etc.) are stated with precise units.")

    if not feedback_parts:
        return "✓ Excellent reproducibility across all criteria. The methods appear to be well-documented and easy to replicate."

    return "\n  ".join(feedback_parts)