# Integration Guide: Combining freight_agent + demo_web_gui

> **Purpose**: This guide provides step-by-step instructions to integrate our two implementations into a unified, production-ready freight quoting system. We're combining the best of both projects! 🚀

---

## Why This Integration?

Both implementations bring unique strengths to the table. By combining them, we create a system that's greater than the sum of its parts.

### What Each Project Contributes

| Project | Key Strengths | What We're Using |
|---------|---------------|------------------|
| **freight_agent** | Optimized AI pipeline, frozen dataclasses for type safety, confidence scoring, streaming support | Core quote generation engine |
| **demo_web_gui** | Beautiful Streamlit UI, comprehensive FastAPI server, database persistence, HITL workflow, webhook integration | User interface, API layer, persistence |

### Why Combine Instead of Choose One?

1. **Complementary Focus Areas**
   - `freight_agent` focused on optimizing the AI pipeline (reducing API calls, deterministic validation)
   - `demo_web_gui` focused on the user experience (UI, database, human review workflow)
   - Together: optimized AI + great UX = production-ready system

2. **Shared Data Models**
   - Using frozen dataclasses from `freight_agent` gives us immutability and type safety across the entire system
   - This prevents subtle bugs from data mutations as quotes flow through the pipeline

3. **Confidence-Driven HITL**
   - `freight_agent`'s confidence scoring (HIGH/MEDIUM/LOW) can drive smarter routing in `demo_web_gui`'s HITL workflow
   - Instead of simple keyword matching, we get nuanced decisions based on data completeness

4. **Cost & Latency Benefits**
   - The optimized pipeline uses fewer API calls, which means lower costs and faster responses
   - These savings compound as we process more quotes

### The Integration Philosophy

We're using the **Adapter Pattern** - a thin translation layer between the two systems. This means:
- ✅ Minimal changes to existing, working code
- ✅ Each project's tests continue to work
- ✅ Easy to debug (clear boundary between systems)
- ✅ Reversible if needed

---

## Overview

We are combining:
- **freight_agent**: Optimized 3-call AI pipeline with frozen dataclasses, tool calling, and confidence scoring
- **demo_web_gui**: Streamlit UI, FastAPI server, database persistence, and HITL workflow

The integration uses `freight_agent`'s pipeline as the core engine while keeping `demo_web_gui`'s excellent UI and persistence layer.

---

## Prerequisites

Before starting:
1. Ensure both `freight_agent/` and `demo_web_gui/` directories exist in `hackathon 2/`
2. All tests in `freight_agent/tests/` should pass
3. Have access to the same Python environment with all dependencies

---

## Step 1: Create Shared Models Package

Create a shared package that both projects can import from. This ensures we have a single source of truth for our data structures.

### 1.1 Create the shared directory structure

```bash
mkdir -p "hackathon 2/shared"
touch "hackathon 2/shared/__init__.py"
```

### 1.2 Create `hackathon 2/shared/__init__.py`

```python
"""Shared models and utilities for freight quote system."""

from pathlib import Path

SHARED_DIR = Path(__file__).parent
PROJECT_ROOT = SHARED_DIR.parent
FREIGHT_AGENT_DIR = PROJECT_ROOT / "freight_agent"
DEMO_WEB_GUI_DIR = PROJECT_ROOT / "demo_web_gui"
HACKATHON_DATA_DIR = PROJECT_ROOT / "hackathon_data"
```

### 1.3 Create `hackathon 2/shared/models.py`

This file consolidates our data models. Copy the frozen dataclasses from `freight_agent/src/models.py`:

```python
"""
Shared data models for the freight quote system.

These are frozen (immutable) dataclasses for type safety across both projects.
Immutability helps prevent subtle bugs as data flows through the pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
from enum import Enum


class ConfidenceScore(Enum):
    """
    Confidence level for generated quotes.

    Used to drive HITL routing decisions:
    - HIGH: Ready to send automatically
    - MEDIUM: Suggest human verification
    - LOW: Requires human review
    """
    HIGH = "high"      # Complete data, rates found, no SOP errors
    MEDIUM = "medium"  # Missing clarification or partial data
    LOW = "low"        # Critical information missing


@dataclass(frozen=True)
class Email:
    """Incoming email request."""
    from_address: str
    to_address: str
    subject: str
    body: str


@dataclass(frozen=True)
class Shipment:
    """Extracted shipment details from email."""
    mode: str  # "sea" or "air"
    origin_raw: str
    destination_raw: str
    container_size_ft: Optional[int] = None  # For sea: 20 or 40
    quantity: int = 1
    weight_kg: Optional[float] = None  # For air
    volume_cbm: Optional[float] = None  # For air


@dataclass(frozen=True)
class ExtractionResult:
    """Result of email extraction step."""
    shipments: List[Shipment]
    missing_fields: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None


@dataclass(frozen=True)
class CustomerSOP:
    """Customer-specific Standard Operating Procedures."""
    customer_name: str
    margin_percent: float = 15.0
    flat_discount_percent: Optional[float] = None
    volume_discount_tiers: Optional[dict] = None
    mode_restriction: Optional[str] = None  # "sea" or "air" only
    origin_restriction: Optional[List[str]] = None
    show_transit_time: bool = True
    show_chargeable_weight: bool = False
    hide_margin: bool = False
    discount_before_margin: bool = True
    warn_transit_over_days: Optional[int] = None


@dataclass(frozen=True)
class Surcharge:
    """Destination-specific surcharge."""
    destination: str
    surcharge_type: str
    amount: float
    description: str


@dataclass(frozen=True)
class EnrichedShipment:
    """Shipment with enrichment data."""
    shipment: Shipment
    origin_normalized: str
    destination_normalized: str
    surcharges: List[Surcharge] = field(default_factory=list)


@dataclass(frozen=True)
class EnrichedRequest:
    """Fully enriched request with customer context."""
    customer_name: str
    customer_sop: CustomerSOP
    shipments: List[EnrichedShipment]
    validation_errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RateMatch:
    """Matched rate from rate sheet."""
    origin: str
    destination: str
    rate_per_container: Optional[float] = None  # For sea
    rate_per_kg: Optional[float] = None  # For air
    min_charge: Optional[float] = None  # For air
    transit_days: Optional[int] = None


@dataclass(frozen=True)
class QuoteLineItem:
    """Single line item in a quote."""
    description: str
    base_price: float
    discount_amount: float = 0.0
    margin_amount: float = 0.0
    surcharge_total: float = 0.0
    line_total: float = 0.0


@dataclass(frozen=True)
class DisplayFlags:
    """Flags controlling what to show in quote response."""
    show_transit_time: bool = True
    show_chargeable_weight: bool = False
    show_margin_breakdown: bool = True
    show_discount_reason: bool = True


@dataclass(frozen=True)
class Quote:
    """Complete quote with all line items."""
    customer_name: str
    line_items: List[QuoteLineItem]
    subtotal: float
    total_discount: float
    total_margin: float
    total_surcharges: float
    grand_total: float
    currency: str = "USD"
    display_flags: DisplayFlags = field(default_factory=DisplayFlags)
    sop_summary: Optional[str] = None


@dataclass(frozen=True)
class QuoteResponse:
    """Final formatted response."""
    subject: str
    body: str


@dataclass(frozen=True)
class PipelineResult:
    """Complete result from the quote pipeline."""
    email: Email
    extraction: ExtractionResult
    enriched_request: Optional[EnrichedRequest]
    rate_matches: List[RateMatch]
    quote: Optional[Quote]
    response: Optional[QuoteResponse]
    confidence: ConfidenceScore
    trace: List[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
```

---

## Step 2: Update freight_agent to Use Shared Models

### 2.1 Modify `freight_agent/src/models.py`

Replace the content with an import from shared (keeps backwards compatibility):

```python
"""
Models for freight_agent.
Re-exports from shared models for backwards compatibility.
"""

# Re-export all models from shared
from shared.models import (
    ConfidenceScore,
    Email,
    Shipment,
    ExtractionResult,
    CustomerSOP,
    Surcharge,
    EnrichedShipment,
    EnrichedRequest,
    RateMatch,
    QuoteLineItem,
    DisplayFlags,
    Quote,
    QuoteResponse,
    PipelineResult,
)

__all__ = [
    "ConfidenceScore",
    "Email",
    "Shipment",
    "ExtractionResult",
    "CustomerSOP",
    "Surcharge",
    "EnrichedShipment",
    "EnrichedRequest",
    "RateMatch",
    "QuoteLineItem",
    "DisplayFlags",
    "Quote",
    "QuoteResponse",
    "PipelineResult",
]
```

### 2.2 Add shared to Python path

Create/update `freight_agent/src/__init__.py`:

```python
"""freight_agent package."""
import sys
from pathlib import Path

# Add shared directory to path
shared_path = Path(__file__).parent.parent.parent / "shared"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path.parent))
```

---

## Step 3: Create Pipeline Adapter in demo_web_gui

This is the key integration piece - a thin adapter that bridges the two systems.

### 3.1 Create `demo_web_gui/src/freight_pipeline_adapter.py`

```python
"""
Adapter to integrate freight_agent's pipeline into demo_web_gui.

This module provides a clean interface between:
- demo_web_gui's expected input/output formats
- freight_agent's optimized pipeline

The adapter pattern keeps both codebases clean and changes minimal.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add freight_agent to path
HACKATHON_DIR = Path(__file__).parent.parent.parent
FREIGHT_AGENT_DIR = HACKATHON_DIR / "freight_agent"
sys.path.insert(0, str(FREIGHT_AGENT_DIR))
sys.path.insert(0, str(HACKATHON_DIR))

# Import from freight_agent
from src.pipeline import process_email as freight_process_email
from src.models import Email, PipelineResult, ConfidenceScore


def adapt_email_input(email_data: dict) -> Email:
    """
    Convert demo_web_gui email format to freight_agent Email dataclass.

    Handles both key naming conventions for compatibility.

    Args:
        email_data: Dict with keys: from/from_address, to/to_address, subject, body

    Returns:
        Frozen Email dataclass
    """
    return Email(
        from_address=email_data.get("from", email_data.get("from_address", "")),
        to_address=email_data.get("to", email_data.get("to_address", "")),
        subject=email_data.get("subject", ""),
        body=email_data.get("body", ""),
    )


def adapt_pipeline_output(result: PipelineResult) -> dict:
    """
    Convert freight_agent PipelineResult to demo_web_gui expected format.

    Maps the frozen dataclass structure to the dict format expected by
    demo_web_gui's UI and database layer.

    Args:
        result: PipelineResult from freight_agent pipeline

    Returns:
        Dict matching demo_web_gui's expected structure
    """
    # Extract inferred fields for database storage
    origin_city = None
    destination_city = None
    transport_type = None

    if result.enriched_request and result.enriched_request.shipments:
        first_shipment = result.enriched_request.shipments[0]
        origin_city = first_shipment.origin_normalized
        destination_city = first_shipment.destination_normalized
        transport_type = first_shipment.shipment.mode

    return {
        "quote_text": result.response.body if result.response else None,
        "subject": result.response.subject if result.response else None,
        "confidence": result.confidence.value,
        "confidence_score": result.confidence,  # Keep enum for programmatic use
        "error": result.error,
        "trace": result.trace,

        # Inferred fields for DB storage (compatible with demo_web_gui schema)
        "inferred": {
            "origin_city": origin_city,
            "destination_city": destination_city,
            "price": result.quote.grand_total if result.quote else None,
            "currency": result.quote.currency if result.quote else "USD",
            "transport_type": transport_type,
            "has_route": len(result.rate_matches) > 0,
        },

        # Full result for debugging
        "pipeline_result": result.to_dict() if hasattr(result, 'to_dict') else None,
    }


def determine_hitl_routing(result: PipelineResult, config: dict = None) -> tuple[bool, str]:
    """
    Determine if a quote should be routed to human review.

    Combines freight_agent's ConfidenceScore with demo_web_gui's business rules
    for comprehensive HITL routing.

    Args:
        result: PipelineResult from pipeline
        config: Optional config with thresholds (e.g., large_order_threshold)

    Returns:
        Tuple of (needs_human: bool, reason: str)
    """
    config = config or {}
    large_order_threshold = config.get("large_order_threshold", 20000)

    # Low confidence always needs human
    if result.confidence == ConfidenceScore.LOW:
        return True, "Low confidence - critical information missing"

    # Medium confidence needs human verification
    if result.confidence == ConfidenceScore.MEDIUM:
        reason = "Medium confidence"
        if result.extraction and result.extraction.needs_clarification:
            reason += f" - {result.extraction.clarification_reason}"
        return True, reason

    # Validation errors need human
    if result.enriched_request and result.enriched_request.validation_errors:
        errors = result.enriched_request.validation_errors
        return True, f"Validation errors: {', '.join(errors)}"

    # Large orders need human approval (using demo_web_gui's threshold logic)
    if result.quote and result.quote.grand_total > large_order_threshold:
        return True, f"Large order (>${large_order_threshold:,})"

    # Check for special keywords in original email (demo_web_gui's keyword detection)
    special_keywords = ["ddp", "dg", "dangerous", "reefer", "refrigerated", "hazmat", "oversized"]
    email_text = (result.email.subject + " " + result.email.body).lower()
    for keyword in special_keywords:
        if keyword in email_text:
            return True, f"Special request detected: {keyword}"

    # High confidence, no issues - can auto-process
    return False, "Auto-approved: high confidence"


def run_freight_pipeline(
    email_data: dict,
    rate_sheet_path: str,
    difficulty: str = "medium",
    enable_sop: bool = True,
    use_streaming: bool = False,
    config: dict = None,
) -> dict:
    """
    Main entry point - runs freight_agent pipeline with demo_web_gui interface.

    This is the function demo_web_gui should call instead of its own pipeline.

    Args:
        email_data: Dict with email fields (from, to, subject, body)
        rate_sheet_path: Path to the rate sheet Excel file
        difficulty: Rate sheet difficulty level ("easy", "medium", "hard")
        enable_sop: Whether to apply SOP rules
        use_streaming: Whether to use streaming response formatter
        config: Additional configuration options

    Returns:
        Dict with quote_text, confidence, trace, inferred fields, and HITL routing
    """
    config = config or {}

    # Convert input format
    email = adapt_email_input(email_data)

    # Run the optimized freight_agent pipeline
    result: PipelineResult = freight_process_email(
        email=email,
        rate_sheet_path=rate_sheet_path,
        difficulty=difficulty,
        enable_sop=enable_sop,
        use_streaming=use_streaming,
    )

    # Convert output format
    output = adapt_pipeline_output(result)

    # Determine HITL routing (combines both projects' logic)
    needs_human, hitl_reason = determine_hitl_routing(result, config)
    output["needs_human_review"] = needs_human
    output["hitl_reason"] = hitl_reason
    output["status"] = "needs_human_decision" if needs_human else "auto_processed"

    return output


# Convenience exports
__all__ = [
    "run_freight_pipeline",
    "adapt_email_input",
    "adapt_pipeline_output",
    "determine_hitl_routing",
]
```

---

## Step 4: Update demo_web_gui Pipeline Integration

### 4.1 Modify `demo_web_gui/src/pipeline.py`

Update to use the adapter while preserving the original implementation:

```python
"""
Pipeline module for demo_web_gui.

This module now delegates to freight_agent's optimized pipeline via the adapter.
The original implementation is preserved in _legacy_run_quote_pipeline for
reference and fallback.
"""

from pathlib import Path
from typing import Optional, Dict, Any

# Import the adapter
from .freight_pipeline_adapter import run_freight_pipeline as _run_freight_pipeline

# NOTE: The original implementation below is preserved for reference.
# You can rename the original function to _legacy_run_quote_pipeline
# and keep it in case we need to compare behavior or debug.


def run_quote_pipeline(
    email_data: dict,
    rate_sheet_path: str,
    difficulty: str = "medium",
    enable_sop: bool = True,
    use_openai: bool = True,
    config: dict = None,
) -> dict:
    """
    Run the quote generation pipeline.

    This function now uses freight_agent's optimized pipeline
    via the adapter layer, combining both projects' strengths.

    Args:
        email_data: Dict with email fields
        rate_sheet_path: Path to rate sheet
        difficulty: Rate sheet difficulty
        enable_sop: Whether to apply SOP rules
        use_openai: Whether to use OpenAI (always True for freight_agent)
        config: Additional config options

    Returns:
        Dict with quote results, trace, and HITL routing info
    """
    # Use the optimized freight_agent pipeline via adapter
    return _run_freight_pipeline(
        email_data=email_data,
        rate_sheet_path=rate_sheet_path,
        difficulty=difficulty,
        enable_sop=enable_sop,
        use_streaming=False,  # Streamlit handles its own streaming
        config=config,
    )
```

---

## Step 5: Update demo_web_gui API Server

### 5.1 Modify `demo_web_gui/api_server.py` quote endpoint

Find the `/quote` endpoint and update it to use the new pipeline output format:

```python
@app.post("/api/v1/quote")
async def generate_quote(request: QuoteRequest):
    """Generate a freight quote from email."""
    try:
        # Run the integrated pipeline
        result = run_quote_pipeline(
            email_data={
                "from": request.email_from,
                "to": request.email_to,
                "subject": request.subject,
                "body": request.body,
            },
            rate_sheet_path=get_rate_sheet_path(request.difficulty),
            difficulty=request.difficulty,
            enable_sop=request.enable_sop,
            config={"large_order_threshold": settings.HITL_LARGE_ORDER_USD},
        )

        # The adapter now returns confidence and HITL routing
        return {
            "success": True,
            "quote_text": result["quote_text"],
            "subject": result.get("subject"),
            "confidence": result["confidence"],
            "needs_human_review": result["needs_human_review"],
            "hitl_reason": result["hitl_reason"],
            "status": result["status"],
            "inferred": result["inferred"],
            "trace": result["trace"],
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## Step 6: Update demo_web_gui Streamlit App

### 6.1 Modify `demo_web_gui/app.py` to display confidence

Find the section that displays quote results and add confidence display:

```python
# After running the pipeline
result = run_quote_pipeline(...)

# Display confidence badge
confidence = result.get("confidence", "unknown")
confidence_colors = {
    "high": "green",
    "medium": "orange",
    "low": "red",
}
st.markdown(
    f"**Confidence:** :{confidence_colors.get(confidence, 'gray')}[{confidence.upper()}]"
)

# Display HITL routing decision
if result.get("needs_human_review"):
    st.warning(f"🔍 Needs Human Review: {result.get('hitl_reason')}")
else:
    st.success("✅ Auto-approved for sending")
```

---

## Step 7: Preserve and Run Tests

### 7.1 Keep freight_agent tests intact

**Important**: The tests in `freight_agent/tests/` should remain unchanged. They test the core pipeline and must continue to pass.

Existing tests:
- `test_extraction.py` - Tests extraction on all 10 emails
- `test_e2e_pipeline.py` - End-to-end tests comparing with solutions
- `test_email02_hard.py` - Specific hard test scenarios

### 7.2 Add integration tests

Create `demo_web_gui/tests/test_integration.py`:

```python
"""
Integration tests to verify freight_agent pipeline works with demo_web_gui.

These tests ensure the adapter correctly bridges the two systems.
"""

import pytest
import sys
from pathlib import Path

# Setup paths
HACKATHON_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(HACKATHON_DIR))
sys.path.insert(0, str(HACKATHON_DIR / "freight_agent"))
sys.path.insert(0, str(HACKATHON_DIR / "demo_web_gui"))

from demo_web_gui.src.freight_pipeline_adapter import (
    run_freight_pipeline,
    adapt_email_input,
    determine_hitl_routing,
)
from shared.models import ConfidenceScore


class TestPipelineAdapter:
    """Test the freight_agent adapter."""

    def test_adapt_email_input(self):
        """Test email format conversion."""
        email_data = {
            "from": "test@example.com",
            "to": "quotes@company.com",
            "subject": "Quote request",
            "body": "Need a quote for shipping.",
        }

        email = adapt_email_input(email_data)

        assert email.from_address == "test@example.com"
        assert email.to_address == "quotes@company.com"
        assert email.subject == "Quote request"
        assert email.body == "Need a quote for shipping."

    def test_adapter_handles_alternate_keys(self):
        """Test email with alternate key names (demo_web_gui format)."""
        email_data = {
            "from_address": "test@example.com",
            "to_address": "quotes@company.com",
            "subject": "Quote request",
            "body": "Need a quote.",
        }

        email = adapt_email_input(email_data)

        assert email.from_address == "test@example.com"


class TestHITLRouting:
    """Test HITL routing logic (combines both projects' rules)."""

    def test_low_confidence_needs_human(self):
        """Low confidence should always route to human."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.confidence = ConfidenceScore.LOW
        result.extraction = None
        result.enriched_request = None
        result.quote = None
        result.email = MagicMock()
        result.email.subject = "Test"
        result.email.body = "Test body"

        needs_human, reason = determine_hitl_routing(result)

        assert needs_human is True
        assert "Low confidence" in reason

    def test_high_confidence_auto_approved(self):
        """High confidence with no issues should auto-approve."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.confidence = ConfidenceScore.HIGH
        result.extraction = MagicMock()
        result.extraction.needs_clarification = False
        result.enriched_request = MagicMock()
        result.enriched_request.validation_errors = []
        result.quote = MagicMock()
        result.quote.grand_total = 5000  # Below threshold
        result.email = MagicMock()
        result.email.subject = "Normal shipment"
        result.email.body = "Regular cargo request"

        needs_human, reason = determine_hitl_routing(result)

        assert needs_human is False
        assert "Auto-approved" in reason

    def test_large_order_needs_human(self):
        """Large orders should route to human (demo_web_gui rule)."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.confidence = ConfidenceScore.HIGH
        result.extraction = MagicMock()
        result.extraction.needs_clarification = False
        result.enriched_request = MagicMock()
        result.enriched_request.validation_errors = []
        result.quote = MagicMock()
        result.quote.grand_total = 50000  # Above threshold
        result.email = MagicMock()
        result.email.subject = "Normal"
        result.email.body = "Normal"

        needs_human, reason = determine_hitl_routing(result, {"large_order_threshold": 20000})

        assert needs_human is True
        assert "Large order" in reason

    def test_special_keywords_need_human(self):
        """Special keywords should route to human (demo_web_gui rule)."""
        from unittest.mock import MagicMock

        result = MagicMock()
        result.confidence = ConfidenceScore.HIGH
        result.extraction = MagicMock()
        result.extraction.needs_clarification = False
        result.enriched_request = MagicMock()
        result.enriched_request.validation_errors = []
        result.quote = MagicMock()
        result.quote.grand_total = 1000
        result.email = MagicMock()
        result.email.subject = "DDP shipment needed"
        result.email.body = "Please quote DDP terms"

        needs_human, reason = determine_hitl_routing(result)

        assert needs_human is True
        assert "ddp" in reason.lower()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 7.3 Run all tests to verify

```bash
# Run freight_agent tests (should still pass - no changes to core logic)
cd "hackathon 2/freight_agent"
python -m pytest tests/ -v

# Run integration tests
cd "hackathon 2/demo_web_gui"
python -m pytest tests/test_integration.py -v
```

---

## Step 8: Update Requirements

### 8.1 Ensure both projects have compatible dependencies

Add to `demo_web_gui/requirements.txt` if not present:

```
# Shared with freight_agent
openai>=1.40.0
pandas>=2.0.0
openpyxl>=3.1.0
```

---

## Summary Checklist

After completing all steps, verify:

- [ ] `shared/models.py` exists with all frozen dataclasses
- [ ] `freight_agent/src/models.py` re-exports from shared
- [ ] `demo_web_gui/src/freight_pipeline_adapter.py` exists
- [ ] `demo_web_gui/src/pipeline.py` calls the adapter
- [ ] `demo_web_gui/api_server.py` returns confidence and HITL info
- [ ] `demo_web_gui/app.py` displays confidence badges
- [ ] All `freight_agent/tests/` still pass ✅
- [ ] Integration tests in `demo_web_gui/tests/test_integration.py` pass ✅

---

## Architecture After Integration

```
hackathon 2/
├── shared/                          # NEW - shared code
│   ├── __init__.py
│   └── models.py                    # Frozen dataclasses (source of truth)
│
├── freight_agent/                   # Core AI engine
│   ├── src/
│   │   ├── models.py                # Re-exports from shared
│   │   ├── pipeline.py              # Optimized 3-call pipeline (unchanged)
│   │   ├── extraction.py            # (unchanged)
│   │   ├── enrichment.py            # (unchanged)
│   │   └── ...
│   └── tests/                       # PRESERVED - all tests kept
│       ├── test_extraction.py
│       ├── test_e2e_pipeline.py
│       └── test_email02_hard.py
│
├── demo_web_gui/                    # UI & persistence layer
│   ├── app.py                       # Streamlit UI (updated for confidence)
│   ├── api_server.py                # FastAPI (updated for confidence)
│   ├── src/
│   │   ├── freight_pipeline_adapter.py  # NEW - bridges the systems
│   │   ├── pipeline.py              # Updated to use adapter
│   │   ├── db_store.py              # (unchanged)
│   │   └── ...
│   └── tests/
│       └── test_integration.py      # NEW - integration tests
│
└── hackathon_data/                  # (unchanged)
```

---

## Notes for Implementation

1. **Preserve existing code** - The adapter pattern means minimal changes to working code
2. **Keep all tests** - Both `freight_agent/tests/` and any existing `demo_web_gui` tests should continue to work
3. **The adapter is the bridge** - It handles format conversion so neither project needs major refactoring
4. **Confidence drives HITL** - The confidence scoring makes human routing decisions smarter
5. **Frozen dataclasses** - These ensure type safety and prevent subtle mutation bugs

---

## Questions?

If anything is unclear or you run into issues during implementation, let's discuss! This integration is about combining our best work into something even better. 🚀
