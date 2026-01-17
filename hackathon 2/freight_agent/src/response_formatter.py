"""
Response Formatter (GPT Call #3)

Takes a Quote and generates a natural-sounding email reply.
Uses GPT for tone matching and professional formatting.
"""

import json
from datetime import datetime
from openai import AsyncOpenAI

from models import Quote, QuoteLineItem, QuoteResponse, Email


FORMATTER_SYSTEM_PROMPT = """
You are a freight quotation assistant. Generate a professional email reply
based on the quote data provided.

## DISPLAY RULES (configured per customer):
- show_transit_time: {show_transit_time} - Include transit days if true
- show_chargeable_weight: {show_chargeable_weight} - Show weight calculation for air freight if true
- show_subtotals: {show_subtotals} - Break down base price, discount, and margin if true
- hide_margin: {hide_margin} - Don't mention margin percentage if true

## FORMATTING GUIDELINES:
1. Start with a warm greeting using the customer's name (extract first name if available)
2. Reference their original request briefly
3. Present the quote clearly:
   - Use a simple table or formatted list for multiple items
   - Include route, size/weight, and total price
   - Add transit time ONLY if show_transit_time=true
   - Add price breakdown ONLY if show_subtotals=true
4. Include any WARNINGS prominently (but professionally)
5. If any routes have ERRORS (no rate found), explain clearly and offer next steps
6. Mention quote validity (7 days from today)
7. End with an offer to answer questions
8. Professional sign-off

## TONE GUIDELINES:
- Match the formality of the customer's original email
- Casual request (typos, informal language) → friendly but professional response
- Formal request → more business-like response
- Always be helpful and clear

## CURRENCY:
- Always show prices in USD
- Format: $X,XXX.XX (with commas for thousands, 2 decimal places)

## HANDLING ERRORS:
If a route has no rate found:
- Acknowledge it clearly but professionally
- Explain we don't currently have rates for that specific route
- Offer to check with carriers or suggest contacting for alternatives
- Don't skip the route silently

## OUTPUT FORMAT:
Return ONLY the email body text. Do not include subject line or metadata.
"""


def _quote_to_dict(quote: Quote) -> dict:
    """Convert Quote to a JSON-serializable dict for the prompt."""
    return {
        "customer_name": quote.customer_name,
        "customer_email": quote.customer_email,
        "line_items": [
            {
                "shipment_index": li.shipment_index,
                "description": li.description,
                "has_rate": li.rate_match is not None,
                "base_price": li.base_price,
                "discount_amount": li.discount_amount,
                "margin_amount": li.margin_amount,
                "surcharge_total": li.surcharge_total,
                "line_total": li.line_total,
                "transit_days": li.rate_match.transit_days if li.rate_match else None,
                "chargeable_weight_kg": li.rate_match.chargeable_weight_kg if li.rate_match else None,
                "warnings": list(li.warnings),
                "errors": list(li.errors),
            }
            for li in quote.line_items
        ],
        "grand_total": quote.grand_total,
        "is_complete": quote.is_complete,
        "has_warnings": quote.has_warnings,
        "has_errors": quote.has_errors,
    }


async def format_response(
    quote: Quote,
    original_email: Email,
    client: AsyncOpenAI,
    model: str = "gpt-4o-mini",
) -> QuoteResponse:
    """
    GPT call #3: Generate natural email response from structured quote.

    Args:
        quote: The calculated quote with all pricing
        original_email: The original customer email for context/tone matching
        client: OpenAI async client
        model: Model to use (default: gpt-4o-mini)

    Returns:
        QuoteResponse with subject and body
    """
    # Build system prompt with display flags
    system_prompt = FORMATTER_SYSTEM_PROMPT.format(
        show_transit_time=quote.show_transit_time,
        show_chargeable_weight=quote.show_chargeable_weight,
        show_subtotals=quote.show_subtotals,
        hide_margin=quote.hide_margin,
    )

    # Build user prompt with email and quote data
    quote_json = json.dumps(_quote_to_dict(quote), indent=2)
    user_prompt = f"""
## ORIGINAL EMAIL FROM CUSTOMER:
From: {original_email.sender}
Subject: {original_email.subject}

{original_email.body}

## QUOTE DATA:
{quote_json}

## TODAY'S DATE: {datetime.now().strftime("%B %d, %Y")}

Generate the email response now. Return only the email body text.
"""

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,  # Slightly creative for natural language
    )

    body = response.choices[0].message.content.strip()

    # Generate subject line
    subject = f"RE: {original_email.subject}"

    return QuoteResponse(
        subject=subject,
        body=body,
        quote=quote,
        generated_at=datetime.now().isoformat(),
        model_used=model,
    )


def format_response_sync(
    quote: Quote,
    original_email: Email,
    client,  # Regular OpenAI client
    model: str = "gpt-4o-mini",
) -> QuoteResponse:
    """
    Synchronous version of format_response for non-async contexts.
    """
    # Build system prompt with display flags
    system_prompt = FORMATTER_SYSTEM_PROMPT.format(
        show_transit_time=quote.show_transit_time,
        show_chargeable_weight=quote.show_chargeable_weight,
        show_subtotals=quote.show_subtotals,
        hide_margin=quote.hide_margin,
    )

    # Build user prompt with email and quote data
    quote_json = json.dumps(_quote_to_dict(quote), indent=2)
    user_prompt = f"""
## ORIGINAL EMAIL FROM CUSTOMER:
From: {original_email.sender}
Subject: {original_email.subject}

{original_email.body}

## QUOTE DATA:
{quote_json}

## TODAY'S DATE: {datetime.now().strftime("%B %d, %Y")}

Generate the email response now. Return only the email body text.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    body = response.choices[0].message.content.strip()
    subject = f"RE: {original_email.subject}"

    return QuoteResponse(
        subject=subject,
        body=body,
        quote=quote,
        generated_at=datetime.now().isoformat(),
        model_used=model,
    )
