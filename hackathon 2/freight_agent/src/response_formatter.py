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
6. End with an offer to answer questions
7. Sign off with EXACTLY:
   Best regards,
   Magus AI

## SOP REFERENCES (IMPORTANT):
When explaining pricing, ALWAYS reference the customer's account agreement:
- If a DISCOUNT was applied, mention it with the discount_reason (e.g., "Your 10% account discount has been applied")
- If SURCHARGES were added, explain each one using the surcharge name and reason (e.g., "Australia Biosecurity Fee: $150")
- If a request was REJECTED due to mode/origin restrictions, explain per their account agreement
- Use phrases like "per your account agreement" or "as per your SOP" when referencing special pricing
- The sop_summary field provides context about the customer's account terms

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
        "sop_summary": quote.sop_summary,  # SOP context for referencing in response
        "line_items": [
            {
                "shipment_index": li.shipment_index,
                "description": li.description,
                "has_rate": li.rate_match is not None,
                "base_price": li.base_price,
                "discount_amount": li.discount_amount,
                "discount_reason": li.discount_reason,  # Why this discount was applied (per SOP)
                "margin_amount": li.margin_amount,
                "surcharge_total": li.surcharge_total,
                "surcharges": [  # Detailed surcharge breakdown with reasons
                    {"name": s.name, "amount": s.amount, "reason": s.reason}
                    for s in li.surcharges
                ] if li.surcharges else [],
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


def format_response_streaming(
    quote: Quote,
    original_email: Email,
    client,  # Regular OpenAI client
    model: str = "gpt-4o-mini",
):
    """
    Streaming version of format_response for real-time output.

    Yields chunks of text as they arrive from the API, providing
    a better user experience with perceived lower latency.

    Args:
        quote: The calculated quote with all pricing
        original_email: The original customer email for context/tone matching
        client: OpenAI client (sync)
        model: Model to use (default: gpt-4o-mini)

    Yields:
        str: Chunks of the response body as they arrive

    Returns:
        After iteration completes, you can call .get_result() on the
        returned generator to get the final QuoteResponse.
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

    # Create streaming response
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        stream=True,  # Enable streaming!
    )

    # Yield chunks as they arrive
    full_body = []
    for chunk in stream:
        # Each chunk has a delta with partial content
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_body.append(content)
            yield content

    # After streaming completes, build and store the final result
    # The caller can access this via the generator's return value
    return QuoteResponse(
        subject=f"RE: {original_email.subject}",
        body="".join(full_body).strip(),
        quote=quote,
        generated_at=datetime.now().isoformat(),
        model_used=model,
    )


def format_response_streaming_with_result(
    quote: Quote,
    original_email: Email,
    client,
    model: str = "gpt-4o-mini",
) -> tuple[callable, callable]:
    """
    Convenience wrapper that returns both a streaming iterator and a way to get the final result.

    Usage:
        stream, get_result = format_response_streaming_with_result(quote, email, client)
        for chunk in stream():
            print(chunk, end="", flush=True)
        response = get_result()

    Returns:
        Tuple of (stream_function, get_result_function)
    """
    result_holder = {"response": None}

    def stream():
        gen = format_response_streaming(quote, original_email, client, model)
        try:
            while True:
                yield next(gen)
        except StopIteration as e:
            # Generator returned the QuoteResponse
            result_holder["response"] = e.value

    def get_result() -> QuoteResponse:
        return result_holder["response"]

    return stream, get_result
