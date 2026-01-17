"""
Qontext API client for retrieving customer context and SOPs.

This module provides a simple interface to query the Qontext knowledge graph
for customer-specific rules, SOPs, and contextual information.
"""
import os
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class QontextResponse:
    """Response from Qontext retrieval API."""
    success: bool
    context: list[str] | None = None
    error: str | None = None
    raw_response: dict | None = None


class QontextClient:
    """
    Client for the Qontext retrieval API.

    Usage:
        client = QontextClient()
        response = client.retrieve("What are the SOPs for Global Imports Ltd?")
        for item in response.context:
            print(item)
    """

    def __init__(
        self,
        api_key: str | None = None,
        vault_id: str | None = None,
        workspace_id: str | None = None,
        base_url: str = "https://api.qontext.ai"
    ):
        self.api_key = api_key or os.getenv("QONTEXT_API_KEY")
        self.vault_id = vault_id or os.getenv("QONTEXT_VAULT_ID")
        self.workspace_id = workspace_id or os.getenv("QONTEXT_WORKSPACE_ID")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("QONTEXT_API_KEY not found in environment variables")
        if not self.vault_id:
            raise ValueError("QONTEXT_VAULT_ID not found in environment variables")
        if not self.workspace_id:
            raise ValueError("QONTEXT_WORKSPACE_ID not found in environment variables")

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        depth: int = 2,
        rerank: bool = True
    ) -> QontextResponse:
        """
        Retrieve context from Qontext based on a natural language query.

        Args:
            query: Natural language query (e.g., "What are the SOPs for customer X?")
            limit: Number of nodes to retrieve (default: 10)
            depth: Depth of graph traversal (default: 2)
            rerank: Whether to rerank results for relevance (default: True)

        Returns:
            QontextResponse with list of context strings or error information
        """
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "workspaceId": self.workspace_id,
            "knowledgeGraphId": self.vault_id,
            "prompt": query,
            "limit": limit,
            "depth": depth,
            "rerank": rerank,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/retrieval",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 201:
                data = response.json()
                # Response is a list of context strings
                return QontextResponse(
                    success=True,
                    context=data if isinstance(data, list) else [str(data)],
                    raw_response={"data": data}
                )
            else:
                return QontextResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )

        except requests.RequestException as e:
            return QontextResponse(
                success=False,
                error=str(e)
            )

    def get_customer_sop(self, customer_name: str) -> QontextResponse:
        """
        Retrieve SOP rules for a specific customer.

        Args:
            customer_name: Name of the customer (e.g., "Global Imports Ltd")

        Returns:
            QontextResponse with customer-specific rules
        """
        query = f"""What are all the rules, discounts, restrictions, and requirements
        for customer {customer_name}? Include:
        - Discount percentages and how to apply them
        - Mode restrictions (sea only, air only)
        - Location equivalences
        - Margin rules
        - Output requirements (what to show in the quote)"""

        return self.retrieve(query, limit=15, depth=2)

    def get_destination_rules(self, destination: str) -> QontextResponse:
        """
        Retrieve rules that apply to a specific destination.

        Args:
            destination: Destination location (e.g., "Australia")

        Returns:
            QontextResponse with destination-specific rules (e.g., surcharges)
        """
        query = f"What surcharges or special rules apply to shipments going to {destination}?"
        return self.retrieve(query, limit=5, depth=1)


def test_qontext():
    """Test the Qontext client with sample queries."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("Testing Qontext API connection...")
    print("-" * 50)

    try:
        client = QontextClient()
        print(f"API Key: {client.api_key[:20]}...")
        print(f"Vault ID: {client.vault_id}")
        print(f"Workspace ID: {client.workspace_id}")
        print("-" * 50)

        # Test 1: General SOP query
        print("\n[Test 1] General SOP query:")
        response = client.retrieve("What customer SOPs exist?")
        if response.success:
            print(f"SUCCESS! Found {len(response.context)} results:")
            for i, item in enumerate(response.context[:3]):
                print(f"  {i+1}. {item[:100]}...")
        else:
            print(f"FAILED: {response.error}")

        # Test 2: Specific customer
        print("\n[Test 2] Global Imports Ltd SOP:")
        response = client.get_customer_sop("Global Imports Ltd")
        if response.success:
            print(f"SUCCESS! Found {len(response.context)} results:")
            for i, item in enumerate(response.context[:3]):
                print(f"  {i+1}. {item[:100]}...")
        else:
            print(f"FAILED: {response.error}")

        # Test 3: Destination rules
        print("\n[Test 3] Australia destination rules:")
        response = client.get_destination_rules("Australia")
        if response.success:
            print(f"SUCCESS! Found {len(response.context)} results:")
            for i, item in enumerate(response.context[:3]):
                print(f"  {i+1}. {item[:100]}...")
        else:
            print(f"FAILED: {response.error}")

    except ValueError as e:
        print(f"Configuration error: {e}")


if __name__ == "__main__":
    test_qontext()
