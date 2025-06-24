"""Data models for the Support Agent system.

This module defines the core data structures used throughout the support agent
application. All models use Pydantic for type safety, validation, and serialization.
These models ensure consistent data handling across MCP server communications and
internal processing.
"""

from typing import List

from pydantic import BaseModel


class SupportRequest(BaseModel):
    """Support request from an engineer requiring assistance.

    Represents an incoming support request with all necessary information
    for analysis and routing. This model captures the essential details
    needed to classify, prioritize, and resolve technical issues.

    Attributes:
        engineer_sid: Unique identifier for the requesting engineer.
        request_id: Unique identifier for this specific request.
        issue_description: Detailed description of the problem or question.
        lob: Line of business (default: "platform") for organizational routing.
    """

    engineer_sid: str
    request_id: str
    issue_description: str
    lob: str = "platform"  # line of business


class Classification(BaseModel):
    """Request classification result from the classification server.

    Contains the categorization and analysis results for a support request.
    This model is used to determine how a request should be handled and
    the appropriate workflow for resolution.

    Attributes:
        category: Primary category of the issue (e.g., 'technical_issue', 'query').
        confidence: Confidence score (0.0-1.0) in the classification accuracy.
        suggested_workflow: Recommended workflow for handling this type of request.
        reasoning: Explanation of why this classification was chosen.
    """

    category: str
    confidence: float
    suggested_workflow: str
    reasoning: str


class AnalysisResult(BaseModel):
    """Complete analysis result combining all server outputs.

    Aggregates results from multiple MCP servers to provide a comprehensive
    analysis of a support request. This model represents the final output
    of the support agent's analysis process, including classification,
    knowledge base findings, system health, and recommendations.

    Attributes:
        request_id: Identifier linking back to the original support request.
        classification: Classification results including category and priority.
        knowledge_results: List of relevant knowledge base entries found.
        health_status: Current health status of affected systems.
        confidence_score: Overall confidence (0.0-1.0) in the analysis quality.
        recommendations: Generated recommendations for resolving the issue.
        sources_consulted: List of information sources used in the analysis.
    """

    request_id: str
    classification: Classification
    knowledge_results: List[dict]
    health_status: dict
    confidence_score: float
    recommendations: str
    sources_consulted: List[str]
