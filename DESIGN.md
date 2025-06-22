# AI Production Support Assistant - System Design

## Executive Summary

This document defines the technical architecture for an AI-powered production support assistant system designed to assist L3 support engineers of a central team. The system employs a Model Context Protocol (MCP) based architecture to provide intelligent analysis and resolution suggestions to support engineers.

**Phase 1 Focus**: The assistant provides analysis and resolution recommendations to human support engineers rather than direct responses to end users.

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│            AI Support Assistant Core                            │
│  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐   │
│  │ Request         │ │  Analysis        │ │ Recommendation  │   │
│  │ Processor       │ │  Engine          │ │ Generator       │   │
│  └─────────────────┘ └──────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐   │
│  │ Context-Aware   │ │  DFS Knowledge   │ │ Section-Level   │   │
│  │ Gap Detection   │ │  Retrieval       │ │ Vector Search   │   │
│  └─────────────────┘ └──────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   │    MCP Integration      │
                   │       Gateway           │
                   └─────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│ Knowledge      │    │ Health Monitor  │    │ Classification  │
│ Retrieval      │    │ & Logs          │    │ & Triage        │
│ MCP Server     │    │ MCP Server      │    │ MCP Server      │
└────────────────┘    └─────────────────┘    └─────────────────┘

```

### Core Design Principles

1. **Modularity**: MCP-based architecture enables independent development and deployment of services
2. **Extensibility**: Plugin architecture allows easy addition of new capabilities
3. **Resilience**: Circuit breakers and fallback mechanisms ensure system stability
4. **Type Safety**: Strongly typed interfaces using Pydantic models
5. **Observability**: Comprehensive logging, monitoring, and metrics collection

## MCP Server Specifications

### MCP Server 1: Knowledge Retrieval Service

**Purpose**: Interface with knowledge bases, runbooks, and documentation

**Core Capabilities**:
- Semantic search across multiple knowledge sources
- Context-aware information retrieval
- Multi-modal content support (text, diagrams, code)
- Relevance scoring and ranking

**API Interface**:
```python
class KnowledgeQuery(BaseModel):
    query: str
    context: Optional[str] = None
    lob: Optional[str] = None
    max_results: int = 10
    filters: Optional[Dict[str, Any]] = None

class KnowledgeResult(BaseModel):
    content: str
    source: str
    relevance_score: float
    metadata: Dict[str, Any]
    last_updated: datetime

class KnowledgeRetrievalService:
    async def search(self, query: KnowledgeQuery) -> List[KnowledgeResult]
    async def get_by_id(self, doc_id: str) -> Optional[KnowledgeResult]
    async def get_related(self, doc_id: str, limit: int = 5) -> List[KnowledgeResult]
```

### MCP Server 2: Health Monitoring & Logs Service

**Purpose**: Component health checking and log analysis

**Core Capabilities**:
- Health endpoint monitoring across services
- Log aggregation and search
- Pattern detection in logs
- Performance metrics collection

**API Interface**:
```python
class HealthCheckRequest(BaseModel):
    service_name: str
    endpoint_url: str
    timeout_seconds: int = 30

class LogQuery(BaseModel):
    service_name: str
    time_range: Tuple[datetime, datetime]
    log_level: Optional[str] = None
    search_terms: Optional[List[str]] = None
    max_lines: int = 1000

class HealthStatus(BaseModel):
    service_name: str
    status: Literal["healthy", "degraded", "unhealthy"]
    response_time_ms: float
    last_check: datetime
    details: Dict[str, Any]

class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    message: str
    service: str
    metadata: Dict[str, Any]

class HealthMonitoringService:
    async def check_health(self, request: HealthCheckRequest) -> HealthStatus
    async def get_service_status(self, service_name: str) -> HealthStatus
    async def query_logs(self, query: LogQuery) -> List[LogEntry]
    async def analyze_logs(self, entries: List[LogEntry]) -> Dict[str, Any]
```

### MCP Server 3: Classification & Triage Service

**Purpose**: Request categorization and intelligent triage

**Core Capabilities**:
- Automatic request classification
- Pattern recognition for common issues
- Custom category management
- Priority assignment

**API Interface**:
```python
class ClassificationRequest(BaseModel):
    user_request: str
    conversation_history: List[str] = []
    metadata: Dict[str, Any] = {}

class Classification(BaseModel):
    category: str
    subcategory: Optional[str]
    confidence: float
    priority: Literal["low", "medium", "high", "critical"]
    suggested_workflow: str
    reasoning: str

class ClassificationService:
    async def classify_request(self, request: ClassificationRequest) -> Classification
    async def get_categories(self) -> List[str]
    async def retrain_model(self, feedback_data: List[Dict[str, Any]]) -> bool
```

**Note**: For Phase 1, we have simplified the MCP architecture to focus on 3 core services that support context gathering and recommendation generation for human engineers.

## Assistant Core Architecture

### Core Components

#### 1. Request Processor
```python
class RequestProcessor:
    """
    Processes incoming support requests from engineers.
    Handles request validation, sanitization, and initial routing.
    """
    
    async def process_request(self, request: SupportRequest) -> ProcessedRequest
    async def validate_request(self, request: SupportRequest) -> ValidationResult
    async def extract_entities(self, request_text: str) -> List[Entity]
    async def enrich_context(self, request: SupportRequest) -> EnrichedContext
```

#### 2. Context Manager
```python
class ContextManager:
    """
    Manages conversation context and cross-session patterns.
    Provides context-aware information retrieval.
    """
    
    async def build_context(self, conversation_id: str) -> ConversationContext
    async def get_relevant_history(self, current_context: str) -> List[ConversationContext]
```

#### 3. Analysis Engine
```python
class AnalysisEngine:
    """
    Core analysis engine that orchestrates MCP services and synthesizes information.
    Implements the main analysis logic and recommendation generation.
    """
    
    async def analyze_issue(self, request: ProcessedRequest) -> AnalysisResult
    async def synthesize_information(self, gathered_data: Dict[str, Any]) -> AnalysisSummary
    async def plan_analysis_steps(self, classified_request: Classification) -> List[AnalysisStep]
    async def execute_analysis_workflow(self, workflow: AnalysisWorkflow) -> AnalysisResult
```

#### 4. Workflow Orchestrator
```python
class WorkflowOrchestrator:
    """
    Manages multi-step workflows and coordinates between MCP services.
    Handles workflow state and execution monitoring.
    """
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any]) -> WorkflowResult
    async def pause_workflow(self, workflow_id: str) -> None
    async def resume_workflow(self, workflow_id: str) -> None
    async def get_workflow_status(self, workflow_id: str) -> WorkflowStatus
```

#### 5. Recommendation Generator
```python
class RecommendationGenerator:
    """
    Generates structured recommendations and analysis summaries for support engineers.
    Handles formatting of technical findings and suggested resolution steps.
    """
    
    async def generate_recommendations(self, analysis: AnalysisResult) -> RecommendationReport
    async def format_analysis_summary(self, analysis_data: Dict[str, Any]) -> str
    async def create_diagnostic_steps(self, issue_category: str, symptoms: List[str]) -> List[DiagnosticStep]
```

#### 6. Vector Search Engine
```python
class VectorSearchEngine:
    """
    Provides section-level semantic search using vector embeddings for knowledge retrieval.
    Handles section parsing, indexing, and context-aware similarity-based search operations.
    Supports DFS recursive searches for implementation detail discovery.
    """
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]
    async def index_documents(self) -> bool  # Parses markdown into sections and creates embeddings
    async def identify_knowledge_gaps(self, content: str, user_request: str) -> List[str]
    async def recursive_search(self, knowledge_data: List, depth: int = 1) -> List[SearchResult]
```

### Data Models

#### Core Models
```python
class SupportRequest(BaseModel):
    """Primary request model from support engineers"""
    engineer_sid: str
    request_id: str
    issue_description: str
    affected_system: Optional[str] = None
    urgency: Literal["low", "medium", "high", "critical"]
    lob: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    attachments: List[Attachment] = []

class ProcessedRequest(BaseModel):
    """Enriched request after initial processing"""
    original_request: SupportRequest
    entities: List[Entity]
    classification: Classification
    context: Dict[str, Any]
    similar_issues: List[str] = []

class AnalysisResult(BaseModel):
    """Comprehensive analysis result for support engineers"""
    request_id: str
    analysis_summary: str
    confidence_score: float
    recommended_actions: List[RecommendedAction]
    diagnostic_steps: List[DiagnosticStep]
    similar_incidents: List[HistoricalIncident]
    escalation_recommended: bool
    sources_consulted: List[str]
    estimated_resolution_time: Optional[int] = None

class RecommendedAction(BaseModel):
    """Individual recommended action for issue resolution"""
    action_id: str
    description: str
    priority: int
    risk_level: Literal["low", "medium", "high"]
    expected_outcome: str
    estimated_time_minutes: int
    prerequisites: List[str] = []

class AnalysisWorkflow(BaseModel):
    """Analysis workflow definition"""
    workflow_id: str
    name: str
    analysis_steps: List[AnalysisStep]
    trigger_conditions: Dict[str, Any]
    timeout_minutes: int = 15
```

## Data Flow Design

### Request Processing Pipeline

```
1. Request Ingestion
   ├── User request validation
   └── Request context extraction

2. Initial Triage & Classification
   ├── Issue classification (Classification MCP Service)
   └── Analysis workflow selection

3. Context Building
   ├── Historical incident lookup
   ├── System status retrieval
   └── Knowledge base search preparation

4. Information Gathering
   ├── Section-level knowledge search (Knowledge Retrieval MCP)
   ├── Context-aware gap identification (LLM-based)
   ├── DFS recursive knowledge retrieval for missing details
   ├── Health status checks (Health Monitor MCP)
   └── Log analysis (Health Monitor MCP)

5. Analysis & Synthesis
   ├── Multi-source data analysis
   ├── Pattern recognition and matching
   └── Resolution recommendation generation

6. Confidence Assessment
   ├── Analysis confidence scoring
   ├── Recommendation reliability evaluation
   └── Escalation recommendation

7. Report Generation
   ├── Analysis summary formatting
   ├── Diagnostic steps compilation
   └── Actionable recommendations

8. Delivery to Support Engineer
   ├── Structured report delivery
   ├── Feedback collection setup
   └── Analysis tracking and metrics
```

### Integration Patterns

#### 1. Async Processing Pattern
```python
class AsyncProcessor:
    """
    Handles long-running operations without blocking the main response flow.
    Used for log analysis, comprehensive health checks, and batch operations.
    """
    
    async def submit_task(self, task: Task) -> str  # Returns task_id
    async def get_task_status(self, task_id: str) -> TaskStatus
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]
```

#### 2. Retry with Backoff Pattern
```python
class RetryHandler:
    """
    Implements intelligent retry logic with exponential backoff.
    Handles transient failures in MCP service communications.
    """
    
    async def retry_with_backoff(
        self, 
        operation: Callable, 
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> Any
```

## Technical Design Decisions

### 1. Conversation Context Management
**Decision**: Hybrid approach combining in-memory context with persistent storage.

**Rationale**: 
- In-memory: Fast access for active conversations
- Persistent storage: Cross-session continuity and pattern recognition
- Context summarization: Manage memory usage while preserving essential information

**Implementation**: 
- Langgraph native memory integration
- Context summarization after conversation completion

### 2. Response Quality and Speed Balance
**Decision**: Implement tiered response strategy with confidence thresholds.

**Rationale**:
- High confidence: Immediate response
- Medium confidence: Additional validation step
- Low confidence: Escalation with context

**Implementation**:
- Parallel information gathering from multiple MCP services
- Streaming responses for long-running queries
- Confidence-based response strategies

### 3. External Knowledge Base Integration
**Decision**: Textual knowledge base with semantic search and hierarchical organization.

**Structure**:
```
Knowledge Base
├── Runbooks (step-by-step procedures)
├── Troubleshooting Guides (diagnostic workflows)
├── System Documentation (architecture, APIs)
├── Historical Incidents (patterns, solutions)
└── Best Practices (recommendations, standards)
```

**Implementation**: MCP Server.

### 4. Confidence Scoring Algorithm
**Decision**: Multi-factor confidence scoring with dynamic thresholds.

**Factors**:
- Source reliability (knowledge base vs. historical incidents)
- Information completeness (all required data available)
- Pattern match strength (similarity to resolved cases)

**Implementation**: Weighted scoring model with category-specific thresholds.

### 5. Learning and Improvement
**Decision**: Continuous learning through feedback loops and pattern recognition.

**Mechanisms**:
- Resolution outcome tracking
- Pattern recognition from successful interventions

**Implementation**: Invoke knowledge updates based on interaction data.

## Security and Deployment Considerations

### Security Requirements
1. **Authentication**: MCP supporting OAuth 2.0 with fid-based access control
2. **Audit Logging**: Comprehensive audit trail for all actions


### Observability Stack
1. **Metrics**: Prometheus with custom business metrics
2. **Tracing**: Distributed tracing with Jaeger
3. **Dashboards**: Grafana dashboards for operational visibility

## Implementation Roadmap

### Phase 1: Core Architecture & Demo 

#### Foundation
- [ ] Project structure and development environment setup
- [ ] Core data models and API contracts
- [ ] MCP integration framework
- [ ] Basic agent engine structure

#### MCP Services (Mock Implementations)
- [ ] Knowledge Retrieval Service with sample runbooks and documentation
- [ ] Classification & Triage Service with basic issue categorization
- [ ] Health monitoring capabilities for system status checks
- [ ] Analysis & Recommendation Service for suggestion generation

#### Analysis Assistant Core
- [ ] Request processor implementation
- [ ] Context management system
- [ ] Analysis workflow orchestration
- [ ] Recommendation generation engine
- [ ] Confidence assessment system

#### Integration & Demo
- [ ] Testbed CLI interface for receiving user requests and showing generated response for support engineer (similar to Claude Code CLI)
- [ ] End-to-end analysis workflow testing
- [ ] Sample scenarios for common support issues

## Technology Stack

### Backend Services
- **Language**: Python 3.12+
- **Framework**: FastAPI with FastMCP for MCP servers
- **Data Validation**: Pydantic v2
- **Vector Search**: sentence-transformers with all-MiniLM-L6-v2 model
- **LLM Integration**: OpenAI/Anthropic APIs with configurable parameters
- **Configuration**: Environment variable-based configuration system

### Development Tools
- **Testing**: pytest with async support and comprehensive functional tests
- **Code Quality**: ruff for linting and formatting
- **Configuration**: Pydantic-based configuration with environment variable overrides
- **Knowledge Base**: Pure markdown with automatic section parsing

## Success Metrics & KPIs

### Primary Metrics
1. **Analysis Accuracy**: > 85% of recommendations deemed helpful by support engineers
2. **Time to Analysis**: < 5 minutes for standard issue analysis
3. **Confidence Reliability**: Analysis confidence scores correlate with engineer feedback
4. **Knowledge Coverage**: > 80% of queries find relevant knowledge base matches

### Secondary Metrics
1. **Engineer Adoption**: Active usage across support teams
2. **Knowledge Base Utilization**: Track most/least used content
3. **Analysis Completeness**: Percentage of recommendations with actionable steps

### Learning Metrics
1. **Model Improvement**: Month-over-month confidence score increases
2. **Pattern Recognition**: Successful identification of recurring issues
3. **Feedback Quality**: Percentage of actionable feedback received
4. **Knowledge Growth**: Rate of knowledge base expansion

---

## Conclusion

This design provides a focused, scalable foundation for the AI Production Support Analysis Assistant system. The MCP-based architecture ensures modularity and extensibility, while the streamlined data flow design addresses the specific requirements of providing intelligent analysis and recommendations to support engineers.

**Phase 1 Benefits:**
- Reduced complexity by focusing on analysis and assitance rather than direct user interaction
- Clear value proposition for support engineers through actionable recommendations
- Simplified testing and validation through engineer feedback
- Lower risk deployment as a support tool rather than customer-facing system

The implementation roadmap balances rapid delivery of core analysis functionality with the need for reliable, production-ready features. The focus on accuracy, confidence assessment, and continuous improvement ensures the system provides genuine value to support engineers.

Key success factors include:
- Strong typing and validation throughout the system
- Comprehensive error handling and fallback mechanisms
- Accurate confidence scoring to build engineer trust
- Clear, actionable recommendations with diagnostic steps
- Robust monitoring and feedback collection
