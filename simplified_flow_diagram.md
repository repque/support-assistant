# Simplified End-to-End Flow

## **High-Level Overview**

```mermaid
sequenceDiagram
    participant User as User
    participant Assistant as Assistant
    participant ClassServer as Classification
    participant LLM as OpenAI
    participant KnowServer as Knowledge
    participant HealthServer as Health
    
    Note over User,HealthServer: "My trade has book2 resolved to 'MarkitWire' but did not feed outbound"
    
    User->>Assistant: Submit request
    
    rect rgb(240, 248, 255)
        Note over Assistant,LLM: CLASSIFICATION (3-Step Process)
        Assistant->>ClassServer: 1. Get prompt for ATRS team
        ClassServer-->>Assistant: Team-specific prompt (~2800 chars)
        Assistant->>LLM: 2. Process prompt
        LLM-->>Assistant: {"category": "query"}
        Assistant->>ClassServer: 3. Parse & validate result
        ClassServer-->>Assistant: Final classification
    end
    
    rect rgb(248, 255, 248)
        Note over Assistant,HealthServer: SECTION-LEVEL SEARCH & DFS RETRIEVAL
        Assistant->>KnowServer: Section-level vector search
        KnowServer-->>Assistant: Top relevant sections from knowledge base
        Assistant->>LLM: Identify knowledge gaps in sections
        LLM-->>Assistant: Gap: "check block events" needs implementation
        Assistant->>KnowServer: DFS search for "check block events"
        KnowServer-->>Assistant: Section with ds.evInfo() code found
        Assistant->>HealthServer: Check system health
        HealthServer-->>Assistant: Service status
        Assistant->>LLM: Should assistant handle this request?
        LLM-->>Assistant: Yes - comprehensive knowledge available
    end
    
    rect rgb(255, 248, 240)
        Note over Assistant,LLM: CONTEXT-AWARE RECOMMENDATIONS
        Assistant->>KnowServer: Get analysis prompt with context awareness
        KnowServer-->>Assistant: Contextual prompt + gap detection
        Assistant->>LLM: Generate recommendations (skip book2 verification)
        LLM-->>Assistant: Context-aware troubleshooting with ds.evInfo()
    end
    
    Assistant-->>User: Display results<br/>6 tools, 5.9k tokens, 6.8s
```

## **Core Architecture Decisions**

### 1. **Client-Side LLM Pattern** 
```mermaid
graph LR
    A[Assistant] -->|1: Get Prompt| B[Classification Server]
    A -->|2: Call LLM| C[OpenAI API]
    A -->|3: Parse Result| B
    
    style A fill:#e3f2fd
    style C fill:#fff3e0
```

### 2. **Vector-Based Knowledge & LLM Decision Tree**
```mermaid
flowchart TD
    A[User Request] --> B[Classify with LLM]
    B --> C[Vector Search Knowledge Base]
    C --> D[LLM: Should Handle Request?]
    D -->|No relevant knowledge| E[Stay Silent]
    D -->|Human review needed| F[Defer to Humans]
    D -->|Knowledge available| G[Context-Aware Recommendations]
    
    style E fill:#ffebee
    style F fill:#fff3e0
    style G fill:#e8f5e8
```

### 3. **Multi-Team Support**
```mermaid
graph TB
    subgraph ATRS[ATRS Team]
        A1[query] 
        A2[outage]
        A3[data_issue]
        A4[bless_request]
        A5[review_request]
    end
    
    subgraph CORE[Core Team]
        C1[query]
        C2[jobs] 
        C3[sdlc]
        C4[database]
        C5[cloud]
        C6[ai]
    end
    
    style ATRS fill:#e3f2fd
    style CORE fill:#f3e5f5
```

## **Performance Summary**

| Metric | Value | Detail |
|--------|-------|--------|
| **Response Time** | 6.8s | End-to-end processing with vector search |
| **Token Usage** | 5,900 | ~$0.02-0.03 cost |
| **Tool Calls** | 6 | Classification, Vector Search, Health, Gap Detection |
| **LLM Calls** | 3 | Classification + Decision + Recommendations |
| **Decision** | Context-Aware | Vector search + LLM decision |

## **Example Output**

```
Issue Classification
   Category: query
   Confidence: 85%

Immediate Actions
   1. **Check for Block Events:**
      ```python
      deal = ro(dealName)
      fs = deal.FeedState("MarkitWire")
      dt = fs._DownstreamTrades()[0]
      dt.validate() # investigate validation failures
      ```

   2. **Check Downstream Events:**
      ```python
      deal = ro(dealName)
      ds = deal.DownstreamState("MarkitWire")
      ds.evInfo() # prints downstream events, block events
      ```
   
Analysis Details
   Decision: Context-aware recommendations (skipped book2 verification)
   Sources: Vector search + Gap detection + Context awareness
   
Performance: 6 tools, 5,900 tokens, 6.8 seconds
```

## **Key Success Factors**

- **Vector Embeddings**: Semantic search with sentence-transformers
- **Context Awareness**: Skips redundant steps based on user's stated facts
- **Generic Feed Support**: Works with any feed type through intelligent parameter substitution
- **LLM-Based Decisions**: All logic decisions made by LLM
- **Gap Detection**: Recursively searches for missing implementation details
- **Smart Silence**: Won't respond to vague requests or categories marked for direct escalation (ex: bless requests)
- **Multi-Team**: Can be configured for use by multiple teams, with team-specific categories
