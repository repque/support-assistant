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
    
    Note over User,HealthServer: "My MarkitWire feed isn't sending outbound messages"
    
    User->>Assistant: Submit request
    
    rect rgb(240, 248, 255)
        Note over Assistant,LLM: CLASSIFICATION (3-Step Process)
        Assistant->>ClassServer: 1. Get prompt for ATRS team
        ClassServer-->>Assistant: Team-specific prompt (~2800 chars)
        Assistant->>LLM: 2. Process prompt
        LLM-->>Assistant: {"category": "query", "confidence": 0.85}
        Assistant->>ClassServer: 3. Parse & validate result
        ClassServer-->>Assistant: Final classification
    end
    
    rect rgb(248, 255, 248)
        Note over Assistant,HealthServer: ANALYSIS & DECISION
        Assistant->>KnowServer: Search knowledge base
        KnowServer-->>Assistant: Relevant solutions (85% relevance)
        Assistant->>HealthServer: Check system health
        HealthServer-->>Assistant: Service status
        Assistant->>Assistant: Calculate confidence: 100%
    end
    
    rect rgb(255, 248, 240)
        Note over Assistant,LLM: RECOMMENDATIONS
        Assistant->>KnowServer: Get analysis prompt
        KnowServer-->>Assistant: Contextual prompt
        Assistant->>LLM: Generate recommendations
        LLM-->>Assistant: Detailed troubleshooting steps
    end
    
    Assistant-->>User: Display results<br/>5 tools, 3.5k tokens, 4.2s
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

### 2. **Confidence-Based Decision Tree**
```mermaid
flowchart TD
    A[User Request] --> B[Classify]
    B --> C{Confidence ≥ 60%?}
    C -->|No| D[Stay Silent]
    C -->|Yes| E[Provide Recommendations]
    
    style D fill:#ffebee
    style E fill:#e8f5e8
```

### 3. **Multi-Team Support**
```mermaid
graph TB
    subgraph ATRS[ATRS Team - Trade Management Services]
        A1[query] 
        A2[outage]
        A3[data_issue]
        A4[bless_request]
        A5[review_request]
    end
    
    subgraph CORE[Core Team - Infrastructure]
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
| **Response Time** | 4.2s | End-to-end processing |
| **Token Usage** | 3,500 | ~$0.01-0.02 cost |
| **Tool Calls** | 5 | Classification, Knowledge, Health |
| **LLM Calls** | 2 | Classification + Recommendations |
| **Confidence** | 100% | High enough to respond |

## **Example Output**

```
Issue Classification
   Category: query/feed_issue 
   Priority: high (85% confidence)

Resolution Steps  
   1. Check feed status: SELECT * FROM trade_feeds...
   2. Verify MarkitWire connectivity: curl -H "Auth..."
   3. Review validation results: SELECT * FROM feed_validations...
   4. Check downstream systems status
   
Analysis Details
   Confidence: 100% (Classification: 85%, Knowledge: 85%, Health: Available)
   Sources: Knowledge Base, Health Monitor
   
Performance: 5 tools, 3,500 tokens, 4.2 seconds
```

## **Key Success Factors**

- ✅ **Smart Silence**: Won't respond to vague requests or review requests  
- ✅ **Real LLM**: Uses actual OpenAI API, no mocking
- ✅ **Multi-Team**: Handles both ATRS and Core teams
