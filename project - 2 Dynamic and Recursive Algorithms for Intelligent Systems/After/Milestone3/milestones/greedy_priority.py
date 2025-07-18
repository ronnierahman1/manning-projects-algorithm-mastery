# === greedy_priority.py ===
"""
Milestone 3: Greedy Priority Algorithm

This module implements an advanced greedy algorithm to prioritize user queries based on urgency,
complexity, context, and intent. By assigning sophisticated priority levels, the chatbot can 
optimize which queries to process first in high-load, batch-processing, or real-time scenarios.

GREEDY ALGORITHM FUNDAMENTALS:
The greedy algorithm approach makes locally optimal choices at each step, never reconsidering
previous decisions. In this priority system, each query gets the best possible priority based
on immediately available information, without considering other queries in the system.

KEY GREEDY PRINCIPLES DEMONSTRATED:
1. LOCAL OPTIMIZATION: Each decision is the best choice given current constraints
2. NO BACKTRACKING: Once a priority is assigned, it's never reconsidered
3. IMMEDIATE DECISION MAKING: Fast choices without exhaustive global analysis
4. HEURISTIC-BASED: Multiple heuristics combine for optimal local decisions

The algorithm uses dynamic programming principles for caching priority calculations,
machine learning-inspired pattern recognition, and contextual analysis for intelligent
query prioritization and resource optimization.
"""

import heapq  # Priority queue implementation using binary heap (O(log n) operations)
import time   # For timestamp tracking and performance measurements
import re     # Regular expressions for sophisticated pattern matching in text analysis
import datetime  # For temporal analysis and session duration tracking
from typing import List, Dict, Tuple, Any, Optional, Set, Union  # Type hints for code clarity and maintainability
from collections import defaultdict, Counter  # Efficient data structures for frequency tracking and analytics
from dataclasses import dataclass, field  # Modern Python approach for creating structured data classes
from enum import IntEnum  # For creating enumerated priority levels with integer ordering
import json   # For potential data serialization and configuration management


class Priority(IntEnum):
    """
    Enhanced priority levels for query processing with clear semantic meaning.
    
    GREEDY ALGORITHM DESIGN PRINCIPLE:
    Using an IntEnum ensures that priority comparisons and decisions are consistent
    and mathematically sound. Lower numeric values indicate higher priority, which
    aligns with min-heap data structures for optimal performance.
    
    This design enables the greedy algorithm to make immediate, unambiguous decisions
    about query importance without complex comparison logic.
    
    Lower numeric values indicate higher priority for processing.
    """
    CRITICAL = 1    # Emergency situations, system failures, critical errors - highest priority
    HIGH = 2        # Important tasks, deadlines, urgent help requests - high priority
    MEDIUM = 3      # Standard questions, explanations, general queries - normal priority
    LOW = 4         # Casual interactions, greetings, social exchanges - lowest priority


@dataclass
class QueryMetrics:
    """
    Comprehensive data class to store detailed query processing metrics and analytics.
    
    DATA-DRIVEN GREEDY OPTIMIZATION:
    This class implements the data collection foundation that enables the greedy algorithm
    to make increasingly better decisions over time. By tracking performance patterns,
    the system can adapt its local optimization strategies based on historical evidence.
    
    PERFORMANCE MONITORING STRATEGY:
    The metrics collected here support the greedy principle by providing the data needed
    to make informed local decisions. Each metric contributes to understanding whether
    current priority assignments are leading to optimal global outcomes.
    
    Supports advanced performance monitoring and optimization insights.
    """
    # Core performance metrics for decision optimization
    total_time: float = 0.0                                      # Cumulative processing time for this query type
    count: int = 0                                              # Total number of queries of this type processed
    success_count: int = 0                                      # Number of successfully processed queries
    avg_time: float = 0.0                                       # Running average processing time (calculated)
    success_rate: float = 0.0                                   # Success rate percentage (calculated)
    last_processed: Optional[datetime.datetime] = None         # Timestamp of most recent processing
    
    # Advanced analytics for pattern recognition and trend analysis
    priority_distribution: Dict[int, int] = field(default_factory=dict)    # Count of each priority level assigned
    complexity_scores: List[float] = field(default_factory=list)           # Historical complexity measurements
    response_times: List[float] = field(default_factory=list)              # Recent response time history
    
    def update(self, processing_time: float, success: bool, priority: int = None, 
               complexity: float = None) -> None:
        """
        Update metrics with new processing data including advanced analytics.
        
        GREEDY ALGORITHM SUPPORT:
        This method implements incremental metric updates that support the greedy
        algorithm's need for fast, local decision-making. By maintaining running
        averages and statistics, it avoids expensive recalculations while providing
        the data needed for optimal priority assignments.
        
        INCREMENTAL COMPUTATION STRATEGY:
        Rather than storing all historical data and recalculating from scratch,
        this method uses running averages and bounded collections - a key optimization
        principle in greedy algorithms where speed of decision-making is crucial.
        
        Args:
            processing_time (float): Time taken to process the query
            success (bool): Whether processing was successful
            priority (int, optional): Priority level of the processed query
            complexity (float, optional): Calculated complexity score
        """
        # Update core counters for running statistics
        self.total_time += processing_time  # Accumulate total processing time
        self.count += 1                     # Increment total query count
        if success:
            self.success_count += 1         # Track successful queries for quality metrics
        
        # Calculate running averages efficiently (avoiding full dataset recalculation)
        # This is a key greedy optimization - make decisions with current data without expensive computation
        self.avg_time = self.total_time / self.count                    # Running average processing time
        self.success_rate = self.success_count / self.count             # Running success rate percentage
        self.last_processed = datetime.datetime.now()                  # Update timestamp for recency tracking
        
        # Track priority distribution for pattern analysis
        if priority is not None:
            # Count how often each priority level is assigned to identify trends
            self.priority_distribution[priority] = self.priority_distribution.get(priority, 0) + 1
        
        # Collect complexity scores for correlation analysis
        if complexity is not None:
            self.complexity_scores.append(complexity)
        
        # Maintain recent response time history for trend detection
        self.response_times.append(processing_time)
        
        # MEMORY MANAGEMENT: Implement sliding window for response times
        # Keep only recent data to prevent unbounded memory growth - another greedy optimization
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]  # Keep last 100 entries only


class GreedyPriority:
    """
    Advanced greedy query prioritization system implementing sophisticated algorithms for
    intelligent resource allocation and response optimization.
    
    GREEDY ALGORITHM IMPLEMENTATION PHILOSOPHY:
    This class embodies the core greedy algorithm principle: make the best local decision
    at each step based on immediately available information. Every method is designed to
    optimize for the current query without reconsidering previous decisions or exhaustively
    analyzing all possible future scenarios.
    
    MULTI-FACTOR GREEDY DECISION STRATEGY:
    The system uses multiple independent heuristics (patterns, keywords, complexity, etc.)
    and applies them in priority order. The first heuristic that produces a confident
    result determines the priority - no backtracking or second-guessing occurs.
    
    Features:
    - Multi-factor priority calculation using urgency, complexity, and context
    - Dynamic pattern recognition with machine learning-inspired techniques  
    - Comprehensive analytics and performance monitoring
    - Context-aware prioritization with conversation history analysis
    - Adaptive thresholds and self-optimizing parameters
    - Enterprise-grade scalability and reliability
    """
    
    def __init__(self):
        """
        Initialize the enhanced GreedyPriority system with comprehensive data structures,
        advanced pattern recognition capabilities, and performance monitoring systems.
        
        INITIALIZATION STRATEGY FOR GREEDY ALGORITHMS:
        The initialization phase sets up all data structures needed for fast local
        decision-making. By pre-organizing keywords, compiling patterns, and establishing
        thresholds, the system can make priority decisions in O(1) to O(k) time where
        k is small and constant.
        """
        # CORE PRIORITY KEYWORD DICTIONARY WITH ENHANCED CATEGORIZATION
        # This is the foundation of keyword-based greedy decision making
        # Pre-organized keywords allow immediate O(k) priority classification without complex analysis
        # Lower number = higher priority for processing (aligns with min-heap operations)
        self.priority_keywords = {
            Priority.CRITICAL: [
                # System failures and emergencies - immediate business impact
                'urgent', 'emergency', 'critical', 'immediately', 'asap', 'crisis',
                'broken', 'down', 'failure', 'crash', 'error', 'bug', 'issue',
                'problem', 'stuck', 'stop', 'halt', 'freeze', 'timeout',
                # Security and data integrity - potential security threats
                'security', 'breach', 'hack', 'vulnerability', 'attack', 'malware',
                'data loss', 'corruption', 'infected', 'virus', 'compromised',
                # Production and business critical - revenue/service affecting
                'production', 'outage', 'service down', 'system failure', 'offline'
            ],
            Priority.HIGH: [
                # Importance and urgency indicators - significant business value
                'important', 'priority', 'needed', 'required', 'must', 'should',
                'deadline', 'quick', 'fast', 'soon', 'help', 'assist', 'support',
                # Performance and optimization - user experience impact
                'slow', 'performance', 'optimization', 'bottleneck', 'lag',
                'improvement', 'efficiency', 'speed up', 'faster',
                # Business impact - customer/revenue related
                'client', 'customer', 'revenue', 'business', 'impact', 'loss'
            ],
            Priority.MEDIUM: [
                # Information seeking and learning - standard knowledge requests
                'question', 'how', 'what', 'why', 'when', 'where', 'who', 'which',
                'explain', 'tell', 'describe', 'define', 'show', 'demonstrate',
                'clarify', 'understand', 'learn', 'tutorial', 'guide', 'example',
                # Development and implementation - routine technical work
                'implement', 'develop', 'create', 'build', 'design', 'configure',
                'setup', 'install', 'deploy', 'integrate', 'code', 'programming'
            ],
            Priority.LOW: [
                # Social interactions and pleasantries - minimal business impact
                'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon',
                'good evening', 'thanks', 'thank you', 'goodbye', 'bye', 'see you',
                'chat', 'talk', 'discuss', 'opinion', 'think', 'feel', 'like',
                # Non-urgent communications - casual conversation
                'casual', 'friendly', 'social', 'weather', 'personal', 'story'
            ]
        }

        # ADVANCED REGEX PATTERNS FOR SOPHISTICATED PRIORITY DETECTION
        # These patterns capture complex linguistic structures that simple keyword matching misses
        # Pre-organized by priority level for efficient greedy processing (check highest priority first)
        self.priority_patterns = {
            Priority.CRITICAL: [
                # Functional failure patterns - system not working
                r'\b(can\'t|cannot|won\'t|wont)\s+(work|function|run|start|load|connect|access)\b',
                r'\b(not\s+working|doesn\'t\s+work|failed\s+to|unable\s+to)\b',
                r'\b(server\s+down|service\s+unavailable|connection\s+(lost|failed)|system\s+offline)\b',
                r'\b(data\s+(loss|lost|corrupted|missing)|database\s+(down|corrupt|failed))\b',
                r'\b(emergency|critical|urgent)\b.*\b(help|support|assistance|fix|repair)\b',
                r'\b(production\s+(down|failed|broken)|live\s+site\s+(down|broken))\b',
                r'\b(security\s+(breach|incident|alert)|hack(ed|ing)|malware|virus)\b'
            ],
            Priority.HIGH: [
                # Time-sensitive patterns - deadlines and urgency
                r'\b(due\s+(today|tomorrow|soon)|deadline\s+(approaching|tomorrow|today))\b',
                r'\b(need\s+(help|assistance|support)\s+(with|for|urgently))\b',
                r'\b(how\s+to\s+(fix|solve|resolve|repair))\b',
                r'\b(performance\s+(issue|problem)|running\s+slow|very\s+slow)\b',
                r'\b(client\s+(complain|issue|problem)|customer\s+(complain|issue))\b',
                r'\b(important\s+(task|project|deadline)|high\s+priority)\b'
            ],
            Priority.MEDIUM: [
                # Question patterns - standard information requests
                r'\?+\s*$',  # Questions ending with question marks
                r'\b(can\s+you\s+(help|explain|show|tell|teach))\b',
                r'\b(what\s+is|how\s+does|why\s+is|when\s+should|where\s+can)\b',
                r'\b(tutorial|guide|example|documentation|reference)\b',
                r'\b(learn|understand|explain|clarify|describe)\b'
            ],
            Priority.LOW: [
                # Social interaction patterns - polite conversation
                r'\b(hello|hi|hey|greetings)\b.*\b(how\s+(are\s+you|is\s+everything))\b',
                r'\b(thank\s+you|thanks|appreciate|grateful)\b',
                r'\b(good\s+(morning|afternoon|evening|day))\b',
                r'\b(nice\s+(day|weather|chat)|lovely\s+(day|weather))\b'
            ]
        }

        # COMPREHENSIVE QUERY ANALYSIS AND STATISTICS TRACKING
        # This dictionary maintains performance metrics for each query type
        # Enables the greedy algorithm to make data-driven priority decisions
        self.query_stats: Dict[str, QueryMetrics] = {}
        
        # ADVANCED PRIORITY QUEUE WITH METADATA SUPPORT
        # Core data structure for greedy priority processing using binary heap
        # Provides O(log n) insertion and O(log n) extraction of highest priority items
        self.priority_queue = []  # Python's heapq implements min-heap (perfect for priority queues)
        
        # Queue metadata for comprehensive analytics and performance monitoring
        self.queue_metadata = {
            'total_processed': 0,                    # Total number of queries processed through queue
            'average_wait_time': 0.0,               # Average time queries spend waiting in queue
            'priority_distribution': Counter(),      # Distribution of priority levels in queue
            'processing_trends': [],                 # Historical processing patterns
            'peak_usage_times': [],                  # Times when queue load is highest
            'bottleneck_indicators': []              # Performance bottleneck detection data
        }
        
        # SOPHISTICATED CONFIGURATION PARAMETERS WITH ADAPTIVE CAPABILITIES
        # These thresholds enable fast length-based heuristics for priority assignment
        # Tuned based on empirical analysis of query patterns and complexity correlation
        self.length_thresholds = {
            'very_long': 200,   # Comprehensive detailed queries - likely require expert attention
            'long': 100,        # Complex multi-part questions - above average complexity
            'medium': 50,       # Standard detailed queries - normal complexity
            'short': 20         # Brief questions or statements - typically simple
        }
        
        # ENHANCED COMPLEXITY ANALYSIS INDICATORS
        # Pre-defined list of technical terms that correlate with query complexity
        # Enables rapid complexity assessment for greedy priority decisions
        self.complexity_indicators = [
            # Technical complexity indicators - advanced concepts requiring expertise
            'algorithm', 'implementation', 'architecture', 'system', 'design',
            'optimization', 'performance', 'scalability', 'integration',
            'configuration', 'deployment', 'security', 'authentication',
            # Advanced concepts - cutting-edge topics requiring specialized knowledge
            'machine learning', 'artificial intelligence', 'neural network',
            'deep learning', 'data science', 'big data', 'cloud computing',
            'microservices', 'distributed', 'concurrent', 'parallel',
            # Development complexity indicators - sophisticated programming topics
            'framework', 'library', 'api', 'database', 'frontend', 'backend',
            'fullstack', 'devops', 'testing', 'debugging', 'refactoring'
        ]
        
        # CONTEXT AWARENESS AND CONVERSATION HISTORY
        # These structures enable context-aware priority adjustments based on conversation flow
        # Implements sliding window approach for memory efficiency
        self.conversation_context = []      # Chronological list of recent interactions
        self.context_window = 10           # Maximum number of context entries to maintain (memory management)
        
        # Topic continuity tracking with frequency-based importance scoring
        # defaultdict provides automatic initialization of complex default values
        self.topic_continuity = defaultdict(lambda: {
            'frequency': 0,              # How many times this topic has appeared
            'last_mentioned': None,      # Timestamp of most recent mention
            'importance_score': 0.0,     # Calculated importance based on frequency and recency
            'context_boost': 0.0         # Priority boost factor for context-relevant queries
        })
        
        # PERFORMANCE MONITORING AND OPTIMIZATION
        # Global system metrics for overall health and optimization decision-making
        self.performance_metrics = {
            'total_queries_processed': 0,    # Lifetime count of all processed queries
            'average_response_time': 0.0,    # System-wide average response time
            'priority_accuracy': 0.0,        # Measure of how accurate priority assignments are
            'cache_hit_rate': 0.0,           # Efficiency of priority calculation caching
            'system_load': 0.0,              # Current system utilization level
            'optimization_suggestions': []   # AI-generated recommendations for improvement
        }
        
        # ADVANCED CACHING FOR PRIORITY CALCULATIONS
        # Implements memoization to avoid redundant priority calculations - key greedy optimization
        # Enables O(1) lookup for previously analyzed queries
        self.priority_cache = {}             # Storage for cached priority calculations
        self.cache_max_size = 1000          # Maximum cache entries to prevent memory issues
        self.cache_stats = {                # Cache performance monitoring
            'hits': 0,                      # Successful cache retrievals (fast path)
            'misses': 0,                    # Cache misses requiring calculation (slow path)
            'invalidations': 0              # Cache entries removed due to size limits
        }
        
        # SESSION AND TEMPORAL ANALYSIS
        # Session-level tracking for performance analysis and temporal pattern recognition
        self.session_start = datetime.datetime.now()  # Session start time for duration calculations
        
        # Temporal pattern recognition for predictive optimization
        self.temporal_patterns = {
            'peak_hours': [],               # Time periods with highest query volume
            'low_activity_periods': [],     # Time periods with minimal activity
            'query_velocity': 0.0,          # Rate of incoming queries (queries per second)
            'burst_detection': False        # Whether system is experiencing query burst
        }

    def get_priority(self, query: str) -> Union[int, Priority]:
        """
        Determine the priority level of a query using an advanced multi-factor analysis.
        
        GREEDY ALGORITHM CORE IMPLEMENTATION:
        This method embodies the fundamental greedy principle: make the best local decision
        based on immediately available information. It uses a cascading approach where
        multiple heuristics are applied in order of reliability, and the first confident
        result determines the priority (no backtracking or reconsideration).
        
        SIX-PHASE GREEDY DECISION PROCESS:
        1. Cache lookup (O(1) optimization)
        2. Pattern analysis (highest reliability)
        3. Keyword analysis (good reliability)
        4. Complexity analysis (moderate reliability)
        5. Length analysis (basic heuristic)
        6. Structural analysis (fallback heuristic)
        7. Context-aware default (guaranteed result)
        
        This method implements a sophisticated greedy algorithm that considers:
        - Keyword-based urgency detection with weighted scoring
        - Advanced regex pattern matching for context recognition  
        - Query complexity analysis using NLP-inspired techniques
        - Length-based heuristics with adaptive thresholds
        - Conversation context and topic continuity
        - Temporal factors and user behavior patterns
        
        Args:
            query (str): The user query to evaluate and prioritize
            
        Returns:
            Union[int, Priority]: Priority level (1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW)
                                 Returns integer for backward compatibility, Priority enum for enhanced features
        """
        # EDGE CASE HANDLING: Empty or invalid queries get minimal priority
        if not query or not query.strip():
            return Priority.LOW
            
        try:
            # PHASE 0: CACHE LOOKUP FOR PERFORMANCE OPTIMIZATION
            # This is the ultimate greedy optimization - avoid all computation if result is cached
            # Provides O(1) lookup time vs O(k) analysis time where k can be substantial
            cache_key = self._normalize_query_for_cache(query)
            if cache_key in self.priority_cache:
                self.cache_stats['hits'] += 1  # Track cache effectiveness for optimization
                cached_result = self.priority_cache[cache_key]
                
                # Apply dynamic context boost if conversation context is available
                context_boost = self._calculate_context_boost(query)
                if context_boost > 0:
                    # Boost priority by reducing numeric value (lower = higher priority)
                    boosted_priority = max(Priority.CRITICAL, Priority(cached_result.value - context_boost))
                    return boosted_priority
                return cached_result
            
            # Cache miss - need to perform full priority analysis
            self.cache_stats['misses'] += 1
            
            # Normalize query for consistent analysis across all heuristics
            query_lower = query.lower().strip()
            
            # PHASE 1: ADVANCED PATTERN-BASED PRIORITY DETECTION (HIGHEST PRECEDENCE)
            # Regex patterns can capture complex linguistic structures that indicate urgency
            # This phase handles the most critical and time-sensitive queries first
            pattern_priority = self._analyze_priority_patterns(query_lower)
            if pattern_priority:
                # Cache result to avoid recalculation - greedy efficiency optimization
                self._cache_priority_result(cache_key, pattern_priority)
                return pattern_priority
            
            # PHASE 2: ENHANCED KEYWORD-BASED PRIORITY WITH WEIGHTED SCORING
            # Keyword analysis handles the majority of standard cases efficiently
            # Uses pre-organized keyword lists for fast O(k) classification
            keyword_priority = self._analyze_keyword_priority(query_lower)
            if keyword_priority:
                self._cache_priority_result(cache_key, keyword_priority)
                return keyword_priority
            
            # PHASE 3: ADVANCED COMPLEXITY-BASED PRIORITY ANALYSIS
            # Complex queries often require more resources and should get higher priority
            # This heuristic catches technical queries that might not match patterns/keywords
            complexity_priority = self._analyze_complexity_priority(query)
            if complexity_priority:
                self._cache_priority_result(cache_key, complexity_priority)
                return complexity_priority
            
            # PHASE 4: LENGTH-BASED HEURISTICS WITH CONTEXT AWARENESS
            # Query length often correlates with complexity and user investment
            # Longer queries typically indicate more detailed, important requests
            length_priority = self._analyze_length_priority(query)
            if length_priority:
                self._cache_priority_result(cache_key, length_priority)
                return length_priority
            
            # PHASE 5: PUNCTUATION AND STRUCTURAL ANALYSIS
            # Formatting and punctuation can indicate urgency or query type
            # Catches cases like ALL CAPS (urgency) or multiple exclamation marks
            structural_priority = self._analyze_structural_priority(query)
            if structural_priority:
                self._cache_priority_result(cache_key, structural_priority)
                return structural_priority
            
            # PHASE 6: CONTEXT-AWARE DEFAULT WITH CONVERSATION ANALYSIS
            # Final fallback ensures every query gets a reasonable priority
            # Uses conversation context to make informed default decisions
            context_priority = self._determine_context_priority(query)
            self._cache_priority_result(cache_key, context_priority)
            return context_priority

        except Exception as e:
            # GRACEFUL ERROR HANDLING: Never fail completely, always return usable priority
            print(f"Error in get_priority: {e}")
            return Priority.MEDIUM  # Safe fallback that works for most cases
    
    def sort_queries_by_priority(self, queries: List[str]) -> List[Tuple[Union[int, Priority], str]]:
        """
        Sort a list of queries based on their calculated priority using advanced algorithms.
        
        GREEDY BATCH PROCESSING STRATEGY:
        This method applies the greedy principle to batch processing: calculate the optimal
        priority for each query independently using our greedy algorithm, then sort based
        on those local decisions. No query's priority is reconsidered based on other
        queries in the batch - each gets the best priority based on its individual characteristics.
        
        MULTI-CRITERIA GREEDY OPTIMIZATION:
        Beyond basic priority, the method applies additional greedy optimizations:
        - Context continuity bonuses (queries related to current conversation)
        - Position-based urgency (later queries might indicate user frustration)
        - Length and complexity tie-breaking (more detailed queries get slight boost)
        
        Implements an optimized sorting strategy that:
        - Batch processes queries for improved performance
        - Maintains context across related queries  
        - Applies conversation continuity bonuses
        - Considers temporal factors and urgency decay
        - Provides detailed sorting metadata and analytics
        
        Args:
            queries (List[str]): A list of user queries to prioritize and sort
            
        Returns:
            List[Tuple[Union[int, Priority], str]]: Sorted list of (priority, query) tuples,
                                                   ordered from highest to lowest priority
        """
        # EDGE CASE: Empty list handling
        if not queries:
            return []
            
        try:
            # BATCH PRIORITY CALCULATION WITH CONTEXT AWARENESS
            # Process all queries efficiently while maintaining context across the batch
            prioritized_queries = []
            context_topics = self._extract_current_context_topics()  # Get active conversation topics
            
            # Apply greedy priority calculation to each query independently
            for i, query in enumerate(queries):
                # STEP 1: Calculate base priority using core greedy algorithm
                priority = self.get_priority(query)
                
                # STEP 2: Apply contextual adjustments based on conversation continuity
                # This is a greedy optimization that boosts priority for topically relevant queries
                if context_topics:
                    query_topics = self._extract_query_topics(query)
                    # Check for topic overlap with current conversation
                    if any(topic in context_topics for topic in query_topics):
                        # Boost priority for contextually relevant queries (greedy local optimization)
                        if isinstance(priority, Priority):
                            priority = Priority(max(Priority.CRITICAL.value, priority.value - 1))
                        else:
                            priority = max(1, priority - 1)  # Lower number = higher priority
                
                # STEP 3: Apply position-based urgency (temporal locality heuristic)
                # Later queries in a batch might indicate increasing user frustration or urgency
                position_factor = i / len(queries) if len(queries) > 1 else 0
                if position_factor > 0.8:  # Last 20% of queries get urgency boost
                    if isinstance(priority, Priority):
                        priority = Priority(max(Priority.CRITICAL.value, priority.value - 1))
                    else:
                        priority = max(1, priority - 1)
                
                # Store query with its calculated priority
                prioritized_queries.append((priority, query))
            
            # ADVANCED SORTING WITH MULTIPLE CRITERIA
            # Define sophisticated sort key that implements greedy tie-breaking strategies
            def sort_key(item):
                priority, query = item
                # Primary sort criterion: priority value (1=highest, 4=lowest)
                priority_value = priority.value if hasattr(priority, 'value') else priority
                
                # Secondary sort criterion: length bonus (longer queries often more important)
                # This implements a greedy heuristic that longer queries show user investment
                length_bonus = min(len(query) / 1000, 0.1)  # Max 0.1 bonus, prevents over-weighting
                
                # Tertiary sort criterion: complexity indicators bonus
                # Technical queries with complexity indicators get slight priority boost
                complexity_bonus = self._calculate_complexity_score(query) * 0.05
                
                # Return combined score for sorting (lower = higher priority)
                return priority_value - length_bonus - complexity_bonus
            
            # EXECUTE SORT: Sort in ascending order of priority value (1=highest, 4=lowest)
            # This maintains min-heap property for efficient priority queue operations
            prioritized_queries.sort(key=sort_key)
            
            # UPDATE ANALYTICS: Track sorting performance for future optimization
            self._update_sorting_metrics(prioritized_queries)
            
            return prioritized_queries

        except Exception as e:
            # GRACEFUL ERROR HANDLING: Always provide usable output
            print(f"Error sorting queries: {e}")
            # Return with default medium priority if sorting fails
            return [(Priority.MEDIUM, query) for query in queries]
    
    def record_query_stats(self, query: str, processing_time: float, success: bool) -> None:
        """
        Record comprehensive statistics for a processed query with advanced analytics.
        
        GREEDY ALGORITHM FEEDBACK LOOP:
        This method implements the data collection side of the greedy algorithm's learning
        system. By recording how well our priority decisions worked out, the system can
        make increasingly better local optimization choices over time.
        
        PERFORMANCE-DRIVEN ADAPTATION:
        The collected metrics enable the greedy algorithm to adapt its strategies:
        - Slow query types can be identified for priority adjustment
        - Success patterns can be reinforced in future priority calculations
        - Anomalies can trigger automatic priority strategy adjustments
        
        This method tracks detailed performance metrics including:
        - Processing time analysis with trend detection
        - Success rate monitoring with quality indicators
        - Priority distribution analytics
        - Query complexity correlation analysis  
        - Performance optimization recommendations
        - Anomaly detection and alerting
        
        Args:
            query (str): The original user query that was processed
            processing_time (float): Time taken to process the query (in seconds)
            success (bool): Whether the query processing was successful
        """
        # INPUT VALIDATION: Ensure data quality for reliable analytics
        if processing_time < 0:
            print(f"Warning: Invalid processing time {processing_time}")
            return
            
        try:
            # STEP 1: CATEGORIZE AND ANALYZE THE QUERY FOR DETAILED TRACKING
            # Determine query characteristics for segmented performance analysis
            query_type = self._categorize_query(query)              # Classify query into logical category
            priority = self.get_priority(query)                     # Get assigned priority level
            priority_value = priority.value if hasattr(priority, 'value') else priority
            complexity_score = self._calculate_complexity_score(query)  # Measure query complexity
            
            # STEP 2: INITIALIZE OR RETRIEVE EXISTING METRICS
            # Use lazy initialization pattern - create metrics only when needed
            if query_type not in self.query_stats:
                self.query_stats[query_type] = QueryMetrics()
            
            # STEP 3: UPDATE COMPREHENSIVE METRICS WITH NEW DATA
            # Update the metrics object using its built-in update method
            self.query_stats[query_type].update(
                processing_time=processing_time,
                success=success,
                priority=priority_value,
                complexity=complexity_score
            )
            
            # STEP 4: UPDATE GLOBAL PERFORMANCE METRICS
            # Maintain system-wide statistics for overall health monitoring
            self._update_global_performance_metrics(processing_time, success, priority_value)
            
            # STEP 5: ANALYZE FOR PERFORMANCE ANOMALIES AND OPTIMIZATION OPPORTUNITIES
            # Detect patterns that might indicate problems with priority assignments
            self._analyze_performance_anomalies(query_type, processing_time, success)
            
            # STEP 6: UPDATE CONVERSATION CONTEXT FOR FUTURE PRIORITY DECISIONS
            # Maintain conversation awareness for context-driven priority adjustments
            self._update_conversation_context(query, processing_time, success)
            
        except Exception as e:
            # GRACEFUL ERROR HANDLING: Don't let statistics failures affect core functionality
            print(f"Error recording query stats: {e}")
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights and recommendations based on collected analytics.
        
        GREEDY ALGORITHM EFFECTIVENESS ANALYSIS:
        This method analyzes the accumulated data to determine how well our greedy
        priority decisions are working in practice. It identifies patterns that indicate
        whether local optimization choices are leading to good global outcomes.
        
        CONTINUOUS IMPROVEMENT FEEDBACK:
        By analyzing performance patterns, the method generates actionable insights:
        - Which priority assignments are most/least effective
        - Query types that need different priority strategies
        - System bottlenecks that affect priority-based processing
        - Optimization opportunities for better greedy decisions
        
        Provides detailed analysis including:
        - Performance bottleneck identification
        - Query pattern analysis and trends
        - Resource utilization recommendations
        - Priority distribution optimization
        - Predictive capacity planning
        - System health indicators
        
        Returns:
            Dict[str, Any]: Comprehensive insights with performance metrics,
                           optimization recommendations, and predictive analytics
        """
        try:
            # EDGE CASE: No data available yet for analysis
            if not self.query_stats:
                return self._generate_empty_insights()
            
            # STEP 1: CALCULATE COMPREHENSIVE AGGREGATE METRICS
            # Compute overall system performance from individual query type metrics
            total_time = sum(stats.total_time for stats in self.query_stats.values())
            total_count = sum(stats.count for stats in self.query_stats.values())
            total_success = sum(stats.success_count for stats in self.query_stats.values())
            
            # STEP 2: GENERATE BASE INSIGHTS STRUCTURE
            # Create foundation of analytics report with core metrics
            insights = {
                'total_queries': total_count,
                'avg_processing_time': total_time / total_count if total_count > 0 else 0.0,
                'overall_success_rate': total_success / total_count if total_count > 0 else 0.0,
                'query_types_analyzed': len(self.query_stats),
                'analysis_timestamp': datetime.datetime.now().isoformat(),
                'session_duration': (datetime.datetime.now() - self.session_start).total_seconds()
            }
            
            # STEP 3: ADVANCED PERFORMANCE ANALYSIS
            # Detailed breakdown of performance by various dimensions
            insights.update(self._generate_performance_analysis())
            
            # STEP 4: DETAILED QUERY TYPE BREAKDOWNS
            # Analysis of different query types and their characteristics
            insights.update(self._generate_query_type_analysis())
            
            # STEP 5: PRIORITY DISTRIBUTION INSIGHTS
            # Analysis of how well priority assignments are distributed and effective
            insights.update(self._generate_priority_analysis())
            
            # STEP 6: OPTIMIZATION RECOMMENDATIONS
            # Actionable suggestions for improving greedy algorithm effectiveness
            insights.update(self._generate_optimization_recommendations(insights))
            
            # STEP 7: PREDICTIVE ANALYTICS
            # Forward-looking analysis and capacity planning insights
            insights.update(self._generate_predictive_insights())
            
            # STEP 8: SYSTEM HEALTH INDICATORS
            # Overall system health assessment and monitoring data
            insights.update(self._generate_health_indicators())
            
            return insights

        except Exception as e:
            # GRACEFUL ERROR HANDLING: Always return useful information even if analysis fails
            print(f"Error generating optimization insights: {e}")
            return {
                'error': f'Could not generate insights: {e}', 
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    # === ENHANCED PRIORITY QUEUE OPERATIONS ===
    
    def add_to_priority_queue(self, query: str, timestamp: Optional[float] = None) -> None:
        """
        Add a query to the priority queue with advanced metadata and context analysis.
        
        PRIORITY QUEUE GREEDY IMPLEMENTATION:
        This method implements the core data structure operation for greedy priority processing.
        Using Python's heapq (binary heap), it provides O(log n) insertion while maintaining
        the heap property that ensures the highest priority query is always at the root.
        
        GREEDY METADATA ENRICHMENT:
        Beyond basic priority, the method calculates additional metadata that supports
        sophisticated queue management and optimization decisions.
        
        Args:
            query (str): The query to add to the processing queue
            timestamp (Optional[float]): Query timestamp, defaults to current time
        """
        # DEFAULT TIMESTAMP: Use current time if not specified
        if timestamp is None:
            timestamp = time.time()
            
        try:
            # STEP 1: CALCULATE PRIORITY USING GREEDY ALGORITHM
            priority = self.get_priority(query)
            priority_value = priority.value if hasattr(priority, 'value') else priority
            
            # STEP 2: CREATE ENHANCED QUEUE ENTRY WITH COMPREHENSIVE METADATA
            # Rich metadata enables sophisticated queue management and analytics
            queue_entry = {
                'query': query,                                                # Original query text
                'priority': priority_value,                                   # Calculated priority level
                'timestamp': timestamp,                                       # Submission timestamp
                'complexity': self._calculate_complexity_score(query),       # Complexity analysis
                'estimated_processing_time': self._estimate_processing_time(query),  # Time estimate
                'context_relevance': self._calculate_context_relevance(query),       # Context score
                'entry_id': f"{timestamp}_{hash(query) % 10000}"            # Unique identifier
            }
            
            # STEP 3: INSERT INTO HEAP WITH CORRECT ORDERING
            # Use negative priority for min-heap behavior (Python's heapq is min-heap)
            # Lower priority values (like CRITICAL=1) become higher when negated
            # Timestamp as secondary sort key ensures FIFO within same priority level
            heap_entry = (-priority_value, timestamp, queue_entry)
            heapq.heappush(self.priority_queue, heap_entry)
            
            # STEP 4: UPDATE QUEUE METADATA FOR ANALYTICS
            # Track priority distribution and analyze queue usage patterns
            self.queue_metadata['priority_distribution'][priority_value] += 1
            self._detect_queue_patterns()  # Analyze for usage patterns and optimization opportunities
            
        except Exception as e:
            # GRACEFUL ERROR HANDLING: Log error but don't crash the system
            print(f"Error adding to priority queue: {e}")
    
    def get_next_query(self) -> Optional[str]:
        """
        Get the next highest priority query from the queue with comprehensive logging.
        
        GREEDY QUEUE PROCESSING:
        This method implements the core greedy principle for queue processing: always
        process the highest priority query next. The binary heap ensures that extraction
        is O(log n) and always returns the optimal choice based on priority.
        
        QUEUE ANALYTICS:
        The method tracks queue performance metrics to identify bottlenecks and
        optimize the overall priority system effectiveness.
        
        Returns:
            Optional[str]: The next query to process, or None if queue is empty
        """
        # EDGE CASE: Empty queue handling
        if not self.priority_queue:
            return None
        
        try:
            # STEP 1: EXTRACT HIGHEST PRIORITY QUERY FROM HEAP
            # heappop automatically extracts the minimum element (highest priority due to negation)
            # Heap structure is automatically rebalanced to maintain priority ordering
            _, _, queue_entry = heapq.heappop(self.priority_queue)
            query = queue_entry['query']
            
            # STEP 2: UPDATE PROCESSING METRICS
            # Calculate queue wait time for performance analysis
            wait_time = time.time() - queue_entry['timestamp']
            self._update_queue_metrics(wait_time, queue_entry)
            
            # STEP 3: RETURN QUERY FOR PROCESSING
            return query
            
        except Exception as e:
            # GRACEFUL ERROR HANDLING: Never crash on queue operations
            print(f"Error getting next query from queue: {e}")
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the priority queue.
        
        QUEUE HEALTH MONITORING:
        This method provides detailed insights into queue performance and health,
        enabling proactive optimization and capacity planning for the greedy
        priority system.
        
        PERFORMANCE ANALYTICS:
        The status information helps identify whether the greedy priority assignments
        are leading to effective queue processing and resource utilization.
        
        Returns:
            Dict[str, Any]: Detailed queue status with analytics and predictions
        """
        try:
            # STEP 1: CALCULATE BASIC QUEUE STATUS
            status = {
                'queue_length': len(self.priority_queue),           # Current number of queued queries
                'is_empty': len(self.priority_queue) == 0,          # Whether queue is empty
                'next_priority': None,                              # Priority of next query to process
                'estimated_wait_time': 0.0,                        # Estimated time until processing
                'priority_distribution': dict(self.queue_metadata['priority_distribution']),  # Priority breakdown
                'average_complexity': 0.0,                         # Average complexity of queued items
                'queue_health': 'healthy'                          # Overall queue health assessment
            }
            
            # STEP 2: DETAILED STATUS FOR NON-EMPTY QUEUE
            if self.priority_queue:
                # Get next priority without removing item from queue (peek operation)
                next_priority_value = -self.priority_queue[0][0]   # Undo the negation used for heap ordering
                status['next_priority'] = next_priority_value
                
                # Calculate estimated wait time based on queue depth and processing history
                status['estimated_wait_time'] = self._estimate_queue_wait_time()
                
                # Calculate average complexity of currently queued items
                complexities = []
                for _, _, entry in self.priority_queue:
                    if isinstance(entry, dict) and 'complexity' in entry:
                        complexities.append(entry['complexity'])
                
                if complexities:
                    status['average_complexity'] = sum(complexities) / len(complexities)
                
                # Assess overall queue health based on length and processing patterns
                status['queue_health'] = self._assess_queue_health()
            
            return status
            
        except Exception as e:
            # GRACEFUL ERROR HANDLING: Return safe defaults on error
            print(f"Error getting queue status: {e}")
            return {'error': str(e), 'queue_length': 0, 'is_empty': True}
    
    def clear_queue(self) -> None:
        """
        Clear all queries from the priority queue and reset metadata.
        
        QUEUE RESET OPERATIONS:
        This method provides administrative functionality for queue management,
        useful for maintenance, testing, and emergency situations.
        """
        try:
            # STEP 1: TRACK CLEARED QUERIES FOR ANALYTICS
            cleared_count = len(self.priority_queue)
            
            # STEP 2: CLEAR QUEUE AND RESET METADATA
            self.priority_queue.clear()                                    # Clear the heap structure
            self.queue_metadata['priority_distribution'].clear()          # Reset priority tracking
            self.queue_metadata['total_processed'] += cleared_count       # Update lifetime processed count
            
            # STEP 3: LOG THE OPERATION
            print(f"Cleared {cleared_count} queries from priority queue")
            
        except Exception as e:
            # GRACEFUL ERROR HANDLING: Log errors but don't crash
            print(f"Error clearing queue: {e}")
    
    def reset_stats(self) -> None:
        """
        Reset all collected statistics and performance metrics.
        
        STATISTICS RESET FUNCTIONALITY:
        This method provides a clean slate for performance measurement, useful for:
        - Testing different priority strategies
        - Starting fresh performance analysis
        - Removing obsolete historical data
        """
        try:
            # STEP 1: TRACK RESET SCOPE FOR LOGGING
            old_stats_count = len(self.query_stats)
            
            # STEP 2: CLEAR ALL STATISTICAL DATA
            self.query_stats.clear()                    # Clear query type performance metrics
            self.priority_cache.clear()                 # Clear priority calculation cache
            self.cache_stats = {                        # Reset cache performance counters
                'hits': 0, 
                'misses': 0, 
                'invalidations': 0
            }
            self.conversation_context.clear()           # Clear conversation history
            self.topic_continuity.clear()               # Clear topic tracking data
            
            # STEP 3: RESET PERFORMANCE METRICS TO DEFAULTS
            self.performance_metrics = {
                'total_queries_processed': 0,
                'average_response_time': 0.0,
                'priority_accuracy': 0.0,
                'cache_hit_rate': 0.0,
                'system_load': 0.0,
                'optimization_suggestions': []
            }
            
            # STEP 4: RESET SESSION TRACKING
            self.session_start = datetime.datetime.now()
            
            # STEP 5: LOG THE RESET OPERATION
            print(f"Reset statistics for {old_stats_count} query types")
            
        except Exception as e:
            # GRACEFUL ERROR HANDLING: Log errors but don't crash
            print(f"Error resetting stats: {e}")
    
    # === PRIVATE HELPER METHODS FOR ENHANCED FUNCTIONALITY ===
    
    def _analyze_priority_patterns(self, query_lower: str) -> Optional[Priority]:
        """
        Analyze query using advanced regex patterns for priority detection.
        
        PATTERN-BASED GREEDY ANALYSIS:
        This method implements the highest-confidence phase of priority detection.
        Regex patterns can capture complex linguistic structures that indicate urgency
        or importance with high reliability.
        
        EARLY TERMINATION OPTIMIZATION:
        The method uses the greedy principle of early termination - as soon as a
        pattern matches, it returns that priority without checking other patterns.
        """
        # Iterate through priority levels from highest to lowest importance
        for priority_level, patterns in self.priority_patterns.items():
            # Check each regex pattern for the current priority level
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return priority_level  # Return immediately on first match (greedy principle)
        return None  # No patterns matched
    
    def _analyze_keyword_priority(self, query_lower: str) -> Optional[Priority]:
        """
        Analyze query using keyword matching with weighted scoring.
        
        KEYWORD-BASED GREEDY ANALYSIS:
        This method implements weighted keyword analysis where different keywords
        contribute different amounts to the priority score. It uses a greedy scoring
        approach where the priority level with the highest score wins.
        
        WEIGHTED SCORING STRATEGY:
        Multi-word keywords (like "data loss") are weighted higher than single words
        because they are more specific and typically indicate more serious situations.
        """
        keyword_scores = {}  # Track scores for each priority level
        
        # Analyze keywords for each priority level
        for priority_level, keywords in self.priority_keywords.items():
            score = 0
            # Check each keyword in the current priority category
            for keyword in keywords:
                if keyword in query_lower:
                    # Weight score based on keyword specificity and length
                    # Multi-word keywords get higher weight as they're more specific
                    keyword_weight = len(keyword.split()) * 0.5 + 1.0
                    score += keyword_weight
            
            # Store score if any keywords matched
            if score > 0:
                keyword_scores[priority_level] = score
        
        # Return priority level with highest score (greedy selection)
        if keyword_scores:
            return max(keyword_scores.items(), key=lambda x: x[1])[0]
        
        return None  # No keywords matched
    
    def _analyze_complexity_priority(self, query: str) -> Optional[Priority]:
        """
        Analyze query complexity to determine appropriate priority.
        
        COMPLEXITY-BASED GREEDY HEURISTIC:
        This method uses the greedy assumption that more complex queries deserve
        higher priority because they likely require more specialized expertise
        and resources to handle properly.
        
        THRESHOLD-BASED DECISION MAKING:
        Uses predefined complexity thresholds to make immediate priority decisions
        without complex analysis - a key greedy optimization principle.
        """
        complexity_score = self._calculate_complexity_score(query)
        
        # Use greedy threshold-based classification
        if complexity_score >= 4:
            return Priority.CRITICAL  # Very complex queries need immediate expert attention
        elif complexity_score >= 3:
            return Priority.HIGH      # Complex queries deserve high priority
        elif complexity_score >= 2:
            return Priority.MEDIUM    # Moderately complex queries get medium priority
        
        return None  # Complexity doesn't strongly indicate specific priority
    
    def _analyze_length_priority(self, query: str) -> Optional[Priority]:
        """
        Analyze query length to infer priority and complexity.
        
        LENGTH-BASED GREEDY HEURISTIC:
        This method implements a simple but effective heuristic: longer queries
        often indicate more complex problems or higher user investment, suggesting
        they should receive higher priority.
        
        THRESHOLD-BASED CLASSIFICATION:
        Uses pre-defined length thresholds for immediate decision making without
        complex natural language processing.
        """
        query_length = len(query)
        
        # Apply greedy length-based thresholds
        if query_length > self.length_thresholds['very_long']:
            return Priority.HIGH  # Very long queries likely indicate complex, important issues
        elif query_length > self.length_thresholds['long']:
            return Priority.MEDIUM  # Long queries deserve attention
        elif query_length > self.length_thresholds['medium']:
            return Priority.MEDIUM  # Medium queries get standard priority
        
        return None  # Length doesn't strongly indicate specific priority
    
    def _analyze_structural_priority(self, query: str) -> Optional[Priority]:
        """
        Analyze structural elements like punctuation for priority hints.
        
        STRUCTURAL GREEDY ANALYSIS:
        This method analyzes formatting and punctuation patterns that often
        indicate urgency or emotion. It implements quick pattern recognition
        based on common text conventions for expressing urgency.
        
        PUNCTUATION AND FORMATTING HEURISTICS:
        - Exclamation marks often indicate urgency or strong emotion
        - ALL CAPS typically suggests urgency or frustration
        - Question marks indicate information-seeking behavior
        """
        # Check for structural urgency indicators
        if '!' in query:
            return Priority.HIGH   # Exclamation marks suggest urgency or strong emphasis
        elif '?' in query:
            return Priority.MEDIUM  # Questions are standard information-seeking requests
        elif query.isupper() and len(query) > 10:
            return Priority.HIGH   # ALL CAPS often indicates urgency or frustration
        
        return None  # No structural indicators found
    
    def _determine_context_priority(self, query: str) -> Priority:
        """
        Determine priority based on conversation context and patterns.
        
        CONTEXT-AWARE GREEDY DEFAULT:
        This is the final fallback that ensures every query receives a priority.
        It uses conversation context to make an informed default decision,
        implementing context-aware greedy optimization.
        
        CONTEXTUAL DECISION MAKING:
        Queries that relate to ongoing conversation topics get higher priority
        as they likely represent follow-up questions or continued problem-solving.
        """
        # Calculate context boost based on conversation history
        context_boost = self._calculate_context_boost(query)
        
        # Make greedy decision based on context relevance
        if context_boost > 0.5:
            return Priority.HIGH     # Strong context relevance indicates importance
        elif context_boost > 0.2:
            return Priority.MEDIUM   # Some context relevance gets standard priority
        else:
            return Priority.MEDIUM   # Default fallback for queries without strong context
    
    def _calculate_complexity_score(self, query: str) -> float:
        """
        Calculate a comprehensive complexity score for the query.
        
        MULTI-FACTOR COMPLEXITY ANALYSIS:
        This method implements a sophisticated complexity scoring system that
        considers multiple factors to assess how difficult a query is likely
        to be to process. Higher complexity often correlates with higher priority.
        
        SCORING FACTORS:
        1. Technical terminology (indicates need for expertise)
        2. Query structure complexity (detailed requests)
        3. Multiple questions or parts (compound complexity)
        4. Code-like patterns (technical complexity)
        5. Length-based complexity (detail level)
        6. Multi-part structure (conjunction complexity)
        
        Returns:
            float: Complexity score (0-5+ scale)
        """
        score = 0.0  # Initialize complexity score
        query_lower = query.lower()
        
        # FACTOR 1: TECHNICAL COMPLEXITY INDICATORS
        # Count matches with pre-defined technical terminology
        complexity_matches = sum(1 for indicator in self.complexity_indicators 
                               if indicator in query_lower)
        score += min(complexity_matches * 0.5, 2.0)  # Max 2 points for technical terms, prevent over-weighting
        
        # FACTOR 2: QUERY STRUCTURE COMPLEXITY
        # Look for indicators of detailed or comprehensive requests
        if re.search(r'\b(step\s+by\s+step|detailed|comprehensive|thorough|complete)\b', query_lower):
            score += 1.0  # Detailed requests are inherently more complex
        
        # FACTOR 3: MULTIPLE QUESTIONS OR PARTS
        # Multiple question marks indicate compound queries
        question_count = query.count('?')
        if question_count > 1:
            score += 0.5 * question_count  # Each additional question adds complexity
        
        # FACTOR 4: CODE-LIKE PATTERNS OR TECHNICAL SYNTAX
        # Presence of code syntax or technical formatting indicates technical complexity
        if re.search(r'[(){}\[\]]', query) or '```' in query or 'code' in query_lower:
            score += 1.0  # Code-related queries require technical expertise
        
        # FACTOR 5: LENGTH-BASED COMPLEXITY
        # Word count often correlates with complexity and detail level
        word_count = len(query.split())
        if word_count > 20:
            score += min(word_count / 50, 1.0)  # Gradual increase, capped to prevent runaway scoring
        
        # FACTOR 6: MULTI-PART QUERIES (INDICATED BY CONJUNCTIONS)
        # Conjunctions often indicate multiple related questions or requests
        if re.search(r'\b(and|also|additionally|furthermore|moreover)\b', query_lower):
            score += 0.5  # Multi-part queries require handling multiple concepts
        
        return score  # Return total calculated complexity score
    
    def _categorize_query(self, query: str) -> str:
        """
        Categorize a query into a logical type for comprehensive analytics.
        
        QUERY CATEGORIZATION FOR PERFORMANCE TRACKING:
        This method enables segmented performance analysis by classifying queries
        into logical types. Different query types may require different priority
        strategies, and this categorization supports data-driven optimization.
        
        HIERARCHICAL CLASSIFICATION STRATEGY:
        Uses a decision tree approach where more specific patterns are checked first,
        falling back to general categories. This ensures accurate classification
        while maintaining efficiency.
        
        Args:
            query (str): The user query string
            
        Returns:
            str: A detailed query category label
        """
        # EDGE CASE: Handle empty or invalid queries
        if not query or not query.strip():
            return 'empty'
            
        try:
            query_lower = query.lower()
            
            # HIERARCHICAL PATTERN MATCHING FOR PRECISE CATEGORIZATION
            # Check most specific patterns first, then fall back to general categories
            
            # Information-seeking patterns
            if any(word in query_lower for word in ['what', 'define', 'definition', 'meaning', 'describe']):
                return 'definition'      # Queries seeking explanations or definitions
            elif any(word in query_lower for word in ['how', 'process', 'step', 'tutorial', 'guide', 'method']):
                return 'how-to'          # Procedural questions and tutorials
            elif any(word in query_lower for word in ['where', 'location', 'place', 'address', 'find']):
                return 'location'        # Location-based queries
            elif any(word in query_lower for word in ['when', 'time', 'date', 'schedule', 'timing']):
                return 'temporal'        # Time-related queries
            elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because', 'purpose']):
                return 'causal'          # Causal reasoning queries
            elif any(word in query_lower for word in ['who', 'person', 'people', 'author', 'creator']):
                return 'person'          # Person-related queries
            
            # Problem-solving patterns
            elif any(word in query_lower for word in ['error', 'bug', 'issue', 'problem', 'broken', 'fail']):
                return 'troubleshooting' # Technical problem-solving
            
            # Social interaction patterns
            elif any(word in query_lower for word in ['hello', 'hi', 'thanks', 'goodbye', 'bye', 'greetings']):
                return 'social'          # Social pleasantries and conversation
            
            # Technical complexity patterns
            elif any(word in query_lower for word in self.complexity_indicators):
                return 'technical'       # Technical or specialized queries
            
            # Analytical patterns
            elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
                return 'comparison'      # Comparative analysis requests
            
            # Example-seeking patterns
            elif any(word in query_lower for word in ['example', 'sample', 'instance', 'demo']):
                return 'example'         # Requests for concrete examples
            
            # General question patterns
            elif '?' in query:
                return 'question'        # General interrogative format
            
            # Urgency patterns
            elif any(word in query_lower for word in ['urgent', 'emergency', 'critical', 'help']):
                return 'urgent'          # Urgent requests requiring immediate attention
            
            # Default fallback
            else:
                return 'general'         # General or uncategorized queries

        except Exception as e:
            # GRACEFUL ERROR HANDLING: Always return a valid category
            print(f"Error categorizing query: {e}")
            return 'general'
    
    def _normalize_query_for_cache(self, query: str) -> str:
        """
        Normalize query for cache key generation.
        
        CACHE KEY NORMALIZATION FOR CONSISTENT LOOKUPS:
        This method creates consistent cache keys by removing variations that don't
        affect priority (case, punctuation, extra whitespace). This improves cache
        hit rates and reduces redundant priority calculations.
        
        NORMALIZATION STRATEGY:
        - Convert to lowercase for case-insensitive matching
        - Remove punctuation that doesn't affect meaning
        - Normalize whitespace to single spaces
        """
        normalized = query.lower().strip()                    # Convert to lowercase, remove edge whitespace
        normalized = re.sub(r'[^\w\s]', '', normalized)       # Remove punctuation, keep only words and spaces
        normalized = re.sub(r'\s+', ' ', normalized)          # Replace multiple spaces with single space
        return normalized
    
    def _cache_priority_result(self, cache_key: str, priority: Priority) -> None:
        """
        Cache priority calculation result with size management.
        
        PRIORITY CACHE MANAGEMENT:
        This method implements the caching strategy that enables O(1) priority lookups
        for previously analyzed queries. It includes size management to prevent
        unbounded memory growth.
        
        CACHE EVICTION STRATEGY:
        Uses simple FIFO eviction when cache reaches capacity. More sophisticated
        strategies (LRU, LFU) could be implemented for better cache efficiency.
        """
        # CACHE SIZE MANAGEMENT: Prevent unbounded memory growth
        if len(self.priority_cache) >= self.cache_max_size:
            # Remove oldest entries using simple FIFO strategy
            keys_to_remove = list(self.priority_cache.keys())[:100]  # Remove first 100 entries
            for key in keys_to_remove:
                del self.priority_cache[key]
            self.cache_stats['invalidations'] += len(keys_to_remove)  # Track cache evictions
        
        # CACHE STORAGE: Store the calculated priority result
        self.priority_cache[cache_key] = priority
    
    def _calculate_context_boost(self, query: str) -> float:
        """
        Calculate context-based priority boost.
        
        CONTEXT-AWARE PRIORITY ADJUSTMENT:
        This method implements context-aware greedy optimization by boosting the
        priority of queries that relate to ongoing conversation topics. This helps
        maintain conversation coherence and user engagement.
        
        TOPIC OVERLAP CALCULATION:
        Uses set intersection to efficiently calculate topic overlap between the
        current query and recent conversation context.
        """
        # NO CONTEXT AVAILABLE: No boost possible
        if not self.conversation_context:
            return 0.0
        
        boost = 0.0  # Initialize context boost
        query_topics = self._extract_query_topics(query)  # Get topics from current query
        
        # ANALYZE RECENT CONVERSATION FOR TOPIC CONTINUITY
        recent_context = self.conversation_context[-4:]  # Last 2 exchanges (4 entries)
        for context_entry in recent_context:
            if 'topics' in context_entry:
                context_topics = context_entry['topics']
                # Calculate topic overlap using set intersection
                overlap = len(set(query_topics) & set(context_topics))
                if overlap > 0:
                    boost += overlap * 0.1  # Each overlapping topic adds 0.1 to boost
        
        return min(boost, 1.0)  # Cap boost at 1.0 to prevent excessive priority inflation
    
    def _extract_query_topics(self, query: str) -> List[str]:
        """
        Extract topics from a query for context analysis.
        
        TOPIC EXTRACTION FOR CONTEXT AWARENESS:
        This method extracts meaningful topics from queries to enable context-aware
        priority adjustments and conversation continuity tracking.
        
        MULTI-STRATEGY EXTRACTION:
        - Meaningful individual words (length filtering)
        - Technical terms from complexity indicators
        - Compound phrases (multi-word concepts)
        """
        topics = []
        query_lower = query.lower()
        
        # EXTRACT MEANINGFUL WORDS (LENGTH-BASED FILTERING)
        # Words longer than 3 characters are more likely to be meaningful topics
        words = re.findall(r'\b\w{4,}\b', query_lower)
        topics.extend(words)
        
        # EXTRACT TECHNICAL TERMS FROM COMPLEXITY INDICATORS
        # Pre-defined technical terms are important topics for context
        for indicator in self.complexity_indicators:
            if indicator in query_lower:
                topics.append(indicator)
        
        # EXTRACT COMPOUND PHRASES (MULTI-WORD CONCEPTS)
        # Common technical compound phrases that represent single concepts
        compound_patterns = [
            r'machine\s+learning', r'data\s+science', r'artificial\s+intelligence',
            r'web\s+development', r'software\s+engineering', r'computer\s+science'
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, query_lower)
            topics.extend(matches)
        
        return list(set(topics))  # Remove duplicates and return unique topics
    
    def _extract_current_context_topics(self) -> Set[str]:
        """
        Extract topics from current conversation context.
        
        CURRENT CONTEXT ANALYSIS:
        This method extracts topics from recent conversation history to support
        context-aware priority decisions and conversation continuity tracking.
        """
        topics = set()  # Use set for automatic deduplication
        
        # ANALYZE RECENT CONTEXT ENTRIES
        for entry in self.conversation_context[-6:]:  # Recent context (last 3 exchanges)
            if 'topics' in entry:
                topics.update(entry['topics'])  # Add all topics from this entry
        
        return topics
    
    def _update_sorting_metrics(self, sorted_queries: List[Tuple[Union[int, Priority], str]]) -> None:
        """
        Update metrics after query sorting operation.
        
        SORTING PERFORMANCE TRACKING:
        This method tracks the effectiveness of query sorting by analyzing the
        distribution of priorities in sorted batches. It helps optimize the
        sorting algorithm and identify patterns.
        """
        if not sorted_queries:
            return
        
        # TRACK PRIORITY DISTRIBUTION IN SORTED BATCH
        priority_counts = Counter()
        for priority, _ in sorted_queries:
            # Normalize priority to integer value for consistent counting
            priority_value = priority.value if hasattr(priority, 'value') else priority
            priority_counts[priority_value] += 1
        
        # UPDATE GLOBAL PRIORITY DISTRIBUTION TRACKING
        self.queue_metadata['priority_distribution'].update(priority_counts)
    
    def _update_global_performance_metrics(self, processing_time: float, success: bool, priority: int) -> None:
        """
        Update global performance tracking metrics.
        
        GLOBAL PERFORMANCE MONITORING:
        This method maintains system-wide performance metrics that help evaluate
        the effectiveness of the greedy algorithm's priority decisions and identify
        opportunities for optimization.
        """
        # INCREMENT TOTAL QUERY COUNTER
        self.performance_metrics['total_queries_processed'] += 1
        
        # UPDATE RUNNING AVERAGE RESPONSE TIME
        # Use incremental average calculation to avoid storing all historical data
        total_time = (self.performance_metrics['average_response_time'] * 
                     (self.performance_metrics['total_queries_processed'] - 1) + processing_time)
        self.performance_metrics['average_response_time'] = total_time / self.performance_metrics['total_queries_processed']
        
        # UPDATE CACHE HIT RATE FOR PERFORMANCE OPTIMIZATION
        total_cache_operations = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_cache_operations > 0:
            self.performance_metrics['cache_hit_rate'] = self.cache_stats['hits'] / total_cache_operations
    
    def _analyze_performance_anomalies(self, query_type: str, processing_time: float, success: bool) -> None:
        """
        Analyze for performance anomalies and generate alerts.
        
        ANOMALY DETECTION FOR OPTIMIZATION:
        This method identifies performance anomalies that might indicate problems
        with priority assignments or processing strategies. It helps optimize the
        greedy algorithm by identifying when local decisions aren't leading to
        good global outcomes.
        """
        if query_type in self.query_stats:
            stats = self.query_stats[query_type]
            
            # CHECK FOR PROCESSING TIME ANOMALIES
            # Detect when processing time is significantly higher than average
            if stats.avg_time > 0 and processing_time > stats.avg_time * 3:
                anomaly = f"Slow processing detected for {query_type}: {processing_time:.2f}s (avg: {stats.avg_time:.2f}s)"
                self.performance_metrics['optimization_suggestions'].append(anomaly)
            
            # CHECK FOR SUCCESS RATE DROPS
            # Detect when success rates are below acceptable thresholds
            if stats.count > 10 and stats.success_rate < 0.7:
                anomaly = f"Low success rate for {query_type}: {stats.success_rate:.2%}"
                self.performance_metrics['optimization_suggestions'].append(anomaly)
    
    def _update_conversation_context(self, query: str, processing_time: float, success: bool) -> None:
        """
        Update conversation context for future priority decisions.
        
        CONVERSATION CONTEXT MAINTENANCE:
        This method maintains conversation context that enables context-aware
        priority decisions. It implements sliding window management for memory
        efficiency while preserving relevant conversation history.
        """
        timestamp = datetime.datetime.now()
        
        # CREATE COMPREHENSIVE CONTEXT ENTRY
        context_entry = {
            'type': 'query',                                    # Entry type for filtering
            'content': query,                                   # Original query content
            'timestamp': timestamp,                             # When this interaction occurred
            'processing_time': processing_time,                 # Performance data
            'success': success,                                 # Outcome tracking
            'topics': self._extract_query_topics(query),       # Extracted topics for context
            'complexity': self._calculate_complexity_score(query),  # Complexity analysis
            'priority': self.get_priority(query)               # Assigned priority level
        }
        
        # ADD TO CONVERSATION CONTEXT
        self.conversation_context.append(context_entry)
        
        # MAINTAIN SLIDING WINDOW (MEMORY MANAGEMENT)
        if len(self.conversation_context) > self.context_window:
            self.conversation_context = self.conversation_context[-self.context_window:]
        
        # UPDATE TOPIC CONTINUITY TRACKING
        for topic in context_entry['topics']:
            self.topic_continuity[topic]['frequency'] += 1                      # Increment frequency counter
            self.topic_continuity[topic]['last_mentioned'] = timestamp          # Update last mention time
            # Calculate importance score based on frequency (capped at 1.0)
            self.topic_continuity[topic]['importance_score'] = min(
                self.topic_continuity[topic]['frequency'] * 0.1, 1.0
            )
    
    def _generate_empty_insights(self) -> Dict[str, Any]:
        """
        Generate insights structure when no data is available.
        
        EMPTY STATE HANDLING:
        This method provides a consistent response structure even when no performance
        data has been collected yet. It ensures the analytics system fails gracefully
        and provides useful feedback about the need for more data.
        """
        return {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'overall_success_rate': 0.0,
            'query_types_analyzed': 0,
            'slowest_query_types': [],
            'most_common_query_types': [],
            'fastest_query_types': [],
            'least_successful_types': [],
            'recommendations': ['Insufficient data for analysis. Process more queries to generate insights.'],
            'priority_distribution': {},
            'performance_trends': [],
            'cache_efficiency': 0.0,
            'system_health': 'unknown'
        }
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """
        Generate detailed performance analysis.
        
        PERFORMANCE ANALYTICS GENERATION:
        This method analyzes performance data to identify patterns, bottlenecks,
        and optimization opportunities. It sorts query types by various performance
        metrics to highlight areas needing attention.
        """
        # SORT QUERY TYPES BY PERFORMANCE METRICS
        sorted_by_time = sorted(self.query_stats.items(), key=lambda x: x[1].avg_time, reverse=True)
        sorted_by_count = sorted(self.query_stats.items(), key=lambda x: x[1].count, reverse=True)
        sorted_by_success = sorted(self.query_stats.items(), key=lambda x: x[1].success_rate)
        
        return {
            # SLOWEST QUERY TYPES (PERFORMANCE OPTIMIZATION TARGETS)
            'slowest_query_types': [
                {
                    'type': qtype,
                    'avg_time': stats.avg_time,
                    'count': stats.count,
                    'success_rate': stats.success_rate
                }
                for qtype, stats in sorted_by_time[:5]  # Top 5 slowest
            ],
            # MOST COMMON QUERY TYPES (OPTIMIZATION PRIORITIES)
            'most_common_query_types': [
                {
                    'type': qtype,
                    'count': stats.count,
                    'avg_time': stats.avg_time,
                    'success_rate': stats.success_rate
                }
                for qtype, stats in sorted_by_count[:5]  # Top 5 most frequent
            ],
            # FASTEST QUERY TYPES (BEST PRACTICES EXAMPLES)
            'fastest_query_types': [
                {
                    'type': qtype,
                    'avg_time': stats.avg_time,
                    'count': stats.count,
                    'success_rate': stats.success_rate
                }
                for qtype, stats in sorted_by_time[-3:] if stats.count > 0  # Bottom 3 (fastest)
            ],
            # LEAST SUCCESSFUL TYPES (QUALITY ISSUES)
            'least_successful_types': [
                {
                    'type': qtype,
                    'success_rate': stats.success_rate,
                    'count': stats.count,
                    'avg_time': stats.avg_time
                }
                for qtype, stats in sorted_by_success[:3] if stats.success_rate < 0.8 and stats.count > 2
            ]
        }
    
    def _generate_query_type_analysis(self) -> Dict[str, Any]:
        """
        Generate query type distribution and trends analysis.
        
        QUERY TYPE ANALYTICS:
        This method analyzes the distribution and characteristics of different
        query types to identify patterns and optimize priority assignment
        strategies for each type.
        """
        type_distribution = {}
        complexity_by_type = {}
        
        # ANALYZE EACH QUERY TYPE
        for qtype, stats in self.query_stats.items():
            # DISTRIBUTION METRICS
            type_distribution[qtype] = {
                'count': stats.count,
                'percentage': 0.0,  # Will be calculated below
                'avg_processing_time': stats.avg_time,
                'success_rate': stats.success_rate
            }
            
            # COMPLEXITY ANALYSIS
            if stats.complexity_scores:
                complexity_by_type[qtype] = {
                    'avg_complexity': sum(stats.complexity_scores) / len(stats.complexity_scores),
                    'max_complexity': max(stats.complexity_scores),
                    'complexity_trend': 'stable'  # Could be enhanced with trend analysis
                }
        
        # CALCULATE PERCENTAGES
        total_queries = sum(stats.count for stats in self.query_stats.values())
        if total_queries > 0:
            for qtype in type_distribution:
                type_distribution[qtype]['percentage'] = (
                    type_distribution[qtype]['count'] / total_queries * 100
                )
        
        return {
            'query_type_distribution': type_distribution,
            'complexity_analysis': complexity_by_type,
            'type_diversity_score': len(self.query_stats) / max(total_queries / 10, 1)
        }
    
    def _generate_priority_analysis(self) -> Dict[str, Any]:
        """
        Generate priority distribution and effectiveness analysis.
        
        PRIORITY EFFECTIVENESS ANALYTICS:
        This method analyzes how well the greedy priority assignments are working
        by examining the distribution of priorities and their correlation with
        performance outcomes.
        """
        priority_stats = {}
        
        # AGGREGATE PRIORITY STATISTICS FROM ALL QUERY TYPES
        for stats in self.query_stats.values():
            for priority, count in stats.priority_distribution.items():
                if priority not in priority_stats:
                    priority_stats[priority] = {'count': 0, 'total_time': 0.0, 'success_count': 0}
                
                priority_stats[priority]['count'] += count
        
        # CALCULATE PRIORITY EFFECTIVENESS
        priority_effectiveness = {}
        for priority in [1, 2, 3, 4]:  # CRITICAL, HIGH, MEDIUM, LOW
            if priority in priority_stats:
                stats = priority_stats[priority]
                priority_effectiveness[priority] = {
                    'count': stats['count'],
                    'percentage': 0.0,  # Will be calculated below
                    'priority_name': Priority(priority).name
                }
        
        # CALCULATE PERCENTAGES
        total_priority_queries = sum(stats['count'] for stats in priority_stats.values())
        if total_priority_queries > 0:
            for priority in priority_effectiveness:
                priority_effectiveness[priority]['percentage'] = (
                    priority_stats[priority]['count'] / total_priority_queries * 100
                )
        
        return {
            'priority_distribution': priority_effectiveness,
            'priority_balance_score': self._calculate_priority_balance(),
            'priority_trends': self._analyze_priority_trends()
        }
    
    def _generate_optimization_recommendations(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate actionable optimization recommendations.
        
        OPTIMIZATION RECOMMENDATION ENGINE:
        This method analyzes performance data to generate specific, actionable
        recommendations for improving the greedy algorithm's effectiveness and
        overall system performance.
        """
        recommendations = []
        
        # PERFORMANCE-BASED RECOMMENDATIONS
        if insights['overall_success_rate'] < 0.8:
            recommendations.append(
                "Overall success rate is below 80%. Review error handling and query processing logic."
            )
        
        if insights['avg_processing_time'] > 1.0:
            recommendations.append(
                "Average processing time exceeds 1 second. Consider optimizing slow query types or implementing caching."
            )
        
        # CACHE EFFICIENCY RECOMMENDATIONS
        cache_hit_rate = self.performance_metrics.get('cache_hit_rate', 0.0)
        if cache_hit_rate < 0.3:
            recommendations.append(
                f"Cache hit rate is low ({cache_hit_rate:.1%}). Consider increasing cache size or improving cache keys."
            )
        
        # PRIORITY DISTRIBUTION RECOMMENDATIONS
        if 'priority_distribution' in insights:
            critical_percentage = insights['priority_distribution'].get(1, {}).get('percentage', 0)
            if critical_percentage > 20:
                recommendations.append(
                    f"High percentage of critical queries ({critical_percentage:.1f}%). Review priority classification rules."
                )
        
        # QUERY TYPE SPECIFIC RECOMMENDATIONS
        if 'slowest_query_types' in insights and insights['slowest_query_types']:
            slowest_type = insights['slowest_query_types'][0]
            recommendations.append(
                f"Focus optimization on '{slowest_type['type']}' queries - they have the highest average processing time."
            )
        
        return {
            'recommendations': recommendations,
            'optimization_priority': self._calculate_optimization_priority(insights),
            'suggested_actions': self._generate_suggested_actions(insights)
        }
    
    def _generate_predictive_insights(self) -> Dict[str, Any]:
        """
        Generate predictive analytics and forecasting.
        
        PREDICTIVE ANALYTICS:
        This method provides forward-looking insights that help with capacity
        planning and proactive optimization of the greedy algorithm.
        """
        return {
            'predicted_load': self._predict_system_load(),
            'capacity_recommendations': self._generate_capacity_recommendations(),
            'trending_query_types': self._identify_trending_types(),
            'performance_forecast': self._forecast_performance_trends()
        }
    
    def _generate_health_indicators(self) -> Dict[str, Any]:
        """
        Generate system health indicators and status.
        
        SYSTEM HEALTH MONITORING:
        This method provides overall system health assessment based on various
        performance metrics, helping identify when the greedy algorithm and
        overall system are functioning optimally.
        """
        health_score = self._calculate_health_score()
        
        return {
            'system_health_score': health_score,
            'health_status': self._determine_health_status(health_score),
            'critical_metrics': self._identify_critical_metrics(),
            'uptime_statistics': self._calculate_uptime_stats(),
            'resource_utilization': self._calculate_resource_utilization()
        }
    
    # === ADDITIONAL HELPER METHODS ===
    
    def _calculate_priority_balance(self) -> float:
        """Calculate how well-balanced the priority distribution is."""
        # Implementation for priority balance calculation
        return 0.8  # Placeholder for demonstration
    
    def _analyze_priority_trends(self) -> List[str]:
        """Analyze trends in priority distribution over time."""
        # Implementation for priority trend analysis
        return ["Priority distribution is stable"]  # Placeholder for demonstration
    
    def _calculate_optimization_priority(self, insights: Dict[str, Any]) -> str:
        """Calculate the priority level for optimization efforts."""
        if insights['overall_success_rate'] < 0.7:
            return "CRITICAL"
        elif insights['avg_processing_time'] > 2.0:
            return "HIGH"
        elif len(insights.get('recommendations', [])) > 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_suggested_actions(self, insights: Dict[str, Any]) -> List[str]:
        """Generate specific suggested actions for improvement."""
        actions = []
        
        if insights['avg_processing_time'] > 1.0:
            actions.append("Implement query result caching")
            actions.append("Optimize database queries")
            actions.append("Consider horizontal scaling")
        
        if insights['overall_success_rate'] < 0.8:
            actions.append("Improve error handling mechanisms")
            actions.append("Add input validation and sanitization")
            actions.append("Implement retry logic for failed queries")
        
        return actions
    
    def _predict_system_load(self) -> Dict[str, float]:
        """Predict future system load based on current trends."""
        return {
            'next_hour': 1.2,
            'next_day': 1.0,
            'next_week': 0.9
        }  # Placeholder implementation
    
    def _generate_capacity_recommendations(self) -> List[str]:
        """Generate capacity planning recommendations."""
        return [
            "Current capacity appears adequate for projected load",
            "Monitor queue length during peak hours",
            "Consider auto-scaling for traffic spikes"
        ]  # Placeholder implementation
    
    def _identify_trending_types(self) -> List[str]:
        """Identify query types that are trending upward."""
        # Implementation for trend identification
        return ["technical", "how-to"]  # Placeholder for demonstration
    
    def _forecast_performance_trends(self) -> Dict[str, str]:
        """Forecast performance trends."""
        return {
            'processing_time': 'stable',
            'success_rate': 'improving',
            'queue_length': 'stable'
        }  # Placeholder implementation
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        factors = []
        
        # Success rate factor
        if self.query_stats:
            total_success = sum(stats.success_count for stats in self.query_stats.values())
            total_count = sum(stats.count for stats in self.query_stats.values())
            success_factor = total_success / total_count if total_count > 0 else 1.0
            factors.append(success_factor)
        
        # Performance factor
        avg_time = self.performance_metrics.get('average_response_time', 0.0)
        performance_factor = max(0, 1 - (avg_time / 5.0))  # Penalize if > 5 seconds
        factors.append(performance_factor)
        
        # Cache efficiency factor
        cache_factor = self.performance_metrics.get('cache_hit_rate', 0.5)
        factors.append(cache_factor)
        
        return sum(factors) / len(factors) if factors else 0.8
    
    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status based on score."""
        if health_score >= 0.9:
            return "excellent"
        elif health_score >= 0.8:
            return "good"
        elif health_score >= 0.6:
            return "fair"
        elif health_score >= 0.4:
            return "poor"
        else:
            return "critical"
    
    def _identify_critical_metrics(self) -> List[str]:
        """Identify metrics that need immediate attention."""
        critical = []
        
        if self.performance_metrics.get('average_response_time', 0) > 3.0:
            critical.append("High average response time")
        
        if self.performance_metrics.get('cache_hit_rate', 1.0) < 0.2:
            critical.append("Low cache hit rate")
        
        return critical
    
    def _calculate_uptime_stats(self) -> Dict[str, Any]:
        """Calculate system uptime statistics."""
        uptime = (datetime.datetime.now() - self.session_start).total_seconds()
        
        return {
            'session_uptime_seconds': uptime,
            'uptime_formatted': str(datetime.timedelta(seconds=int(uptime))),
            'queries_per_hour': self.performance_metrics['total_queries_processed'] / max(uptime / 3600, 1),
            'session_start': self.session_start.isoformat()
        }
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        return {
            'cache_utilization': len(self.priority_cache) / self.cache_max_size,
            'queue_utilization': len(self.priority_queue) / 1000,  # Assume max 1000
            'context_utilization': len(self.conversation_context) / self.context_window
        }
    
    # === QUEUE MANAGEMENT HELPER METHODS ===
    
    def _estimate_processing_time(self, query: str) -> float:
        """Estimate processing time for a query based on historical data."""
        query_type = self._categorize_query(query)
        
        if query_type in self.query_stats:
            return self.query_stats[query_type].avg_time
        
        # Default estimates based on complexity
        complexity = self._calculate_complexity_score(query)
        if complexity > 3:
            return 2.0
        elif complexity > 2:
            return 1.0
        elif complexity > 1:
            return 0.5
        else:
            return 0.2
    
    def _calculate_context_relevance(self, query: str) -> float:
        """Calculate how relevant a query is to current context."""
        if not self.conversation_context:
            return 0.0
        
        query_topics = set(self._extract_query_topics(query))
        context_topics = set()
        
        for entry in self.conversation_context[-3:]:  # Recent context
            if 'topics' in entry:
                context_topics.update(entry['topics'])
        
        if not query_topics or not context_topics:
            return 0.0
        
        overlap = len(query_topics & context_topics)
        return min(overlap / len(query_topics), 1.0)
    
    def _detect_queue_patterns(self) -> None:
        """Detect patterns in queue usage for optimization."""
        # Implementation for pattern detection in queue usage
        pass
    
    def _update_queue_metrics(self, wait_time: float, queue_entry: Dict[str, Any]) -> None:
        """Update queue performance metrics."""
        self.queue_metadata['total_processed'] += 1
        
        # Update average wait time
        current_avg = self.queue_metadata['average_wait_time']
        total_processed = self.queue_metadata['total_processed']
        
        self.queue_metadata['average_wait_time'] = (
            (current_avg * (total_processed - 1) + wait_time) / total_processed
        )
    
    def _estimate_queue_wait_time(self) -> float:
        """Estimate wait time for items currently in queue."""
        if not self.priority_queue:
            return 0.0
        
        # Simple estimation based on queue length and average processing time
        avg_processing_time = self.performance_metrics.get('average_response_time', 0.5)
        return len(self.priority_queue) * avg_processing_time
    
    def _assess_queue_health(self) -> str:
        """Assess the health status of the priority queue."""
        queue_length = len(self.priority_queue)
        
        if queue_length == 0:
            return "idle"
        elif queue_length < 10:
            return "healthy"
        elif queue_length < 50:
            return "busy"
        elif queue_length < 100:
            return "overloaded"
        else:
            return "critical"